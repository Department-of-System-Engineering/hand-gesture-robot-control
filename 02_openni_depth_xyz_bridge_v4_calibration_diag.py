import math
import os
import socket
import struct
import time

import cv2
import numpy as np
from openni import openni2


# OpenNI2 mélységfeldolgozó és XYZ-adatküldő a V4 kalibrációhoz.
# A MediaPipe külön folyamatban fut, és az 50011-es porton továbbítja
# a normalizált tenyérkoordinátákat. Az Astra XYZ [mm] az 50010-es,
# a mélységi minták diagnosztikája az 50012-es UDP-portra kerül.


# Az OpenNI2 redistributable elérési útja az OPENNI2_REDIST
# környezeti változóval állítható be, ha az alapértelmezett keresés sikertelen.
OPENNI2_REDIST = os.environ.get("OPENNI2_REDIST")

PALM_UDP_IP = "127.0.0.1"
PALM_UDP_PORT = 50011
PALM_PACKET_FORMAT = "<IffB"
PALM_PACKET_SIZE = struct.calcsize(
    PALM_PACKET_FORMAT
)

XYZ_UDP_IP = "127.0.0.1"
XYZ_UDP_PORT = 50010
XYZ_PACKET_FORMAT = "<IfffB"

# Kalibrációs diagnosztika, az alábbi mezősorrendben:
# bridge_seq, palm_seq, palm_x, palm_y,
# depth_px, depth_py, patch_valid_count,
# depth_median_mm, patch_min_mm, patch_max_mm, valid_xyz
DIAG_UDP_PORT = 50012
DIAG_PACKET_FORMAT = "<IIffiiHHHHB"

DEPTH_X_OFFSET_PX = -8
DEPTH_Y_OFFSET_PX = 0

# Mélységi kép és diagnosztikai segédjelölések.
CENTER_CROSS_HALF_PX = 12

REFERENCE_ZONE_CENTER_X_NORM = 0.58
REFERENCE_ZONE_CENTER_Y_NORM = 0.50
REFERENCE_ZONE_HALF_W_NORM = 0.075
REFERENCE_ZONE_HALF_H_NORM = 0.11

DEPTH_PATCH_RADIUS = 4

DISPLAY_MIN_MM = 450
DISPLAY_MAX_MM = 2500

MIN_VALID_DEPTH_MM = 500
MAX_VALID_DEPTH_MM = 2500

PALM_STALE_TIMEOUT_S = 0.25


def median_depth_mm(
    depth_mm,
    x,
    y,
    radius
):
    h, w = depth_mm.shape

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[patch > 0]

    if valid.size == 0:
        return 0

    return int(
        np.median(valid)
    )


def depth_patch_diagnostics(
    depth_mm,
    x,
    y,
    radius
):
    h, w = depth_mm.shape

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[patch > 0]

    if valid.size == 0:
        return 0, 0, 0, 0

    return (
        int(np.median(valid)),
        int(valid.size),
        int(np.min(valid)),
        int(np.max(valid)),
    )


def depth_to_colormap(depth_mm):
    valid_mask = depth_mm > 0

    clipped = np.clip(
        depth_mm,
        DISPLAY_MIN_MM,
        DISPLAY_MAX_MM
    ).astype(np.float32)

    norm = (
        (clipped - DISPLAY_MIN_MM)
        / (DISPLAY_MAX_MM - DISPLAY_MIN_MM)
    )

    norm = 1.0 - norm

    gray = (
        norm * 255.0
    ).astype(np.uint8)

    gray[~valid_mask] = 0

    view = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_TURBO
    )

    view[~valid_mask] = (
        0,
        0,
        0
    )

    return view


def rgb_norm_to_depth_pixel(
    palm_x_norm,
    palm_y_norm,
    depth_w,
    depth_h,
    offset_x,
    offset_y
):
    # A kipróbált képpontirányt megtartjuk, további X-tükrözés nélkül.
    dx = int(
        palm_x_norm * depth_w
    ) + offset_x

    dy = int(
        palm_y_norm * depth_h
    ) + offset_y

    dx = max(
        0,
        min(depth_w - 1, dx)
    )

    dy = max(
        0,
        min(depth_h - 1, dy)
    )

    return dx, dy


def metric_xyz_from_depth_pixel(
    px,
    py,
    z_mm,
    width,
    height,
    hfov_rad,
    vfov_rad
):
    fx = width / (
        2.0
        * math.tan(
            hfov_rad / 2.0
        )
    )

    fy = height / (
        2.0
        * math.tan(
            vfov_rad / 2.0
        )
    )

    cx = (
        width - 1
    ) / 2.0

    cy = (
        height - 1
    ) / 2.0

    x_mm = (
        (px - cx)
        * z_mm
        / fx
    )

    y_mm = (
        (py - cy)
        * z_mm
        / fy
    )

    return (
        float(x_mm),
        float(y_mm),
        float(z_mm)
    )


def drain_palm_udp(sock):
    latest = None

    while True:
        try:
            data, _ = sock.recvfrom(
                128
            )
        except BlockingIOError:
            break

        if (
            len(data)
            != PALM_PACKET_SIZE
        ):
            continue

        latest = struct.unpack(
            PALM_PACKET_FORMAT,
            data
        )

    return latest


def send_xyz(
    sock,
    sequence,
    x_mm,
    y_mm,
    z_mm,
    valid
):
    packet = struct.pack(
        XYZ_PACKET_FORMAT,
        int(sequence) & 0xFFFFFFFF,
        float(x_mm),
        float(y_mm),
        float(z_mm),
        1 if valid else 0
    )

    sock.sendto(
        packet,
        (
            XYZ_UDP_IP,
            XYZ_UDP_PORT
        )
    )



def send_diag(
    sock,
    bridge_seq,
    palm_seq,
    palm_x,
    palm_y,
    depth_px,
    depth_py,
    patch_valid_count,
    depth_median_mm,
    patch_min_mm,
    patch_max_mm,
    valid_xyz
):
    packet = struct.pack(
        DIAG_PACKET_FORMAT,
        int(bridge_seq) & 0xFFFFFFFF,
        int(palm_seq) & 0xFFFFFFFF,
        float(palm_x),
        float(palm_y),
        int(depth_px),
        int(depth_py),
        int(max(0, min(65535, patch_valid_count))),
        int(max(0, min(65535, depth_median_mm))),
        int(max(0, min(65535, patch_min_mm))),
        int(max(0, min(65535, patch_max_mm))),
        1 if valid_xyz else 0,
    )

    sock.sendto(
        packet,
        (
            XYZ_UDP_IP,
            DIAG_UDP_PORT,
        )
    )


def draw_center_guides(
    frame
):
    h, w = frame.shape[:2]

    cx = int(
        w / 2
    )
    cy = int(
        h / 2
    )

    # A kamera képének geometriai közepe.
    cv2.line(
        frame,
        (
            cx - CENTER_CROSS_HALF_PX,
            cy
        ),
        (
            cx + CENTER_CROSS_HALF_PX,
            cy
        ),
        (
            255,
            255,
            255
        ),
        1
    )

    cv2.line(
        frame,
        (
            cx,
            cy - CENTER_CROSS_HALF_PX
        ),
        (
            cx,
            cy + CENTER_CROSS_HALF_PX
        ),
        (
            255,
            255,
            255
        ),
        1
    )

    # Az RGB-ablakban jelölt referenciaterület megfelelője.
    zcx = int(
        REFERENCE_ZONE_CENTER_X_NORM
        * w
    )

    zcy = int(
        REFERENCE_ZONE_CENTER_Y_NORM
        * h
    )

    hw = int(
        REFERENCE_ZONE_HALF_W_NORM
        * w
    )

    hh = int(
        REFERENCE_ZONE_HALF_H_NORM
        * h
    )

    x0 = max(
        0,
        zcx - hw
    )
    x1 = min(
        w - 1,
        zcx + hw
    )
    y0 = max(
        0,
        zcy - hh
    )
    y1 = min(
        h - 1,
        zcy + hh
    )

    cv2.rectangle(
        frame,
        (
            x0,
            y0
        ),
        (
            x1,
            y1
        ),
        (
            230,
            230,
            230
        ),
        1
    )

    hatch_step = 18

    start = (
        x0
        - (y1 - y0)
    )

    for sx in range(
        start,
        x1 + hatch_step,
        hatch_step
    ):
        p1x = max(
            x0,
            sx
        )

        p1y = y1 - max(
            0,
            x0 - sx
        )

        p2x = min(
            x1,
            sx + (y1 - y0)
        )

        p2y = y1 - (
            p2x - sx
        )

        if (
            x0 <= p1x <= x1
            and y0 <= p1y <= y1
            and x0 <= p2x <= x1
            and y0 <= p2y <= y1
        ):
            cv2.line(
                frame,
                (
                    int(p1x),
                    int(p1y)
                ),
                (
                    int(p2x),
                    int(p2y)
                ),
                (
                    180,
                    180,
                    180
                ),
                1
            )


def main():
    print("=" * 78)
    print("2/3 OPENNI DEPTH -> XYZ UDP BRIDGE")
    print("=" * 78)
    print()
    print(
        f"Palm be: {PALM_UDP_IP}:{PALM_UDP_PORT}"
    )

    print(
        f"XYZ ki:  {XYZ_UDP_IP}:{XYZ_UDP_PORT}"
    )
    print()

    palm_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    palm_sock.bind(
        (
            PALM_UDP_IP,
            PALM_UDP_PORT
        )
    )

    palm_sock.setblocking(
        False
    )

    xyz_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    diag_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    depth_stream = None
    openni_loaded = False

    offset_x = DEPTH_X_OFFSET_PX
    offset_y = DEPTH_Y_OFFSET_PX

    latest_palm = None
    latest_palm_time = None
    latest_palm_seq = 0

    sequence = 0

    try:
        print("OpenNI2 inicializalas...")

        if OPENNI2_REDIST:
            openni2.initialize(
                OPENNI2_REDIST
            )
        else:
            openni2.initialize()

        openni_loaded = True

        device = (
            openni2.Device.open_any()
        )

        depth_stream = (
            device.create_depth_stream()
        )

        depth_stream.start()

        try:
            hfov = float(
                depth_stream
                .get_horizontal_fov()
            )

            vfov = float(
                depth_stream
                .get_vertical_fov()
            )

            print(
                "[OK] Depth | "
                f"HFOV={math.degrees(hfov):.1f} | "
                f"VFOV={math.degrees(vfov):.1f}"
            )

        except Exception:
            hfov = math.radians(
                60.0
            )

            vfov = math.radians(
                49.5
            )

            print(
                "[WARN] FOV fallback"
            )

        print()

        cv2.namedWindow(
            "2 - OpenNI Depth XYZ Bridge V4 DIAG",
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            "2 - OpenNI Depth XYZ Bridge V4 DIAG",
            800,
            600
        )

        print(
            "ESC = kilepes"
        )
        print(
            "J/L = depth pont balra/jobbra"
        )
        print(
            "I/K = depth pont fel/le"
        )
        print(
            "R   = offset nullazas"
        )
        print()

        last_frame_time = (
            time.perf_counter()
        )

        fps = 0.0

        while True:
            now_mono = (
                time.monotonic()
            )

            palm_packet = (
                drain_palm_udp(
                    palm_sock
                )
            )

            if palm_packet is not None:
                (
                    palm_seq,
                    palm_x,
                    palm_y,
                    palm_valid
                ) = palm_packet

                latest_palm_time = (
                    now_mono
                )

                latest_palm_seq = int(
                    palm_seq
                )

                if palm_valid:
                    latest_palm = (
                        float(palm_x),
                        float(palm_y)
                    )
                else:
                    latest_palm = None

            palm_fresh = (
                latest_palm is not None
                and latest_palm_time is not None
                and (
                    now_mono
                    - latest_palm_time
                    <= PALM_STALE_TIMEOUT_S
                )
            )

            depth_frame = (
                depth_stream.read_frame()
            )

            depth_w = (
                depth_frame.width
            )

            depth_h = (
                depth_frame.height
            )

            depth_buffer = (
                depth_frame
                .get_buffer_as_uint16()
            )

            # Az OpenNI natív pufferéről másolat készül a saját NumPy-memóriába,
            # így a további feldolgozás nem hivatkozik az eredeti pufferre.
            depth_mm = np.frombuffer(
                depth_buffer,
                dtype=np.uint16
            ).reshape(
                (
                    depth_h,
                    depth_w
                )
            ).copy()

            depth_view = (
                depth_to_colormap(
                    depth_mm
                )
            )

            draw_center_guides(
                depth_view
            )

            valid_xyz = False

            x_mm = 0.0
            y_mm = 0.0
            z_mm = 0.0

            depth_value_mm = 0
            depth_px = -1
            depth_py = -1
            patch_valid_count = 0
            patch_min_mm = 0
            patch_max_mm = 0
            diag_palm_x = 0.0
            diag_palm_y = 0.0

            if palm_fresh:
                (
                    palm_x,
                    palm_y
                ) = latest_palm

                (
                    depth_px,
                    depth_py
                ) = (
                    rgb_norm_to_depth_pixel(
                        palm_x,
                        palm_y,
                        depth_w,
                        depth_h,
                        offset_x,
                        offset_y
                    )
                )

                diag_palm_x = float(
                    palm_x
                )
                diag_palm_y = float(
                    palm_y
                )

                (
                    depth_value_mm,
                    patch_valid_count,
                    patch_min_mm,
                    patch_max_mm,
                ) = depth_patch_diagnostics(
                    depth_mm,
                    depth_px,
                    depth_py,
                    DEPTH_PATCH_RADIUS
                )

                if (
                    MIN_VALID_DEPTH_MM
                    <= depth_value_mm
                    <= MAX_VALID_DEPTH_MM
                ):
                    (
                        x_mm,
                        y_mm,
                        z_mm
                    ) = (
                        metric_xyz_from_depth_pixel(
                            depth_px,
                            depth_py,
                            depth_value_mm,
                            depth_w,
                            depth_h,
                            hfov,
                            vfov
                        )
                    )

                    valid_xyz = True

                cv2.circle(
                    depth_view,
                    (
                        depth_px,
                        depth_py
                    ),
                    10,
                    (
                        255,
                        255,
                        255
                    ),
                    2
                )

                cv2.line(
                    depth_view,
                    (
                        depth_px - 15,
                        depth_py
                    ),
                    (
                        depth_px + 15,
                        depth_py
                    ),
                    (
                        255,
                        255,
                        255
                    ),
                    1
                )

                cv2.line(
                    depth_view,
                    (
                        depth_px,
                        depth_py - 15
                    ),
                    (
                        depth_px,
                        depth_py + 15
                    ),
                    (
                        255,
                        255,
                        255
                    ),
                    1
                )

            sequence += 1

            send_xyz(
                xyz_sock,
                sequence,
                x_mm,
                y_mm,
                z_mm,
                valid_xyz
            )

            send_diag(
                diag_sock,
                sequence,
                latest_palm_seq,
                diag_palm_x,
                diag_palm_y,
                depth_px,
                depth_py,
                patch_valid_count,
                depth_value_mm,
                patch_min_mm,
                patch_max_mm,
                valid_xyz
            )

            frame_now = (
                time.perf_counter()
            )

            dt = (
                frame_now
                - last_frame_time
            )

            last_frame_time = (
                frame_now
            )

            if dt > 0:
                inst = 1.0 / dt

                fps = (
                    0.90 * fps
                    + 0.10 * inst
                    if fps > 0
                    else inst
                )

            status = (
                "XYZ OK -> ROBOT UDP"
                if valid_xyz
                else (
                    "VAROM A PALM UDP-T"
                    if not palm_fresh
                    else "INVALID DEPTH"
                )
            )

            cv2.putText(
                depth_view,
                status,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    255,
                    255,
                    255
                ),
                2
            )

            cv2.putText(
                depth_view,
                (
                    f"Depth: "
                    f"{depth_value_mm} mm"
                ),
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    255,
                    255,
                    255
                ),
                2
            )

            if valid_xyz:
                cv2.putText(
                    depth_view,
                    (
                        "XYZ [mm]: "
                        f"{x_mm:+.1f}, "
                        f"{y_mm:+.1f}, "
                        f"{z_mm:+.1f}"
                    ),
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (
                        255,
                        255,
                        255
                    ),
                    2
                )

            cv2.putText(
                depth_view,
                (
                    f"Offset: "
                    f"{offset_x:+d}/"
                    f"{offset_y:+d}px | "
                    f"FPS: {fps:.1f}"
                ),
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (
                    255,
                    255,
                    255
                ),
                2
            )

            cv2.imshow(
                "2 - OpenNI Depth XYZ Bridge V4 DIAG",
                depth_view
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == 27:
                break

            elif key in (
                ord("j"),
                ord("J")
            ):
                offset_x -= 2

            elif key in (
                ord("l"),
                ord("L")
            ):
                offset_x += 2

            elif key in (
                ord("i"),
                ord("I")
            ):
                offset_y -= 2

            elif key in (
                ord("k"),
                ord("K")
            ):
                offset_y += 2

            elif key in (
                ord("r"),
                ord("R")
            ):
                offset_x = 0
                offset_y = 0

    except Exception as e:
        print()
        print("[PYTHON HIBA]")
        print(type(e).__name__ + ":", e)

    finally:
        try:
            sequence += 1

            send_xyz(
                xyz_sock,
                sequence,
                0.0,
                0.0,
                0.0,
                False
            )
        except Exception:
            pass

        try:
            palm_sock.close()
        except Exception:
            pass

        try:
            xyz_sock.close()
        except Exception:
            pass

        try:
            diag_sock.close()
        except Exception:
            pass

        if depth_stream is not None:
            try:
                depth_stream.stop()
            except Exception:
                pass

        if openni_loaded:
            try:
                openni2.unload()
            except Exception:
                pass

        cv2.destroyAllWindows()

        print("Depth bridge vege.")


if __name__ == "__main__":
    main()
