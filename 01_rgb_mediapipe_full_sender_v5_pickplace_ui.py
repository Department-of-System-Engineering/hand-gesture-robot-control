import math
import socket
import struct
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# ASTRA RGB + MEDIAPIPE FULL HAND SENDER V5 PICKPLACE UI
#
# PORT 50011:
#   regi palm X/Y packet -> a stabil OpenNI depth bridge-nek
#
# PORT 50012:
#   3x3 kezorientacios matrix + normalizalt pinch ratio
#   -> a fo launchernek
#
# Egyetlen process nyitja az Astra RGB kamerat.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"

UDP_IP = "127.0.0.1"

PALM_UDP_PORT = 50011
PALM_PACKET_FORMAT = "<IffB"

FEATURE_UDP_PORT = 50012
# seq + 9 db matrix float + pinch_ratio + valid
FEATURE_PACKET_FORMAT = "<I10fB"

# Launcher -> RGB status:
# tracking_on, gripper_on
STATUS_UDP_PORT = 50013
STATUS_PACKET_FORMAT = "<BB"
STATUS_PACKET_SIZE = struct.calcsize(STATUS_PACKET_FORMAT)
PICK_STATUS_PACKET_FORMAT = "<BBBBBBBf"
PICK_STATUS_PACKET_SIZE = struct.calcsize(PICK_STATUS_PACKET_FORMAT)

# Shadow-only palm stream a fo programnak. A depth tovabbra is egyedul
# az 50011-et fogadja. A fo program SOHA nem bindolja azt.
SHADOW_UDP_PORT = 50014
SHADOW_PACKET_FORMAT = "<IffB"

# ------------------------------------------------------------
# DEMO UI
# ------------------------------------------------------------
# Pontosan a kamera kepkozepe.
CENTER_CROSS_HALF_PX = 12

# Ajanlott C-referencia-zona:
# szandekosan kicsit JOBBRA van a geometriai kozephez kepest,
# hogy balra nagyobb hasznalhato keztartalek maradjon.
REFERENCE_ZONE_CENTER_X_NORM = 0.58
REFERENCE_ZONE_CENTER_Y_NORM = 0.50
REFERENCE_ZONE_HALF_W_NORM = 0.075
REFERENCE_ZONE_HALF_H_NORM = 0.11

PALM_IDS = [0, 5, 9, 13, 17]

PALM_FILTER_ALPHA = 0.25
PINCH_FILTER_ALPHA = 0.22

PREFERRED_NAME_PARTS = [
    "orbbec",
    "astra",
]

FALLBACK_EXACT_NAMES = [
    "USB Camera",
]


def find_astra_camera_index():
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        raise RuntimeError(
            "Hianyzik a pygrabber.\n"
            "Telepites: pip install pygrabber"
        )

    devices = FilterGraph().get_input_devices()

    print()
    print("DirectShow kamerak:")

    for index, name in enumerate(devices):
        print(f"  [{index}] {name}")

    for index, name in enumerate(devices):
        lower = name.lower()

        if any(
            part in lower
            for part in PREFERRED_NAME_PARTS
        ):
            print()
            print(
                f"[OK] Astra RGB: [{index}] {name}"
            )
            return index

    exact_matches = [
        (index, name)
        for index, name in enumerate(devices)
        if name.strip() in FALLBACK_EXACT_NAMES
    ]

    if len(exact_matches) == 1:
        index, name = exact_matches[0]

        print()
        print(
            f"[OK] Astra RGB: [{index}] {name}"
        )
        return index

    raise RuntimeError(
        "Nem talaltam egyertelmuen az Astra RGB kamerat.\n"
        f"Eszkozok: {devices}"
    )


def create_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return HandLandmarker.create_from_options(
        options
    )


def normalize(v):
    v = np.asarray(
        v,
        dtype=np.float64
    )

    n = float(
        np.linalg.norm(v)
    )

    if (
        not np.isfinite(n)
        or n < 1e-8
    ):
        raise ValueError(
            "Ervenytelen orientacios vektor."
        )

    return v / n


def hand_rotation_matrix(
    world_landmarks
):
    pts = np.array(
        [
            [
                float(lm.x),
                float(lm.y),
                float(lm.z)
            ]
            for lm in world_landmarks
        ],
        dtype=np.float64
    )

    if pts.shape[0] < 18:
        raise ValueError(
            "Keves world landmark."
        )

    wrist = pts[0]
    index_mcp = pts[5]
    middle_mcp = pts[9]
    pinky_mcp = pts[17]

    x_axis = normalize(
        index_mcp - pinky_mcp
    )

    y_raw = normalize(
        middle_mcp - wrist
    )

    z_axis = normalize(
        np.cross(
            x_axis,
            y_raw
        )
    )

    y_axis = normalize(
        np.cross(
            z_axis,
            x_axis
        )
    )

    return np.column_stack(
        (
            x_axis,
            y_axis,
            z_axis
        )
    )


def distance3(a, b):
    return math.sqrt(
        (float(a.x) - float(b.x)) ** 2
        + (float(a.y) - float(b.y)) ** 2
        + (float(a.z) - float(b.z)) ** 2
    )


def pinch_ratio(
    world_landmarks
):
    pinch = distance3(
        world_landmarks[4],
        world_landmarks[8]
    )

    palm_width = distance3(
        world_landmarks[5],
        world_landmarks[17]
    )

    if palm_width < 1e-8:
        raise ValueError(
            "Ervenytelen tenyermetret."
        )

    return pinch / palm_width


def draw_hand(
    frame,
    landmarks
):
    h, w = frame.shape[:2]

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    pts = [
        (
            int(lm.x * w),
            int(lm.y * h)
        )
        for lm in landmarks
    ]

    for a, b in connections:
        cv2.line(
            frame,
            pts[a],
            pts[b],
            (180, 180, 180),
            2
        )

    for p in pts:
        cv2.circle(
            frame,
            p,
            3,
            (0, 255, 0),
            -1
        )

    palm_x = int(
        np.mean(
            [
                pts[i][0]
                for i in PALM_IDS
            ]
        )
    )

    palm_y = int(
        np.mean(
            [
                pts[i][1]
                for i in PALM_IDS
            ]
        )
    )

    cv2.circle(
        frame,
        (palm_x, palm_y),
        8,
        (0, 0, 255),
        -1
    )

    # Pinch pontok.
    cv2.circle(
        frame,
        pts[4],
        8,
        (0, 255, 255),
        -1
    )

    cv2.circle(
        frame,
        pts[8],
        8,
        (0, 255, 255),
        -1
    )

    cv2.line(
        frame,
        pts[4],
        pts[8],
        (0, 0, 255),
        2
    )


def send_palm(
    sock,
    sequence,
    palm_x,
    palm_y,
    valid
):
    packet = struct.pack(
        PALM_PACKET_FORMAT,
        int(sequence) & 0xFFFFFFFF,
        float(palm_x),
        float(palm_y),
        1 if valid else 0
    )

    sock.sendto(
        packet,
        (
            UDP_IP,
            PALM_UDP_PORT
        )
    )


def send_shadow_palm(sock, sequence, palm_x, palm_y, valid):
    packet = struct.pack(
        SHADOW_PACKET_FORMAT, int(sequence), float(palm_x), float(palm_y), int(valid)
    )
    try:
        sock.sendto(packet, (UDP_IP, SHADOW_UDP_PORT))
    except Exception:
        pass


def send_features(
    sock,
    sequence,
    rotation_matrix,
    pinch,
    valid
):
    if valid:
        values = [
            float(x)
            for x in rotation_matrix.reshape(-1)
        ]

        pinch_value = float(
            pinch
        )

    else:
        values = [
            0.0
        ] * 9

        pinch_value = 0.0

    packet = struct.pack(
        FEATURE_PACKET_FORMAT,
        int(sequence) & 0xFFFFFFFF,
        *values,
        pinch_value,
        1 if valid else 0
    )

    sock.sendto(
        packet,
        (
            UDP_IP,
            FEATURE_UDP_PORT
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

    # Kamera geometriai kozepe: vekony kereszt.
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

    # Ajanlott C-zona.
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

    # Finom ferde vonalkazas, hogy ne legyen osszekeverheto
    # a tenyleges landmarkokkal.
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

    cv2.putText(
        frame,
        "C ZONA",
        (
            x0 + 4,
            max(
                16,
                y0 - 5
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (
            255,
            255,
            255
        ),
        1
    )


def drain_status_udp(sock, previous):
    latest = dict(previous)
    while True:
        try:
            data, _ = sock.recvfrom(128)
        except BlockingIOError:
            break
        if len(data) == STATUS_PACKET_SIZE:
            tracking_on, gripper_on = struct.unpack(STATUS_PACKET_FORMAT, data)
            latest["tracking"] = bool(tracking_on)
            latest["gripper"] = bool(gripper_on)
        elif len(data) == PICK_STATUS_PACKET_SIZE:
            (tracking_on, gripper_on, mode, source, flow, trial, gate, loss_elapsed) = struct.unpack(
                PICK_STATUS_PACKET_FORMAT, data
            )
            latest.update({
                "tracking": bool(tracking_on),
                "gripper": bool(gripper_on),
                "mode": int(mode),
                "source": int(source),
                "flow": int(flow),
                "trial": int(trial),
                "gate": int(gate),
                "loss_elapsed": float(loss_elapsed),
            })
    return latest

def draw_demo_status(frame, status):
    h, w = frame.shape[:2]
    x0 = max(0, w - 290)
    y0 = 8
    panel_h = 132
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (w-8, y0+panel_h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    mode_txt = "DUAL" if status.get("mode") == 2 else "ASTRA"
    source_names = {0:"IDLE",1:"ASTRA",2:"TOP",3:"HOLD",4:"PAUSE"}
    src_txt = source_names.get(status.get("source",0), "?")
    flow_txt = "FLOW" if status.get("flow",1) else "STRICT"
    lines = [
        f"MODE {mode_txt} | SRC {src_txt}",
        f"TRACK {'ON' if status.get('tracking') else 'OFF'} | RG2 {'ON' if status.get('gripper') else 'OFF'}",
        f"REACQ {flow_txt} | TRIAL {'ON' if status.get('trial') else 'OFF'}",
        f"TOP GATE {'OK' if status.get('gate') else 'NO'} | LOSS {status.get('loss_elapsed',0.0):.2f}s",
    ]
    for i,line in enumerate(lines):
        cv2.putText(frame,line,(x0+10,y0+28+i*27),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,255),1,cv2.LINE_AA)


def main():

    print("=" * 78)
    print("ASTRA RGB + MEDIAPIPE FULL HAND SENDER V5 PICKPLACE UI")
    print("=" * 78)
    print()
    print(
        f"Palm UDP:    {UDP_IP}:{PALM_UDP_PORT}"
    )
    print(
        f"Feature UDP: {UDP_IP}:{FEATURE_UDP_PORT}"
    )

    cap = None
    landmarker = None

    udp_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    status_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    status_sock.bind(
        (
            UDP_IP,
            STATUS_UDP_PORT
        )
    )

    status_sock.setblocking(
        False
    )

    status = {
        "tracking": False,
        "gripper": False,
        "mode": 1,
        "source": 0,
        "flow": 1,
        "trial": 0,
        "gate": 0,
        "loss_elapsed": 0.0,
    }

    filtered_x = None
    filtered_y = None
    filtered_pinch = None

    sequence = 0

    try:
        camera_index = (
            find_astra_camera_index()
        )

        print()
        print("MediaPipe inicializalas...")

        landmarker = (
            create_landmarker()
        )

        print("[OK] MediaPipe")

        cap = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW
        )

        if not cap.isOpened():
            raise RuntimeError(
                "Astra RGB nem nyithato."
            )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        print("[OK] Astra RGB")
        print()
        print("ESC = kilepes")

        cv2.namedWindow(
            "Astra Full Hand Sender V5 PICKPLACE",
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            "Astra Full Hand Sender V5 PICKPLACE",
            960,
            720
        )

        last_time = time.perf_counter()
        fps = 0.0

        while True:
            ok, frame = (
                cap.read()
            )

            if not ok:
                raise RuntimeError(
                    "RGB frame olvasasi hiba."
                )

            frame = cv2.flip(
                frame,
                1
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                time.monotonic()
                * 1000
            )

            result = (
                landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

            palm_valid = False
            feature_valid = False

            palm_x = 0.0
            palm_y = 0.0

            rotation_matrix = (
                np.eye(
                    3,
                    dtype=np.float64
                )
            )

            pinch_value = 0.0

            if result.hand_landmarks:
                landmarks = (
                    result.hand_landmarks[0]
                )

                draw_hand(
                    frame,
                    landmarks
                )

                palm_x_raw = float(
                    np.mean(
                        [
                            landmarks[i].x
                            for i in PALM_IDS
                        ]
                    )
                )

                palm_y_raw = float(
                    np.mean(
                        [
                            landmarks[i].y
                            for i in PALM_IDS
                        ]
                    )
                )

                if filtered_x is None:
                    filtered_x = (
                        palm_x_raw
                    )

                    filtered_y = (
                        palm_y_raw
                    )

                else:
                    filtered_x = (
                        PALM_FILTER_ALPHA
                        * palm_x_raw
                        + (
                            1.0
                            - PALM_FILTER_ALPHA
                        )
                        * filtered_x
                    )

                    filtered_y = (
                        PALM_FILTER_ALPHA
                        * palm_y_raw
                        + (
                            1.0
                            - PALM_FILTER_ALPHA
                        )
                        * filtered_y
                    )

                palm_x = filtered_x
                palm_y = filtered_y
                palm_valid = True

            else:
                filtered_x = None
                filtered_y = None

            if (
                result.hand_landmarks
                and result.hand_world_landmarks
            ):
                try:
                    world = (
                        result.hand_world_landmarks[0]
                    )

                    rotation_matrix = (
                        hand_rotation_matrix(
                            world
                        )
                    )

                    pinch_raw = (
                        pinch_ratio(
                            world
                        )
                    )

                    if filtered_pinch is None:
                        filtered_pinch = (
                            pinch_raw
                        )

                    else:
                        filtered_pinch = (
                            PINCH_FILTER_ALPHA
                            * pinch_raw
                            + (
                                1.0
                                - PINCH_FILTER_ALPHA
                            )
                            * filtered_pinch
                        )

                    pinch_value = (
                        filtered_pinch
                    )

                    feature_valid = True

                except Exception:
                    feature_valid = False

            else:
                filtered_pinch = None

            sequence += 1

            send_palm(
                udp_sock, sequence, palm_x, palm_y, palm_valid
            )
            send_shadow_palm(
                udp_sock, sequence, palm_x, palm_y, palm_valid
            )

            send_features(
                udp_sock,
                sequence,
                rotation_matrix,
                pinch_value,
                feature_valid
            )

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            if dt > 0:
                instant = 1.0 / dt

                fps = (
                    0.90 * fps
                    + 0.10 * instant
                    if fps > 0
                    else instant
                )

            status = drain_status_udp(status_sock, status)

            draw_center_guides(
                frame
            )

            draw_demo_status(frame, status)

            cv2.putText(
                frame,
                (
                    "PALM OK / 3D FEATURE OK"
                    if (
                        palm_valid
                        and feature_valid
                    )
                    else (
                        "PALM OK / FEATURE LOST"
                        if palm_valid
                        else "NINCS KEZ"
                    )
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"PALM u={palm_x:.3f} v={palm_y:.3f}",
                (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"PINCH ratio: {pinch_value:.3f}",
                (20, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | SHADOW->50014",
                (20, 126),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Astra Full Hand Sender V5 PICKPLACE",
                frame
            )

            if (
                cv2.waitKey(1)
                & 0xFF
                == 27
            ):
                break

    except Exception as e:
        print()
        print("[HIBA]")
        print(
            type(e).__name__ + ":",
            e
        )

    finally:
        try:
            sequence += 1

            send_palm(
                udp_sock,
                sequence,
                0.0,
                0.0,
                False
            )

            send_features(
                udp_sock,
                sequence,
                np.eye(3),
                0.0,
                False
            )

        except Exception:
            pass

        try:
            udp_sock.close()
        except Exception:
            pass

        try:
            status_sock.close()
        except Exception:
            pass

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass

        cv2.destroyAllWindows()

        print(
            "RGB full sender vege."
        )


if __name__ == "__main__":
    main()
