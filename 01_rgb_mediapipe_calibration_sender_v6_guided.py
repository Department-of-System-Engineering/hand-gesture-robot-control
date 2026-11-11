"""Astra RGB-kézdetektálás a háromsíkú kamerakalibrációhoz.

Az Orbbec/Astra RGB-kamerát DirectShow eszköznév alapján választja ki.
Képolvasási hiba esetén megkísérli a kamera újranyitását. A MediaPipe
által meghatározott tenyérpontot az 50011-es UDP-porton továbbítja a
mélységi folyamatnak. Az 50031-es porton fogadja a kalibrációs állapotot.

A kezelőfelület a LOW, MID és HIGH síkot, a rögzített pontok számát,
a következő ajánlott 3×3 rácspontot és a mentés visszajelzését mutatja.
A kamera újranyitásainak eseménynaplója az
``astra_rgb_reconnect_log_v6.csv`` fájlba kerül.
"""

import csv
import json
import socket
import struct
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"

UDP_IP = "127.0.0.1"

PALM_UDP_PORT = 50011
PALM_PACKET_FORMAT = "<IffB"

CAL_STATUS_PORT = 50031
CAL_KEY_PORT = 50033

RECONNECT_LOG = BASE_DIR / "astra_rgb_reconnect_log_v6.csv"

PALM_IDS = [0, 5, 9, 13, 17]
PALM_FILTER_ALPHA = 0.25

PREFERRED_NAME_PARTS = [
    "orbbec",
    "astra",
]

FALLBACK_EXACT_NAMES = [
    "USB Camera",
]

WIDTH = 640
HEIGHT = 480
FPS = 30

REOPEN_DELAY_S = 0.50
INVALID_SEND_PERIOD_S = 0.10

# Az Astra-kép középső részén kijelölt kalibrációs terület.
GUIDE_X1 = 0.22
GUIDE_X2 = 0.78
GUIDE_Y1 = 0.22
GUIDE_Y2 = 0.78

FLASH_DURATION_S = 1.50


def outlined_text(
    frame,
    text,
    pos,
    scale=0.65,
    color=(255, 255, 255),
    thickness=2,
):
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 4,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def find_astra_camera():
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        raise RuntimeError(
            "Hianyzik a pygrabber. Telepites: pip install pygrabber"
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
            print(f"[OK] Astra RGB: [{index}] {name}")
            return index, name

    exact_matches = [
        (index, name)
        for index, name in enumerate(devices)
        if name.strip() in FALLBACK_EXACT_NAMES
    ]

    if len(exact_matches) == 1:
        index, name = exact_matches[0]
        print()
        print(f"[OK] Astra RGB fallback: [{index}] {name}")
        return index, name

    raise RuntimeError(
        "Nem talaltam egyertelmuen az Astra RGB kamerat. "
        f"Eszkozok: {devices}"
    )


def create_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return HandLandmarker.create_from_options(options)


def open_astra_camera():
    index, name = find_astra_camera()

    print(
        f"[INFO] Astra RGB nyitas: DirectShow index={index} | {name}"
    )

    cap = cv2.VideoCapture(
        index,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Astra RGB nem nyithato DirectShow-val: [{index}] {name}"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH,
    )
    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT,
    )
    cap.set(
        cv2.CAP_PROP_FPS,
        FPS,
    )
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    for _ in range(40):
        ok, frame = cap.read()

        if ok and frame is not None:
            print(f"[OK] Astra RGB megnyitva: [{index}] {name}")
            return cap, index, name

        time.sleep(0.05)

    cap.release()

    raise RuntimeError(
        f"Astra RGB megnyilt, de nem adott kepet: [{index}] {name}"
    )


def send_palm(
    sock,
    seq,
    x,
    y,
    valid,
):
    packet = struct.pack(
        PALM_PACKET_FORMAT,
        int(seq),
        float(x),
        float(y),
        int(bool(valid)),
    )

    sock.sendto(
        packet,
        (
            UDP_IP,
            PALM_UDP_PORT,
        ),
    )


def drain_cal_status(
    sock,
    previous,
):
    latest = previous

    while True:
        try:
            data, _ = sock.recvfrom(4096)
        except BlockingIOError:
            break

        try:
            payload = json.loads(
                data.decode("utf-8")
            )
        except Exception:
            continue

        latest = payload

    return latest


def guide_points_px(frame):
    h, w = frame.shape[:2]

    xs = [
        GUIDE_X1,
        (GUIDE_X1 + GUIDE_X2) * 0.5,
        GUIDE_X2,
    ]

    ys = [
        GUIDE_Y1,
        (GUIDE_Y1 + GUIDE_Y2) * 0.5,
        GUIDE_Y2,
    ]

    pts = []

    for yy in ys:
        for xx in xs:
            pts.append(
                (
                    int(round(xx * w)),
                    int(round(yy * h)),
                )
            )

    return pts


def draw_calibration_ui(
    frame,
    status,
    local_hand_ok,
    flash_state,
):
    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # Kalibrációs segédkeret és 3×3 ajánlott pontháló.
    # --------------------------------------------------------
    x1 = int(round(GUIDE_X1 * w))
    x2 = int(round(GUIDE_X2 * w))
    y1 = int(round(GUIDE_Y1 * h))
    y2 = int(round(GUIDE_Y2 * h))

    can_save = bool(
        status.get(
            "can_save",
            False,
        )
    )

    guide_color = (
        (0, 220, 0)
        if can_save
        else (0, 215, 255)
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        guide_color,
        2,
        cv2.LINE_AA,
    )

    pts = guide_points_px(frame)

    plane_counts = status.get(
        "counts",
        {
            "LOW": 0,
            "MID": 0,
            "HIGH": 0,
        },
    )

    active_plane = str(
        status.get(
            "plane",
            "MID",
        )
    )

    point_count = int(
        plane_counts.get(
            active_plane,
            0,
        )
    )

    next_index = min(
        8,
        max(
            0,
            point_count,
        ),
    )

    for i, p in enumerate(pts):
        color = (
            (0, 255, 255)
            if i == next_index
            else (160, 160, 160)
        )

        radius = (
            12
            if i == next_index
            else 5
        )

        thickness = (
            3
            if i == next_index
            else 1
        )

        cv2.circle(
            frame,
            p,
            radius,
            color,
            thickness,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Bal felső állapotjelző panel.
    # --------------------------------------------------------
    overlay = frame.copy()

    panel_w = min(
        340,
        w - 20,
    )

    panel_h = 190

    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_w, 10 + panel_h),
        (15, 15, 15),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.58,
        frame,
        0.42,
        0,
        frame,
    )

    astra_ok = bool(
        status.get(
            "astra_ok",
            local_hand_ok,
        )
    )

    top_ok = bool(
        status.get(
            "top_ok",
            False,
        )
    )

    ok_color = (0, 230, 0)
    bad_color = (0, 0, 235)
    wait_color = (0, 210, 255)

    outlined_text(
        frame,
        "ASTRA OK" if astra_ok else "ASTRA LOST",
        (24, 42),
        0.72,
        ok_color if astra_ok else bad_color,
        2,
    )

    outlined_text(
        frame,
        "TOP OK" if top_ok else "TOP LOST",
        (190, 42),
        0.72,
        ok_color if top_ok else bad_color,
        2,
    )

    counts_text = (
        f"{active_plane}  "
        f"L/M/H="
        f"{int(plane_counts.get('LOW', 0))}/"
        f"{int(plane_counts.get('MID', 0))}/"
        f"{int(plane_counts.get('HIGH', 0))}"
    )

    outlined_text(
        frame,
        counts_text,
        (24, 78),
        0.58,
        (255, 255, 255),
        2,
    )

    current_y = status.get(
        "current_y_mm",
        None,
    )

    target_y = status.get(
        "target_y_mm",
        None,
    )

    if (
        current_y is not None
        and target_y is not None
    ):
        y_text = (
            f"Y={float(current_y):+.0f}  "
            f"cel={float(target_y):+.0f} mm"
        )
    elif current_y is not None:
        y_text = (
            f"Y={float(current_y):+.0f} mm  "
            "(elso pont allitja a sikot)"
        )
    else:
        y_text = "Y=--"

    outlined_text(
        frame,
        y_text,
        (24, 112),
        0.50,
        (230, 230, 230),
        1,
    )

    reason = str(
        status.get(
            "reason",
            "VARJ",
        )
    )

    outlined_text(
        frame,
        "MENTHETO - C"
        if can_save
        else reason,
        (24, 148),
        0.70,
        ok_color if can_save else wait_color,
        2,
    )

    confirm_stage = int(
        status.get(
            "confirm_stage",
            1,
        )
    )

    if confirm_stage == 2:
        outlined_text(
            frame,
            "UGYANITT ISMET C (2/2)",
            (24, 178),
            0.62,
            (0, 215, 255),
            2,
        )

    # --------------------------------------------------------
    # A mentés eredményét jelző középső üzenet.
    # --------------------------------------------------------
    if (
        flash_state["until"] is not None
        and time.monotonic()
        < flash_state["until"]
    ):
        text = flash_state["text"]
        kind = flash_state["kind"]

        if kind == "ok":
            color = (0, 235, 0)
        elif kind == "error":
            color = (0, 0, 245)
        else:
            color = (0, 215, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.65
        thickness = 4

        (tw, th), _ = cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )

        tx = max(
            10,
            (w - tw) // 2,
        )

        ty = max(
            th + 20,
            h // 2,
        )

        # Sötét háttér a visszajelzés olvashatóságához.
        cv2.rectangle(
            frame,
            (
                max(0, tx - 24),
                max(0, ty - th - 24),
            ),
            (
                min(w - 1, tx + tw + 24),
                min(h - 1, ty + 24),
            ),
            (0, 0, 0),
            -1,
        )

        outlined_text(
            frame,
            text,
            (tx, ty),
            scale,
            color,
            thickness,
        )


def ensure_reconnect_log():
    exists = RECONNECT_LOG.exists()

    f = open(
        RECONNECT_LOG,
        "a",
        newline="",
        encoding="utf-8-sig",
    )

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "pc_time",
            "event",
            "reconnect_count",
            "camera_index",
            "camera_name",
            "note",
        ],
    )

    if not exists:
        writer.writeheader()
        f.flush()

    return f, writer


def log_reconnect(
    writer,
    file_obj,
    event,
    reconnect_count,
    index=None,
    name=None,
    note="",
):
    writer.writerow({
        "pc_time": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "event": event,
        "reconnect_count": reconnect_count,
        "camera_index": (
            ""
            if index is None
            else index
        ),
        "camera_name": (
            ""
            if name is None
            else name
        ),
        "note": note,
    })

    file_obj.flush()


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hianyzik: {MODEL_PATH}"
        )

    print("=" * 72)
    print("ASTRA RGB CALIBRATION SENDER V6 - GUIDED + KEY FORWARD")
    print("=" * 72)

    palm_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    key_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    status_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )
    status_sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )
    status_sock.bind(
        (
            UDP_IP,
            CAL_STATUS_PORT,
        )
    )
    status_sock.setblocking(False)

    landmarker = create_landmarker()

    cv2.namedWindow(
        "ASTRA CALIBRATION",
        cv2.WINDOW_NORMAL,
    )
    cv2.resizeWindow(
        "ASTRA CALIBRATION",
        960,
        720,
    )

    log_file, log_writer = ensure_reconnect_log()

    cap = None
    camera_index = None
    camera_name = None

    sequence = 0
    reconnect_count = 0
    filtered_x = None
    filtered_y = None

    last_invalid_send = 0.0

    status = {}
    last_flash_id = None

    flash_state = {
        "until": None,
        "text": "",
        "kind": "info",
    }

    try:
        while True:
            # -------------------------------------------------
            # Kamera újranyitása képvesztés után.
            # -------------------------------------------------
            if cap is None:
                try:
                    (
                        cap,
                        camera_index,
                        camera_name,
                    ) = open_astra_camera()

                    log_reconnect(
                        log_writer,
                        log_file,
                        "OPEN_OK",
                        reconnect_count,
                        camera_index,
                        camera_name,
                        "",
                    )

                except Exception as e:
                    reconnect_count += 1

                    log_reconnect(
                        log_writer,
                        log_file,
                        "OPEN_FAIL",
                        reconnect_count,
                        None,
                        None,
                        f"{type(e).__name__}: {e}",
                    )

                    now = time.monotonic()

                    if (
                        now - last_invalid_send
                        >= INVALID_SEND_PERIOD_S
                    ):
                        sequence += 1
                        send_palm(
                            palm_sock,
                            sequence,
                            0.0,
                            0.0,
                            False,
                        )
                        last_invalid_send = now

                    time.sleep(REOPEN_DELAY_S)
                    continue

            ok, frame = cap.read()

            if (
                not ok
                or frame is None
            ):
                reconnect_count += 1

                log_reconnect(
                    log_writer,
                    log_file,
                    "FRAME_READ_FAIL",
                    reconnect_count,
                    camera_index,
                    camera_name,
                    "cap.read() failed",
                )

                try:
                    cap.release()
                except Exception:
                    pass

                cap = None
                filtered_x = None
                filtered_y = None

                sequence += 1
                send_palm(
                    palm_sock,
                    sequence,
                    0.0,
                    0.0,
                    False,
                )

                time.sleep(REOPEN_DELAY_S)
                continue

            # A kalibrációs mélységi feldolgozás tükrözött RGB-koordinátát vár.
            frame = cv2.flip(
                frame,
                1,
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms = int(
                time.monotonic()
                * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            palm_valid = False
            palm_x = 0.0
            palm_y = 0.0

            if result.hand_landmarks:
                landmarks = (
                    result.hand_landmarks[0]
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
                    filtered_x = palm_x_raw
                    filtered_y = palm_y_raw
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

                h, w = frame.shape[:2]

                # A kezelőfelületen csak a tenyérreferenciát és öt kézpontot jelöljük.
                for idx in PALM_IDS:
                    px = int(
                        round(
                            landmarks[idx].x
                            * w
                        )
                    )
                    py = int(
                        round(
                            landmarks[idx].y
                            * h
                        )
                    )

                    cv2.circle(
                        frame,
                        (px, py),
                        4,
                        (255, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )

                cx = int(
                    round(
                        palm_x
                        * w
                    )
                )
                cy = int(
                    round(
                        palm_y
                        * h
                    )
                )

                cv2.circle(
                    frame,
                    (cx, cy),
                    11,
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

            else:
                filtered_x = None
                filtered_y = None

            sequence += 1

            send_palm(
                palm_sock,
                sequence,
                palm_x,
                palm_y,
                palm_valid,
            )

            # -------------------------------------------------
            # A fő kalibrációs programtól érkező állapotadatok.
            # -------------------------------------------------
            status = drain_cal_status(
                status_sock,
                status,
            )

            flash_id = status.get(
                "flash_id",
                None,
            )

            if (
                flash_id is not None
                and flash_id != last_flash_id
            ):
                last_flash_id = flash_id

                flash_state = {
                    "until": (
                        time.monotonic()
                        + FLASH_DURATION_S
                    ),
                    "text": str(
                        status.get(
                            "flash_text",
                            "",
                        )
                    ),
                    "kind": str(
                        status.get(
                            "flash_kind",
                            "info",
                        )
                    ),
                }

            draw_calibration_ui(
                frame,
                status,
                palm_valid,
                flash_state,
            )

            cv2.imshow(
                "ASTRA CALIBRATION",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key != 255:
                try:
                    ch = chr(key).lower()
                except Exception:
                    ch = ""

                if ch in "123cufvrq":
                    try:
                        key_sock.sendto(
                            ch.encode("ascii"),
                            (
                                UDP_IP,
                                CAL_KEY_PORT,
                            ),
                        )
                    except Exception:
                        pass

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        try:
            sequence += 1
            send_palm(
                palm_sock,
                sequence,
                0.0,
                0.0,
                False,
            )
        except Exception:
            pass

        try:
            landmarker.close()
        except Exception:
            pass

        try:
            palm_sock.close()
        except Exception:
            pass

        try:
            key_sock.close()
        except Exception:
            pass

        try:
            status_sock.close()
        except Exception:
            pass

        try:
            log_file.close()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
