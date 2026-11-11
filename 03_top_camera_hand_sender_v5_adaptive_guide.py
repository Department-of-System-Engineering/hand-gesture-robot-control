import os
import json
import socket
import struct
import time
from pathlib import Path

os.environ.setdefault(
    "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS",
    "0"
)

import cv2
import mediapipe as mp


# Eternico ET201 felsőkamerás kézdetektálás a V4 kalibrációhoz.
# Az 50020-as UDP-porton normalizált tenyérpozíciót, tenyérszélességet
# és érvényességi jelzést küld. A kamera önmagában nem mér metrikus XYZ-t.
# A síkfüggő segédkeret a 3×3 kalibrációs pontok kijelölését támogatja.
# A felsőkamera-adatokat a fő kalibrációs program dolgozza fel.

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"

UDP_IP = "127.0.0.1"
UDP_PORT = 50020
PACKET_FORMAT = "<IfffB"

# A Windows kameraindexek változhatnak, ezért az Eternico kamerát
# a DirectShow eszközlistában szereplő neve alapján azonosítjuk.
ETERNICO_NAME_PARTS = (
    "eternico",
    "et201",
)

# Az Astra és Orbbec eszközök kizárása a felső kamera kiválasztásakor.
ASTRA_NAME_PARTS = (
    "orbbec",
    "astra",
)

WIDTH = 640
HEIGHT = 480
FPS = 30

SHOW_UI = True

# A fő kalibrációs programtól érkező állapotinformáció.
CAL_STATUS_PORT = 50032
CAL_KEY_PORT = 50033

# A segédkeret a kézmagassággal együtt változik a perspektíva miatt.
# LOW síkon kisebb, HIGH síkon nagyobb képi területet jelölünk.
# A keretek képi középpontja mindhárom síkon azonos.
GUIDE_BOX_BY_PLANE = {
    "LOW": {
        "cx": 0.50,
        "cy": 0.50,
        "w": 0.40,
        "h": 0.40,
    },
    "MID": {
        "cx": 0.50,
        "cy": 0.50,
        "w": 0.50,
        "h": 0.50,
    },
    "HIGH": {
        "cx": 0.50,
        "cy": 0.50,
        "w": 0.58,
        "h": 0.58,
    },
}

GUIDE_BOX_COLOR = (0, 220, 255)
GUIDE_BOX_THICKNESS = 2
CENTER_CROSS_HALF = 12

PALM_IDS = [0, 5, 9, 13, 17]

LANDMARK_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def outlined_text(
    frame,
    text,
    pos,
    scale=0.65,
    thickness=1
):
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def drain_cal_status(
    sock,
    previous
):
    latest = previous

    while True:
        try:
            data, _ = sock.recvfrom(4096)
        except BlockingIOError:
            break

        try:
            latest = json.loads(
                data.decode("utf-8")
            )
        except Exception:
            continue

    return latest


def guide_config(status):
    plane = str(
        status.get(
            "plane",
            "MID"
        )
    )

    return GUIDE_BOX_BY_PLANE.get(
        plane,
        GUIDE_BOX_BY_PLANE["MID"]
    )


def guide_box_rect(frame, status):
    h, w = frame.shape[:2]
    cfg = guide_config(status)

    box_w = int(round(w * cfg["w"]))
    box_h = int(round(h * cfg["h"]))

    cx = int(round(w * cfg["cx"]))
    cy = int(round(h * cfg["cy"]))

    x1 = max(0, cx - box_w // 2)
    y1 = max(0, cy - box_h // 2)
    x2 = min(w - 1, x1 + box_w)
    y2 = min(h - 1, y1 + box_h)

    return x1, y1, x2, y2


def guide_points_px(frame, status):
    x1, y1, x2, y2 = guide_box_rect(
        frame,
        status
    )

    xs = [
        x1,
        (x1 + x2) // 2,
        x2,
    ]

    ys = [
        y1,
        (y1 + y2) // 2,
        y2,
    ]

    pts = []

    for yy in ys:
        for xx in xs:
            pts.append(
                (int(xx), int(yy))
            )

    return pts


def draw_guide_box(frame, status):
    x1, y1, x2, y2 = guide_box_rect(
        frame,
        status
    )

    can_save = bool(
        status.get(
            "can_save",
            False
        )
    )

    color = (
        (0, 230, 0)
        if can_save
        else GUIDE_BOX_COLOR
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        GUIDE_BOX_THICKNESS,
        cv2.LINE_AA
    )

    pts = guide_points_px(
        frame,
        status
    )

    counts = status.get(
        "counts",
        {}
    )

    plane = str(
        status.get(
            "plane",
            "MID"
        )
    )

    next_index = min(
        8,
        max(
            0,
            int(
                counts.get(
                    plane,
                    0
                )
            )
        )
    )

    for i, p in enumerate(pts):
        is_next = (
            i == next_index
        )

        cv2.circle(
            frame,
            p,
            12 if is_next else 5,
            (0, 255, 255)
            if is_next
            else (170, 170, 170),
            3 if is_next else 1,
            cv2.LINE_AA
        )

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    cv2.line(
        frame,
        (cx - CENTER_CROSS_HALF, cy),
        (cx + CENTER_CROSS_HALF, cy),
        color,
        2,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (cx, cy - CENTER_CROSS_HALF),
        (cx, cy + CENTER_CROSS_HALF),
        color,
        2,
        cv2.LINE_AA
    )

    return x1, y1, x2, y2


def palm_in_guide_box(
    frame,
    palm_x,
    palm_y,
    status
):
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = guide_box_rect(
        frame,
        status
    )

    px = int(round(palm_x * w))
    py = int(round(palm_y * h))

    return (
        x1 <= px <= x2
        and y1 <= py <= y2
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

    return HandLandmarker.create_from_options(
        options
    )


def find_eternico_dshow_device():
    """Az Eternico kamerát a DirectShow eszközlistában név szerint keresi.

    Az MSMF és a DirectShow indexei eltérhetnek, ezért a kamerát ugyanazzal
    a DirectShow backenddel nyitja meg, amelyből az eszközlistát lekérte.
    """
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

    matches = []

    for index, name in enumerate(devices):
        lower = name.lower()

        is_eternico = any(
            part in lower
            for part in ETERNICO_NAME_PARTS
        )

        is_astra = any(
            part in lower
            for part in ASTRA_NAME_PARTS
        )

        if is_eternico and not is_astra:
            matches.append((index, name))

    if len(matches) == 1:
        index, name = matches[0]
        print()
        print(f"[OK] Eternico top kamera: [{index}] {name}")
        return index, name

    if len(matches) > 1:
        raise RuntimeError(
            "Tobb Eternico/ET201 nevu kamera talalhato, "
            "nem valasztok talalomra: "
            + str(matches)
        )

    raise RuntimeError(
        "Nem talaltam nev szerint Eternico / ET201 kamerat. "
        "Nem nyitok meg vakon egy regi kameraindexet. "
        f"Eszkozok: {devices}"
    )


def open_camera():
    index, device_name = find_eternico_dshow_device()

    print(
        f"[INFO] Eternico nyitas NEV ALAPJAN: "
        f"DirectShow index={index} | {device_name}"
    )

    # A kiválasztott DirectShow-indexet DirectShow backenddel nyitjuk meg.
    cap = cv2.VideoCapture(
        index,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Az Eternico nev szerint megvan, de DirectShow-val "
            f"nem nyithato: [{index}] {device_name}. "
            "Nem valtok automatikusan masik indexre, nehogy az Astrat nyissuk meg."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FPS
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    for _ in range(40):
        ok, frame = cap.read()

        if ok and frame is not None:
            print(
                f"[OK] Eternico ET201 megnyitva: "
                f"[{index}] {device_name}"
            )
            return cap

        time.sleep(0.05)

    cap.release()

    raise RuntimeError(
        "Az Eternico nev szerint megnyilt, de nem adott ervenyes kepet: "
        f"[{index}] {device_name}"
    )


def detect(
    landmarker,
    frame_bgr,
    timestamp_ms
):
    rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if not result.hand_landmarks:
        return None

    return result.hand_landmarks[0]


def palm_data(
    frame,
    landmarks
):
    h, w = frame.shape[:2]

    pts = [
        (
            float(lm.x),
            float(lm.y)
        )
        for lm in landmarks
    ]

    palm_x = sum(
        pts[i][0]
        for i in PALM_IDS
    ) / len(PALM_IDS)

    palm_y = sum(
        pts[i][1]
        for i in PALM_IDS
    ) / len(PALM_IDS)

    x5, y5 = pts[5]
    x17, y17 = pts[17]

    palm_width = (
        (
            (x5 - x17) ** 2
            + (y5 - y17) ** 2
        ) ** 0.5
    )

    pixel_pts = [
        (
            int(round(x * w)),
            int(round(y * h))
        )
        for x, y in pts
    ]

    return (
        palm_x,
        palm_y,
        palm_width,
        pixel_pts
    )


def draw_hand(
    frame,
    pts,
    palm_x,
    palm_y
):
    h, w = frame.shape[:2]

    for a, b in LANDMARK_CONNECTIONS:
        cv2.line(
            frame,
            pts[a],
            pts[b],
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    for x, y in pts:
        cv2.circle(
            frame,
            (x, y),
            3,
            (0, 0, 0),
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            (x, y),
            1,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

    cx = int(
        round(
            palm_x * w
        )
    )

    cy = int(
        round(
            palm_y * h
        )
    )

    cv2.circle(
        frame,
        (cx, cy),
        9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hianyzik: {MODEL_PATH}"
        )

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    status_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )
    status_sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )
    status_sock.bind(
        (
            UDP_IP,
            CAL_STATUS_PORT
        )
    )
    status_sock.setblocking(False)

    key_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    if SHOW_UI:
        try:
            cv2.namedWindow("TOP CAMERA ASSIST", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("TOP CAMERA ASSIST", WIDTH, HEIGHT)
        except Exception:
            pass

    landmarker = create_landmarker()

    cap = None
    seq = 0
    start = time.perf_counter()
    status = {}

    print(
        f"[OK] UDP sender -> {UDP_IP}:{UDP_PORT}"
    )

    try:
        while True:
            # -------------------------------------------------
            # Rövid USB- vagy DirectShow-hiba esetén a kamera újranyitása.
            # -------------------------------------------------
            if cap is None:
                try:
                    cap = open_camera()
                except Exception as e:
                    print(
                        "[TOP CAM UJRANYITASI HIBA] "
                        + type(e).__name__
                        + ": "
                        + str(e)
                    )
                    time.sleep(1.0)
                    continue

            ok, frame = cap.read()

            if (
                not ok
                or frame is None
            ):
                print(
                    "[FIGYELEM] Eternico kepolvasas megszakadt. "
                    "Automatikus ujranyitas..."
                )

                try:
                    cap.release()
                except Exception:
                    pass

                cap = None

                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

                time.sleep(0.40)
                continue

            status = drain_cal_status(
                status_sock,
                status
            )

            seq += 1

            timestamp_ms = int(
                (
                    time.perf_counter()
                    - start
                )
                * 1000.0
            )

            guide_rect = None
            if SHOW_UI:
                guide_rect = draw_guide_box(frame, status)

            landmarks = detect(
                landmarker,
                frame,
                timestamp_ms
            )

            valid = 0
            palm_x = 0.0
            palm_y = 0.0
            palm_width = 0.0

            palm_inside_box = False

            if landmarks is not None:
                (
                    palm_x,
                    palm_y,
                    palm_width,
                    pts
                ) = palm_data(
                    frame,
                    landmarks
                )

                valid = 1
                palm_inside_box = palm_in_guide_box(
                    frame,
                    palm_x,
                    palm_y,
                    status
                )

                if SHOW_UI:
                    draw_hand(
                        frame,
                        pts,
                        palm_x,
                        palm_y
                    )

            packet = struct.pack(
                PACKET_FORMAT,
                int(seq),
                float(palm_x),
                float(palm_y),
                float(palm_width),
                int(valid)
            )

            sock.sendto(
                packet,
                (
                    UDP_IP,
                    UDP_PORT
                )
            )

            if SHOW_UI:
                outlined_text(
                    frame,
                    "ETERNICO TOP CAM V5 - ADAPTIV GUIDE",
                    (18, 32),
                    0.68,
                    2
                )

                outlined_text(
                    frame,
                    "HAND OK"
                    if valid
                    else "HAND LOST",
                    (18, 66),
                    0.72,
                    2
                )

                plane = str(
                    status.get(
                        "plane",
                        "MID"
                    )
                )

                confirm_stage = int(
                    status.get(
                        "confirm_stage",
                        1
                    )
                )

                outlined_text(
                    frame,
                    f"{plane} | meres {confirm_stage}/2",
                    (18, 100),
                    0.62,
                    2
                )

                if valid:
                    outlined_text(
                        frame,
                        "IN BOX"
                        if palm_inside_box
                        else "HAND OUTSIDE BOX",
                        (18, 134),
                        0.68,
                        2
                    )

                    outlined_text(
                        frame,
                        f"u={palm_x:.3f}  v={palm_y:.3f}  w={palm_width:.3f}",
                        (18, 168),
                        0.58,
                        1
                    )
                else:
                    outlined_text(
                        frame,
                        "Tedd a kezedet a sarga keret kozelebe.",
                        (18, 134),
                        0.58,
                        1
                    )

                outlined_text(
                    frame,
                    "A keret csak tajekoztato, nem szigoru feltetel.",
                    (18, frame.shape[0] - 44),
                    0.48,
                    1
                )

                outlined_text(
                    frame,
                    "Fo programbol Q = teljes kilepes",
                    (18, frame.shape[0] - 18),
                    0.48,
                    1
                )

                cv2.imshow(
                    "TOP CAMERA ASSIST",
                    frame
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
                                    CAL_KEY_PORT
                                )
                            )
                        except Exception:
                            pass

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        landmarker.close()

        try:
            sock.close()
        except Exception:
            pass

        try:
            status_sock.close()
        except Exception:
            pass

        try:
            key_sock.close()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
