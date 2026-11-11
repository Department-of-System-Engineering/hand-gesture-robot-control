import os
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


# ============================================================
# Eternico ET201 felso kamera -> tenyerdetekcio UDP sender V5 PICKPLACE UI - NEV ALAPU ESZKOZVALASZTAS
#
# FELADAT:
# - a felso kamera masodlagos, eltero nezopontu kamera
# - NEM ad metrikus XYZ-t
# - elkuldi:
#     seq, palm_x_norm, palm_y_norm, palm_width_norm, valid
# - UI-ban nagy GUIDE BOX-ot rajzolunk, hogy a kalibracios
#   teruletet konnyebb legyen beloni.
#
# A fo TDK robotprogram a normalizalt tenyerpontot a V4 kalibracioval
# XZ-becslesre hasznalhatja. A sender maga nem szamit metrikus XYZ-t.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"

UDP_IP = "127.0.0.1"
UDP_PORT = 50020
PACKET_FORMAT = "<IfffB"

STATUS_UDP_PORT = 50021
PICK_STATUS_PACKET_FORMAT = "<BBBBBBBf"
PICK_STATUS_PACKET_SIZE = struct.calcsize(PICK_STATUS_PACKET_FORMAT)

# NINCS fix kameraindex. A Windows kameraindexek backendtol es
# inditasi sorrendtol fuggoen el tudnak csuszni.
# A mar bevalt Astra-detektorhoz hasonloan DirectShow eszkoznev
# alapjan keressuk meg a felso Eternico kamerat.
ETERNICO_NAME_PARTS = (
    "eternico",
    "et201",
)

# Ezeket kifejezetten nem fogadjuk el top kameranak.
ASTRA_NAME_PARTS = (
    "orbbec",
    "astra",
)

WIDTH = 640
HEIGHT = 480
FPS = 30

SHOW_UI = True

# Nagy vizualis segedkeret a felso kamera kepen.
GUIDE_BOX_W_FRAC = 0.58
GUIDE_BOX_H_FRAC = 0.58
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


def guide_box_rect(frame):
    h, w = frame.shape[:2]

    box_w = int(round(w * GUIDE_BOX_W_FRAC))
    box_h = int(round(h * GUIDE_BOX_H_FRAC))

    x1 = max(0, (w - box_w) // 2)
    y1 = max(0, (h - box_h) // 2)
    x2 = min(w - 1, x1 + box_w)
    y2 = min(h - 1, y1 + box_h)

    return x1, y1, x2, y2


def draw_guide_box(frame):
    x1, y1, x2, y2 = guide_box_rect(frame)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        GUIDE_BOX_COLOR,
        GUIDE_BOX_THICKNESS,
        cv2.LINE_AA
    )

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    cv2.line(
        frame,
        (cx - CENTER_CROSS_HALF, cy),
        (cx + CENTER_CROSS_HALF, cy),
        GUIDE_BOX_COLOR,
        2,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (cx, cy - CENTER_CROSS_HALF),
        (cx, cy + CENTER_CROSS_HALF),
        GUIDE_BOX_COLOR,
        2,
        cv2.LINE_AA
    )

    return x1, y1, x2, y2


def palm_in_guide_box(frame, palm_x, palm_y):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = guide_box_rect(frame)

    px = int(round(palm_x * w))
    py = int(round(palm_y * h))

    return x1 <= px <= x2 and y1 <= py <= y2


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
    """
    Ugyanaz az elv, mint a mar bevalt Astra RGB senderben:
    a DirectShow eszkozlistabol NEV szerint valasztunk.

    Fontos: nem hasznalunk regi fix MSMF indexet, mert a backendek
    indexelese nem azonos, es a Windows eszkozsorrend valtozhat.
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

    # A pygrabber DirectShow listajanak indexet ugyanazzal a
    # DirectShow backenddel nyitjuk meg. Nem keverjuk MSMF indexszel.
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


def drain_pick_status(sock, previous):
    latest = dict(previous)
    while True:
        try:
            data, _ = sock.recvfrom(128)
        except BlockingIOError:
            break
        if len(data) != PICK_STATUS_PACKET_SIZE:
            continue
        tracking, gripper, mode, source, flow, trial, gate, loss_elapsed = struct.unpack(
            PICK_STATUS_PACKET_FORMAT, data
        )
        latest.update({
            "tracking":bool(tracking), "gripper":bool(gripper), "mode":int(mode),
            "source":int(source), "flow":int(flow), "trial":int(trial),
            "gate":int(gate), "loss_elapsed":float(loss_elapsed)
        })
    return latest


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hianyzik: {MODEL_PATH}"
        )

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    status_sock.bind((UDP_IP, STATUS_UDP_PORT))
    status_sock.setblocking(False)
    status = {"tracking":False,"gripper":False,"mode":1,"source":0,"flow":1,"trial":0,"gate":0,"loss_elapsed":0.0}

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

    print(
        f"[OK] UDP sender -> {UDP_IP}:{UDP_PORT}"
    )

    try:
        while True:
            # -------------------------------------------------
            # Kamera automatikus ujranyitas.
            # Egy rovid USB/DirectShow kepolvasasi hiba ne allitsa le
            # az egesz top-camera processt.
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
                guide_rect = draw_guide_box(frame)

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
                    palm_y
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

            status = drain_pick_status(status_sock, status)

            if SHOW_UI:
                outlined_text(
                    frame,
                    "ETERNICO TOP CAM V5 - PICKPLACE",
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

                if valid:
                    outlined_text(
                        frame,
                        "IN BOX"
                        if palm_inside_box
                        else "HAND OUTSIDE BOX",
                        (18, 100),
                        0.68,
                        2
                    )

                    outlined_text(
                        frame,
                        f"u={palm_x:.3f}  v={palm_y:.3f}  w={palm_width:.3f}",
                        (18, 134),
                        0.58,
                        1
                    )
                else:
                    outlined_text(
                        frame,
                        "Tedd a kezedet a sarga keret kozelebe.",
                        (18, 100),
                        0.58,
                        1
                    )

                mode_txt = "DUAL" if status.get("mode") == 2 else "ASTRA"
                source_names = {0:"IDLE",1:"ASTRA",2:"TOP",3:"HOLD",4:"PAUSE"}
                src_txt = source_names.get(status.get("source",0), "?")
                flow_txt = "FLOW" if status.get("flow",1) else "STRICT"
                outlined_text(frame, f"MODE={mode_txt} SRC={src_txt} REACQ={flow_txt}", (18, 170), 0.52, 1)
                outlined_text(frame, f"TOP GATE={'OK' if status.get('gate') else 'NO'}  LOSS={status.get('loss_elapsed',0.0):.2f}s", (18, 198), 0.52, 1)
                outlined_text(frame, f"TRIAL={'ON' if status.get('trial') else 'OFF'}  RG2={'ON' if status.get('gripper') else 'OFF'}", (18, 226), 0.52, 1)

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

                # A top ablak sajat Q-ja mar nem allitja le a processt.
                # A teljes rendszert a fo launcher Q-ja zarja le.
                cv2.waitKey(1)

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        landmarker.close()
        sock.close()
        status_sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
