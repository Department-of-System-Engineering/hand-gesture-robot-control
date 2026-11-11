"""
TDK KETKAMERAS VAKFOLT / KITAKARAS TESZT V5
2026-10-21

Cel:
- Astra RGB vs. Eternico ET201 detektalasi folytonossag
- objektiv Astra-only vs. Astra+top osszehasonlitas
- HIGH / MID / LOW kulon futamok
- automatikus BASELINE -> OCCLUSION -> RECOVERY fazis
- frame-szintu CSV + futam-osszefoglalo CSV
- egy automatikus "Astra LOST, TOP OK" pelda-kep futamonkent

Billentyuk:
1 = HIGH
2 = MID
3 = LOW
S = 75 masodperces futam inditasa
R = aktualis futam megszakitasa
Q / ESC = kilepes

Futam:
0-15 s   BASELINE  : termeszetes kezmozgas, szandekos kitakaras nelkul
15-60 s  OCCLUSION : termeszetes mozgas + ismetelt realis Astra-kitakaras
60-75 s  RECOVERY  : ujra normal mozgas, kitakaras nelkul
"""

import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"

WIDTH = 640
HEIGHT = 480
FPS = 30

PREFERRED_ASTRA = ("orbbec", "astra")
PREFERRED_TOP = ("eternico", "et201")
ASTRA_FALLBACK_EXACT = ("USB Camera",)

ENUM_RETRY_S = 1.0
ENUM_TIMEOUT_S = 20.0

RUN_DURATION_S = 75.0
BASELINE_END_S = 15.0
OCCLUSION_END_S = 60.0

FRAME_LOG = BASE_DIR / "blindspot_tdk_v5_frames.csv"
RUN_LOG = BASE_DIR / "blindspot_tdk_v5_runs.csv"
EXAMPLE_DIR = BASE_DIR / "blindspot_tdk_v5_examples"

PALM_IDS = [0, 5, 9, 13, 17]

LANDMARK_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def list_devices():
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        raise RuntimeError(
            "Hianyzik a pygrabber. Telepites: pip install pygrabber"
        )

    return FilterGraph().get_input_devices()


def find_named_cameras():
    deadline = time.monotonic() + ENUM_TIMEOUT_S
    last_devices = []

    while time.monotonic() < deadline:
        devices = list_devices()
        last_devices = devices

        print("\nDirectShow kamerak:")
        for i, name in enumerate(devices):
            print(f"  [{i}] {name}")

        astra = None
        top = None

        for i, name in enumerate(devices):
            low = name.lower()

            if astra is None and any(p in low for p in PREFERRED_ASTRA):
                astra = (i, name)

            if top is None and any(p in low for p in PREFERRED_TOP):
                top = (i, name)

        if astra is None:
            fallback = [
                (i, name)
                for i, name in enumerate(devices)
                if name.strip() in ASTRA_FALLBACK_EXACT
            ]
            if len(fallback) == 1:
                astra = fallback[0]

        if astra is not None and top is not None and astra[0] != top[0]:
            print(f"\n[OK] Astra: [{astra[0]}] {astra[1]}")
            print(f"[OK] Top  : [{top[0]}] {top[1]}")
            return astra, top

        print("[VARJ] Kamera-nevek meg nem egyertelmuek, ujraproba...")
        time.sleep(ENUM_RETRY_S)

    raise RuntimeError(
        "Nem sikerult egyertelmuen megtalalni mindket kamerat.\n"
        f"Utolso lista: {last_devices}"
    )


def open_camera(index, label):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"{label} nem nyithato DirectShow-val.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(40):
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"[OK] {label} megnyitva.")
            return cap
        time.sleep(0.05)

    cap.release()
    raise RuntimeError(f"{label} megnyilt, de nem ad ervenyes kepet.")


def create_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return HandLandmarker.create_from_options(options)


def detect(landmarker, frame_bgr, timestamp_ms):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks:
        return False, None, None

    lm = result.hand_landmarks[0]

    xs = [float(lm[i].x) for i in PALM_IDS]
    ys = [float(lm[i].y) for i in PALM_IDS]

    palm = (
        float(np.mean(xs)),
        float(np.mean(ys)),
    )

    return True, lm, palm


def draw_hand(frame, landmarks):
    if landmarks is None:
        return

    h, w = frame.shape[:2]
    pts = [
        (int(lm.x * w), int(lm.y * h))
        for lm in landmarks
    ]

    for a, b in LANDMARK_CONNECTIONS:
        cv2.line(
            frame,
            pts[a],
            pts[b],
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)


def outlined_text(frame, text, pos, scale=0.7, thickness=2):
    cv2.putText(
        frame, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, (0, 0, 0),
        thickness + 3, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, (255, 255, 255),
        thickness, cv2.LINE_AA,
    )


def phase_for_elapsed(elapsed):
    if elapsed < BASELINE_END_S:
        return "BASELINE"
    if elapsed < OCCLUSION_END_S:
        return "OCCLUSION"
    return "RECOVERY"


def phase_instruction(phase):
    if phase == "BASELINE":
        return "NORMAL MOZGAS - NE TAKARD KI"
    if phase == "OCCLUSION":
        return "ISMETELTEN TAKARD KI AZ ASTRAT"
    return "NORMAL MOZGAS - NE TAKARD KI"


def open_csv_append(path, fields):
    exists = path.exists() and path.stat().st_size > 0

    f = open(path, "a", newline="", encoding="utf-8-sig")
    w = csv.DictWriter(f, fieldnames=fields)

    if not exists:
        w.writeheader()
        f.flush()

    return f, w


def safe_float(value):
    return "" if value is None else f"{float(value):.6f}"


def summarize(rows):
    phases = ["ALL", "BASELINE", "OCCLUSION", "RECOVERY"]
    out = {}

    for phase in phases:
        subset = rows if phase == "ALL" else [r for r in rows if r["phase"] == phase]
        n = len(subset)

        if n == 0:
            out[phase] = {
                "n": 0,
                "astra_pct": 0.0,
                "top_pct": 0.0,
                "combined_pct": 0.0,
                "rescue_pct": 0.0,
                "both_lost_pct": 0.0,
            }
            continue

        astra_ok = sum(int(r["astra_ok"]) for r in subset)
        top_ok = sum(int(r["top_ok"]) for r in subset)
        combined_ok = sum(
            1
            for r in subset
            if int(r["astra_ok"]) or int(r["top_ok"])
        )
        both_lost = sum(
            1
            for r in subset
            if not int(r["astra_ok"]) and not int(r["top_ok"])
        )
        astra_lost = n - astra_ok
        rescued = sum(
            1
            for r in subset
            if not int(r["astra_ok"]) and int(r["top_ok"])
        )

        out[phase] = {
            "n": n,
            "astra_pct": 100.0 * astra_ok / n,
            "top_pct": 100.0 * top_ok / n,
            "combined_pct": 100.0 * combined_ok / n,
            "rescue_pct": (
                100.0 * rescued / astra_lost
                if astra_lost
                else 0.0
            ),
            "both_lost_pct": 100.0 * both_lost / n,
        }

    return out


def save_example(frame_a, frame_b, run_id, plane, elapsed):
    EXAMPLE_DIR.mkdir(exist_ok=True)

    fa = frame_a.copy()
    fb = frame_b.copy()

    outlined_text(fa, "ASTRA: HAND LOST", (18, 36), 0.8, 2)
    outlined_text(fb, "TOP: HAND OK", (18, 36), 0.8, 2)

    h = min(fa.shape[0], fb.shape[0])
    fa = fa[:h]
    fb = fb[:h]
    combined = np.hstack([fa, fb])

    filename = (
        f"run_{run_id:02d}_{plane}_rescue_"
        f"{elapsed:05.1f}s.jpg"
    )
    cv2.imwrite(str(EXAMPLE_DIR / filename), combined)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Hianyzik: {MODEL_PATH}")

    (astra_idx, _), (top_idx, _) = find_named_cameras()

    # RGB-only teszt, OpenNI/depth NEM fut.
    cap_a = open_camera(astra_idx, "Astra RGB")
    cap_b = open_camera(top_idx, "Eternico TOP")

    lm_a = create_landmarker()
    lm_b = create_landmarker()

    frame_fields = [
        "pc_time",
        "run_id",
        "plane",
        "elapsed_s",
        "phase",
        "astra_ok",
        "top_ok",
        "combined_ok",
        "rescue",
        "both_lost",
        "astra_u",
        "astra_v",
        "top_u",
        "top_v",
    ]

    run_fields = [
        "pc_time",
        "run_id",
        "plane",
        "phase",
        "n",
        "astra_detection_pct",
        "top_detection_pct",
        "combined_detection_pct",
        "astra_lost_rescued_by_top_pct",
        "both_lost_pct",
    ]

    frame_file, frame_writer = open_csv_append(FRAME_LOG, frame_fields)
    run_file, run_writer = open_csv_append(RUN_LOG, run_fields)

    current_plane = "MID"
    running = False
    run_id = 0
    run_start = None
    rows = []
    example_saved = False
    rescue_streak = 0
    seq = 0

    print("\n" + "=" * 70)
    print("TDK KETKAMERAS KITAKARAS TESZT V5")
    print("=" * 70)
    print("1=HIGH  2=MID  3=LOW")
    print("S = 75 s futam inditasa")
    print("R = futam megszakitasa")
    print("Q/ESC = kilepes")
    print("\nAjánlott sorrend: MID -> HIGH -> LOW\n")

    try:
        while True:
            ok_a, frame_a = cap_a.read()
            ok_b, frame_b = cap_b.read()

            if not ok_a or frame_a is None:
                raise RuntimeError("Astra RGB kepolvasas megszakadt.")
            if not ok_b or frame_b is None:
                raise RuntimeError("Eternico kepolvasas megszakadt.")

            # A fo rendszerhez hasonloan az Astrat tukrozzuk.
            frame_a = cv2.flip(frame_a, 1)

            seq += 1
            timestamp_ms = seq * 33

            hand_a, lmks_a, palm_a = detect(lm_a, frame_a, timestamp_ms)
            hand_b, lmks_b, palm_b = detect(lm_b, frame_b, timestamp_ms)

            draw_hand(frame_a, lmks_a)
            draw_hand(frame_b, lmks_b)

            now = time.monotonic()

            if running:
                elapsed = now - run_start
                phase = phase_for_elapsed(elapsed)

                row = {
                    "pc_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "run_id": run_id,
                    "plane": current_plane,
                    "elapsed_s": f"{elapsed:.3f}",
                    "phase": phase,
                    "astra_ok": int(hand_a),
                    "top_ok": int(hand_b),
                    "combined_ok": int(hand_a or hand_b),
                    "rescue": int((not hand_a) and hand_b),
                    "both_lost": int((not hand_a) and (not hand_b)),
                    "astra_u": safe_float(None if palm_a is None else palm_a[0]),
                    "astra_v": safe_float(None if palm_a is None else palm_a[1]),
                    "top_u": safe_float(None if palm_b is None else palm_b[0]),
                    "top_v": safe_float(None if palm_b is None else palm_b[1]),
                }

                rows.append(row)
                frame_writer.writerow(row)

                # Egy stabilabb, legalabb 3 frame-es rescue pillanat kepmentese.
                if (not hand_a) and hand_b and phase == "OCCLUSION":
                    rescue_streak += 1
                else:
                    rescue_streak = 0

                if rescue_streak >= 3 and not example_saved:
                    save_example(
                        frame_a,
                        frame_b,
                        run_id,
                        current_plane,
                        elapsed,
                    )
                    example_saved = True

                remaining = max(0.0, RUN_DURATION_S - elapsed)

                for frame, label, ok in (
                    (frame_a, "ASTRA RGB", hand_a),
                    (frame_b, "TOP / ETERNICO", hand_b),
                ):
                    outlined_text(
                        frame,
                        f"{label} | {current_plane} | {phase}",
                        (18, 32),
                        0.72, 2,
                    )
                    outlined_text(
                        frame,
                        "HAND OK" if ok else "HAND LOST",
                        (18, 68),
                        0.86, 2,
                    )
                    outlined_text(
                        frame,
                        phase_instruction(phase),
                        (18, 108),
                        0.62, 2,
                    )
                    outlined_text(
                        frame,
                        f"RUN {run_id} | {remaining:04.1f}s",
                        (18, 145),
                        0.66, 2,
                    )

                if elapsed >= RUN_DURATION_S:
                    running = False
                    frame_file.flush()

                    stats = summarize(rows)

                    print(f"\n\n=== RUN {run_id} {current_plane} KESZ ===")
                    for phase_name in ("BASELINE", "OCCLUSION", "RECOVERY", "ALL"):
                        s = stats[phase_name]
                        print(
                            f"{phase_name:9s} | "
                            f"Astra {s['astra_pct']:.1f}% | "
                            f"Top {s['top_pct']:.1f}% | "
                            f"Kombinalt {s['combined_pct']:.1f}% | "
                            f"Rescue {s['rescue_pct']:.1f}% | "
                            f"BothLost {s['both_lost_pct']:.1f}%"
                        )

                        run_writer.writerow({
                            "pc_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "run_id": run_id,
                            "plane": current_plane,
                            "phase": phase_name,
                            "n": s["n"],
                            "astra_detection_pct": f"{s['astra_pct']:.3f}",
                            "top_detection_pct": f"{s['top_pct']:.3f}",
                            "combined_detection_pct": f"{s['combined_pct']:.3f}",
                            "astra_lost_rescued_by_top_pct": f"{s['rescue_pct']:.3f}",
                            "both_lost_pct": f"{s['both_lost_pct']:.3f}",
                        })

                    run_file.flush()
                    rows = []
                    rescue_streak = 0

            else:
                for frame, label, ok in (
                    (frame_a, "ASTRA RGB", hand_a),
                    (frame_b, "TOP / ETERNICO", hand_b),
                ):
                    outlined_text(
                        frame,
                        f"{label} | VALASZTOTT: {current_plane}",
                        (18, 32),
                        0.72, 2,
                    )
                    outlined_text(
                        frame,
                        "HAND OK" if ok else "HAND LOST",
                        (18, 68),
                        0.86, 2,
                    )
                    outlined_text(
                        frame,
                        "1/2/3 sik | S = START",
                        (18, 108),
                        0.64, 2,
                    )

            cv2.imshow("ASTRA RGB - TDK kitakaras teszt", frame_a)
            cv2.imshow("TOP CAMERA - TDK kitakaras teszt", frame_b)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("1") and not running:
                current_plane = "HIGH"
            elif key == ord("2") and not running:
                current_plane = "MID"
            elif key == ord("3") and not running:
                current_plane = "LOW"
            elif key == ord("s") and not running:
                run_id += 1
                run_start = time.monotonic()
                running = True
                rows = []
                example_saved = False
                rescue_streak = 0
                print(f"\n[START] RUN {run_id} | {current_plane}")
            elif key == ord("r") and running:
                print(f"\n[ABORT] RUN {run_id} | {current_plane}")
                running = False
                rows = []
                rescue_streak = 0
                frame_file.flush()

    finally:
        cap_a.release()
        cap_b.release()
        lm_a.close()
        lm_b.close()
        frame_file.close()
        run_file.close()
        cv2.destroyAllWindows()

    print("\nKesz.")
    print(f"Frame log: {FRAME_LOG.name}")
    print(f"Run log:   {RUN_LOG.name}")
    print(f"Peldakepek: {EXAMPLE_DIR.name}/")


if __name__ == "__main__":
    main()
