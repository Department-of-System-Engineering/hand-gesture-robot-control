"""
TOP -> ASTRA MULTIPLANE CALIBRATION V4 QUALITY
2026-10-21

Cel:
- 3 magassagi sik (LOW/MID/HIGH)
- sikonkent 9 elfogadott pont
- minden pont KETSZER merendo ugyanott
- automatikus konzisztenciaellenorzes
- kozeli top-kepi pont + nagy Astra XZ elteres -> automatikus gyanus/elutasitott pont
- sikonkent OpenCV homografia
- Astra-Y szerinti sikkozi interpolacio
- reszletes depth diagnosztika
- OpenNI packet watchdog: befagyasnal teljes kamera-stack restart,
  az addig elfogadott pontok megtartasaval
- automatikus resume a v4 samples CSV-bol
- billentyuk a terminalbol VAGY az Astra/top kameraablakbol

BILLENTYUK:
1 LOW
2 MID
3 HIGH
C meres / megerosites
U pending meres torlese, vagy utolso elfogadott pont undo
F fit + mentes
V validacio ki/be
R teljes v4 meres reset
Q kilepes

Pontfelvetel:
1) Menj a kijelolt ponthoz, varj amig MENTHETO.
2) C -> "1/2 OK - UGYANITT ISMET C"
3) Maradj ugyanott, ujra MENTHETO -> C
4) csak ekkor lesz vegleges MENTVE.

A topkamera guide magassagfuggo:
LOW kisebb, MID kozepes, HIGH nagyobb.
"""

import csv
import json
import math
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import msvcrt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
UDP_IP = "127.0.0.1"

# ============================================================
# CHILDREN
# ============================================================

DEPTH_SCRIPT = BASE_DIR / "02_openni_depth_xyz_bridge_v4_calibration_diag.py"
RGB_SCRIPT = BASE_DIR / "01_rgb_mediapipe_calibration_sender_v6_guided.py"
TOP_SCRIPT = BASE_DIR / "03_top_camera_hand_sender_v5_adaptive_guide.py"

DEPTH_WARMUP_S = 5.0
RGB_WARMUP_S = 3.0
TOP_WARMUP_S = 1.5

CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

# ============================================================
# UDP
# ============================================================

ASTRA_XYZ_PORT = 50010
ASTRA_XYZ_FORMAT = "<IfffB"
ASTRA_XYZ_SIZE = struct.calcsize(ASTRA_XYZ_FORMAT)

TOP_PORT = 50020
TOP_FORMAT = "<IfffB"
TOP_SIZE = struct.calcsize(TOP_FORMAT)

DIAG_PORT = 50012
DIAG_FORMAT = "<IIffiiHHHHB"
DIAG_SIZE = struct.calcsize(DIAG_FORMAT)

ASTRA_STATUS_PORT = 50031
TOP_STATUS_PORT = 50032
CAL_KEY_PORT = 50033

# ============================================================
# OUTPUT
# ============================================================

CAL_JSON = BASE_DIR / "top_astra_multiplane_calibration_v4.json"
SAMPLES_CSV = BASE_DIR / "top_astra_multiplane_samples_v4.csv"
VALIDATION_CSV = BASE_DIR / "top_astra_multiplane_validation_v4.csv"
EVENT_CSV = BASE_DIR / "top_astra_multiplane_events_v4.csv"

# ============================================================
# PARAMETERS
# ============================================================

MAX_ASTRA_AGE_S = 0.18
MAX_TOP_AGE_S = 0.18
MAX_DIAG_AGE_S = 0.18
MAX_PAIR_SKEW_S = 0.080

CAPTURE_WINDOW_S = 0.55
MIN_WINDOW_SAMPLES = 6

MAX_ASTRA_X_STD_MM = 12.0
MAX_ASTRA_Y_STD_MM = 12.0
MAX_ASTRA_Z_STD_MM = 15.0
MAX_TOP_U_STD = 0.015
MAX_TOP_V_STD = 0.015
MAX_PAIR_SKEW_CAPTURE_MS = 60.0

PLANE_Y_TOLERANCE_MM = 25.0
MIN_PLANE_SEPARATION_MM = 55.0
REQUIRED_POINTS_PER_PLANE = 9

# Ket egymast koveto, ugyanazon pontra vegzett meres.
CONFIRM_TOP_DISTANCE_MAX = 0.018
CONFIRM_X_DIFF_MAX_MM = 45.0
CONFIRM_Y_DIFF_MAX_MM = 22.0
CONFIRM_Z_DIFF_MAX_MM = 55.0

# Korabbi tipusu hiba: top-kepen szinte ugyanott, Astra XZ-ben nagyon tavol.
NEAR_TOP_DISTANCE_MAX = 0.030
NEAR_TOP_XZ_DISTANCE_MAX_MM = 100.0

DEPTH_PACKET_WATCHDOG_S = 1.30
STACK_START_GRACE_S = 8.0

UI_PERIOD_S = 0.10
STATUS_SEND_PERIOD_S = 0.05
VALIDATION_LOG_PERIOD_S = 0.10

PLANE_KEYS = {
    "1": "LOW",
    "2": "MID",
    "3": "HIGH",
}
PLANE_ORDER = ["LOW", "MID", "HIGH"]


# ============================================================
# PROCESS MANAGEMENT
# ============================================================

def check_required_scripts():
    missing = [
        p.name
        for p in (DEPTH_SCRIPT, RGB_SCRIPT, TOP_SCRIPT)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Hianyzo fajl(ok): " + ", ".join(missing)
        )


def launch_child(script_path, label):
    print(f"[START] {label}: {script_path.name}")
    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        creationflags=CREATE_NEW_CONSOLE,
    )
    time.sleep(0.35)
    if proc.poll() is not None:
        raise RuntimeError(
            f"{label} azonnal leallt (exit={proc.returncode})."
        )
    print(f"[OK] {label} | PID={proc.pid}")
    return proc


def wait_child(seconds, proc, label):
    deadline = time.monotonic() + float(seconds)

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"{label} warm-up alatt leallt "
                f"(exit={proc.returncode})."
            )

        remaining = max(0.0, deadline - time.monotonic())
        print(
            f"\r[WARM-UP] {label}: {remaining:4.1f}s ",
            end="",
            flush=True,
        )
        time.sleep(0.10)

    print(f"\r[WARM-UP] {label}: kesz.     ")


def start_camera_stack():
    check_required_scripts()
    children = []

    try:
        p = launch_child(DEPTH_SCRIPT, "OPENNI DEPTH")
        children.append(("OPENNI DEPTH", p))
        wait_child(DEPTH_WARMUP_S, p, "OPENNI DEPTH")

        p = launch_child(RGB_SCRIPT, "ASTRA RGB")
        children.append(("ASTRA RGB", p))
        wait_child(RGB_WARMUP_S, p, "ASTRA RGB")

        p = launch_child(TOP_SCRIPT, "ETERNICO TOP")
        children.append(("ETERNICO TOP", p))
        wait_child(TOP_WARMUP_S, p, "ETERNICO TOP")

        print()
        print("[OK] Kamera stack kesz.")
        return children

    except Exception:
        stop_children(children)
        raise


def stop_children(children):
    if not children:
        return

    for _, proc in reversed(children):
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    deadline = time.monotonic() + 2.0

    while time.monotonic() < deadline:
        if all(proc.poll() is not None for _, proc in children):
            break
        time.sleep(0.05)

    for _, proc in reversed(children):
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass


def check_children_alive(children):
    dead = []
    for label, proc in children:
        if proc.poll() is not None:
            dead.append(f"{label} (exit={proc.returncode})")

    if dead:
        raise RuntimeError(
            "Alrendszer leallt: " + "; ".join(dead)
        )


# ============================================================
# UDP
# ============================================================

def make_listener(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_IP, port))
    sock.setblocking(False)
    return sock


def drain_latest(sock, size, fmt):
    latest = None
    arrival = None

    while True:
        try:
            data, _ = sock.recvfrom(512)
        except BlockingIOError:
            break

        if len(data) != size:
            continue

        latest = struct.unpack(fmt, data)
        arrival = time.monotonic()

    return latest, arrival


def drain_key_events(sock):
    keys = []

    while True:
        try:
            data, _ = sock.recvfrom(32)
        except BlockingIOError:
            break

        try:
            ch = data.decode("ascii").strip().lower()
        except Exception:
            continue

        if ch:
            keys.append(ch[0])

    return keys


def send_status(sock, port, payload):
    try:
        data = json.dumps(
            payload,
            separators=(",", ":"),
        ).encode("utf-8")

        sock.sendto(data, (UDP_IP, port))
    except Exception:
        pass


# ============================================================
# LOGS / RESUME
# ============================================================

SAMPLE_FIELDS = [
    "sample_id",
    "plane",
    "plane_point_index",
    "pc_time",
    "top_u_norm",
    "top_v_norm",
    "top_palm_width_norm",
    "astra_x_mm",
    "astra_y_mm",
    "astra_z_mm",
    "astra_seq",
    "top_seq",
    "window_n",
    "astra_x_std_mm",
    "astra_y_std_mm",
    "astra_z_std_mm",
    "top_u_std",
    "top_v_std",
    "pair_skew_median_ms",
    "pair_skew_max_ms",
    "plane_reference_y_mm",
    "plane_y_error_mm",
    "confirm_top_distance",
    "confirm_x_diff_mm",
    "confirm_y_diff_mm",
    "confirm_z_diff_mm",
    "diag_bridge_seq",
    "diag_palm_seq",
    "astra_rgb_palm_u",
    "astra_rgb_palm_v",
    "depth_px",
    "depth_py",
    "depth_patch_valid_count",
    "depth_median_mm",
    "depth_patch_min_mm",
    "depth_patch_max_mm",
]


def write_samples(samples):
    with open(
        SAMPLES_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        w = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
        w.writeheader()
        w.writerows(samples)


def load_existing_samples():
    if not SAMPLES_CSV.exists():
        return []

    rows = []

    with open(
        SAMPLES_CSV,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        for row in csv.DictReader(f):
            if row:
                rows.append(dict(row))

    # A numerikus mezoket float/int tipussa visszaallitjuk.
    int_fields = {
        "sample_id",
        "plane_point_index",
        "astra_seq",
        "top_seq",
        "window_n",
        "diag_bridge_seq",
        "diag_palm_seq",
        "depth_px",
        "depth_py",
        "depth_patch_valid_count",
        "depth_median_mm",
        "depth_patch_min_mm",
        "depth_patch_max_mm",
    }

    for row in rows:
        for key in SAMPLE_FIELDS:
            if key in ("plane", "pc_time"):
                continue

            value = row.get(key, "")
            if value in ("", None):
                row[key] = ""
                continue

            try:
                if key in int_fields:
                    row[key] = int(float(value))
                else:
                    row[key] = float(value)
            except Exception:
                pass

    return rows


def restore_plane_refs(samples):
    refs = {p: None for p in PLANE_ORDER}

    for plane in PLANE_ORDER:
        vals = [
            float(s["plane_reference_y_mm"])
            for s in samples
            if (
                s.get("plane") == plane
                and s.get("plane_reference_y_mm") not in ("", None)
            )
        ]

        if vals:
            refs[plane] = float(np.median(vals))

    return refs


def open_append_csv(path, fields):
    exists = path.exists() and path.stat().st_size > 0

    f = open(
        path,
        "a",
        newline="",
        encoding="utf-8-sig",
    )

    w = csv.DictWriter(f, fieldnames=fields)

    if not exists:
        w.writeheader()
        f.flush()

    return f, w


# ============================================================
# CAPTURE
# ============================================================

def capture_stats(history):
    cutoff = time.monotonic() - CAPTURE_WINDOW_S
    window = [r for r in history if r["t"] >= cutoff]

    if len(window) < MIN_WINDOW_SAMPLES:
        return None

    keys = [
        "top_u",
        "top_v",
        "top_width",
        "astra_x",
        "astra_y",
        "astra_z",
        "pair_skew_ms",
    ]

    arr = {
        k: np.array([float(r[k]) for r in window], dtype=np.float64)
        for k in keys
    }

    return {
        "window": window,
        "med": {k: float(np.median(v)) for k, v in arr.items()},
        "std": {k: float(np.std(v)) for k, v in arr.items()},
        "max": {k: float(np.max(v)) for k, v in arr.items()},
    }


def evaluate_ready(
    history,
    active_plane,
    plane_refs,
    astra_fresh,
    top_fresh,
):
    if not astra_fresh:
        return False, "ASTRA LOST", None

    if not top_fresh:
        return False, "TOP LOST", None

    st = capture_stats(history)

    if st is None:
        return False, "VARJ 1-2 SEC", None

    sd = st["std"]

    if (
        sd["astra_x"] > MAX_ASTRA_X_STD_MM
        or sd["astra_y"] > MAX_ASTRA_Y_STD_MM
        or sd["astra_z"] > MAX_ASTRA_Z_STD_MM
        or sd["top_u"] > MAX_TOP_U_STD
        or sd["top_v"] > MAX_TOP_V_STD
    ):
        return False, "MOZOG A KEZ", st

    if st["max"]["pair_skew_ms"] > MAX_PAIR_SKEW_CAPTURE_MS:
        return False, "KAMERA SZINKRON VARJ", st

    ref_y = plane_refs.get(active_plane)

    if ref_y is not None:
        dy = st["med"]["astra_y"] - ref_y

        if abs(dy) > PLANE_Y_TOLERANCE_MM:
            return False, f"ROSSZ MAGASSAG {dy:+.0f}mm", st

    return True, "MENTHETO - C", st


def diag_fields_from_window(window):
    # Csak a matching bridge-seq diag rekordokat tesszuk a history-ba,
    # igy itt egyszeruen medianolhatunk.
    fields = [
        "diag_bridge_seq",
        "diag_palm_seq",
        "astra_rgb_palm_u",
        "astra_rgb_palm_v",
        "depth_px",
        "depth_py",
        "depth_patch_valid_count",
        "depth_median_mm",
        "depth_patch_min_mm",
        "depth_patch_max_mm",
    ]

    result = {}

    for field in fields:
        vals = [
            float(r[field])
            for r in window
            if r.get(field) not in (None, "")
        ]

        if vals:
            result[field] = float(np.median(vals))
        else:
            result[field] = ""

    return result


def build_capture_row(stats):
    window = stats["window"]
    med = stats["med"]
    sd = stats["std"]
    last = window[-1]

    row = {
        "pc_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_u_norm": med["top_u"],
        "top_v_norm": med["top_v"],
        "top_palm_width_norm": med["top_width"],
        "astra_x_mm": med["astra_x"],
        "astra_y_mm": med["astra_y"],
        "astra_z_mm": med["astra_z"],
        "astra_seq": int(last["astra_seq"]),
        "top_seq": int(last["top_seq"]),
        "window_n": len(window),
        "astra_x_std_mm": sd["astra_x"],
        "astra_y_std_mm": sd["astra_y"],
        "astra_z_std_mm": sd["astra_z"],
        "top_u_std": sd["top_u"],
        "top_v_std": sd["top_v"],
        "pair_skew_median_ms": med["pair_skew_ms"],
        "pair_skew_max_ms": stats["max"]["pair_skew_ms"],
    }

    row.update(diag_fields_from_window(window))
    return row


def compare_repeat(a, b):
    du = float(a["top_u_norm"]) - float(b["top_u_norm"])
    dv = float(a["top_v_norm"]) - float(b["top_v_norm"])
    top_dist = math.hypot(du, dv)

    dx = abs(float(a["astra_x_mm"]) - float(b["astra_x_mm"]))
    dy = abs(float(a["astra_y_mm"]) - float(b["astra_y_mm"]))
    dz = abs(float(a["astra_z_mm"]) - float(b["astra_z_mm"]))

    ok = (
        top_dist <= CONFIRM_TOP_DISTANCE_MAX
        and dx <= CONFIRM_X_DIFF_MAX_MM
        and dy <= CONFIRM_Y_DIFF_MAX_MM
        and dz <= CONFIRM_Z_DIFF_MAX_MM
    )

    return ok, {
        "top_dist": top_dist,
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }


def merge_repeat(a, b, cmp):
    out = {}

    average_fields = [
        "top_u_norm",
        "top_v_norm",
        "top_palm_width_norm",
        "astra_x_mm",
        "astra_y_mm",
        "astra_z_mm",
        "astra_x_std_mm",
        "astra_y_std_mm",
        "astra_z_std_mm",
        "top_u_std",
        "top_v_std",
        "pair_skew_median_ms",
        "pair_skew_max_ms",
        "astra_rgb_palm_u",
        "astra_rgb_palm_v",
        "depth_px",
        "depth_py",
        "depth_patch_valid_count",
        "depth_median_mm",
        "depth_patch_min_mm",
        "depth_patch_max_mm",
    ]

    for key in average_fields:
        vals = []

        for src in (a, b):
            value = src.get(key, "")
            if value not in ("", None):
                vals.append(float(value))

        out[key] = (
            float(np.mean(vals))
            if vals
            else ""
        )

    out["pc_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["astra_seq"] = int(b["astra_seq"])
    out["top_seq"] = int(b["top_seq"])
    out["window_n"] = int(a["window_n"]) + int(b["window_n"])

    for key in ("diag_bridge_seq", "diag_palm_seq"):
        value = b.get(key, "")
        out[key] = (
            int(round(float(value)))
            if value not in ("", None)
            else ""
        )

    out["confirm_top_distance"] = cmp["top_dist"]
    out["confirm_x_diff_mm"] = cmp["dx"]
    out["confirm_y_diff_mm"] = cmp["dy"]
    out["confirm_z_diff_mm"] = cmp["dz"]

    return out


def close_point_consistency(candidate, samples, plane):
    for s in samples:
        if s.get("plane") != plane:
            continue

        du = float(candidate["top_u_norm"]) - float(s["top_u_norm"])
        dv = float(candidate["top_v_norm"]) - float(s["top_v_norm"])
        dtop = math.hypot(du, dv)

        if dtop > NEAR_TOP_DISTANCE_MAX:
            continue

        dx = float(candidate["astra_x_mm"]) - float(s["astra_x_mm"])
        dz = float(candidate["astra_z_mm"]) - float(s["astra_z_mm"])
        dxz = math.hypot(dx, dz)

        if dxz > NEAR_TOP_XZ_DISTANCE_MAX_MM:
            return False, (
                f"KOZELI TOP PONT, DE XZ ELTER {dxz:.0f}mm "
                f"(top d={dtop:.4f})"
            )

    return True, ""


# ============================================================
# HOMOGRAPHY
# ============================================================

def predict_homography(H, u, v):
    src = np.array(
        [[[
            float(u),
            float(v),
        ]]],
        dtype=np.float32,
    )

    dst = cv2.perspectiveTransform(src, H)

    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


def fit_plane(plane_samples):
    if len(plane_samples) < REQUIRED_POINTS_PER_PLANE:
        raise ValueError(
            f"{plane_samples[0]['plane'] if plane_samples else '?'}: "
            f"{len(plane_samples)}/{REQUIRED_POINTS_PER_PLANE} pont."
        )

    src = np.array(
        [
            [float(s["top_u_norm"]), float(s["top_v_norm"])]
            for s in plane_samples
        ],
        dtype=np.float32,
    )

    dst = np.array(
        [
            [float(s["astra_x_mm"]), float(s["astra_z_mm"])]
            for s in plane_samples
        ],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(src, dst, method=0)

    if H is None:
        raise ValueError("cv2.findHomography() nem adott megoldast.")

    pred = cv2.perspectiveTransform(
        src.reshape(-1, 1, 2),
        H,
    ).reshape(-1, 2)

    err = pred - dst
    e2d = np.linalg.norm(err, axis=1)

    stats = {
        "n": len(plane_samples),
        "rmse_x_mm": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rmse_z_mm": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "rmse_2d_mm": float(np.sqrt(np.mean(e2d ** 2))),
        "median_2d_mm": float(np.median(e2d)),
        "max_2d_mm": float(np.max(e2d)),
    }

    plane_y = float(np.median(
        [float(s["astra_y_mm"]) for s in plane_samples]
    ))

    plane_y_std = float(np.std(
        [float(s["astra_y_mm"]) for s in plane_samples]
    ))

    return H, plane_y, plane_y_std, stats


def fit_all(samples):
    models = {}

    for plane in PLANE_ORDER:
        subset = [s for s in samples if s.get("plane") == plane]

        if len(subset) != REQUIRED_POINTS_PER_PLANE:
            raise ValueError(
                f"{plane}: {len(subset)}/{REQUIRED_POINTS_PER_PLANE} pont."
            )

        H, y, ystd, stats = fit_plane(subset)

        models[plane] = {
            "label": plane,
            "H": H,
            "plane_y_mm": y,
            "plane_y_std_mm": ystd,
            "stats": stats,
        }

    ordered = sorted(models.values(), key=lambda m: m["plane_y_mm"])
    gaps = [
        ordered[i + 1]["plane_y_mm"] - ordered[i]["plane_y_mm"]
        for i in range(2)
    ]

    if min(gaps) < MIN_PLANE_SEPARATION_MM:
        raise ValueError(
            f"Sikok tul kozel vannak: min gap={min(gaps):.1f}mm."
        )

    return models


def predict_interpolated(models, u, v, y):
    ordered = sorted(models.values(), key=lambda m: m["plane_y_mm"])
    y = float(y)

    if y <= ordered[0]["plane_y_mm"]:
        lo, hi = ordered[0], ordered[1]
    elif y >= ordered[-1]["plane_y_mm"]:
        lo, hi = ordered[-2], ordered[-1]
    else:
        lo, hi = ordered[0], ordered[1]

        for a, b in zip(ordered[:-1], ordered[1:]):
            if a["plane_y_mm"] <= y <= b["plane_y_mm"]:
                lo, hi = a, b
                break

    y0 = lo["plane_y_mm"]
    y1 = hi["plane_y_mm"]

    alpha = 0.0 if abs(y1 - y0) < 1e-9 else (y - y0) / (y1 - y0)
    alpha = max(-0.5, min(1.5, alpha))

    x0, z0 = predict_homography(lo["H"], u, v)
    x1, z1 = predict_homography(hi["H"], u, v)

    return (
        x0 + alpha * (x1 - x0),
        z0 + alpha * (z1 - z0),
        lo["label"],
        hi["label"],
        alpha,
    )


def save_calibration(models, samples):
    payload = {
        "version": 4,
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": (
            "Quality-controlled three-plane OpenCV homography: "
            "top(u,v)->Astra(X,Z), interpolation by Astra Y."
        ),
        "required_points_per_plane": REQUIRED_POINTS_PER_PLANE,
        "double_measurement": True,
        "plane_y_tolerance_mm": PLANE_Y_TOLERANCE_MM,
        "sample_count": len(samples),
        "planes": {},
    }

    for label, model in models.items():
        payload["planes"][label] = {
            "plane_y_mm": model["plane_y_mm"],
            "plane_y_std_mm": model["plane_y_std_mm"],
            "homography_3x3": model["H"].tolist(),
            "fit_stats": model["stats"],
        }

    CAL_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("TOP -> ASTRA MULTIPLANE CALIBRATION V4 QUALITY")
    print("=" * 78)

    astra_sock = make_listener(ASTRA_XYZ_PORT)
    top_sock = make_listener(TOP_PORT)
    diag_sock = make_listener(DIAG_PORT)
    key_sock = make_listener(CAL_KEY_PORT)

    status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    children = start_camera_stack()
    stack_started = time.monotonic()

    samples = load_existing_samples()
    plane_refs = restore_plane_refs(samples)

    if samples:
        counts0 = {
            p: sum(1 for s in samples if s.get("plane") == p)
            for p in PLANE_ORDER
        }
        print(
            "[RESUME] "
            f"L/M/H={counts0['LOW']}/{counts0['MID']}/{counts0['HIGH']} "
            f"| osszes={len(samples)}"
        )

    active_plane = "MID"
    models = {}
    validation_on = False
    pending = None

    latest_astra = None
    latest_astra_arrival = None
    latest_top = None
    latest_top_arrival = None
    latest_diag = None
    latest_diag_arrival = None

    history = []
    last_pair = None

    flash_id = 0
    flash_text = ""
    flash_kind = "info"

    run_t0 = time.monotonic()
    last_status_send = 0.0
    last_terminal = 0.0
    last_validation_log = 0.0

    event_fields = [
        "pc_time",
        "elapsed_s",
        "event",
        "plane",
        "detail",
    ]
    event_file, event_writer = open_append_csv(EVENT_CSV, event_fields)

    validation_fields = [
        "pc_time",
        "elapsed_s",
        "top_u_norm",
        "top_v_norm",
        "top_palm_width_norm",
        "astra_x_mm",
        "astra_y_mm",
        "astra_z_mm",
        "pred_x_mm",
        "pred_z_mm",
        "error_x_mm",
        "error_z_mm",
        "error_2d_mm",
        "lower_plane",
        "upper_plane",
        "interp_alpha",
        "pair_skew_ms",
        "diag_bridge_seq",
        "diag_palm_seq",
        "astra_rgb_palm_u",
        "astra_rgb_palm_v",
        "depth_px",
        "depth_py",
        "depth_patch_valid_count",
        "depth_median_mm",
        "depth_patch_min_mm",
        "depth_patch_max_mm",
    ]
    validation_file, validation_writer = open_append_csv(
        VALIDATION_CSV,
        validation_fields,
    )

    def log_event(event, detail=""):
        event_writer.writerow({
            "pc_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": f"{time.monotonic()-run_t0:.3f}",
            "event": event,
            "plane": active_plane,
            "detail": detail,
        })
        event_file.flush()

    try:
        while True:
            now = time.monotonic()

            check_children_alive(children)

            a, ta = drain_latest(
                astra_sock,
                ASTRA_XYZ_SIZE,
                ASTRA_XYZ_FORMAT,
            )
            if a is not None:
                seq, x, y, z, valid = a
                latest_astra = {
                    "seq": int(seq),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "valid": bool(valid),
                }
                latest_astra_arrival = ta

            b, tb = drain_latest(
                top_sock,
                TOP_SIZE,
                TOP_FORMAT,
            )
            if b is not None:
                seq, u, v, width, valid = b
                latest_top = {
                    "seq": int(seq),
                    "u": float(u),
                    "v": float(v),
                    "width": float(width),
                    "valid": bool(valid),
                }
                latest_top_arrival = tb

            d, td = drain_latest(
                diag_sock,
                DIAG_SIZE,
                DIAG_FORMAT,
            )
            if d is not None:
                (
                    bridge_seq,
                    palm_seq,
                    palm_u,
                    palm_v,
                    depth_px,
                    depth_py,
                    patch_count,
                    depth_median,
                    patch_min,
                    patch_max,
                    diag_valid,
                ) = d

                latest_diag = {
                    "bridge_seq": int(bridge_seq),
                    "palm_seq": int(palm_seq),
                    "palm_u": float(palm_u),
                    "palm_v": float(palm_v),
                    "depth_px": int(depth_px),
                    "depth_py": int(depth_py),
                    "patch_count": int(patch_count),
                    "depth_median": int(depth_median),
                    "patch_min": int(patch_min),
                    "patch_max": int(patch_max),
                    "valid": bool(diag_valid),
                }
                latest_diag_arrival = td

            # OpenNI befagyas: processz elhet, de packet nem jon.
            if (
                now - stack_started > STACK_START_GRACE_S
                and (
                    latest_astra_arrival is None
                    or now - latest_astra_arrival > DEPTH_PACKET_WATCHDOG_S
                )
            ):
                log_event(
                    "DEPTH_WATCHDOG_RESTART",
                    (
                        "no XYZ packets for "
                        f">{DEPTH_PACKET_WATCHDOG_S:.2f}s"
                    ),
                )

                flash_id += 1
                flash_text = "DEPTH RESTART"
                flash_kind = "error"

                stop_children(children)
                time.sleep(0.5)
                children = start_camera_stack()
                stack_started = time.monotonic()

                latest_astra = None
                latest_astra_arrival = None
                latest_top = None
                latest_top_arrival = None
                latest_diag = None
                latest_diag_arrival = None
                history.clear()
                last_pair = None
                pending = None
                continue

            astra_fresh = bool(
                latest_astra is not None
                and latest_astra_arrival is not None
                and now - latest_astra_arrival <= MAX_ASTRA_AGE_S
                and latest_astra["valid"]
            )

            top_fresh = bool(
                latest_top is not None
                and latest_top_arrival is not None
                and now - latest_top_arrival <= MAX_TOP_AGE_S
                and latest_top["valid"]
            )

            # Csak uj es idoben kozel allo packet-par kerul a capture history-ba.
            if astra_fresh and top_fresh:
                pair_id = (
                    latest_astra["seq"],
                    latest_top["seq"],
                )
                skew_s = abs(
                    latest_astra_arrival - latest_top_arrival
                )

                if (
                    pair_id != last_pair
                    and skew_s <= MAX_PAIR_SKEW_S
                ):
                    row = {
                        "t": now,
                        "astra_seq": latest_astra["seq"],
                        "top_seq": latest_top["seq"],
                        "astra_x": latest_astra["x"],
                        "astra_y": latest_astra["y"],
                        "astra_z": latest_astra["z"],
                        "top_u": latest_top["u"],
                        "top_v": latest_top["v"],
                        "top_width": latest_top["width"],
                        "pair_skew_ms": skew_s * 1000.0,
                    }

                    if (
                        latest_diag is not None
                        and latest_diag_arrival is not None
                        and now - latest_diag_arrival <= MAX_DIAG_AGE_S
                        and latest_diag["bridge_seq"] == latest_astra["seq"]
                    ):
                        row.update({
                            "diag_bridge_seq": latest_diag["bridge_seq"],
                            "diag_palm_seq": latest_diag["palm_seq"],
                            "astra_rgb_palm_u": latest_diag["palm_u"],
                            "astra_rgb_palm_v": latest_diag["palm_v"],
                            "depth_px": latest_diag["depth_px"],
                            "depth_py": latest_diag["depth_py"],
                            "depth_patch_valid_count": latest_diag["patch_count"],
                            "depth_median_mm": latest_diag["depth_median"],
                            "depth_patch_min_mm": latest_diag["patch_min"],
                            "depth_patch_max_mm": latest_diag["patch_max"],
                        })

                    history.append(row)
                    last_pair = pair_id

                    cutoff = now - 2.0
                    while history and history[0]["t"] < cutoff:
                        history.pop(0)

            ready, reason, st = evaluate_ready(
                history,
                active_plane,
                plane_refs,
                astra_fresh,
                top_fresh,
            )

            # Ha van pending elso meres, megnezzuk, hogy ugyanazon ponton maradt-e.
            confirm_stage = 1
            if pending is not None:
                confirm_stage = 2

                if ready and st is not None:
                    preview = build_capture_row(st)
                    ok_repeat, cmp = compare_repeat(pending, preview)

                    if not ok_repeat:
                        reason = (
                            "TERJ VISSZA A PONTRA "
                            f"dZ={cmp['dz']:.0f}mm"
                        )
                        ready = False
                    else:
                        reason = "UGYANITT ISMET C"

            # Billentyuk: kameraablakok + terminal.
            keys = drain_key_events(key_sock)

            if msvcrt.kbhit():
                keys.append(msvcrt.getwch().lower())

            for key in keys:
                if key == "q":
                    return

                if key in PLANE_KEYS:
                    active_plane = PLANE_KEYS[key]
                    pending = None
                    flash_id += 1
                    flash_text = active_plane
                    flash_kind = "info"
                    continue

                if key == "u":
                    if pending is not None:
                        pending = None
                        flash_id += 1
                        flash_text = "1/2 TOROLVE"
                        flash_kind = "info"
                        continue

                    if samples:
                        removed = samples.pop()

                        if not any(
                            s.get("plane") == removed.get("plane")
                            for s in samples
                        ):
                            plane_refs[removed["plane"]] = None

                        # ujraszamozas
                        pc = {p: 0 for p in PLANE_ORDER}
                        for i, s in enumerate(samples, 1):
                            s["sample_id"] = i
                            pc[s["plane"]] += 1
                            s["plane_point_index"] = pc[s["plane"]]

                        write_samples(samples)
                        flash_id += 1
                        flash_text = "UNDO"
                        flash_kind = "info"
                    continue

                if key == "r":
                    samples.clear()
                    plane_refs = {p: None for p in PLANE_ORDER}
                    models = {}
                    validation_on = False
                    pending = None
                    write_samples(samples)

                    flash_id += 1
                    flash_text = "RESET"
                    flash_kind = "info"
                    log_event("RESET", "")
                    continue

                if key == "c":
                    ready_now, reason_now, st_now = evaluate_ready(
                        history,
                        active_plane,
                        plane_refs,
                        astra_fresh,
                        top_fresh,
                    )

                    if not ready_now or st_now is None:
                        flash_id += 1
                        flash_text = "NEM MENTVE"
                        flash_kind = "error"
                        log_event("SAVE_REJECT", reason_now)
                        continue

                    current = build_capture_row(st_now)

                    if pending is None:
                        pending = current
                        flash_id += 1
                        flash_text = "1/2 OK - UGYANITT MARADJ"
                        flash_kind = "info"
                        log_event(
                            "POINT_FIRST_MEASUREMENT",
                            (
                                f"u={current['top_u_norm']:.4f}; "
                                f"v={current['top_v_norm']:.4f}; "
                                f"X={current['astra_x_mm']:.1f}; "
                                f"Y={current['astra_y_mm']:.1f}; "
                                f"Z={current['astra_z_mm']:.1f}"
                            ),
                        )
                        continue

                    ok_repeat, cmp = compare_repeat(pending, current)

                    if not ok_repeat:
                        log_event(
                            "POINT_REPEAT_REJECT",
                            (
                                f"top={cmp['top_dist']:.4f}; "
                                f"dX={cmp['dx']:.1f}; "
                                f"dY={cmp['dy']:.1f}; "
                                f"dZ={cmp['dz']:.1f}"
                            ),
                        )
                        pending = None
                        flash_id += 1
                        flash_text = "GYANUS - MERD UJRA 1/2"
                        flash_kind = "error"
                        continue

                    candidate = merge_repeat(pending, current, cmp)

                    ok_near, near_reason = close_point_consistency(
                        candidate,
                        samples,
                        active_plane,
                    )

                    if not ok_near:
                        log_event("POINT_NEAR_CONSISTENCY_REJECT", near_reason)
                        pending = None
                        flash_id += 1
                        flash_text = "GYANUS PONT - MERD UJRA"
                        flash_kind = "error"
                        continue

                    if plane_refs[active_plane] is None:
                        plane_refs[active_plane] = float(
                            candidate["astra_y_mm"]
                        )

                    ref_y = plane_refs[active_plane]
                    yerr = float(candidate["astra_y_mm"]) - ref_y

                    plane_count = (
                        sum(
                            1
                            for s in samples
                            if s.get("plane") == active_plane
                        )
                        + 1
                    )

                    if plane_count > REQUIRED_POINTS_PER_PLANE:
                        pending = None
                        flash_id += 1
                        flash_text = "EZ A SIK MAR 9/9"
                        flash_kind = "error"
                        continue

                    candidate.update({
                        "sample_id": len(samples) + 1,
                        "plane": active_plane,
                        "plane_point_index": plane_count,
                        "plane_reference_y_mm": ref_y,
                        "plane_y_error_mm": yerr,
                    })

                    # int-szeru diag mezok szepen.
                    for k in (
                        "diag_bridge_seq",
                        "diag_palm_seq",
                        "depth_px",
                        "depth_py",
                        "depth_patch_valid_count",
                        "depth_median_mm",
                        "depth_patch_min_mm",
                        "depth_patch_max_mm",
                    ):
                        if candidate.get(k, "") not in ("", None):
                            candidate[k] = int(round(float(candidate[k])))

                    samples.append(candidate)
                    write_samples(samples)
                    pending = None

                    flash_id += 1
                    flash_text = f"MENTVE {active_plane} #{plane_count}"
                    flash_kind = "ok"

                    log_event(
                        "POINT_SAVED",
                        (
                            f"idx={plane_count}; "
                            f"Y={candidate['astra_y_mm']:.1f}; "
                            f"dY={yerr:+.1f}; "
                            f"confirm_dZ={cmp['dz']:.1f}; "
                            f"depth={candidate.get('depth_median_mm','')}"
                        ),
                    )
                    continue

                if key == "f":
                    try:
                        pending = None
                        models = fit_all(samples)
                        save_calibration(models, samples)

                        print()
                        print("\nFIT V4:")
                        for m in sorted(
                            models.values(),
                            key=lambda x: x["plane_y_mm"],
                        ):
                            stf = m["stats"]
                            print(
                                f"  {m['label']:>4} | "
                                f"Y={m['plane_y_mm']:+.1f} "
                                f"stdY={m['plane_y_std_mm']:.1f} | "
                                f"RMSE2D={stf['rmse_2d_mm']:.1f}mm | "
                                f"MAX={stf['max_2d_mm']:.1f}mm"
                            )

                        flash_id += 1
                        flash_text = "FIT KESZ"
                        flash_kind = "ok"
                        log_event("FIT_OK", "")

                    except Exception as e:
                        flash_id += 1
                        flash_text = "FIT HIBA"
                        flash_kind = "error"
                        log_event("FIT_ERROR", str(e))
                        print(f"\n[FIT HIBA] {e}")
                    continue

                if key == "v":
                    if len(models) != 3:
                        flash_id += 1
                        flash_text = "ELOBB F"
                        flash_kind = "error"
                    else:
                        validation_on = not validation_on
                        flash_id += 1
                        flash_text = (
                            "VALIDACIO ON"
                            if validation_on
                            else "VALIDACIO OFF"
                        )
                        flash_kind = "info"
                    continue

            # Live validation.
            live_error = None

            if len(models) == 3 and astra_fresh and top_fresh:
                pred_x, pred_z, lo, hi, alpha = predict_interpolated(
                    models,
                    latest_top["u"],
                    latest_top["v"],
                    latest_astra["y"],
                )

                ex = pred_x - latest_astra["x"]
                ez = pred_z - latest_astra["z"]
                live_error = math.hypot(ex, ez)

                if (
                    validation_on
                    and now - last_validation_log >= VALIDATION_LOG_PERIOD_S
                ):
                    last_validation_log = now

                    row = {
                        "pc_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "elapsed_s": f"{now-run_t0:.3f}",
                        "top_u_norm": f"{latest_top['u']:.6f}",
                        "top_v_norm": f"{latest_top['v']:.6f}",
                        "top_palm_width_norm": f"{latest_top['width']:.6f}",
                        "astra_x_mm": f"{latest_astra['x']:.3f}",
                        "astra_y_mm": f"{latest_astra['y']:.3f}",
                        "astra_z_mm": f"{latest_astra['z']:.3f}",
                        "pred_x_mm": f"{pred_x:.3f}",
                        "pred_z_mm": f"{pred_z:.3f}",
                        "error_x_mm": f"{ex:.3f}",
                        "error_z_mm": f"{ez:.3f}",
                        "error_2d_mm": f"{live_error:.3f}",
                        "lower_plane": lo,
                        "upper_plane": hi,
                        "interp_alpha": f"{alpha:.6f}",
                        "pair_skew_ms": (
                            ""
                            if (
                                latest_astra_arrival is None
                                or latest_top_arrival is None
                            )
                            else f"{abs(latest_astra_arrival-latest_top_arrival)*1000.0:.3f}"
                        ),
                    }

                    if latest_diag is not None:
                        row.update({
                            "diag_bridge_seq": latest_diag["bridge_seq"],
                            "diag_palm_seq": latest_diag["palm_seq"],
                            "astra_rgb_palm_u": f"{latest_diag['palm_u']:.6f}",
                            "astra_rgb_palm_v": f"{latest_diag['palm_v']:.6f}",
                            "depth_px": latest_diag["depth_px"],
                            "depth_py": latest_diag["depth_py"],
                            "depth_patch_valid_count": latest_diag["patch_count"],
                            "depth_median_mm": latest_diag["depth_median"],
                            "depth_patch_min_mm": latest_diag["patch_min"],
                            "depth_patch_max_mm": latest_diag["patch_max"],
                        })

                    validation_writer.writerow(row)
                    validation_file.flush()

            counts = {
                p: sum(1 for s in samples if s.get("plane") == p)
                for p in PLANE_ORDER
            }

            current_y = (
                None
                if st is None
                else st["med"]["astra_y"]
            )
            target_y = plane_refs[active_plane]

            if now - last_status_send >= STATUS_SEND_PERIOD_S:
                last_status_send = now

                payload = {
                    "astra_ok": astra_fresh,
                    "top_ok": top_fresh,
                    "can_save": ready,
                    "reason": reason,
                    "plane": active_plane,
                    "counts": counts,
                    "current_y_mm": current_y,
                    "target_y_mm": target_y,
                    "validation_on": validation_on,
                    "live_error_mm": live_error,
                    "confirm_stage": confirm_stage,
                    "flash_id": flash_id,
                    "flash_text": flash_text,
                    "flash_kind": flash_kind,
                }

                send_status(status_sock, ASTRA_STATUS_PORT, payload)
                send_status(status_sock, TOP_STATUS_PORT, payload)

            if now - last_terminal >= UI_PERIOD_S:
                last_terminal = now

                ytxt = "--" if current_y is None else f"{current_y:+.0f}"
                reftxt = "--" if target_y is None else f"{target_y:+.0f}"

                line = (
                    f"{active_plane:<4} "
                    f"L/M/H={counts['LOW']}/{counts['MID']}/{counts['HIGH']} | "
                    f"A:{'OK' if astra_fresh else '--'} "
                    f"T:{'OK' if top_fresh else '--'} | "
                    f"Y={ytxt}/{reftxt} | "
                    f"{reason} | {confirm_stage}/2"
                )

                if validation_on and live_error is not None:
                    line += f" | VAL={live_error:.0f}mm"

                print(
                    "\r" + line[:160] + " " * 20,
                    end="",
                    flush=True,
                )

            time.sleep(0.002)

    finally:
        print()
        try:
            astra_sock.close()
            top_sock.close()
            diag_sock.close()
            key_sock.close()
            status_sock.close()
        except Exception:
            pass

        try:
            event_file.close()
            validation_file.close()
        except Exception:
            pass

        stop_children(children)
        print("V4 quality kalibralo vege.")


if __name__ == "__main__":
    main()
