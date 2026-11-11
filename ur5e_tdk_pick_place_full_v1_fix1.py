import ctypes
import csv
import json
import math
import socket
import struct
import subprocess
import serial
import serial.tools.list_ports
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIO


# ============================================================
# UR5e TDK PICK-AND-PLACE + ASTRA/TOP + IMU + RG2 - V1
#
# Inditasi lanc:
#
# 1) OpenNI depth KULON WINDOWS KONZOLBAN
# 2) 5 masodperc warm-up
# 3) RGB + MediaPipe KULON WINDOWS KONZOLBAN
# 4) 3 masodperc warm-up
# 5) varunk tenyleges, ervenyes XYZ UDP-re
# 6) csak ezutan RTDE robot
#
# Ez pontosan a laborban kezzel stabilnak bizonyult sorrendet koveti:
# DEPTH -> varakozas -> RGB -> ervenyes XYZ -> ROBOT.
#
# V13 - DEMO SAFE / PLUSSMACI A -> B
#
# DEMO CEL:
#   megbizhato XYZ kezkovetes + ESP32 IMU orientacio + stabil OPEN/CLOSED RG2
#
# TUDATOS EGYSZERUSITES:
# - XYZ: Astra depth + kamera
# - orientacio: ESP32 quaternion IMU
# - C-kor a HOME TCP-orientacioja a robot referencia
# - a robot megfogojat C elott allitsd a pluss megfogasahoz
#   megfelelo, pl. lefele nezo biztonsagos helyzetbe
#
# ROBUSZTUSSAG:
# - IK elorellenorzes minden uj servo-cel elott
# - az aktualis joint-allashoz kozeli IK megoldast kerunk
# - elerhetetlen / mas IK-agra atugro / joint-hataros celt nem engedunk
# - rossz celnal a robot az utolso jo pozt tartja, tracking ON marad
# - Wrist 2 singularitasvedo zona
# - nagy kamera XYZ frame-ugras eldobasa
# - elforditott/elerol rosszul lathato keznel POSE HOLD
#
# RG2:
# - az integralt RTDE-I/O worker a V19 rendszerben mar hasznalt kommunikacios alapra epul
# - ero: 10 N
# - a kalibralt potmeter jel folyamatosan 0..110 mm celnyilasra kepzodik
# - a kulon V5 RG2 robotoldali bridge gyorsabb diagnosztikai ag maradt
# - a regi pinch OPEN/CLOSED logika referencia celbol a fajlban maradt, de nem ez hajtja az RG2-t
#
# A regi K/M iranykalibracios JSON fajlokat tovabbra is hasznalja.
# ============================================================


ROBOT_IP = "192.168.0.20"

BASE_DIR = Path(__file__).resolve().parent

RGB_SCRIPT = (
    BASE_DIR
    / "01_rgb_mediapipe_full_sender_v5_pickplace_ui.py"
)

DEPTH_SCRIPT = (
    BASE_DIR
    / "02_openni_depth_xyz_bridge_v5_adaptive_robust.py"
)

TOP_CAMERA_SCRIPT = (
    BASE_DIR
    / "03_top_camera_hand_sender_v5_pickplace_ui.py"
)

MODEL_FILE = (
    BASE_DIR
    / "hand_landmarker.task"
)

COORD_CALIBRATION_FILE = (
    BASE_DIR
    / "camera_robot_coord_calibration.json"
)

ROBOT_DIRECTION_MAP_FILE = (
    BASE_DIR
    / "robot_operator_direction_map.json"
)

PINCH_CALIBRATION_FILE = (
    BASE_DIR
    / "pinch_calibration.json"
)

ROBOT_HOME_FILE = (
    BASE_DIR
    / "robot_demo_home.json"
)

RECOVERY_HOME_FILE = (
    BASE_DIR
    / "robot_recovery_home.json"
)

POT_CALIBRATION_FILE = (
    BASE_DIR
    / "pot_rg2_calibration.json"
)

TOP_METRIC_CALIBRATION_FILE = (
    BASE_DIR
    / "top_astra_multiplane_calibration_v4.json"
)

MEASUREMENT_DIR = (
    BASE_DIR
    / "tdk_pick_place_logs"
)

UDP_IP = "127.0.0.1"
UDP_PORT = 50010
PACKET_FORMAT = "<IfffB"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

PALM_UDP_IP = "127.0.0.1"
PALM_UDP_PORT = 50011
PALM_PACKET_FORMAT = "<IffB"
PALM_PACKET_SIZE = struct.calcsize(
    PALM_PACKET_FORMAT
)

FEATURE_UDP_IP = "127.0.0.1"
FEATURE_UDP_PORT = 50012
FEATURE_PACKET_FORMAT = "<I10fB"
FEATURE_PACKET_SIZE = struct.calcsize(
    FEATURE_PACKET_FORMAT
)

# Launcher -> RGB ablak allapot overlay
STATUS_UDP_IP = "127.0.0.1"
STATUS_UDP_PORT = 50013
STATUS_PACKET_FORMAT = "<BB"

# Eternico felso kamera. A sender normalizalt 2D tenyeradatot kuld.
TOP_UDP_IP = "127.0.0.1"
TOP_UDP_PORT = 50020
TOP_PACKET_FORMAT = "<IfffB"
TOP_PACKET_SIZE = struct.calcsize(
    TOP_PACKET_FORMAT
)

# VS Code terminalban ne fusson folyamatos telemetria.
# Események/gombok/hibak tovabbra is kulon sorban megjelennek.
TERMINAL_LIVE_STATUS = False

CAMERA_START_TIMEOUT_S = 30.0

# A kezi tesztben ez a sorrend volt stabil:
# 1) OpenNI depth indul
# 2) varunk, hogy teljesen felalljon
# 3) RGB + MediaPipe indul
# 4) csak ervenyes XYZ utan kapcsolodunk a robothoz
DEPTH_WARMUP_S = 5.0
RGB_WARMUP_S = 3.0
TOP_CAMERA_WARMUP_S = 1.5
TOP_RESTART_COOLDOWN_S = 2.0

# A felso kamera sender nem becsul metrikus XYZ-t.
# A fo program a V4 kalibraciobol XZ-becslesre kepes, es csak ervenyes gate eseten
# enged rovid TOP_FALLBACK allapotot. Egyebkent HOLD vagy PAUSE kovetkezik.
TOP_HAND_FRESH_TIMEOUT_S = 0.35
TOP_ASSIST_MAX_HOLD_S = 1.50

# ------------------------------------------------------------
# TDK A/B MERES - KONZERVATIV TOP-KAMERA METRIKUS FALLBACK
# ------------------------------------------------------------
# A jelenlegi validacios eredmenyek alapjan a metrikus top-fallbacket
# NEM engedjuk a teljes munkaterben. Csak a legerosebben validalt
# tartomanyban hasznaljuk.
TOP_METRIC_FALLBACK_MAX_S = 1.50
TOP_FALLBACK_Z_MIN_MM = 1200.0
TOP_FALLBACK_Z_MAX_MM = 1300.0
TOP_FALLBACK_MAX_NEAREST_PLANE_Y_MM = 50.0
TOP_FALLBACK_MAX_START_XZ_DIFF_MM = 80.0
TOP_FALLBACK_MAX_FRAME_XZ_JUMP_MM = 80.0

# A V4 kalibracio soran hasznalt magassagfuggo top-kepi guide boxok.
# Ez NEM teljes geometriai konvex-burok vizsgalat, hanem konzervativ
# runtime gate a kalibralt kepteruleten belul maradashoz.
TOP_FALLBACK_GUIDE_BOUNDS = {
    "LOW": (0.30, 0.70, 0.30, 0.70),
    "MID": (0.25, 0.75, 0.25, 0.75),
    "HIGH": (0.21, 0.79, 0.21, 0.79),
}

MEASUREMENT_LOG_HZ = 50.0
MEASUREMENT_LOG_PERIOD_S = 1.0 / MEASUREMENT_LOG_HZ

FREQUENCY = 250.0
DT = 1.0 / FREQUENCY

VELOCITY = 0.20
ACCELERATION = 0.50
LOOKAHEAD_TIME = 0.10
GAIN = 400

# A kalibracio utan mar NEM a kamera nyers X/Y/Z tengelyeit
# kotjuk kozvetlenul a robothoz.
#
# A kez szemantikus iranyai:
#   RIGHT   = a sajat jobb oldalad fele
#   UP      = felfele
#   FORWARD = a kamera fele
#
# Ezeket a launcher a kalibralt kamera-bazisbol szamolja.
HAND_RIGHT_SCALE = 0.90
HAND_UP_SCALE = 0.90
HAND_FORWARD_SCALE = 0.75

# ------------------------------------------------------------
# ROBOT-IRANY TANITAS (M gomb)
# ------------------------------------------------------------
# A robot a sajat BASE koordinatarendszereben kis +X/+Y/+Z
# probamozgasokat vegez. A kezelo megmondja, hogy az neki
# melyik irany: R/L/U/D/F/B.
#
# R = right / jobbra
# L = left / balra
# U = up / fel
# D = down / le
# F = forward / elore
# B = backward / hatra
#
# Ezek a tesztmozgashoz tartoznak, NEM a kezkovetes normal
# sebessegei.
ROBOT_MAP_TEST_MOVE_M = 0.015
ROBOT_MAP_TEST_SPEED_M_S = 0.030
ROBOT_MAP_TEST_ACCEL_M_S2 = 0.10

# Kalibracios minimum mozdulat.
# Ennel kisebb elmozdulasbol nem fogadunk el tengelyt.
CALIBRATION_MIN_MOVE_MM = 80.0

MAX_OFFSET_X_M = 0.190
MAX_OFFSET_Y_M = 0.400
MAX_OFFSET_Z_M = 0.380

MAX_TARGET_SPEED_M_S = 0.080
TARGET_CHANGE_DEADBAND_M = 0.0007

PACKET_STALE_TIMEOUT_S = 0.35
HAND_LOSS_GRACE_S = 0.75

# ------------------------------------------------------------
# PICK-AND-PLACE REACQUIRE POLICY
# ------------------------------------------------------------
# STRICT: a regi 0.75 s grace utan P kell.
# FLOW: 3.0 s-ig automatikus HOLD, majd keztalalatkor ugyanahhoz
# az eredeti C referenciahoz terunk vissza (NINCS automatikus C/rebase),
# de a visszaindulas 0.8 s-os sebessegrampot kap.
REACQUIRE_POLICY_DEFAULT = "FLOW"
FLOW_REACQUIRE_WINDOW_S = 3.00
REACQUIRE_RESUME_RAMP_S = 0.80
REACQUIRE_RAMP_START_FACTOR = 0.25

# Shadow megfigyeles: az Astra RGB palm 2D kulon porton csak logcelra.
# Nem szol bele a robotvezerlesbe.
ASTRA_SHADOW_UDP_IP = "127.0.0.1"
ASTRA_SHADOW_UDP_PORT = 50014
ASTRA_SHADOW_PACKET_FORMAT = "<IffB"
ASTRA_SHADOW_PACKET_SIZE = struct.calcsize(ASTRA_SHADOW_PACKET_FORMAT)
SHADOW_PAIR_MAX_SKEW_S = 0.080

# Gazdagabb UI-statusz a ket kameraablaknak.
# tracking, gripper, mode, source, flow, trial, top_gate, loss_elapsed_s
PICK_STATUS_PACKET_FORMAT = "<BBBBBBBf"
PICK_STATUS_RGB_PORT = 50013
PICK_STATUS_TOP_PORT = 50021

# ------------------------------------------------------------
# DEMO SAFE - KEZ / KAMERA MINOSEGKAPUK
# ------------------------------------------------------------
# Tracking kozben az egymas utani ervenyes XYZ mintak kozti
# irrealis ugrasokat eldobjuk. A regi depth bridge hatterugrasai
# ellen ved.
MAX_HAND_FRAME_JUMP_MM = 100.0
MAX_HAND_Z_FRAME_JUMP_MM = 75.0

# A MediaPipe tenyernormal kamerairanyu komponense.
# 1.0 = tenyer/kezhat nagyjabol a kamera fele nez
# 0.0 = a kez sikja elerol/oldalrol latszik.
#
# Demo alatt ilyen rossz nezeti helyzetben nem frissitjuk a robot
# XYZ celjat es a gripper allapotat sem.
MIN_PALM_CAMERA_ALIGNMENT = 0.38

# MediaPipe feature-matrix csak pose-quality/pinch segedadat.
# Az IMU adja a robot orientaciojat, de ezt a regi V15/V16
# kamera-feature szuro tovabbra is hasznalja.
MAX_HAND_ORIENTATION_JUMP_DEG = 35.0

# ------------------------------------------------------------
# ROBOT MOZGAS - V15
# ------------------------------------------------------------
# Tracking kozben nincs IK eloreellenorzes.
# A fo vedelmet a HOME, a fix orientacio es az XYZ offset-limit adja.

# ------------------------------------------------------------
# IMU ORIENTACIO - ESP32 + ADXL345 + L3G4200D
# ------------------------------------------------------------
# Az ESP32-n ez fusson:
#   esp32_imu_quaternion_pot_v5_pickplace.ino
#
# A C gomb:
#   - rogzit egy XYZ kezreferenciat
#   - a HOME TCP-orientaciojat veszi robot referencianak
#   - Z parancsot kuld az ESP32-nek, tehat az aktualis kezallas
#     lesz az IMU relativ 0 orientacioja
#
# A kamera tovabbra is XYZ-t ad. Az orientaciohoz a MediaPipe
# 3D kezmatrixat NEM hasznaljuk robotvezerlesre.

IMU_BAUD = 115200
IMU_READY_TIMEOUT_S = 30.0
IMU_STALE_TIMEOUT_S = 0.35

# COM port automatikus keresesnel ezeket reszesitjuk elonyben.
IMU_PORT_DESCRIPTION_HINTS = (
    "CP210",
    "Silicon Labs",
)

# V3-ban letesztelt tengelylekepzes:
#   robot helyi X <- szenzor Z
#   robot helyi Y <- -szenzor Y
#   robot helyi Z <- szenzor X
IMU_TO_ROBOT_LOCAL = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64
)

# 1:1 orientacios erzekenyseg.
ORIENTATION_GAIN = 1.00

# Teljes 3D relativ elfordulas korlatja a C referenciahoz kepest.
MAX_REL_ANGLE_DEG = 45.0

# A szenzor X tengelye a fenti mapping szerint a robot helyi Z-t
# hajtja. Ezt kerted legalabb 2x gyorsabbra.
# robot local X / Y / Z max szogsebesseg:
ROBOT_LOCAL_ORIENTATION_SPEED_DEG_S = np.array(
    [10.0, 10.0, 20.0],
    dtype=np.float64
)

# ------------------------------------------------------------
# ROBOT JOINT SAFETY GUARD
# ------------------------------------------------------------
# UR5e modell-specifikus tartomany.
# J0 Base, J1 Shoulder, J2 Elbow, J3 Wrist1, J4 Wrist2, J5 Wrist3.
#
# A program nem varja meg a hard limitet:
# minden tengelynel 10 fokkal elotte tracking OFF + servoStop.
JOINT_NAMES = [
    "Base",
    "Shoulder",
    "Elbow",
    "Wrist 1",
    "Wrist 2",
    "Wrist 3",
]

JOINT_HARD_MIN_DEG = [
    -360.0,
    -360.0,
    -160.0,
    -360.0,
    -360.0,
    -360.0,
]

JOINT_HARD_MAX_DEG = [
    +360.0,
    +360.0,
    +160.0,
    +360.0,
    +360.0,
    +360.0,
]

JOINT_LIMIT_MARGIN_DEG = 10.0

# Klasszikus UR wrist-singularity kozeleben a Wrist 2 kb. 0 fok.
# Inkabb megallitjuk a kezi kovetest, mint hogy a csuklo atbillenjen.
WRIST2_SINGULARITY_GUARD_DEG = 12.0

# ------------------------------------------------------------
# TANITHATO DEMO HOME
# ------------------------------------------------------------
# H = aktualis robotallapot mentese
# R = visszaallas erre a joint-konfiguraciora
#
# moveJ joint-sebesseg / gyorsulas [rad/s, rad/s^2].
HOME_MOVE_SPEED_RAD_S = 0.35
HOME_MOVE_ACCEL_RAD_S2 = 0.50

# C referencia csak akkor veheto fel, ha a robot jointjai
# legfeljebb ennyire vannak a HOME jointjaitol.
HOME_READY_TOL_DEG = 5.0

# Ha valaki nagyon messze van a HOME joint-konfiguraciotol,
# R-rel ne inditsunk automatikusan egy hatalmas joint-soprest.
HOME_RETURN_MAX_JOINT_DELTA_DEG = 100.0

# ------------------------------------------------------------
# TANITHATO RECOVERY HOME
# ------------------------------------------------------------
# J = biztonsagos, kompakt koztes joint-pozt ment.
# R:
#   - ha a normal HOME kozvetlenul biztonsagos -> oda megy
#   - kulonben lassan RECOVERY HOME -> normal HOME
#
# FONTOS: kornyezeti 3D utkozesi modell nelkul "barmilyen" pozbol
# matematikailag nem garantalhato az utkozesmentes ut.
RECOVERY_MOVE_SPEED_RAD_S = 0.20
RECOVERY_MOVE_ACCEL_RAD_S2 = 0.30

# ------------------------------------------------------------
# POT -> RG2
# ------------------------------------------------------------
GRIPPER_FORCE_N = 10.0
GRIPPER_CLOSED_WIDTH_MM = 0.0
GRIPPER_OPEN_WIDTH_MM = 110.0

POT_FILTER_ALPHA = 0.60
POT_MIN_CALIBRATION_SPAN_RAW = 250.0

# A regi pinch logika kodja megmaradt a fajlban referencia celbol,
# de V18-ban az RG2-t a kacsacsor potmetere hajtja.
# A kezcsippentesbol NEM folytonos szelesseget kerunk.
# Ket stabil allapot van: OPEN / CLOSED.
#
# A Z/O kalibraciobol 0..1 normalizalt pinch ertek keszul:
#   0 = zart ujjak
#   1 = nyitott ujjak
#
# Hiszterezis V11:
#   OPEN -> CLOSED csak 0.25 alatt
#   CLOSED -> OPEN csak 0.80 felett
#
# A feltetelnek egy ideig folyamatosan igaznak kell lennie.
PINCH_CLOSE_THRESHOLD = 0.25
PINCH_OPEN_THRESHOLD = 0.80

PINCH_CLOSE_DWELL_S = 0.20
PINCH_OPEN_DWELL_S = 0.45

# Egy allapotvaltas utan ennyi ideig nem engedunk azonnali
# ellenkezo parancsot. Ez a "bezar, aztan kinyit" ellen van.
GRIPPER_REVERSE_LOCKOUT_S = 1.00

# Ketallapotu vezerlesnel csak allapotvaltaskor kuldunk parancsot.
MIN_GRIPPER_COMMAND_INTERVAL_S = 0.20
MIN_GRIPPER_TARGET_CHANGE_MM = 2.0

# A legacy URScript DI8/DI9 es DO8/DO9 a UR dokumentacio szerint
# a tool digital I/O 0/1 parjanak felel meg.
# RTDEReceive global digital index: tool DI0/DI1 -> 16/17.
TOOL_DI0_GLOBAL_ID = 16
TOOL_DI1_GLOBAL_ID = 17

RG2_BIT_HALF_PERIOD_S = 0.012
RG2_READY_TIMEOUT_S = 2.0
RG2_BUSY_TIMEOUT_S = 4.0


user32 = ctypes.windll.user32


def key_down(letter):
    return bool(
        user32.GetAsyncKeyState(
            ord(letter.upper())
        ) & 0x8000
    )


class KeyEdge:
    def __init__(self):
        self.previous = {}

    def pressed(self, letter):
        now = key_down(letter)
        before = self.previous.get(
            letter,
            False
        )
        self.previous[letter] = now
        return now and not before


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def fmt_pose(pose):
    return (
        f"X={pose[0]*1000:+8.2f} "
        f"Y={pose[1]*1000:+8.2f} "
        f"Z={pose[2]*1000:+8.2f} mm"
    )


def vector_distance_xyz(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]

    return math.sqrt(
        dx * dx
        + dy * dy
        + dz * dz
    )


def rate_limit_xyz(
    current,
    desired,
    max_step
):
    dx = desired[0] - current[0]
    dy = desired[1] - current[1]
    dz = desired[2] - current[2]

    distance = math.sqrt(
        dx * dx
        + dy * dy
        + dz * dz
    )

    if (
        distance <= max_step
        or distance == 0
    ):
        current[0] = desired[0]
        current[1] = desired[1]
        current[2] = desired[2]
        return

    scale = max_step / distance

    current[0] += dx * scale
    current[1] += dy * scale
    current[2] += dz * scale


def vec_sub(a, b):
    return [
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2]
    ]


def vec_dot(a, b):
    return (
        a[0] * b[0]
        + a[1] * b[1]
        + a[2] * b[2]
    )


def vec_cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]


def vec_norm(v):
    return math.sqrt(
        v[0] * v[0]
        + v[1] * v[1]
        + v[2] * v[2]
    )


def vec_scale(v, s):
    return [
        v[0] * s,
        v[1] * s,
        v[2] * s
    ]


def vec_normalize(v):
    n = vec_norm(v)

    if n < 1e-9:
        raise ValueError(
            "Nullahosszu kalibracios vektor."
        )

    return [
        v[0] / n,
        v[1] / n,
        v[2] / n
    ]


def build_coord_basis(
    right_raw,
    up_raw,
    forward_raw
):
    """
    A felhasznalo 3 ismert kezmozdulatabol ortonormalis
    bazist keszit a kamera XYZ koordinatarendszereben.

    Kimenet:
        {
            "right":   [x,y,z],
            "up":      [x,y,z],
            "forward": [x,y,z]
        }

    Ezutan tetszoleges kamera-delta projekcioja ezekre a
    vektorokra megadja a kez "jobbra / fel / elore" komponenseit.
    """

    lengths = {
        "RIGHT": vec_norm(right_raw),
        "UP": vec_norm(up_raw),
        "FORWARD": vec_norm(forward_raw)
    }

    for name, length in lengths.items():
        if length < CALIBRATION_MIN_MOVE_MM:
            raise ValueError(
                f"{name} kalibracios mozgas tul kicsi: "
                f"{length:.1f} mm. Minimum "
                f"{CALIBRATION_MIN_MOVE_MM:.0f} mm."
            )

    # 1) RIGHT legyen az elso biztos tengely.
    e_right = vec_normalize(
        right_raw
    )

    # 2) UP-bol kivesszuk a RIGHT iranyu komponenst.
    up_rejected = vec_sub(
        up_raw,
        vec_scale(
            e_right,
            vec_dot(
                up_raw,
                e_right
            )
        )
    )

    if vec_norm(up_rejected) < 40.0:
        raise ValueError(
            "A RIGHT es UP kalibracios mozgas tul kozel "
            "azonos iranyu. Ismeteld meg nagyobb, tisztabb "
            "mozdulatokkal."
        )

    e_up = vec_normalize(
        up_rejected
    )

    # 3) A harmadik tengelyt kereszt-szorzattal kenyszeritjuk
    # pontosan merolegesre.
    e_forward = vec_normalize(
        vec_cross(
            e_right,
            e_up
        )
    )

    # A kereszt-szorzat ketfele iranyba mutathat.
    # A felhasznalo FORWARD mintaja donti el az elojelt.
    if vec_dot(
        e_forward,
        forward_raw
    ) < 0.0:
        e_forward = vec_scale(
            e_forward,
            -1.0
        )

    # Az UP tengelyt ujraszamoljuk, hogy mindharom pontosan
    # ortogonalis legyen, mikozben az eredeti UP irany elojelere figyelunk.
    corrected_up = vec_normalize(
        vec_cross(
            e_forward,
            e_right
        )
    )

    if vec_dot(
        corrected_up,
        up_raw
    ) < 0.0:
        corrected_up = vec_scale(
            corrected_up,
            -1.0
        )
        e_forward = vec_scale(
            e_forward,
            -1.0
        )

    e_up = corrected_up

    # Ellenorzes: a FORWARD nyers minta ne legyen majdnem a
    # RIGHT/UP sikban.
    forward_alignment = abs(
        vec_dot(
            vec_normalize(
                forward_raw
            ),
            e_forward
        )
    )

    if forward_alignment < 0.45:
        raise ValueError(
            "A FORWARD kalibracios mozgas nem eleg kulonul el "
            "a RIGHT/UP siktol. Ismeteld meg tisztabb "
            "elore-hatra mozdulattal."
        )

    return {
        "right": e_right,
        "up": e_up,
        "forward": e_forward,
        "raw_lengths_mm": lengths
    }


def save_coord_calibration(
    basis
):
    payload = {
        "version": 1,
        "meaning": {
            "right": "user right",
            "up": "user up",
            "forward": "toward camera"
        },
        "basis_camera_xyz": {
            "right": basis["right"],
            "up": basis["up"],
            "forward": basis["forward"]
        },
        "raw_lengths_mm": (
            basis.get(
                "raw_lengths_mm",
                {}
            )
        )
    }

    COORD_CALIBRATION_FILE.write_text(
        json.dumps(
            payload,
            indent=2
        ),
        encoding="utf-8"
    )


def load_coord_calibration():
    if not COORD_CALIBRATION_FILE.exists():
        return None

    try:
        payload = json.loads(
            COORD_CALIBRATION_FILE.read_text(
                encoding="utf-8"
            )
        )

        basis_raw = payload[
            "basis_camera_xyz"
        ]

        basis = {
            "right": vec_normalize(
                basis_raw["right"]
            ),
            "up": vec_normalize(
                basis_raw["up"]
            ),
            "forward": vec_normalize(
                basis_raw["forward"]
            )
        }

        return basis

    except Exception as e:
        print(
            "[FIGYELEM] A mentett koordinata-kalibracio "
            "nem toltheto be:"
        )
        print(
            type(e).__name__ + ":",
            e
        )
        return None


def camera_delta_to_operator(
    dx_mm,
    dy_mm,
    dz_mm,
    basis
):
    delta = [
        dx_mm,
        dy_mm,
        dz_mm
    ]

    right_mm = vec_dot(
        delta,
        basis["right"]
    )

    up_mm = vec_dot(
        delta,
        basis["up"]
    )

    forward_mm = vec_dot(
        delta,
        basis["forward"]
    )

    return (
        right_mm,
        up_mm,
        forward_mm
    )


def save_robot_direction_map(
    robot_map
):
    payload = {
        "version": 1,
        "meaning": (
            "operator semantic directions "
            "mapped to UR robot BASE axes"
        ),
        "mapping": robot_map
    }

    ROBOT_DIRECTION_MAP_FILE.write_text(
        json.dumps(
            payload,
            indent=2
        ),
        encoding="utf-8"
    )


def load_robot_direction_map():
    if not ROBOT_DIRECTION_MAP_FILE.exists():
        return None

    try:
        payload = json.loads(
            ROBOT_DIRECTION_MAP_FILE.read_text(
                encoding="utf-8"
            )
        )

        mapping = payload["mapping"]

        for semantic in (
            "right",
            "up",
            "forward"
        ):
            item = mapping[semantic]

            axis = int(item["axis"])
            sign = float(item["sign"])

            if axis not in (0, 1, 2):
                raise ValueError(
                    f"Hibas tengely: {semantic} -> {axis}"
                )

            if sign not in (-1.0, 1.0):
                raise ValueError(
                    f"Hibas elojel: {semantic} -> {sign}"
                )

        axes = [
            int(mapping[name]["axis"])
            for name in (
                "right",
                "up",
                "forward"
            )
        ]

        if len(set(axes)) != 3:
            raise ValueError(
                "A mentett robot mappingben "
                "nem harom kulon tengely szerepel."
            )

        return mapping

    except Exception as e:
        print(
            "[FIGYELEM] A mentett robot-irany mapping "
            "nem toltheto be:"
        )
        print(
            type(e).__name__ + ":",
            e
        )
        return None


def robot_map_text(
    robot_map
):
    if robot_map is None:
        return "nincs"

    axis_names = "XYZ"

    parts = []

    for semantic, short_name in (
        ("right", "RIGHT"),
        ("up", "UP"),
        ("forward", "FORWARD")
    ):
        item = robot_map[semantic]

        axis_name = axis_names[
            int(item["axis"])
        ]

        sign_char = (
            "+"
            if float(item["sign"]) > 0
            else "-"
        )

        parts.append(
            f"{short_name}->{sign_char}{axis_name}"
        )

    return ", ".join(parts)


def teach_robot_directions(
    rtde_r,
    rtde_c
):
    """
    A robot +X, +Y, +Z BASE iranyba 15 mm-es probamozgast vegez.
    A felhasznalo a terminalban megadja, hogy a mozgast milyen
    operatori iranynak latja.

    Pelda:
        +Y a kezelo szerint BALRA megy -> L
        ebbol kovetkezik:
        kez JOBBRA -> robot -Y
    """

    print()
    print("=" * 72)
    print("ROBOT BASE IRANYOK KEZI TANITASA")
    print("=" * 72)
    print(
        "A robot harom KIS, 15 mm-es probamozgast fog vegezni."
    )
    print(
        "Minden probanal figyeld a TCP/kezfej HELYVALTOZTATASAT."
    )
    print()
    print(
        "A terminalba ezt ird:"
    )
    print(
        "  R = jobbra     L = balra"
    )
    print(
        "  U = fel        D = le"
    )
    print(
        "  F = elore      B = hatra"
    )
    print(
        "  Q = megszakitas"
    )
    print()
    print(
        "FONTOS: legyen szabad legalabb nehany cm hely "
        "a robot korul."
    )

    label_info = {
        "R": ("right", +1.0),
        "L": ("right", -1.0),
        "U": ("up", +1.0),
        "D": ("up", -1.0),
        "F": ("forward", +1.0),
        "B": ("forward", -1.0)
    }

    axis_names = "XYZ"
    taught = {}

    for axis in range(3):
        start_pose = (
            rtde_r.getActualTCPPose()
        )

        target_pose = start_pose[:]
        target_pose[axis] += (
            ROBOT_MAP_TEST_MOVE_M
        )

        print()
        print(
            f"TESZT +{axis_names[axis]}: "
            f"{ROBOT_MAP_TEST_MOVE_M*1000:.0f} mm"
        )
        print(
            "Robot mozog..."
        )

        moved_out = False

        try:
            ok = rtde_c.moveL(
                target_pose,
                ROBOT_MAP_TEST_SPEED_M_S,
                ROBOT_MAP_TEST_ACCEL_M_S2
            )

            moved_out = True

            if ok is False:
                raise RuntimeError(
                    f"+{axis_names[axis]} tesztmozgast "
                    "a robot nem fogadta el."
                )

            while True:
                answer = input(
                    f"A +{axis_names[axis]} mozdulat "
                    "neked melyik irany? "
                    "[R/L/U/D/F/B, Q=kilep]: "
                ).strip().upper()

                if answer == "Q":
                    raise KeyboardInterrupt

                if answer in label_info:
                    break

                print(
                    "Ervenytelen valasz. "
                    "R/L/U/D/F/B vagy Q."
                )

        finally:
            if moved_out:
                try:
                    print(
                        "Vissza a teszt kezdopontjara..."
                    )

                    rtde_c.moveL(
                        start_pose,
                        ROBOT_MAP_TEST_SPEED_M_S,
                        ROBOT_MAP_TEST_ACCEL_M_S2
                    )
                except Exception:
                    print(
                        "[FIGYELEM] A teszt utan az automatikus "
                        "visszaallas nem sikerult."
                    )

        semantic, observed_sign = (
            label_info[answer]
        )

        if semantic in taught:
            previous = taught[semantic]

            raise ValueError(
                "Ket robot-tengelyt ugyanannak az operatori "
                f"iranynak jeloltel: {semantic}. "
                f"Korabbi axis={axis_names[previous['axis']]}, "
                f"most axis={axis_names[axis]}. "
                "Inditsd ujra az M kalibraciot."
            )

        # Ha +robot-axis = +semantic, akkor +semantic -> +robot-axis.
        # Ha +robot-axis = -semantic, akkor +semantic -> -robot-axis.
        taught[semantic] = {
            "axis": axis,
            "sign": observed_sign
        }

    missing = [
        name
        for name in (
            "right",
            "up",
            "forward"
        )
        if name not in taught
    ]

    if missing:
        raise ValueError(
            "Nem lett mindharom operatori tengely megtanitva: "
            + ", ".join(missing)
        )

    save_robot_direction_map(
        taught
    )

    print()
    print("=" * 72)
    print("[OK] ROBOT IRANYKALIBRACIO KESZ")
    print("=" * 72)
    print(
        robot_map_text(
            taught
        )
    )
    print(
        "Elmentve ide:"
    )
    print(
        ROBOT_DIRECTION_MAP_FILE
    )
    print()
    print(
        "Ezutan C = referencia, majd P = kovetes."
    )

    return taught


def camera_delta_to_robot_offset(
    dx_mm,
    dy_mm,
    dz_mm,
    basis,
    robot_map
):
    (
        right_mm,
        up_mm,
        forward_mm
    ) = camera_delta_to_operator(
        dx_mm,
        dy_mm,
        dz_mm,
        basis
    )

    semantic_mm = {
        "right": (
            right_mm
            * HAND_RIGHT_SCALE
        ),
        "up": (
            up_mm
            * HAND_UP_SCALE
        ),
        "forward": (
            forward_mm
            * HAND_FORWARD_SCALE
        )
    }

    robot_raw = [
        0.0,
        0.0,
        0.0
    ]

    for semantic in (
        "right",
        "up",
        "forward"
    ):
        item = robot_map[semantic]

        axis = int(
            item["axis"]
        )

        sign = float(
            item["sign"]
        )

        robot_raw[axis] += (
            sign
            * semantic_mm[semantic]
            / 1000.0
        )

    limited_x = (
        abs(robot_raw[0])
        > MAX_OFFSET_X_M
    )

    limited_y = (
        abs(robot_raw[1])
        > MAX_OFFSET_Y_M
    )

    limited_z = (
        abs(robot_raw[2])
        > MAX_OFFSET_Z_M
    )

    return (
        [
            clamp(
                robot_raw[0],
                -MAX_OFFSET_X_M,
                MAX_OFFSET_X_M
            ),
            clamp(
                robot_raw[1],
                -MAX_OFFSET_Y_M,
                MAX_OFFSET_Y_M
            ),
            clamp(
                robot_raw[2],
                -MAX_OFFSET_Z_M,
                MAX_OFFSET_Z_M
            )
        ],
        (
            limited_x,
            limited_y,
            limited_z
        ),
        (
            right_mm,
            up_mm,
            forward_mm
        )
    )


def rotvec_to_matrix(
    rotvec
):
    rv = np.asarray(
        rotvec,
        dtype=np.float64
    ).reshape(3, 1)

    R, _ = cv2.Rodrigues(
        rv
    )

    return R


def matrix_to_rotvec(
    R
):
    rv, _ = cv2.Rodrigues(
        np.asarray(
            R,
            dtype=np.float64
        )
    )

    return rv.reshape(3)


def rotation_distance_rad(
    R_from,
    R_to
):
    relative_R = (
        np.asarray(
            R_from,
            dtype=np.float64
        ).T
        @ np.asarray(
            R_to,
            dtype=np.float64
        )
    )

    rv = matrix_to_rotvec(
        relative_R
    )

    return float(
        np.linalg.norm(
            rv
        )
    )


def rotation_distance_deg(
    R_from,
    R_to
):
    return math.degrees(
        rotation_distance_rad(
            R_from,
            R_to
        )
    )


def rate_limit_orientation_rotvec(
    current_rotvec,
    desired_rotvec,
    max_step_rad
):
    """
    A robot aktualis command-orientaciojat a leheto legrovidebb
    3D forgatassal kozeliti a celhoz. Nem Euler-szogeket vagunk,
    ezert nincs +180/-180 wrap problema.
    """

    current_R = (
        rotvec_to_matrix(
            current_rotvec
        )
    )

    desired_R = (
        rotvec_to_matrix(
            desired_rotvec
        )
    )

    relative_R = (
        current_R.T
        @ desired_R
    )

    relative_rv = (
        matrix_to_rotvec(
            relative_R
        )
    )

    angle = float(
        np.linalg.norm(
            relative_rv
        )
    )

    if (
        angle <= max_step_rad
        or angle < 1e-9
    ):
        return np.asarray(
            desired_rotvec,
            dtype=np.float64
        ).copy()

    step_rv = (
        relative_rv
        * (
            max_step_rad
            / angle
        )
    )

    next_R = (
        current_R
        @ rotvec_to_matrix(
            step_rv
        )
    )

    return matrix_to_rotvec(
        next_R
    )


def angle_delta_wrapped_rad(
    target,
    current
):
    return math.atan2(
        math.sin(
            float(target)
            - float(current)
        ),
        math.cos(
            float(target)
            - float(current)
        )
    )


def distance_to_interval(
    value,
    lo,
    hi
):
    value = float(value)

    if value < lo:
        return (
            lo - value
        )

    if value > hi:
        return (
            value - hi
        )

    return 0.0


def check_joint_soft_limits(
    joint_q_rad
):
    """
    Soft joint limit ellenorzes.

    Visszaad:
      safe: bool
      reasons: lista
      q_deg: aktualis joint szogek fokban
    """

    if (
        joint_q_rad is None
        or len(joint_q_rad) < 6
    ):
        return (
            False,
            ["Nincs ervenyes 6D joint adat."],
            []
        )

    q_deg = [
        math.degrees(
            float(q)
        )
        for q in joint_q_rad[:6]
    ]

    reasons = []

    for i in range(6):
        soft_min = (
            JOINT_HARD_MIN_DEG[i]
            + JOINT_LIMIT_MARGIN_DEG
        )

        soft_max = (
            JOINT_HARD_MAX_DEG[i]
            - JOINT_LIMIT_MARGIN_DEG
        )

        if (
            q_deg[i] <= soft_min
            or q_deg[i] >= soft_max
        ):
            reasons.append(
                f"{JOINT_NAMES[i]}={q_deg[i]:+.1f} deg "
                f"(soft tartomany: {soft_min:+.1f} .. {soft_max:+.1f})"
            )

    # Wrist 2 = q[4].
    if (
        abs(
            q_deg[4]
        )
        <= WRIST2_SINGULARITY_GUARD_DEG
    ):
        reasons.append(
            f"Wrist 2 singularitas-vedo zona: "
            f"{q_deg[4]:+.1f} deg "
            f"(tiltott: +/-{WRIST2_SINGULARITY_GUARD_DEG:.1f} deg)"
        )

    return (
        len(reasons) == 0,
        reasons,
        q_deg
    )


def matrix_to_euler_zyx(
    R
):
    sy = math.sqrt(
        R[0, 0] * R[0, 0]
        + R[1, 0] * R[1, 0]
    )

    singular = (
        sy < 1e-6
    )

    if not singular:
        roll = math.atan2(
            R[2, 1],
            R[2, 2]
        )

        pitch = math.atan2(
            -R[2, 0],
            sy
        )

        yaw = math.atan2(
            R[1, 0],
            R[0, 0]
        )

    else:
        roll = math.atan2(
            -R[1, 2],
            R[1, 1]
        )

        pitch = math.atan2(
            -R[2, 0],
            sy
        )

        yaw = 0.0

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw)
    )


def scaled_limited_orientation_rotvec(
    relative_R
):
    rv = matrix_to_rotvec(
        relative_R
    )

    rv = (
        rv
        * ORIENTATION_GAIN
    )

    angle = float(
        np.linalg.norm(rv)
    )

    max_angle = math.radians(
        MAX_REL_ANGLE_DEG
    )

    if (
        angle > max_angle
        and angle > 1e-9
    ):
        rv *= (
            max_angle
            / angle
        )

    return rv


def ema_np_vector(
    previous,
    current,
    alpha
):
    current = np.asarray(
        current,
        dtype=np.float64
    )

    if previous is None:
        return current.copy()

    return (
        alpha * current
        + (1.0 - alpha)
        * previous
    )


def drain_feature_udp(
    sock
):
    latest = None

    while True:
        try:
            data, _ = sock.recvfrom(
                256
            )
        except BlockingIOError:
            break

        if len(data) != FEATURE_PACKET_SIZE:
            continue

        latest = struct.unpack(
            FEATURE_PACKET_FORMAT,
            data
        )

    return latest


def save_pinch_calibration(
    closed_ratio,
    open_ratio
):
    if (
        closed_ratio is None
        or open_ratio is None
    ):
        return

    payload = {
        "version": 1,
        "closed_ratio": float(
            closed_ratio
        ),
        "open_ratio": float(
            open_ratio
        )
    }

    PINCH_CALIBRATION_FILE.write_text(
        json.dumps(
            payload,
            indent=2
        ),
        encoding="utf-8"
    )


def load_pinch_calibration():
    if not PINCH_CALIBRATION_FILE.exists():
        return (
            None,
            None
        )

    try:
        payload = json.loads(
            PINCH_CALIBRATION_FILE.read_text(
                encoding="utf-8"
            )
        )

        closed_ratio = float(
            payload["closed_ratio"]
        )

        open_ratio = float(
            payload["open_ratio"]
        )

        if (
            open_ratio
            - closed_ratio
            < 0.10
        ):
            raise ValueError(
                "Tul kicsi pinch kalibracios tartomany."
            )

        return (
            closed_ratio,
            open_ratio
        )

    except Exception as e:
        print(
            "[FIGYELEM] Pinch kalibracio nem toltheto be:"
        )
        print(
            type(e).__name__ + ":",
            e
        )

        return (
            None,
            None
        )


def normalized_pinch(
    ratio,
    closed_ratio,
    open_ratio
):
    span = (
        open_ratio
        - closed_ratio
    )

    if span < 0.10:
        raise ValueError(
            "A pinch kalibracio ervenytelen."
        )

    return clamp(
        (
            ratio
            - closed_ratio
        )
        / span,
        0.0,
        1.0
    )


def pinch_to_gripper_width(
    ratio,
    closed_ratio,
    open_ratio
):
    span = (
        open_ratio
        - closed_ratio
    )

    if span < 0.10:
        raise ValueError(
            "A pinch kalibracio ervenytelen."
        )

    t = (
        ratio
        - closed_ratio
    ) / span

    t = clamp(
        t,
        0.0,
        1.0
    )

    return (
        GRIPPER_CLOSED_WIDTH_MM
        + t
        * (
            GRIPPER_OPEN_WIDTH_MM
            - GRIPPER_CLOSED_WIDTH_MM
        )
    )



def save_pot_calibration(
    closed_raw,
    open_raw
):
    if (
        closed_raw is None
        or open_raw is None
    ):
        return

    payload = {
        "version": 1,
        "closed_raw": float(
            closed_raw
        ),
        "open_raw": float(
            open_raw
        ),
    }

    POT_CALIBRATION_FILE.write_text(
        json.dumps(
            payload,
            indent=2
        ),
        encoding="utf-8"
    )


def load_pot_calibration():
    if not POT_CALIBRATION_FILE.exists():
        return (
            None,
            None
        )

    try:
        payload = json.loads(
            POT_CALIBRATION_FILE.read_text(
                encoding="utf-8"
            )
        )

        closed_raw = float(
            payload["closed_raw"]
        )

        open_raw = float(
            payload["open_raw"]
        )

        if (
            abs(
                open_raw
                - closed_raw
            )
            < POT_MIN_CALIBRATION_SPAN_RAW
        ):
            raise ValueError(
                "Tul kicsi potmeter kalibracios tartomany."
            )

        return (
            closed_raw,
            open_raw
        )

    except Exception as e:
        print(
            "[FIGYELEM] Potmeter kalibracio nem toltheto be:"
        )
        print(
            type(e).__name__ + ":",
            e
        )

        return (
            None,
            None
        )


def pot_calibration_valid(
    closed_raw,
    open_raw
):
    return (
        closed_raw is not None
        and open_raw is not None
        and abs(
            open_raw
            - closed_raw
        )
        >= POT_MIN_CALIBRATION_SPAN_RAW
    )


def pot_raw_to_normalized(
    raw,
    closed_raw,
    open_raw
):
    span = (
        open_raw
        - closed_raw
    )

    if abs(span) < 1.0e-9:
        raise ValueError(
            "Nulla potmeter kalibracios tartomany."
        )

    return clamp(
        (
            raw
            - closed_raw
        )
        / span,
        0.0,
        1.0
    )


def pot_raw_to_gripper_width(
    raw,
    closed_raw,
    open_raw
):
    t = pot_raw_to_normalized(
        raw,
        closed_raw,
        open_raw
    )

    return (
        GRIPPER_CLOSED_WIDTH_MM
        + t
        * (
            GRIPPER_OPEN_WIDTH_MM
            - GRIPPER_CLOSED_WIDTH_MM
        )
    )


def save_robot_home(
    joint_q,
    tcp_pose
):
    payload = {
        "version": 1,
        "joint_q_rad": [
            float(x)
            for x in joint_q[:6]
        ],
        "tcp_pose": [
            float(x)
            for x in tcp_pose[:6]
        ],
    }

    ROBOT_HOME_FILE.write_text(
        json.dumps(
            payload,
            indent=2
        ),
        encoding="utf-8"
    )

    return payload


def load_robot_home():
    if not ROBOT_HOME_FILE.exists():
        return None

    try:
        payload = json.loads(
            ROBOT_HOME_FILE.read_text(
                encoding="utf-8"
            )
        )

        q = [
            float(x)
            for x in payload["joint_q_rad"]
        ]

        pose = [
            float(x)
            for x in payload["tcp_pose"]
        ]

        if (
            len(q) != 6
            or len(pose) != 6
        ):
            raise ValueError(
                "A HOME fajl nem 6D adatot tartalmaz."
            )

        return {
            "version": int(
                payload.get(
                    "version",
                    1
                )
            ),
            "joint_q_rad": q,
            "tcp_pose": pose,
        }

    except Exception as e:
        print(
            "[FIGYELEM] Mentett HOME nem toltheto be:"
        )
        print(
            type(e).__name__ + ":",
            e
        )

        return None



def save_recovery_home(
    joint_q,
    tcp_pose
):
    payload = {
        "version": 1,
        "joint_q_rad": [
            float(x)
            for x in joint_q[:6]
        ],
        "tcp_pose": [
            float(x)
            for x in tcp_pose[:6]
        ],
    }

    RECOVERY_HOME_FILE.write_text(
        json.dumps(
            payload,
            indent=2
        ),
        encoding="utf-8"
    )

    return payload


def load_recovery_home():
    if not RECOVERY_HOME_FILE.exists():
        return None

    try:
        payload = json.loads(
            RECOVERY_HOME_FILE.read_text(
                encoding="utf-8"
            )
        )

        q = [
            float(x)
            for x in payload["joint_q_rad"]
        ]

        pose = [
            float(x)
            for x in payload["tcp_pose"]
        ]

        if (
            len(q) != 6
            or len(pose) != 6
        ):
            raise ValueError(
                "A RECOVERY HOME fajl nem 6D adatot tartalmaz."
            )

        return {
            "version": int(
                payload.get(
                    "version",
                    1
                )
            ),
            "joint_q_rad": q,
            "tcp_pose": pose,
        }

    except Exception as e:
        print(
            "[FIGYELEM] Mentett RECOVERY HOME nem toltheto be:"
        )
        print(
            type(e).__name__ + ":",
            e
        )

        return None


def max_joint_delta_deg(
    q_a,
    q_b
):
    if (
        q_a is None
        or q_b is None
        or len(q_a) < 6
        or len(q_b) < 6
    ):
        return 9999.0

    return max(
        abs(
            math.degrees(
                angle_delta_wrapped_rad(
                    q_a[i],
                    q_b[i]
                )
            )
        )
        for i in range(6)
    )


def robot_near_home(
    actual_q,
    home_data
):
    if home_data is None:
        return False

    return (
        max_joint_delta_deg(
            actual_q,
            home_data["joint_q_rad"]
        )
        <= HOME_READY_TOL_DEG
    )


def home_move_is_safe(
    actual_q,
    home_data
):
    """
    V19:
    Ezt mar NEM joint-soft-limit / Wrist2 singularitas tiltasra
    hasznaljuk HOME moveJ kozben.

    A tanitott HOME egy korabban tenylegesen felvett robot joint-poz.
    A robot sajat vezerloje a fizikai joint limiteket ugyis ervenyesiti.

    Itt csak azt dontjuk el, hogy:
      - kozel van -> kozvetlen NORMAL HOME
      - messze van -> RECOVERY HOME-on keresztul menjunk
    """

    if home_data is None:
        return (
            False,
            "nincs mentett HOME"
        )

    target_q = (
        home_data["joint_q_rad"]
    )

    if (
        actual_q is None
        or len(actual_q) < 6
        or target_q is None
        or len(target_q) < 6
    ):
        return (
            False,
            "nincs ervenyes 6D joint adat"
        )

    max_delta = (
        max_joint_delta_deg(
            actual_q,
            target_q
        )
    )

    if (
        max_delta
        > HOME_RETURN_MAX_JOINT_DELTA_DEG
    ):
        return (
            False,
            (
                "messze van a NORMAL HOME: "
                f"max joint elteres {max_delta:.1f} deg; "
                "RECOVERY utvonal javasolt"
            )
        )

    return (
        True,
        "OK"
    )



def recovery_move_is_safe(
    actual_q,
    recovery_data
):
    """
    V19:
    A RECOVERY HOME egy kezelo altal betanitott tenyleges joint-pozt.
    HOME/RECOVERY moveJ-nel nem alkalmazzuk a trackinghez valo
    Wrist2-singularity es soft-limit guardokat.

    Kornyezeti utkozesmodellt tovabbra sem tudunk garantalni:
    csak olyan RECOVERY HOME-ot tanits, amely az ismert munkaterben
    biztonsagos koztes poz.
    """

    if recovery_data is None:
        return (
            False,
            "nincs mentett RECOVERY HOME"
        )

    target_q = (
        recovery_data["joint_q_rad"]
    )

    if (
        target_q is None
        or len(target_q) < 6
    ):
        return (
            False,
            "RECOVERY HOME nem ervenyes 6D joint adat"
        )

    return (
        True,
        "OK"
    )


class RG2RTDEWorker:
    """
    RG2 legacy bit-protokoll RTDE I/O-n.

    V19 FONTOS VALTOZAS:
    - NINCS kulon RTDEReceive kapcsolat a workerben.
    - A fo 250 Hz-es rtde_r kapcsolat olvassa a tool DI0/DI1
      allapotot, es update_inputs()-szal atadja a workernek.
    - A worker csak egy RTDEIO kapcsolatot nyit a kimenetekhez.

    Ezzel elkeruljuk a V18-ban latott masodik
    RTDEReceiveInterface "End of file" kapcsolateldobast.
    """

    def __init__(
        self,
        robot_ip
    ):
        self.robot_ip = (
            robot_ip
        )

        self.lock = (
            threading.Lock()
        )

        self.running = True
        self.enabled = False

        self.latest_target = None
        self.last_sent_target = None
        self.last_send_time = 0.0

        self.last_error = ""
        self.last_grip_detected = False

        self.tool_di0 = False
        self.tool_di1 = False
        self.last_input_update_time = 0.0

        self.rtde_io = None

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True
        )

        self.thread.start()

    def enable(
        self
    ):
        with self.lock:
            self.enabled = True

    def disable(
        self
    ):
        with self.lock:
            self.enabled = False
            self.latest_target = None

    def update_inputs(
        self,
        di0,
        di1
    ):
        with self.lock:
            self.tool_di0 = bool(
                di0
            )

            self.tool_di1 = bool(
                di1
            )

            self.last_input_update_time = (
                time.monotonic()
            )

    def set_target(
        self,
        width_mm
    ):
        with self.lock:
            if not self.enabled:
                return

            self.latest_target = float(
                clamp(
                    width_mm,
                    GRIPPER_CLOSED_WIDTH_MM,
                    GRIPPER_OPEN_WIDTH_MM
                )
            )

    def get_status(self):
        with self.lock:
            return (
                self.last_sent_target,
                self.last_grip_detected,
                self.last_error
            )

    def stop(self):
        self.disable()
        self.running = False

        try:
            self.thread.join(
                timeout=2.0
            )
        except Exception:
            pass

        try:
            if self.rtde_io is not None:
                self.rtde_io.setToolDigitalOut(
                    0,
                    False
                )
        except Exception:
            pass

    def _tool_input(
        self,
        pin
    ):
        with self.lock:
            if pin == 0:
                return bool(
                    self.tool_di0
                )

            return bool(
                self.tool_di1
            )

    def _set_tool_output(
        self,
        pin,
        value
    ):
        ok = self.rtde_io.setToolDigitalOut(
            int(pin),
            bool(value)
        )

        if ok is False:
            raise RuntimeError(
                f"setToolDigitalOut({pin}) sikertelen."
            )

    def _wait_input(
        self,
        pin,
        wanted,
        timeout_s
    ):
        deadline = (
            time.monotonic()
            + timeout_s
        )

        while (
            time.monotonic()
            < deadline
        ):
            if (
                self._tool_input(
                    pin
                )
                == wanted
            ):
                return True

            time.sleep(
                0.008
            )

        return False

    def _send_rg2_command(
        self,
        width_mm,
        force_n
    ):
        target_width = float(
            clamp(
                width_mm,
                0.0,
                110.0
            )
        )

        target_force = float(
            clamp(
                force_n,
                4.0,
                40.0
            )
        )

        encoded_width = (
            target_width
            + 9.2
        )

        encoded_width = clamp(
            encoded_width,
            0.0,
            110.0
        )

        rg_data = (
            math.floor(
                encoded_width
            )
            * 4
        )

        rg_data += (
            math.floor(
                target_force / 2.0
            )
            * 4
            * 111
        )

        rg_data += 32768

        # DI1 = RG2 ready.
        if not self._wait_input(
            1,
            True,
            RG2_READY_TIMEOUT_S
        ):
            raise TimeoutError(
                "RG2 nem lett READY (tool DI1 nem HIGH)."
            )

        value = int(
            rg_data
        )

        msb = 65536

        for _ in range(17):
            self._set_tool_output(
                0,
                True
            )

            if value >= msb:
                value -= msb

                self._set_tool_output(
                    1,
                    False
                )

            else:
                self._set_tool_output(
                    1,
                    True
                )

            time.sleep(
                RG2_BIT_HALF_PERIOD_S
            )

            self._set_tool_output(
                0,
                False
            )

            time.sleep(
                RG2_BIT_HALF_PERIOD_S
            )

            value *= 2

        self._wait_input(
            1,
            False,
            0.60
        )

        if not self._wait_input(
            1,
            True,
            RG2_BUSY_TIMEOUT_S
        ):
            raise TimeoutError(
                "RG2 parancs utan nem tert vissza READY allapotba."
            )

        grip_detected = (
            self._tool_input(
                0
            )
        )

        return grip_detected

    def _connect(self):
        self.rtde_io = (
            RTDEIO(
                self.robot_ip
            )
        )

        # Clock LOW kezdoallapot.
        self._set_tool_output(
            0,
            False
        )

    def _loop(self):
        try:
            self._connect()

        except Exception as e:
            with self.lock:
                self.last_error = (
                    type(e).__name__
                    + ": "
                    + str(e)
                )

            return

        while self.running:
            with self.lock:
                enabled = (
                    self.enabled
                )

                target = (
                    self.latest_target
                )

                last_target = (
                    self.last_sent_target
                )

                last_send = (
                    self.last_send_time
                )

            if (
                not enabled
                or target is None
            ):
                time.sleep(
                    0.03
                )
                continue

            now = (
                time.monotonic()
            )

            if (
                now - last_send
                < MIN_GRIPPER_COMMAND_INTERVAL_S
            ):
                time.sleep(
                    0.03
                )
                continue

            if (
                last_target is not None
                and abs(
                    target - last_target
                )
                < MIN_GRIPPER_TARGET_CHANGE_MM
            ):
                time.sleep(
                    0.03
                )
                continue

            try:
                with self.lock:
                    if not self.enabled:
                        continue

                grip_detected = (
                    self._send_rg2_command(
                        target,
                        GRIPPER_FORCE_N
                    )
                )

                with self.lock:
                    self.last_sent_target = (
                        target
                    )

                    self.last_send_time = (
                        time.monotonic()
                    )

                    self.last_grip_detected = (
                        grip_detected
                    )

                    self.last_error = ""

                print()
                print(
                    f"RG2 -> target={target:.1f} mm "
                    f"| GRIP={grip_detected}"
                )

            except Exception as e:
                with self.lock:
                    self.last_error = (
                        type(e).__name__
                        + ": "
                        + str(e)
                    )

                    self.last_send_time = (
                        time.monotonic()
                    )

                print()
                print(
                    "[RG2 RTDE-I/O HIBA] "
                    + type(e).__name__
                    + ": "
                    + str(e)
                )

            time.sleep(
                0.03
            )



def drain_astra_shadow_udp(sock):
    latest = None
    while True:
        try:
            data, _ = sock.recvfrom(64)
        except BlockingIOError:
            break
        if len(data) != ASTRA_SHADOW_PACKET_SIZE:
            continue
        latest = struct.unpack(ASTRA_SHADOW_PACKET_FORMAT, data)
    return latest


def _control_source_code(source):
    source = str(source or "IDLE")
    if source == "ASTRA":
        return 1
    if source == "TOP_FALLBACK":
        return 2
    if source == "HOLD":
        return 3
    if source.startswith("PAUSE"):
        return 4
    return 0


def send_pickplace_status(
    sock, tracking_on, gripper_on, control_mode, control_source,
    reacquire_policy, trial_active, top_gate_valid, loss_elapsed_s
):
    mode_code = 2 if control_mode == "DUAL_TOP_FALLBACK" else 1
    flow_code = 1 if reacquire_policy == "FLOW" else 0
    packet = struct.pack(
        PICK_STATUS_PACKET_FORMAT,
        1 if tracking_on else 0,
        1 if gripper_on else 0,
        mode_code,
        _control_source_code(control_source),
        flow_code,
        1 if trial_active else 0,
        1 if top_gate_valid else 0,
        float(max(0.0, loss_elapsed_s)),
    )
    for port in (PICK_STATUS_RGB_PORT, PICK_STATUS_TOP_PORT):
        try:
            sock.sendto(packet, (STATUS_UDP_IP, port))
        except Exception:
            pass



class NonBlockingSerialLines:
    def __init__(self, ser):
        self.ser = ser
        self.buffer = bytearray()

    def drain(self):
        lines = []

        waiting = self.ser.in_waiting

        if waiting > 0:
            self.buffer.extend(
                self.ser.read(
                    waiting
                )
            )

        while True:
            idx = self.buffer.find(
                b"\n"
            )

            if idx < 0:
                break

            raw = self.buffer[:idx]
            del self.buffer[:idx + 1]

            line = raw.decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if line:
                lines.append(
                    line
                )

        if len(self.buffer) > 8192:
            self.buffer.clear()

        return lines


def find_imu_port():
    ports = list(
        serial.tools.list_ports.comports()
    )

    if not ports:
        raise RuntimeError(
            "Nem talaltam COM portot az ESP32 IMU-hoz."
        )

    print()
    print("Elerheto COM portok:")

    for i, port in enumerate(ports):
        print(
            f"  [{i}] {port.device}  {port.description}"
        )

    preferred = []

    for port in ports:
        desc = (
            port.description
            or ""
        ).lower()

        if any(
            hint.lower() in desc
            for hint in IMU_PORT_DESCRIPTION_HINTS
        ):
            preferred.append(
                port.device
            )

    if len(preferred) == 1:
        print(
            f"[OK] ESP32 IMU port automatikusan: {preferred[0]}"
        )
        return preferred[0]

    if len(ports) == 1:
        print(
            f"[OK] Egyetlen COM port: {ports[0].device}"
        )
        return ports[0].device

    while True:
        answer = input(
            "ESP32 IMU port? index vagy COM nev: "
        ).strip()

        if not answer:
            continue

        if answer.upper().startswith(
            "COM"
        ):
            return answer.upper()

        try:
            idx = int(
                answer
            )

            if 0 <= idx < len(ports):
                return ports[idx].device

        except ValueError:
            pass


def open_imu_serial_no_reset(
    port
):
    """
    A CP210x/ESP32 auto-resetet probaljuk elkerulni azzal,
    hogy a DTR/RTS vonalakat megnyitas elott inaktiv allapotba
    tesszuk.

    Ha az ESP mar fut es IMUP sorokat kuld, a fo kod nem kenyszerit
    uj kalibraciot minden Python ujrainditasnal.
    """

    ser = serial.Serial(
        port=None,
        baudrate=IMU_BAUD,
        timeout=0.05,
        write_timeout=0.25
    )

    ser.port = port
    ser.dtr = False
    ser.rts = False
    ser.open()

    time.sleep(
        0.25
    )

    return ser


def wait_for_imu_ready(
    ser,
    timeout_s=IMU_READY_TIMEOUT_S
):
    """
    Ket ervenyes eset:
      1) az ESP eppen bootol/kalibral -> megvarjuk a # READY sort;
      2) az ESP mar korabban elindult -> egy IMUP sor mar eleg,
         NEM varunk ujabb resetre / kalibraciora.
    """

    print()
    print(
        "ESP32 IMU + pot kapcsolat ellenorzese..."
    )
    print(
        "Ha az ESP mar fut, a meglevo IMUP streamet hasznaljuk."
    )
    print(
        "Ha most bootol, varjuk a # READY uzenetet."
    )

    deadline = (
        time.monotonic()
        + timeout_s
    )

    saw_startup = False

    while time.monotonic() < deadline:
        raw = ser.readline()

        if not raw:
            continue

        line = raw.decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not line:
            continue

        if line.startswith(
            "IMUP,"
        ):
            if saw_startup:
                print(
                    "[OK] ESP32 IMU+POT stream elindult."
                )
            else:
                print(
                    "[OK] ESP32 mar futott; IMU+POT stream atveve ujrakalibralas nelkul."
                )

            return

        if line.startswith(
            "#"
        ):
            saw_startup = True
            print(
                line
            )

        if line == "# READY":
            print(
                "[OK] ESP32 IMU+POT READY"
            )
            return

    raise RuntimeError(
        "ESP32 IMU+POT stream/READY timeout. "
        "Ellenorizd a COM portot; ha kell, egyszer nyomj EN-t, "
        "de Arduino Serial Monitor ne maradjon nyitva."
    )



def parse_imup_line(
    line
):
    if not line.startswith(
        "IMUP,"
    ):
        return None

    parts = line.split(
        ","
    )

    if len(parts) != 16:
        return None

    try:
        return {
            "ms": int(parts[1]),
            "qw": float(parts[2]),
            "qx": float(parts[3]),
            "qy": float(parts[4]),
            "qz": float(parts[5]),
            "roll": float(parts[6]),
            "pitch": float(parts[7]),
            "yaw": float(parts[8]),
            "ax": float(parts[9]),
            "ay": float(parts[10]),
            "az": float(parts[11]),
            "gx": float(parts[12]),
            "gy": float(parts[13]),
            "gz": float(parts[14]),
            "pot_raw": float(parts[15]),
        }

    except ValueError:
        return None


def quaternion_to_matrix(
    qw,
    qx,
    qy,
    qz
):
    q = np.asarray(
        [qw, qx, qy, qz],
        dtype=np.float64
    )

    n = float(
        np.linalg.norm(
            q
        )
    )

    if (
        not np.isfinite(n)
        or n < 1e-10
    ):
        return np.eye(
            3,
            dtype=np.float64
        )

    q /= n

    w, x, y, z = q

    return np.array(
        [
            [
                1.0 - 2.0 * (y*y + z*z),
                2.0 * (x*y - w*z),
                2.0 * (x*z + w*y),
            ],
            [
                2.0 * (x*y + w*z),
                1.0 - 2.0 * (x*x + z*z),
                2.0 * (y*z - w*x),
            ],
            [
                2.0 * (x*z - w*y),
                2.0 * (y*z + w*x),
                1.0 - 2.0 * (x*x + y*y),
            ],
        ],
        dtype=np.float64
    )


def imu_relative_to_robot_local_rotvec(
    imu_R
):
    # A rotacios vektor szenzor koordinataban.
    sensor_rv = matrix_to_rotvec(
        imu_R
    )

    # A letesztelt X/Z csere + Y elojelcsere.
    robot_local_rv = (
        IMU_TO_ROBOT_LOCAL
        @ sensor_rv
    )

    robot_local_rv *= (
        ORIENTATION_GAIN
    )

    angle = float(
        np.linalg.norm(
            robot_local_rv
        )
    )

    max_angle = math.radians(
        MAX_REL_ANGLE_DEG
    )

    if (
        angle > max_angle
        and angle > 1e-9
    ):
        robot_local_rv *= (
            max_angle
            / angle
        )

    return robot_local_rv


def rate_limit_robot_local_orientation(
    current_rv,
    desired_rv,
    dt
):
    current = np.asarray(
        current_rv,
        dtype=np.float64
    ).copy()

    desired = np.asarray(
        desired_rv,
        dtype=np.float64
    )

    max_step = np.deg2rad(
        ROBOT_LOCAL_ORIENTATION_SPEED_DEG_S
    ) * float(dt)

    delta = (
        desired
        - current
    )

    delta = np.clip(
        delta,
        -max_step,
        max_step
    )

    return (
        current
        + delta
    )


class CoordCalibrationWizard:
    """
    Egyetlen K gombbal vegigvezetheto kalibracio.

    1. K: kozep rogzitese RIGHT-hoz
    2. K: jobb oldali pont
    3. K: uj kozep UP-hoz
    4. K: felso pont
    5. K: uj kozep FORWARD-hoz
    6. K: kamera fele mutato pont -> szamitas + mentes
    """

    def __init__(self):
        self.active = False
        self.stage = 0
        self.right_center = None
        self.right_end = None
        self.up_center = None
        self.up_end = None
        self.forward_center = None

    def cancel(self):
        self.__init__()

    def handle_k(
        self,
        hand_xyz
    ):
        if not self.active:
            self.active = True
            self.stage = 1
            self.right_center = (
                hand_xyz.copy()
            )

            print()
            print("=" * 72)
            print("3D KOORDINATA-KALIBRACIO INDUL")
            print("=" * 72)
            print(
                "1/3 RIGHT: a kozephelyzet rogzitve."
            )
            print(
                "Mozgasd a kezed a SAJAT JOBB oldalad fele "
                "legalabb kb. 10-15 cm-t,"
            )
            print(
                "majd nyomj K-t."
            )
            return None

        if self.stage == 1:
            self.right_end = (
                hand_xyz.copy()
            )
            self.stage = 2

            right_len = vec_norm(
                vec_sub(
                    self.right_end,
                    self.right_center
                )
            )

            print()
            print(
                f"RIGHT minta: {right_len:.1f} mm."
            )
            print(
                "Tedd vissza a kezed a kozephelyzetbe, "
                "majd nyomj K-t."
            )
            return None

        if self.stage == 2:
            self.up_center = (
                hand_xyz.copy()
            )
            self.stage = 3

            print()
            print(
                "2/3 UP: kozephelyzet rogzitve."
            )
            print(
                "Mozgasd a kezed FELFELE legalabb kb. "
                "10-15 cm-t, majd nyomj K-t."
            )
            return None

        if self.stage == 3:
            self.up_end = (
                hand_xyz.copy()
            )
            self.stage = 4

            up_len = vec_norm(
                vec_sub(
                    self.up_end,
                    self.up_center
                )
            )

            print()
            print(
                f"UP minta: {up_len:.1f} mm."
            )
            print(
                "Tedd vissza a kezed a kozephelyzetbe, "
                "majd nyomj K-t."
            )
            return None

        if self.stage == 4:
            self.forward_center = (
                hand_xyz.copy()
            )
            self.stage = 5

            print()
            print(
                "3/3 FORWARD: kozephelyzet rogzitve."
            )
            print(
                "Mozgasd a kezed ELORE, A KAMERA FELE "
                "legalabb kb. 10-15 cm-t,"
            )
            print(
                "majd nyomj K-t."
            )
            return None

        if self.stage == 5:
            forward_end = (
                hand_xyz.copy()
            )

            right_raw = vec_sub(
                self.right_end,
                self.right_center
            )

            up_raw = vec_sub(
                self.up_end,
                self.up_center
            )

            forward_raw = vec_sub(
                forward_end,
                self.forward_center
            )

            basis = build_coord_basis(
                right_raw,
                up_raw,
                forward_raw
            )

            save_coord_calibration(
                basis
            )

            print()
            print("=" * 72)
            print("[OK] 3D KOORDINATA-KALIBRACIO KESZ")
            print("=" * 72)
            print(
                "RIGHT   kamera-vektor: "
                f"{basis['right'][0]:+.3f}, "
                f"{basis['right'][1]:+.3f}, "
                f"{basis['right'][2]:+.3f}"
            )
            print(
                "UP      kamera-vektor: "
                f"{basis['up'][0]:+.3f}, "
                f"{basis['up'][1]:+.3f}, "
                f"{basis['up'][2]:+.3f}"
            )
            print(
                "FORWARD kamera-vektor: "
                f"{basis['forward'][0]:+.3f}, "
                f"{basis['forward'][1]:+.3f}, "
                f"{basis['forward'][2]:+.3f}"
            )
            print(
                "Elmentve ide:"
            )
            print(
                COORD_CALIBRATION_FILE
            )
            print()
            print(
                "Most nyomj C-t uj kez+robot referenciahoz, "
                "aztan P-t a koveteshez."
            )

            self.cancel()
            return basis

        return None


def drain_xyz_udp(sock):
    latest = None

    while True:
        try:
            data, _ = sock.recvfrom(128)
        except BlockingIOError:
            break

        if len(data) != PACKET_SIZE:
            continue

        latest = struct.unpack(
            PACKET_FORMAT,
            data
        )

    return latest



def drain_top_udp(
    sock
):
    latest = None

    while True:
        try:
            data, _ = sock.recvfrom(
                128
            )
        except BlockingIOError:
            break

        if len(data) != TOP_PACKET_SIZE:
            continue

        latest = struct.unpack(
            TOP_PACKET_FORMAT,
            data
        )

    return latest


def load_top_metric_calibration():
    if not TOP_METRIC_CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            "Hianyzik a V4 top->Astra kalibracio: "
            + str(TOP_METRIC_CALIBRATION_FILE)
        )

    payload = json.loads(
        TOP_METRIC_CALIBRATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    planes_raw = payload.get("planes", {})
    required = ("LOW", "MID", "HIGH")
    planes = {}

    for label in required:
        item = planes_raw.get(label)
        if item is None:
            raise ValueError(
                f"A V4 kalibraciobol hianyzik a {label} sik."
            )

        H = np.asarray(
            item["homography_3x3"],
            dtype=np.float64
        )

        if H.shape != (3, 3) or not np.all(np.isfinite(H)):
            raise ValueError(
                f"Ervenytelen homografia: {label}"
            )

        planes[label] = {
            "label": label,
            "plane_y_mm": float(item["plane_y_mm"]),
            "H": H,
        }

    return {
        "version": payload.get("version"),
        "planes": planes,
    }


def predict_top_homography(H, u, v):
    src = np.array(
        [[[float(u), float(v)]]],
        dtype=np.float32
    )

    dst = cv2.perspectiveTransform(
        src,
        np.asarray(H, dtype=np.float64)
    )

    x_mm = float(dst[0, 0, 0])
    z_mm = float(dst[0, 0, 1])

    if not (
        math.isfinite(x_mm)
        and math.isfinite(z_mm)
    ):
        raise ValueError("Nem veges top-kamera predikcio.")

    return x_mm, z_mm


def predict_top_interpolated(calibration, u, v, y_mm):
    ordered = sorted(
        calibration["planes"].values(),
        key=lambda m: m["plane_y_mm"]
    )

    y = float(y_mm)

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

    y0 = float(lo["plane_y_mm"])
    y1 = float(hi["plane_y_mm"])

    if abs(y1 - y0) < 1e-9:
        alpha = 0.0
    else:
        alpha = (y - y0) / (y1 - y0)

    alpha = max(-0.5, min(1.5, alpha))

    x0, z0 = predict_top_homography(lo["H"], u, v)
    x1, z1 = predict_top_homography(hi["H"], u, v)

    x_mm = x0 + alpha * (x1 - x0)
    z_mm = z0 + alpha * (z1 - z0)

    nearest = min(
        ordered,
        key=lambda m: abs(y - float(m["plane_y_mm"]))
    )

    nearest_y_dist = abs(
        y - float(nearest["plane_y_mm"])
    )

    return {
        "x_mm": float(x_mm),
        "z_mm": float(z_mm),
        "lo": lo["label"],
        "hi": hi["label"],
        "alpha": float(alpha),
        "nearest_plane": nearest["label"],
        "nearest_y_dist_mm": float(nearest_y_dist),
    }


def evaluate_top_metric_fallback(
    calibration,
    top_palm,
    held_y_mm,
    last_astra_hand,
    previous_fallback_hand=None
):
    result = {
        "valid": False,
        "reason": "NO_DATA",
        "x_mm": None,
        "y_mm": None,
        "z_mm": None,
        "nearest_plane": "",
        "nearest_y_dist_mm": None,
        "alpha": None,
        "top_vs_astra_xz_error_mm": None,
    }

    if (
        top_palm is None
        or held_y_mm is None
        or last_astra_hand is None
    ):
        return result

    u = float(top_palm[0])
    v = float(top_palm[1])

    try:
        pred = predict_top_interpolated(
            calibration,
            u,
            v,
            held_y_mm
        )
    except Exception as e:
        result["reason"] = (
            "PREDICT_ERROR:" + type(e).__name__
        )
        return result

    x_mm = float(pred["x_mm"])
    z_mm = float(pred["z_mm"])
    nearest_plane = str(pred["nearest_plane"])
    nearest_y_dist = float(pred["nearest_y_dist_mm"])

    result.update({
        "x_mm": x_mm,
        "y_mm": float(held_y_mm),
        "z_mm": z_mm,
        "nearest_plane": nearest_plane,
        "nearest_y_dist_mm": nearest_y_dist,
        "alpha": float(pred["alpha"]),
    })

    try:
        result["top_vs_astra_xz_error_mm"] = math.hypot(
            x_mm - float(last_astra_hand[0]),
            z_mm - float(last_astra_hand[2])
        )
    except Exception:
        pass

    if nearest_y_dist > TOP_FALLBACK_MAX_NEAREST_PLANE_Y_MM:
        result["reason"] = "Y_OUTSIDE_VALIDATED_PLANE"
        return result

    bounds = TOP_FALLBACK_GUIDE_BOUNDS.get(
        nearest_plane
    )

    if bounds is None:
        result["reason"] = "NO_GUIDE_BOUNDS"
        return result

    u0, u1, v0, v1 = bounds

    if not (u0 <= u <= u1 and v0 <= v <= v1):
        result["reason"] = "TOP_UV_OUTSIDE_CALIBRATED_GUIDE"
        return result

    last_z = float(last_astra_hand[2])

    if not (
        TOP_FALLBACK_Z_MIN_MM
        <= last_z
        <= TOP_FALLBACK_Z_MAX_MM
    ):
        result["reason"] = "LAST_ASTRA_Z_OUTSIDE_VALIDATED_RANGE"
        return result

    if not (
        TOP_FALLBACK_Z_MIN_MM
        <= z_mm
        <= TOP_FALLBACK_Z_MAX_MM
    ):
        result["reason"] = "PREDICTED_Z_OUTSIDE_VALIDATED_RANGE"
        return result

    if previous_fallback_hand is None:
        dxz = math.hypot(
            x_mm - float(last_astra_hand[0]),
            z_mm - float(last_astra_hand[2])
        )

        if dxz > TOP_FALLBACK_MAX_START_XZ_DIFF_MM:
            result["reason"] = "FALLBACK_START_JUMP_TOO_LARGE"
            return result

    else:
        dxz = math.hypot(
            x_mm - float(previous_fallback_hand[0]),
            z_mm - float(previous_fallback_hand[2])
        )

        if dxz > TOP_FALLBACK_MAX_FRAME_XZ_JUMP_MM:
            result["reason"] = "FALLBACK_FRAME_JUMP_TOO_LARGE"
            return result

    result["valid"] = True
    result["reason"] = "OK"
    return result


class ExperimentLogger:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        stamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        self.samples_path = (
            self.base_dir
            / f"tdk_pickplace_samples_{stamp}.csv"
        )
        self.events_path = (
            self.base_dir
            / f"tdk_pickplace_events_{stamp}.csv"
        )
        self.trials_path = (
            self.base_dir
            / f"tdk_pickplace_trials_{stamp}.csv"
        )

        self.sample_fields = [
            "pc_time", "mono_s", "trial_id", "trial_active", "mode",
            "control_source", "tracking", "calibrated", "servo_active",
            "astra_seq", "astra_packet_fresh", "astra_control_valid",
            "latest_xyz_packet_valid", "astra_rgb_hand_valid",
            "astra_loss_reason", "pose_guard_good", "palm_camera_alignment",
            "astra_hand_age_s", "astra_x_mm", "astra_y_mm", "astra_z_mm",
            "top_seq", "top_packet_valid_raw", "top_packet_age_s",
            "top_fresh", "top_hand_age_s", "top_u", "top_v",
            "top_width", "top_pred_x_mm", "top_pred_y_mm", "top_pred_z_mm",
            "top_gate_valid", "top_gate_reason", "top_nearest_plane",
            "top_nearest_plane_y_dist_mm", "top_interp_alpha",
            "top_vs_astra_xz_error_mm", "imu_fresh", "imu_qw", "imu_qx",
            "imu_qy", "imu_qz", "imu_roll_deg", "imu_pitch_deg",
            "imu_yaw_deg", "imu_des_rvx", "imu_des_rvy", "imu_des_rvz",
            "imu_cmd_rvx", "imu_cmd_rvy", "imu_cmd_rvz", "pot_fresh",
            "pot_raw", "pot_filtered", "rg2_tracking", "rg2_mapped_width_mm",
            "rg2_last_sent_target_mm", "rg2_grip_detected", "rg2_ready_di1",
            "rg2_grip_di0", "rg2_error", "desired_tcp_x", "desired_tcp_y",
            "desired_tcp_z", "desired_tcp_rx", "desired_tcp_ry", "desired_tcp_rz",
            "commanded_tcp_x", "commanded_tcp_y", "commanded_tcp_z",
            "commanded_tcp_rx", "commanded_tcp_ry", "commanded_tcp_rz",
            "actual_tcp_x", "actual_tcp_y", "actual_tcp_z", "actual_tcp_rx",
            "actual_tcp_ry", "actual_tcp_rz", "tcp_xyz_error_mm",
            "reacquire_policy", "reacquire_window_s", "auto_reacquire_wait",
            "resume_ramp_active", "resume_ramp_factor", "loss_elapsed_s",
            "shadow_astra_seq", "shadow_astra_valid", "shadow_astra_u",
            "shadow_astra_v", "shadow_astra_age_s", "shadow_pair_skew_ms",
            "shadow_pair_ready", "shadow_stereo_status"
        ]

        self.event_fields = [
            "pc_time", "mono_s", "trial_id", "mode", "event", "detail"
        ]

        self.trial_fields = [
            "trial_id", "mode", "reacquire_policy", "start_pc_time", "end_pc_time",
            "duration_s", "outcome", "astra_dropout_episodes",
            "astra_unavailable_s", "top_fallback_entries", "top_fallback_s",
            "hold_entries", "hold_s", "pause_entries", "pause_s"
        ]

        self.samples_f = open(
            self.samples_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        )
        self.events_f = open(
            self.events_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        )
        self.trials_f = open(
            self.trials_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        )

        self.samples_w = csv.DictWriter(
            self.samples_f,
            fieldnames=self.sample_fields
        )
        self.events_w = csv.DictWriter(
            self.events_f,
            fieldnames=self.event_fields
        )
        self.trials_w = csv.DictWriter(
            self.trials_f,
            fieldnames=self.trial_fields
        )

        self.samples_w.writeheader()
        self.events_w.writeheader()
        self.trials_w.writeheader()

        self.trial_id = 0
        self.active_trial = None
        self.sample_counter = 0

    @staticmethod
    def _pc_time():
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def event(self, mono_s, mode, event, detail=""):
        self.events_w.writerow({
            "pc_time": self._pc_time(),
            "mono_s": f"{float(mono_s):.6f}",
            "trial_id": (
                "" if self.active_trial is None
                else self.active_trial["trial_id"]
            ),
            "mode": mode,
            "event": event,
            "detail": detail,
        })
        self.events_f.flush()

    def start_trial(self, mono_s, mode, reacquire_policy="FLOW"):
        if self.active_trial is not None:
            return False

        self.trial_id += 1
        self.active_trial = {
            "trial_id": self.trial_id,
            "mode": mode,
            "reacquire_policy": str(reacquire_policy),
            "start_mono": float(mono_s),
            "start_pc_time": self._pc_time(),
            "last_update": float(mono_s),
            "prev_astra_valid": None,
            "prev_source": None,
            "astra_dropout_episodes": 0,
            "astra_unavailable_s": 0.0,
            "top_fallback_entries": 0,
            "top_fallback_s": 0.0,
            "hold_entries": 0,
            "hold_s": 0.0,
            "pause_entries": 0,
            "pause_s": 0.0,
        }
        self.event(mono_s, mode, "TRIAL_START")
        return True

    def update_trial(self, mono_s, astra_valid, source):
        # V2 LOGGERFIX: az elso mintanal prev_source=None lehet.
        # startswith csak normalizalt stringen fusson.
        t = self.active_trial
        if t is None:
            return

        now = float(mono_s)
        dt = max(0.0, now - float(t["last_update"]))

        if t["prev_astra_valid"] is False:
            t["astra_unavailable_s"] += dt

        prev_source = t["prev_source"]
        prev_source_s = "" if prev_source is None else str(prev_source)
        source_s = "IDLE" if source is None else str(source)

        if prev_source_s == "TOP_FALLBACK":
            t["top_fallback_s"] += dt
        elif prev_source_s == "HOLD":
            t["hold_s"] += dt
        elif prev_source_s.startswith("PAUSE"):
            t["pause_s"] += dt

        if (
            t["prev_astra_valid"] is True
            and not astra_valid
        ):
            t["astra_dropout_episodes"] += 1

        if source_s != prev_source_s:
            if source_s == "TOP_FALLBACK":
                t["top_fallback_entries"] += 1
            elif source_s == "HOLD":
                t["hold_entries"] += 1
            elif source_s.startswith("PAUSE"):
                t["pause_entries"] += 1

        t["prev_astra_valid"] = bool(astra_valid)
        t["prev_source"] = source_s
        t["last_update"] = now

    def finish_trial(self, mono_s, outcome):
        t = self.active_trial
        if t is None:
            return False

        self.update_trial(
            mono_s,
            bool(t["prev_astra_valid"])
            if t["prev_astra_valid"] is not None
            else False,
            t["prev_source"] or "IDLE"
        )

        end_pc = self._pc_time()
        duration = max(
            0.0,
            float(mono_s) - float(t["start_mono"])
        )

        self.trials_w.writerow({
            "trial_id": t["trial_id"],
            "mode": t["mode"],
            "reacquire_policy": t.get("reacquire_policy", ""),
            "start_pc_time": t["start_pc_time"],
            "end_pc_time": end_pc,
            "duration_s": f"{duration:.6f}",
            "outcome": outcome,
            "astra_dropout_episodes": t["astra_dropout_episodes"],
            "astra_unavailable_s": f"{t['astra_unavailable_s']:.6f}",
            "top_fallback_entries": t["top_fallback_entries"],
            "top_fallback_s": f"{t['top_fallback_s']:.6f}",
            "hold_entries": t["hold_entries"],
            "hold_s": f"{t['hold_s']:.6f}",
            "pause_entries": t["pause_entries"],
            "pause_s": f"{t['pause_s']:.6f}",
        })
        self.trials_f.flush()
        self.event(
            mono_s,
            t["mode"],
            "TRIAL_END",
            outcome
        )
        self.active_trial = None
        return True

    def write_sample(self, row):
        safe = {
            key: row.get(key, "")
            for key in self.sample_fields
        }
        self.samples_w.writerow(safe)
        self.sample_counter += 1

        if self.sample_counter % 25 == 0:
            self.samples_f.flush()

    def close(self, mono_s=None, mode=""):
        if self.active_trial is not None:
            if mono_s is None:
                mono_s = time.monotonic()
            self.finish_trial(
                mono_s,
                "ABORTED_SESSION_END"
            )

        for f in (
            self.samples_f,
            self.events_f,
            self.trials_f
        ):
            try:
                f.flush()
                f.close()
            except Exception:
                pass


def pose_or_blank(pose):
    if pose is None or len(pose) < 6:
        return [""] * 6
    return [float(v) for v in pose[:6]]


def start_new_console(
    name,
    script_path
):
    flags = (
        subprocess.CREATE_NEW_CONSOLE
        | subprocess.CREATE_NEW_PROCESS_GROUP
    )

    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(script_path)
        ],
        cwd=str(BASE_DIR),
        creationflags=flags,
        close_fds=True
    )

    print(
        f"[START] {name} "
        f"(PID={proc.pid})"
    )

    return proc


def stop_child(
    name,
    proc
):
    if (
        proc is None
        or proc.poll() is not None
    ):
        return

    print(
        f"Leallitas: {name}..."
    )

    try:
        proc.terminate()
        proc.wait(timeout=2.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def wait_for_palm_ready(
    rgb_proc
):
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    sock.bind(
        (
            PALM_UDP_IP,
            PALM_UDP_PORT
        )
    )

    sock.settimeout(0.25)

    deadline = (
        time.monotonic()
        + CAMERA_START_TIMEOUT_S
    )

    print(
        "Varom az Astra RGB + MediaPipe ervenyes tenyerpontjat..."
    )

    try:
        while (
            time.monotonic()
            < deadline
        ):
            if rgb_proc.poll() is not None:
                raise RuntimeError(
                    "RGB process leallt: "
                    f"exit={rgb_proc.returncode}"
                )

            try:
                data, _ = sock.recvfrom(
                    128
                )
            except socket.timeout:
                continue

            if len(data) != PALM_PACKET_SIZE:
                continue

            (
                seq,
                palm_x,
                palm_y,
                valid
            ) = struct.unpack(
                PALM_PACKET_FORMAT,
                data
            )

            if valid:
                print(
                    "[OK] Astra RGB + MediaPipe kesz: "
                    f"{palm_x:.3f}, {palm_y:.3f}"
                )
                return

        raise TimeoutError(
            "30 s alatt nem jott ervenyes palm UDP."
        )

    finally:
        sock.close()


def wait_for_xyz_ready(
    xyz_sock,
    rgb_proc,
    depth_proc
):
    deadline = (
        time.monotonic()
        + CAMERA_START_TIMEOUT_S
    )

    print(
        "Varom az ervenyes OpenNI XYZ adatot..."
    )

    while (
        time.monotonic()
        < deadline
    ):
        if rgb_proc.poll() is not None:
            raise RuntimeError(
                "RGB process leallt: "
                f"exit={rgb_proc.returncode}"
            )

        if depth_proc.poll() is not None:
            raise RuntimeError(
                "Depth process leallt: "
                f"exit={depth_proc.returncode}"
            )

        packet = drain_xyz_udp(
            xyz_sock
        )

        if packet is not None:
            (
                seq,
                x_mm,
                y_mm,
                z_mm,
                valid
            ) = packet

            if valid:
                print(
                    "[OK] Kamera rendszer kesz: "
                    f"XYZ={x_mm:+.0f},"
                    f"{y_mm:+.0f},"
                    f"{z_mm:+.0f} mm"
                )

                return [
                    float(x_mm),
                    float(y_mm),
                    float(z_mm)
                ]

        time.sleep(0.02)

    raise TimeoutError(
        "30 s alatt nem jott ervenyes XYZ."
    )


def verify_files():
    missing = []

    for path in (
        RGB_SCRIPT,
        DEPTH_SCRIPT,
        TOP_CAMERA_SCRIPT,
        MODEL_FILE,
        TOP_METRIC_CALIBRATION_FILE
    ):
        if not path.exists():
            missing.append(
                path.name
            )

    if missing:
        raise FileNotFoundError(
            "Hianyzik a launcher mappajabol: "
            + ", ".join(missing)
        )


def main():
    print("=" * 78)
    print("UR5e TDK PICK-AND-PLACE + ASTRA/TOP + IMU + RG2 - V1")
    print("=" * 78)
    print()
    print(
        f"Mappa: {BASE_DIR}"
    )
    print()

    verify_files()

    top_metric_calibration = load_top_metric_calibration()

    print(
        "[OK] V4 top->Astra kalibracio betoltve: "
        f"version={top_metric_calibration.get('version')}"
    )

    measurement_logger = ExperimentLogger(
        MEASUREMENT_DIR
    )

    print(
        "[OK] Meresi logok: "
        + str(MEASUREMENT_DIR)
    )
    print(
        "      " + measurement_logger.samples_path.name
    )
    print(
        "      " + measurement_logger.events_path.name
    )
    print(
        "      " + measurement_logger.trials_path.name
    )

    session_t0 = time.monotonic()

    rgb_proc = None
    depth_proc = None
    top_proc = None
    top_camera_available = False
    last_top_restart_attempt = 0.0

    rtde_r = None
    rtde_c = None

    imu_ser = None
    imu_lines = None
    imu_port = None

    xyz_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    xyz_sock.bind(
        (
            UDP_IP,
            UDP_PORT
        )
    )

    xyz_sock.setblocking(False)

    feature_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    feature_sock.bind(
        (
            FEATURE_UDP_IP,
            FEATURE_UDP_PORT
        )
    )

    feature_sock.setblocking(
        False
    )

    astra_shadow_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )
    astra_shadow_sock.bind((ASTRA_SHADOW_UDP_IP, ASTRA_SHADOW_UDP_PORT))
    astra_shadow_sock.setblocking(False)

    top_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    top_sock.bind(
        (
            TOP_UDP_IP,
            TOP_UDP_PORT
        )
    )

    top_sock.setblocking(
        False
    )

    status_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    keys = KeyEdge()

    coord_basis = (
        load_coord_calibration()
    )

    robot_direction_map = (
        load_robot_direction_map()
    )

    coord_wizard = (
        CoordCalibrationWizard()
    )

    latest_hand = None
    latest_sequence = None
    latest_xyz_packet_valid = False

    # V13: tracking kozbeni kamera-ugras szuro.
    last_accepted_hand = None
    last_rejected_hand_jump_mm = 0.0

    last_packet_time = None
    last_valid_time = None

    latest_top_sequence = None
    latest_top_palm = None
    latest_top_packet_valid_raw = False
    last_top_packet_time = None
    last_top_valid_time = None
    top_assist_active = False

    # TDK A/B kiserleti allapot.
    control_mode = "ASTRA_ONLY"
    control_source = "IDLE"
    previous_control_source = "IDLE"
    control_hand = None
    top_metric_info = {
        "valid": False,
        "reason": "NOT_EVALUATED",
        "x_mm": None,
        "y_mm": None,
        "z_mm": None,
        "nearest_plane": "",
        "nearest_y_dist_mm": None,
        "alpha": None,
        "top_vs_astra_xz_error_mm": None,
    }
    top_fallback_start_time = None
    last_top_fallback_hand = None
    last_measurement_log_time = -999.0

    # Pick-and-place FLOW / reacquire allapot.
    reacquire_policy = REACQUIRE_POLICY_DEFAULT
    resume_ramp_start_time = None
    resume_ramp_reason = ""
    resume_ramp_factor = 1.0
    auto_reacquire_wait = False

    # Shadow 2D parositas: csak meres/log, soha nem vezerel.
    latest_astra_shadow_seq = None
    latest_astra_shadow_uv = None
    latest_astra_shadow_valid = False
    last_astra_shadow_packet_time = None

    latest_feature_sequence = None
    latest_feature_packet_valid_raw = False
    latest_hand_R = None
    latest_pinch_ratio = None
    last_feature_packet_time = None
    last_feature_valid_time = None

    # ESP32 quaternion IMU + potmeter allapot.
    latest_imu = None
    latest_imu_R = None
    last_imu_packet_time = None

    latest_pot_raw = None
    filtered_pot_raw = None
    last_pot_packet_time = None
    desired_imu_robot_local_rv = np.zeros(
        3,
        dtype=np.float64
    )
    commanded_imu_robot_local_rv = np.zeros(
        3,
        dtype=np.float64
    )

    # Utolso elfogadott orientacio a frame-to-frame ugrasok szuresehez.
    last_accepted_hand_R = None
    last_rejected_orientation_jump_deg = 0.0

    hand_reference = None
    hand_orientation_reference_R = None

    robot_reference = None
    robot_reference_R = None

    desired_pose = None
    commanded_pose = None

    filtered_orientation_rotvec = None

    (
        pot_closed_raw,
        pot_open_raw
    ) = load_pot_calibration()

    robot_home = (
        load_robot_home()
    )

    recovery_home = (
        load_recovery_home()
    )

    gripper_tracking = False
    gripper_worker = None
    latest_pot_gripper_width = None

    # Regi pinch allapotvaltozok megmaradnak, de V18-ban nem ezek
    # hajtjak az RG2-t.
    # None / OPEN / CLOSED
    gripper_state = None
    last_gripper_requested_state = None

    gripper_transition_candidate = None
    gripper_transition_since = None
    last_gripper_state_change_time = -999.0

    calibrated = False
    tracking = False
    servo_active = False

    # V13 IK guard statusz.
    ik_guard_blocked = False
    ik_guard_reason = ""
    last_ik_reject_print = 0.0

    # V14: ritkitott IK approval.
    approved_pose = None
    last_ik_check_time = -999.0

    operator_components_mm = (
        0.0,
        0.0,
        0.0
    )

    last_print = 0.0
    child_check = 0.0

    try:
        print("1/5 OpenNI depth...")

        depth_proc = start_new_console(
            "OpenNI depth",
            DEPTH_SCRIPT
        )

        print(
            f"Varakozas {DEPTH_WARMUP_S:.0f} masodpercet, "
            "hogy az OpenNI teljesen felalljon..."
        )

        time.sleep(
            DEPTH_WARMUP_S
        )

        if depth_proc.poll() is not None:
            raise RuntimeError(
                "Depth process mar a bemelegedesi ido alatt leallt: "
                f"exit={depth_proc.returncode}"
            )

        print("[OK] Depth process tovabbra is fut.")

        print()
        print("2/5 Astra RGB + MediaPipe...")

        rgb_proc = start_new_console(
            "Astra RGB MediaPipe",
            RGB_SCRIPT
        )

        print(
            f"Varakozas {RGB_WARMUP_S:.0f} masodpercet, "
            "hogy az RGB + MediaPipe felalljon..."
        )

        time.sleep(
            RGB_WARMUP_S
        )

        if rgb_proc.poll() is not None:
            raise RuntimeError(
                "RGB process mar a bemelegedesi ido alatt leallt: "
                f"exit={rgb_proc.returncode}"
            )

        print("[OK] RGB process tovabbra is fut.")

        print()
        print("3/5 Eternico felso kamera (nev alapjan, reconnect)...")

        try:
            top_proc = start_new_console(
                "Eternico top camera",
                TOP_CAMERA_SCRIPT
            )

            time.sleep(
                TOP_CAMERA_WARMUP_S
            )

            if top_proc.poll() is None:
                top_camera_available = True
                print(
                    "[OK] Felso kamera process fut. "
                    "A/B mereshez passziv log vagy metrikus fallback hasznalhato."
                )
            else:
                print(
                    "[FIGYELEM] Felso kamera process leallt. "
                    "A fo Astra rendszer ettol meg folytatodik."
                )

        except Exception as e:
            top_proc = None
            top_camera_available = False

            print(
                "[FIGYELEM] Felso kamera nem indult: "
                + type(e).__name__
                + ": "
                + str(e)
            )

        latest_hand = wait_for_xyz_ready(
            xyz_sock,
            rgb_proc,
            depth_proc
        )

        last_accepted_hand = (
            latest_hand.copy()
            if latest_hand is not None
            else None
        )

        print()
        print("4/5 ESP32 quaternion IMU + potmeter...")

        imu_port = find_imu_port()

        try:
            imu_ser = open_imu_serial_no_reset(
                imu_port
            )
        except serial.SerialException as e:
            raise RuntimeError(
                "ESP32 COM port nem nyithato. "
                "Zard be az Arduino Serial Monitort. "
                + str(e)
            )

        wait_for_imu_ready(
            imu_ser
        )

        imu_ser.timeout = 0
        imu_lines = NonBlockingSerialLines(
            imu_ser
        )

        print()
        print("5/5 RTDE robot...")

        rtde_r = RTDEReceive(
            ROBOT_IP,
            FREQUENCY
        )

        print("[OK] RTDE Receive")

        rtde_c = RTDEControl(
            ROBOT_IP,
            FREQUENCY
        )

        print("[OK] RTDE Control")
        print()
        print("=" * 78)
        print("RENDSZER KESZ")
        print("=" * 78)
        print()
        if coord_basis is None:
            print(
                "[NINCS] Mentett kamera/kez 3D iranykalibracio."
            )
        else:
            print(
                "[OK] Kamera/kez 3D iranykalibracio betoltve."
            )

        if robot_direction_map is None:
            print(
                "[NINCS] Mentett robot operatori irany-mapping."
            )
        else:
            print(
                "[OK] Robot irany-mapping: "
                + robot_map_text(
                    robot_direction_map
                )
            )

        if robot_home is None:
            print(
                "[NINCS] Mentett demo HOME. Allitsd be a robotot, majd H."
            )
        else:
            print(
                "[OK] Demo HOME betoltve. R = visszaallas."
            )
            print(
                "HOME TCP: "
                + fmt_pose(
                    robot_home["tcp_pose"]
                )
            )

        if recovery_home is None:
            print(
                "[NINCS] Mentett RECOVERY HOME. "
                "Allitsd biztonsagos kompakt pozba, majd J."
            )
        else:
            print(
                "[OK] RECOVERY HOME betoltve. "
                "R szukseg eseten ezen keresztul ter vissza."
            )

        print()
        print(
            "K = kamera/kez iranykalibracio"
        )
        print(
            "M = robot +X/+Y/+Z operatori iranyainak megtanitasa"
        )
        print("H = AKTUALIS robotallapot -> NORMAL HOME mentese")
        print("J = AKTUALIS robotallapot -> RECOVERY HOME mentese")
        print("R = HOME: kozvetlenul vagy RECOVERY HOME-on keresztul")
        print("C = kez XYZ referencia felvetele HOME kozeleben")
        print("P = biztonsagos XYZ + IMU tracking BE/KI")
        print(
            "DEMO V15: HOME-orientacio FIX, XYZ kovetes gyors, "
            "tracking kozbeni IK guard NINCS."
        )
        print("Z = kacsacsor POT teljesen ZART allasanak rogzitese")
        print("O = kacsacsor POT teljesen NYITOTT allasanak rogzitese")
        print("G = RG2 POT-aranyos kovetes BE/KI")
        print("Q = teljes kilepes")
        print()
        print("TDK PICK-AND-PLACE MERES:")
        print("1 = ASTRA_ONLY mod (csak tracking OFF mellett)")
        print("2 = DUAL_TOP_FALLBACK mod (csak tracking OFF mellett)")
        print("T = uj meresi futam START")
        print("S = futam vege: SUCCESS")
        print("F = futam vege: FAIL")
        print("X = futam megszakitasa: ABORT")
        print("V = aktualis top-fallback gate diagnosztika")
        print("Y = REACQUIRE POLICY: STRICT / FLOW (tracking OFF mellett)")
        print("FLOW = max 3 s HOLD, auto visszatalalas, NINCS rebase/C, 0.8 s soft resume")
        print("SHADOW = Astra RGB 2D + top 2D parositas/log; robotot nem vezerli")
        print(
            "DUAL fallback gate: last/pred Z=1200..1300 mm, "
            "nearest Y-plane <=50 mm, top guide-boxon belul, max 1.5 s."
        )

        if pot_calibration_valid(
            pot_closed_raw,
            pot_open_raw
        ):
            print(
                "[OK] Potmeter kalibracio betoltve: "
                f"closed={pot_closed_raw:.1f}, "
                f"open={pot_open_raw:.1f}"
            )
        else:
            print(
                "[NINCS] Potmeter kalibracio. "
                "Kacsacsor ZART -> Z, NYITOTT -> O."
            )

        print()

        running = True

        while running:
            cycle_start = (
                rtde_c.initPeriod()
            )

            now = time.monotonic()

            # -------------------------------------------------
            # RG2 tool digital inputok a FO RTDEReceive-bol.
            # A worker V19-ben mar nem nyit masodik Receive kapcsolatot.
            # -------------------------------------------------
            if gripper_worker is not None:
                try:
                    gripper_worker.update_inputs(
                        rtde_r.getDigitalInState(
                            TOOL_DI0_GLOBAL_ID
                        ),
                        rtde_r.getDigitalInState(
                            TOOL_DI1_GLOBAL_ID
                        )
                    )
                except Exception:
                    # A fo RTDE hibat mashol ugyis kezeljuk. Itt ne
                    # spam-eljuk 250 Hz-cel a terminalt.
                    pass

            # -------------------------------------------------
            # ESP32 quaternion IMU - non-blocking serial
            # -------------------------------------------------
            if imu_lines is not None:
                for line in imu_lines.drain():
                    if line.startswith("#"):
                        print(line)
                        continue

                    imu_packet = parse_imup_line(
                        line
                    )

                    if imu_packet is not None:
                        latest_imu = imu_packet
                        last_imu_packet_time = now

                        latest_pot_raw = float(
                            imu_packet["pot_raw"]
                        )

                        last_pot_packet_time = now

                        if filtered_pot_raw is None:
                            filtered_pot_raw = (
                                latest_pot_raw
                            )
                        else:
                            filtered_pot_raw = (
                                POT_FILTER_ALPHA
                                * latest_pot_raw
                                + (
                                    1.0
                                    - POT_FILTER_ALPHA
                                )
                                * filtered_pot_raw
                            )

                        latest_imu_R = quaternion_to_matrix(
                            imu_packet["qw"],
                            imu_packet["qx"],
                            imu_packet["qy"],
                            imu_packet["qz"]
                        )

                        desired_imu_robot_local_rv = (
                            imu_relative_to_robot_local_rotvec(
                                latest_imu_R
                            )
                        )

            imu_fresh = (
                latest_imu is not None
                and last_imu_packet_time is not None
                and (
                    now - last_imu_packet_time
                    <= IMU_STALE_TIMEOUT_S
                )
            )

            pot_fresh = (
                filtered_pot_raw is not None
                and last_pot_packet_time is not None
                and (
                    now - last_pot_packet_time
                    <= IMU_STALE_TIMEOUT_S
                )
            )

            if (
                now - child_check
                >= 0.25
            ):
                child_check = now

                if (
                    rgb_proc.poll()
                    is not None
                ):
                    raise RuntimeError(
                        "RGB process leallt: "
                        f"exit={rgb_proc.returncode}"
                    )

                if (
                    depth_proc.poll()
                    is not None
                ):
                    raise RuntimeError(
                        "Depth process leallt: "
                        f"exit={depth_proc.returncode}"
                    )

                top_dead = (
                    top_proc is None
                    or top_proc.poll() is not None
                )

                if top_dead:
                    if top_camera_available:
                        print()
                        print(
                            "[FIGYELEM] Felso kamera process leallt. "
                            "Automatikus ujrainditas kovetkezik."
                        )

                    top_camera_available = False
                    top_assist_active = False

                    if (
                        now - last_top_restart_attempt
                        >= TOP_RESTART_COOLDOWN_S
                    ):
                        last_top_restart_attempt = now

                        try:
                            top_proc = start_new_console(
                                "Eternico top camera RESTART",
                                TOP_CAMERA_SCRIPT
                            )

                            top_camera_available = True

                            print(
                                "[OK] Felso kamera process ujrainditva. "
                                "Az ASSIST csak friss ervenyes UDP utan aktiv."
                            )

                        except Exception as e:
                            top_proc = None

                            print(
                                "[TOP CAM RESTART HIBA] "
                                + type(e).__name__
                                + ": "
                                + str(e)
                            )

            top_packet = drain_top_udp(
                top_sock
            )

            if top_packet is not None:
                (
                    top_seq,
                    top_x,
                    top_y,
                    top_width,
                    top_valid
                ) = top_packet

                latest_top_sequence = (
                    top_seq
                )

                last_top_packet_time = (
                    now
                )
                latest_top_packet_valid_raw = bool(top_valid)

                if top_valid:
                    latest_top_palm = (
                        float(top_x),
                        float(top_y),
                        float(top_width)
                    )

                    last_top_valid_time = (
                        now
                    )

            shadow_packet = drain_astra_shadow_udp(astra_shadow_sock)
            if shadow_packet is not None:
                sh_seq, sh_u, sh_v, sh_valid = shadow_packet
                latest_astra_shadow_seq = int(sh_seq)
                latest_astra_shadow_valid = bool(sh_valid)
                last_astra_shadow_packet_time = now
                if sh_valid:
                    latest_astra_shadow_uv = (float(sh_u), float(sh_v))

            packet = drain_xyz_udp(
                xyz_sock
            )

            if packet is not None:
                (
                    sequence,
                    x_mm,
                    y_mm,
                    z_mm,
                    valid
                ) = packet

                latest_sequence = sequence
                last_packet_time = now
                latest_xyz_packet_valid = bool(valid)

                if valid:
                    candidate_hand = [
                        float(x_mm),
                        float(y_mm),
                        float(z_mm)
                    ]

                    accept_hand = True

                    if (
                        tracking
                        and last_accepted_hand is not None
                    ):
                        dx_h = (
                            candidate_hand[0]
                            - last_accepted_hand[0]
                        )
                        dy_h = (
                            candidate_hand[1]
                            - last_accepted_hand[1]
                        )
                        dz_h = (
                            candidate_hand[2]
                            - last_accepted_hand[2]
                        )

                        jump_mm = math.sqrt(
                            dx_h * dx_h
                            + dy_h * dy_h
                            + dz_h * dz_h
                        )

                        last_rejected_hand_jump_mm = (
                            jump_mm
                        )

                        if (
                            jump_mm
                            > MAX_HAND_FRAME_JUMP_MM
                            or abs(dz_h)
                            > MAX_HAND_Z_FRAME_JUMP_MM
                        ):
                            accept_hand = False

                    if accept_hand:
                        latest_hand = (
                            candidate_hand
                        )

                        last_accepted_hand = (
                            candidate_hand.copy()
                        )

                        last_valid_time = (
                            now
                        )

            feature_packet = (
                drain_feature_udp(
                    feature_sock
                )
            )

            if feature_packet is not None:
                latest_feature_sequence = (
                    feature_packet[0]
                )

                matrix_values = (
                    feature_packet[1:10]
                )

                pinch_value = float(
                    feature_packet[10]
                )

                feature_valid = bool(
                    feature_packet[11]
                )
                latest_feature_packet_valid_raw = bool(feature_valid)

                last_feature_packet_time = (
                    now
                )

                if feature_valid:
                    candidate_hand_R = (
                        np.asarray(
                            matrix_values,
                            dtype=np.float64
                        ).reshape(
                            3,
                            3
                        )
                    )

                    accept_orientation = True

                    # Csak akkor hasonlitjuk az elozo mintahoz, ha az
                    # tenyleg friss. Kezvesztes utani ujrafelvetelnel
                    # engedjuk az uj orientaciot.
                    if (
                        last_accepted_hand_R is not None
                        and last_feature_valid_time is not None
                        and (
                            now
                            - last_feature_valid_time
                            <= 0.30
                        )
                    ):
                        jump_deg = (
                            rotation_distance_deg(
                                last_accepted_hand_R,
                                candidate_hand_R
                            )
                        )

                        if (
                            jump_deg
                            > MAX_HAND_ORIENTATION_JUMP_DEG
                        ):
                            accept_orientation = False
                            last_rejected_orientation_jump_deg = (
                                jump_deg
                            )

                    if accept_orientation:
                        latest_hand_R = (
                            candidate_hand_R
                        )

                        last_accepted_hand_R = (
                            candidate_hand_R.copy()
                        )

                        latest_pinch_ratio = (
                            pinch_value
                        )

                        last_feature_valid_time = (
                            now
                        )

            packet_fresh = (
                last_packet_time
                is not None
                and (
                    now - last_packet_time
                    <= PACKET_STALE_TIMEOUT_S
                )
            )

            feature_packet_fresh = (
                last_feature_packet_time
                is not None
                and (
                    now
                    - last_feature_packet_time
                    <= PACKET_STALE_TIMEOUT_S
                )
            )

            astra_rgb_hand_valid = (
                feature_packet_fresh
                and latest_feature_packet_valid_raw
            )

            feature_age = (
                999.0
                if last_feature_valid_time is None
                else (
                    now
                    - last_feature_valid_time
                )
            )

            feature_fresh = (
                feature_packet_fresh
                and feature_age <= 0.20
                and latest_hand_R is not None
            )

            if (
                feature_fresh
                and latest_hand_R is not None
            ):
                # hand_rotation_matrix harmadik oszlopa a tenyernormal.
                # Z-komponens abszolut erteke mutatja, mennyire nez
                # a kez sikja a kamera fele.
                palm_camera_alignment = abs(
                    float(
                        latest_hand_R[2, 2]
                    )
                )
            else:
                palm_camera_alignment = 0.0

            demo_hand_pose_good = (
                feature_fresh
                and palm_camera_alignment
                >= MIN_PALM_CAMERA_ALIGNMENT
            )

            hand_age = (
                999.0
                if last_valid_time is None
                else now - last_valid_time
            )

            top_hand_age = (
                999.0
                if last_top_valid_time is None
                else now - last_top_valid_time
            )

            top_packet_age = (
                999.0
                if last_top_packet_time is None
                else now - last_top_packet_time
            )

            top_hand_fresh = (
                top_camera_available
                and top_hand_age
                <= TOP_HAND_FRESH_TIMEOUT_S
            )

            # -------------------------------------------------
            # TDK A/B meresi mod es futamvezerles
            # -------------------------------------------------
            if keys.pressed("1"):
                if tracking:
                    print()
                    print("1 -> modvaltas csak TRACKING OFF mellett.")
                elif measurement_logger.active_trial is not None:
                    print()
                    print("1 -> aktiv futam kozben nincs modvaltas.")
                else:
                    control_mode = "ASTRA_ONLY"
                    measurement_logger.event(
                        now - session_t0,
                        control_mode,
                        "MODE_SELECT",
                        "ASTRA_ONLY"
                    )
                    print()
                    print("[MODE] ASTRA_ONLY")

            if keys.pressed("2"):
                if tracking:
                    print()
                    print("2 -> modvaltas csak TRACKING OFF mellett.")
                elif measurement_logger.active_trial is not None:
                    print()
                    print("2 -> aktiv futam kozben nincs modvaltas.")
                elif not top_camera_available:
                    print()
                    print("2 -> top kamera process nem elerheto.")
                else:
                    control_mode = "DUAL_TOP_FALLBACK"
                    measurement_logger.event(
                        now - session_t0,
                        control_mode,
                        "MODE_SELECT",
                        "DUAL_TOP_FALLBACK"
                    )
                    print()
                    print("[MODE] DUAL_TOP_FALLBACK")

            if keys.pressed("Y"):
                if tracking or measurement_logger.active_trial is not None:
                    print()
                    print("Y -> policy-valtas csak TRACKING OFF es futamon kivul.")
                else:
                    reacquire_policy = (
                        "STRICT" if reacquire_policy == "FLOW" else "FLOW"
                    )
                    measurement_logger.event(
                        now - session_t0, control_mode, "REACQUIRE_POLICY", reacquire_policy
                    )
                    print()
                    print(f"[REACQUIRE] {reacquire_policy}")

            if keys.pressed("V"):
                print()
                print("-" * 72)
                print("TOP FALLBACK DIAG")
                print(f"mode={control_mode}")
                print(
                    "Astra XYZ [mm]: "
                    + (
                        "nincs"
                        if latest_hand is None
                        else (
                            f"{latest_hand[0]:+.1f}, "
                            f"{latest_hand[1]:+.1f}, "
                            f"{latest_hand[2]:+.1f}"
                        )
                    )
                )
                print(
                    "Top palm: "
                    + (
                        "nincs/frissseg elveszett"
                        if not top_hand_fresh or latest_top_palm is None
                        else (
                            f"u={latest_top_palm[0]:.3f}, "
                            f"v={latest_top_palm[1]:.3f}, "
                            f"w={latest_top_palm[2]:.3f}"
                        )
                    )
                )
                print(
                    "Pred XZ [mm]: "
                    + (
                        "nincs"
                        if top_metric_info.get("x_mm") is None
                        else (
                            f"{float(top_metric_info['x_mm']):+.1f}, "
                            f"{float(top_metric_info['z_mm']):+.1f}"
                        )
                    )
                )
                print(
                    f"nearest_plane={top_metric_info.get('nearest_plane', '')} | "
                    f"Ydist={top_metric_info.get('nearest_y_dist_mm', '')} mm | "
                    f"gate={top_metric_info.get('valid', False)} | "
                    f"reason={top_metric_info.get('reason', '')}"
                )
                print(
                    "top-vs-Astra XZ error [mm]: "
                    + str(
                        top_metric_info.get(
                            "top_vs_astra_xz_error_mm",
                            ""
                        )
                    )
                )
                print("-" * 72)

            if keys.pressed("T"):
                if measurement_logger.active_trial is not None:
                    print()
                    print("T -> mar fut meresi futam.")
                elif not tracking:
                    print()
                    print("T -> elobb kapcsold be a trackinget P-vel.")
                elif (
                    control_mode == "DUAL_TOP_FALLBACK"
                    and not top_hand_fresh
                ):
                    print()
                    print("T -> DUAL futamhoz most nem friss a top kamera keze.")
                elif not gripper_tracking:
                    print()
                    print("T -> pick-and-place futamhoz kapcsold be a POT->RG2 kovetest G-vel.")
                else:
                    measurement_logger.start_trial(
                        now - session_t0,
                        control_mode,
                        reacquire_policy
                    )
                    print()
                    print(
                        f"[TRIAL START] #{measurement_logger.trial_id} "
                        f"| {control_mode} | reacquire={reacquire_policy}"
                    )

            if keys.pressed("S"):
                if measurement_logger.finish_trial(
                    now - session_t0,
                    "SUCCESS"
                ):
                    print()
                    print("[TRIAL END] SUCCESS")

            if keys.pressed("F"):
                if measurement_logger.finish_trial(
                    now - session_t0,
                    "FAIL"
                ):
                    print()
                    print("[TRIAL END] FAIL")

            if keys.pressed("X"):
                if measurement_logger.finish_trial(
                    now - session_t0,
                    "ABORT"
                ):
                    print()
                    print("[TRIAL END] ABORT")

            if keys.pressed("Q"):
                print()
                print(
                    "Q -> teljes kilepes"
                )
                running = False

            if keys.pressed("H"):
                # HOME tanitas alatt sem robot-, sem gripper-kovetes ne fusson.
                tracking = False

                if servo_active:
                    try:
                        rtde_c.servoStop()
                    except Exception:
                        pass

                    servo_active = False

                gripper_tracking = False

                if gripper_worker is not None:
                    gripper_worker.disable()

                try:
                    home_q = (
                        rtde_r.getActualQ()
                    )

                    home_pose = (
                        rtde_r.getActualTCPPose()
                    )

                    (
                        home_safe,
                        home_reasons,
                        home_q_deg
                    ) = check_joint_soft_limits(
                        home_q
                    )

                    if not home_safe:
                        print()
                        print(
                            "H -> HOME NEM MENTVE: joint safety guard."
                        )

                        for reason in home_reasons:
                            print(
                                " - "
                                + reason
                            )

                    else:
                        robot_home = (
                            save_robot_home(
                                home_q,
                                home_pose
                            )
                        )

                        # Uj HOME utan a regi C referencia ne maradjon aktiv.
                        calibrated = False
                        hand_reference = None
                        robot_reference = None
                        desired_pose = None
                        commanded_pose = None

                        print()
                        print(
                            "H -> UJ DEMO HOME ELMENTVE"
                        )
                        print(
                            "HOME TCP: "
                            + fmt_pose(
                                home_pose
                            )
                        )
                        print(
                            "HOME q [deg]: "
                            + ", ".join(
                                f"{name}={value:+.1f}"
                                for name, value in zip(
                                    JOINT_NAMES,
                                    home_q_deg
                                )
                            )
                        )
                        print(
                            "Mostantol ennek a HOME-nak az orientacioja "
                            "lesz a fix demo-megfogoallas."
                        )

                except Exception as e:
                    print()
                    print(
                        "[HOME MENTESI HIBA] "
                        + type(e).__name__
                        + ": "
                        + str(e)
                    )

            if keys.pressed("J"):
                tracking = False

                if servo_active:
                    try:
                        rtde_c.servoStop()
                    except Exception:
                        pass

                    servo_active = False

                gripper_tracking = False

                if gripper_worker is not None:
                    gripper_worker.disable()

                try:
                    recovery_q = (
                        rtde_r.getActualQ()
                    )

                    recovery_pose = (
                        rtde_r.getActualTCPPose()
                    )

                    if (
                        recovery_q is None
                        or len(recovery_q) < 6
                    ):
                        raise RuntimeError(
                            "Nincs ervenyes 6D aktualis joint adat."
                        )

                    recovery_q_deg = [
                        math.degrees(
                            float(q)
                        )
                        for q in recovery_q[:6]
                    ]

                    # V19: J mentest NEM blokkolja Wrist2 singularitas-
                    # vagy tracking soft-limit guard. Ez egy tenyleges,
                    # kezelo altal beallitott joint-pozt ment.
                    recovery_home = (
                        save_recovery_home(
                            recovery_q,
                            recovery_pose
                        )
                    )

                    print()
                    print(
                        "J -> RECOVERY HOME ELMENTVE"
                    )
                    print(
                        "RECOVERY q [deg]: "
                        + ", ".join(
                            f"{name}={value:+.1f}"
                            for name, value in zip(
                                JOINT_NAMES,
                                recovery_q_deg
                            )
                        )
                    )
                    print(
                        "HOME/RECOVERY moveJ-nel a tracking Wrist2/"
                        "soft-limit guard nem blokkol. "
                        "Csak olyan kompakt/ures-terben levo pozt tanits, "
                        "amelyhez az ismert munkaterben biztonsagos az ut."
                    )

                except Exception as e:
                    print()
                    print(
                        "[RECOVERY HOME MENTESI HIBA] "
                        + type(e).__name__
                        + ": "
                        + str(e)
                    )

            if keys.pressed("R"):
                tracking = False

                if servo_active:
                    try:
                        rtde_c.servoStop()
                    except Exception:
                        pass

                    servo_active = False

                gripper_tracking = False

                if gripper_worker is not None:
                    gripper_worker.disable()

                if robot_home is None:
                    print()
                    print(
                        "R -> nincs mentett NORMAL HOME. "
                        "Allitsd be a robotot, majd H."
                    )

                else:
                    try:
                        actual_q = (
                            rtde_r.getActualQ()
                        )

                        (
                            direct_home_safe,
                            direct_home_reason
                        ) = home_move_is_safe(
                            actual_q,
                            robot_home
                        )

                        if direct_home_safe:
                            print()
                            print(
                                "R -> kozvetlen, lassu visszaallas NORMAL HOME-ra..."
                            )

                            ok_home = (
                                rtde_c.moveJ(
                                    robot_home["joint_q_rad"],
                                    HOME_MOVE_SPEED_RAD_S,
                                    HOME_MOVE_ACCEL_RAD_S2
                                )
                            )

                            if ok_home is False:
                                raise RuntimeError(
                                    "moveJ NORMAL HOME sikertelen."
                                )

                        else:
                            print()
                            print(
                                "R -> kozvetlen HOME ut nem megfelelo:"
                            )
                            print(
                                "   "
                                + direct_home_reason
                            )

                            if recovery_home is None:
                                raise RuntimeError(
                                    "Nincs RECOVERY HOME. "
                                    "Tanits egy biztonsagos koztes pozt J-vel."
                                )

                            (
                                recovery_safe,
                                recovery_reason
                            ) = recovery_move_is_safe(
                                actual_q,
                                recovery_home
                            )

                            if not recovery_safe:
                                raise RuntimeError(
                                    "RECOVERY HOME ut blokkolva: "
                                    + recovery_reason
                                )

                            print(
                                "R -> 1/2 lassu RECOVERY HOME..."
                            )

                            ok_recovery = (
                                rtde_c.moveJ(
                                    recovery_home["joint_q_rad"],
                                    RECOVERY_MOVE_SPEED_RAD_S,
                                    RECOVERY_MOVE_ACCEL_RAD_S2
                                )
                            )

                            if ok_recovery is False:
                                raise RuntimeError(
                                    "moveJ RECOVERY HOME sikertelen."
                                )

                            # V19: ha a kezelo altal tanitott RECOVERY
                            # HOME-ot elertuk, innen nem alkalmazunk ujabb
                            # Wrist2/soft-limit/distance tiltast.
                            print(
                                "R -> 2/2 NORMAL HOME..."
                            )

                            ok_home = (
                                rtde_c.moveJ(
                                    robot_home["joint_q_rad"],
                                    HOME_MOVE_SPEED_RAD_S,
                                    HOME_MOVE_ACCEL_RAD_S2
                                )
                            )

                            if ok_home is False:
                                raise RuntimeError(
                                    "moveJ NORMAL HOME sikertelen."
                                )

                        actual_home_pose = (
                            rtde_r.getActualTCPPose()
                        )

                        # HOME utan mindig uj C kell.
                        calibrated = False
                        hand_reference = None
                        robot_reference = None
                        desired_pose = None
                        commanded_pose = None

                        print()
                        print(
                            "R -> NORMAL HOME KESZ"
                        )
                        print(
                            "Robot: "
                            + fmt_pose(
                                actual_home_pose
                            )
                        )
                        print(
                            "Kovetkezo: kez referenciahelyzet -> C -> P"
                        )

                    except Exception as e:
                        print()
                        print(
                            "[HOME/RECOVERY MOZGASI HIBA] "
                            + type(e).__name__
                            + ": "
                            + str(e)
                        )

            if keys.pressed("M"):
                tracking = False
                auto_reacquire_wait = False

                if servo_active:
                    rtde_c.servoStop()
                    servo_active = False

                # Egy regi C referencia az uj mapping utan
                # mar ne legyen automatikusan felhasznalva.
                calibrated = False
                hand_reference = None
                robot_reference = None
                desired_pose = None
                commanded_pose = None
                approved_pose = None

                try:
                    robot_direction_map = (
                        teach_robot_directions(
                            rtde_r,
                            rtde_c
                        )
                    )

                except KeyboardInterrupt:
                    print()
                    print(
                        "M -> robot iranytanitas megszakitva."
                    )

                except Exception as e:
                    print()
                    print(
                        "[ROBOT IRANYTANITASI HIBA]"
                    )
                    print(
                        type(e).__name__ + ":",
                        e
                    )
                    print(
                        "A regi mentett mapping valtozatlan maradt."
                    )

            if keys.pressed("K"):
                if (
                    not packet_fresh
                    or hand_age > 0.20
                    or latest_hand is None
                ):
                    print()
                    print(
                        "K -> nincs friss ervenyes kezpozicio."
                    )
                else:
                    # Kalibracio kozben a robot SOHA ne kovesse a kezet.
                    tracking = False

                    if servo_active:
                        rtde_c.servoStop()
                        servo_active = False

                    # A regi C referencia innentol ervenytelen,
                    # mert a koordinata-transzformacio megvaltozhat.
                    calibrated = False
                    hand_reference = None
                    robot_reference = None
                    desired_pose = None
                    commanded_pose = None
                    approved_pose = None

                    try:
                        new_basis = (
                            coord_wizard.handle_k(
                                latest_hand
                            )
                        )

                        if new_basis is not None:
                            coord_basis = (
                                new_basis
                            )

                    except Exception as e:
                        coord_wizard.cancel()

                        print()
                        print(
                            "[KALIBRACIOS HIBA]"
                        )
                        print(
                            type(e).__name__ + ":",
                            e
                        )
                        print(
                            "A kalibracio megszakadt. "
                            "Nyomj K-t az ujrakezdeshez."
                        )

            if keys.pressed("Z"):
                if (
                    not pot_fresh
                    or filtered_pot_raw is None
                ):
                    print()
                    print(
                        "Z -> nincs friss potmeter adat."
                    )

                else:
                    pot_closed_raw = float(
                        filtered_pot_raw
                    )

                    gripper_tracking = False

                    if gripper_worker is not None:
                        gripper_worker.disable()

                    print()
                    print(
                        "Z -> KACSACSOR ZART pot ertek: "
                        f"{pot_closed_raw:.1f}"
                    )

                    if pot_calibration_valid(
                        pot_closed_raw,
                        pot_open_raw
                    ):
                        save_pot_calibration(
                            pot_closed_raw,
                            pot_open_raw
                        )

                        print(
                            "[OK] Potmeter kalibracio elmentve."
                        )

            if keys.pressed("O"):
                if (
                    not pot_fresh
                    or filtered_pot_raw is None
                ):
                    print()
                    print(
                        "O -> nincs friss potmeter adat."
                    )

                else:
                    pot_open_raw = float(
                        filtered_pot_raw
                    )

                    gripper_tracking = False

                    if gripper_worker is not None:
                        gripper_worker.disable()

                    print()
                    print(
                        "O -> KACSACSOR NYITOTT pot ertek: "
                        f"{pot_open_raw:.1f}"
                    )

                    if pot_calibration_valid(
                        pot_closed_raw,
                        pot_open_raw
                    ):
                        save_pot_calibration(
                            pot_closed_raw,
                            pot_open_raw
                        )

                        print(
                            "[OK] Potmeter kalibracio elmentve."
                        )

                    elif pot_closed_raw is not None:
                        print(
                            "[FIGYELEM] Tul kicsi a Z/O pot kulonbseg."
                        )

            if keys.pressed("G"):
                pot_cal_ok = (
                    pot_calibration_valid(
                        pot_closed_raw,
                        pot_open_raw
                    )
                )

                if not pot_cal_ok:
                    print()
                    print(
                        "G -> elobb kacsacsor ZART -> Z es NYITOTT -> O kell."
                    )

                elif not pot_fresh:
                    print()
                    print(
                        "G -> nincs friss potmeter adat."
                    )

                else:
                    if gripper_worker is None:
                        print()
                        print(
                            "RG2 RTDE-I/O worker inditasa..."
                        )

                        gripper_worker = (
                            RG2RTDEWorker(
                                ROBOT_IP
                            )
                        )

                        try:
                            gripper_worker.update_inputs(
                                rtde_r.getDigitalInState(
                                    TOOL_DI0_GLOBAL_ID
                                ),
                                rtde_r.getDigitalInState(
                                    TOOL_DI1_GLOBAL_ID
                                )
                            )
                        except Exception:
                            pass

                        time.sleep(
                            0.25
                        )

                    gripper_tracking = (
                        not gripper_tracking
                    )

                    if gripper_tracking:
                        gripper_worker.enable()
                    else:
                        gripper_worker.disable()

                    print()
                    print(
                        "G -> RG2 POT TRACKING "
                        + (
                            "ON"
                            if gripper_tracking
                            else "OFF"
                        )
                    )


            if keys.pressed("C"):
                if robot_home is None:
                    print()
                    print(
                        "C -> nincs mentett HOME. Allitsd be a robotot es nyomj H-t."
                    )

                elif not imu_fresh:
                    print()
                    print(
                        "C -> nincs friss ESP32 IMU quaternion adat."
                    )

                elif (
                    not latest_xyz_packet_valid
                    or latest_hand is None
                    or hand_age > 0.30
                ):
                    print()
                    print(
                        "C -> kell friss Astra XYZ. A tenyertartas/POSE mar nem tiltja a referenciat."
                    )

                else:
                    try:
                        actual_q_for_c = (
                            rtde_r.getActualQ()
                        )

                        near_home = (
                            robot_near_home(
                                actual_q_for_c,
                                robot_home
                            )
                        )

                    except Exception:
                        near_home = False

                    if not near_home:
                        print()
                        print(
                            "C -> robot nincs HOME kozeleben. Nyomj R-t."
                        )

                    else:
                        hand_reference = (
                            latest_hand.copy()
                        )

                        robot_actual_pose = (
                            rtde_r.getActualTCPPose()
                        )

                        robot_reference = (
                            robot_actual_pose[:]
                        )

                        # A HOME TCP-orientacio a robot orientacios referencia.
                        robot_reference[3] = float(
                            robot_home["tcp_pose"][3]
                        )
                        robot_reference[4] = float(
                            robot_home["tcp_pose"][4]
                        )
                        robot_reference[5] = float(
                            robot_home["tcp_pose"][5]
                        )

                        robot_reference_R = rotvec_to_matrix(
                            robot_reference[3:6]
                        )

                        desired_pose = (
                            robot_reference[:]
                        )

                        commanded_pose = (
                            robot_actual_pose[:]
                        )

                        commanded_pose[3] = (
                            robot_reference[3]
                        )
                        commanded_pose[4] = (
                            robot_reference[4]
                        )
                        commanded_pose[5] = (
                            robot_reference[5]
                        )

                        desired_imu_robot_local_rv[:] = 0.0
                        commanded_imu_robot_local_rv[:] = 0.0

                        # Aktualis IMU kezallas = relativ orientacios nulla.
                        if imu_ser is not None:
                            imu_ser.write(
                                b"Z\n"
                            )
                            imu_ser.flush()

                        calibrated = True
                        tracking = False

                        if servo_active:
                            rtde_c.servoStop()
                            servo_active = False

                        print()
                        print(
                            "C -> KEZ XYZ REFERENCIA KESZ"
                        )
                        print(
                            "Robot orientacio referencia = HOME, kovetes = ESP32 IMU quaternion."
                        )
                        print(
                            "Hand H0 [mm]: "
                            f"{hand_reference[0]:+.1f}, "
                            f"{hand_reference[1]:+.1f}, "
                            f"{hand_reference[2]:+.1f}"
                        )
                        print(
                            "Robot R0: "
                            + fmt_pose(
                                robot_reference
                            )
                        )
                        print(
                            "Tracking OFF. P inditja."
                        )

            if keys.pressed("P"):
                if coord_basis is None:
                    print()
                    print(
                        "P -> elobb K-val kamera/kez kalibracio kell."
                    )

                elif robot_direction_map is None:
                    print()
                    print(
                        "P -> elobb M-mel robot iranytanitas kell."
                    )

                elif not calibrated:
                    print()
                    print(
                        "P -> elobb C referencia kell."
                    )

                elif not imu_fresh:
                    print()
                    print(
                        "P -> nincs friss ESP32 IMU adat."
                    )

                elif not tracking:
                    if (
                        not latest_xyz_packet_valid
                        or latest_hand is None
                        or hand_age > 0.30
                    ):
                        print()
                        print(
                            "P -> nincs friss Astra XYZ. A POSE guard nem tiltja a tracking inditasat."
                        )
                    else:
                        tracking = True

                        commanded_pose = (
                            rtde_r.getActualTCPPose()
                        )

                        # C-kor nullazott IMU orientacio innen indul.
                        commanded_pose[3] = (
                            robot_reference[3]
                        )
                        commanded_pose[4] = (
                            robot_reference[4]
                        )
                        commanded_pose[5] = (
                            robot_reference[5]
                        )

                        commanded_imu_robot_local_rv[:] = 0.0

                        servo_active = False
                        resume_ramp_start_time = now
                        resume_ramp_reason = "MANUAL_P"
                        resume_ramp_factor = REACQUIRE_RAMP_START_FACTOR

                        print()
                        print(
                            "P -> XYZ + IMU ORIENTACIO TRACKING ON | SOFT RESUME"
                        )

                else:
                    tracking = False
                    auto_reacquire_wait = False

                    if servo_active:
                        rtde_c.servoStop()
                        servo_active = False

                    print()
                    print(
                        "P -> TRACKING OFF"
                    )

            # -------------------------------------------------
            # POZICIOFORRAS: ASTRA / TOP_FALLBACK / HOLD / PAUSE
            # -------------------------------------------------
            # Robust depth verzio: a MediaPipe POSE guard diagnosztika, nem pozicio-tiltas.
            astra_control_valid = (
                latest_xyz_packet_valid
                and hand_age <= 0.20
                and latest_hand is not None
            )

            if not feature_packet_fresh:
                astra_loss_reason = "RGB_FEATURE_PACKET_STALE"
            elif not astra_rgb_hand_valid:
                astra_loss_reason = "RGB_HAND_LOST"
            elif not packet_fresh:
                astra_loss_reason = "XYZ_PACKET_STALE"
            elif not latest_xyz_packet_valid:
                astra_loss_reason = "XYZ_MISSING"
            elif latest_hand is None:
                astra_loss_reason = "XYZ_NO_LAST_HAND"
            elif hand_age > 0.20:
                astra_loss_reason = "XYZ_VALID_TOO_OLD"
            else:
                astra_loss_reason = "OK"

            control_hand = None
            control_source = "IDLE" if not tracking else "HOLD"

            # A top predikciot akkor is kiszamoljuk, amikor Astra valid,
            # mert igy a log kesobb plusz cross-validation adatot ad.
            if (
                top_hand_fresh
                and latest_top_palm is not None
                and latest_hand is not None
            ):
                top_metric_info = evaluate_top_metric_fallback(
                    top_metric_calibration,
                    latest_top_palm,
                    latest_hand[1],
                    latest_hand,
                    (
                        last_top_fallback_hand
                        if top_assist_active
                        else None
                    )
                )
            else:
                top_metric_info = {
                    "valid": False,
                    "reason": "TOP_NOT_FRESH",
                    "x_mm": None,
                    "y_mm": None,
                    "z_mm": None,
                    "nearest_plane": "",
                    "nearest_y_dist_mm": None,
                    "alpha": None,
                    "top_vs_astra_xz_error_mm": None,
                }

            if tracking:
                if astra_control_valid:
                    control_source = "ASTRA"
                    auto_reacquire_wait = False
                    control_hand = latest_hand.copy()
                    top_assist_active = False
                    top_fallback_start_time = None
                    last_top_fallback_hand = None

                else:
                    can_metric_fallback = False

                    if (
                        control_mode == "DUAL_TOP_FALLBACK"
                        and top_hand_fresh
                        and top_metric_info.get("valid", False)
                    ):
                        if top_fallback_start_time is None:
                            fallback_elapsed = 0.0
                        else:
                            fallback_elapsed = (
                                now - top_fallback_start_time
                            )

                        can_metric_fallback = (
                            fallback_elapsed
                            <= TOP_METRIC_FALLBACK_MAX_S
                        )

                    if can_metric_fallback:
                        if top_fallback_start_time is None:
                            top_fallback_start_time = now

                        if (
                            now - top_fallback_start_time
                            <= TOP_METRIC_FALLBACK_MAX_S
                        ):
                            control_hand = [
                                float(top_metric_info["x_mm"]),
                                float(top_metric_info["y_mm"]),
                                float(top_metric_info["z_mm"]),
                            ]
                            last_top_fallback_hand = control_hand.copy()
                            control_source = "TOP_FALLBACK"
                            auto_reacquire_wait = False
                            top_assist_active = True
                        else:
                            control_source = "HOLD"

                    else:
                        top_assist_active = False

                        # STRICT = regi 0.75 s. FLOW = max 3 s automatikus HOLD.
                        # A robot HOLD alatt nem extrapolal es nem mozog vakon.
                        reacquire_window_s = (
                            FLOW_REACQUIRE_WINDOW_S
                            if reacquire_policy == "FLOW"
                            else HAND_LOSS_GRACE_S
                        )
                        if hand_age <= reacquire_window_s:
                            control_source = "HOLD"
                            auto_reacquire_wait = True
                        else:
                            tracking = False
                            auto_reacquire_wait = False
                            control_source = "PAUSE_ASTRA_LOST"

                            if servo_active:
                                rtde_c.servoStop()
                                servo_active = False

                            detail = (
                                "ASTRA lost; "
                                + (
                                    "top disabled by mode"
                                    if control_mode == "ASTRA_ONLY"
                                    else "top gate="
                                    + str(top_metric_info.get("reason", "?"))
                                )
                            )

                            print()
                            print(
                                "AUTO PAUSE -> " + detail
                            )

            if control_source != previous_control_source:
                measurement_logger.event(
                    now - session_t0,
                    control_mode,
                    "CONTROL_SOURCE",
                    control_source
                    + (
                        " | top_gate="
                        + str(top_metric_info.get("reason", ""))
                        if control_source != "ASTRA"
                        else ""
                    )
                )

                if control_source == "TOP_FALLBACK":
                    print()
                    print(
                        "TOP METRIC FALLBACK -> XZ top kamerabol, "
                        "Y utolso Astra erteken, IMU orientacio aktiv."
                    )
                elif (
                    previous_control_source == "TOP_FALLBACK"
                    and control_source == "ASTRA"
                ):
                    print()
                    print(
                        "TOP METRIC FALLBACK -> Astra visszatert."
                    )

                if (
                    previous_control_source == "HOLD"
                    and control_source in ("ASTRA", "TOP_FALLBACK")
                ):
                    # NINCS automatikus C/rebase. Ugyanaz az eredeti kez/robot
                    # referencia marad. csak a visszaindulas kap lagy sebessegrampot.
                    resume_ramp_start_time = now
                    resume_ramp_reason = "AUTO_REACQUIRE"
                    resume_ramp_factor = REACQUIRE_RAMP_START_FACTOR
                    auto_reacquire_wait = False
                    measurement_logger.event(
                        now - session_t0, control_mode, "AUTO_REACQUIRE",
                        f"source={control_source}; policy={reacquire_policy}; no_rebase=1"
                    )
                    print()
                    print("AUTO REACQUIRE -> eredeti C referencia marad | SOFT RESUME")

                previous_control_source = control_source

            if (
                tracking
                and not imu_fresh
            ):
                tracking = False

                if servo_active:
                    rtde_c.servoStop()
                    servo_active = False

                control_source = "PAUSE_IMU_LOST"
                print()
                print(
                    "AUTO PAUSE -> ESP32 IMU adat elveszett."
                )

            if (
                gripper_tracking
                and not pot_fresh
            ):
                gripper_tracking = False

                if gripper_worker is not None:
                    gripper_worker.disable()

                print()
                print(
                    "AUTO PAUSE -> RG2 potmeter adat elveszett."
                )

            limits = (
                False,
                False,
                False
            )

            if (
                coord_basis is not None
                and robot_direction_map is not None
                and calibrated
                and tracking
                and control_hand is not None
            ):
                dx_mm = (
                    control_hand[0]
                    - hand_reference[0]
                )
                dy_mm = (
                    control_hand[1]
                    - hand_reference[1]
                )
                dz_mm = (
                    control_hand[2]
                    - hand_reference[2]
                )

                (
                    offset,
                    limits,
                    operator_components_mm
                ) = camera_delta_to_robot_offset(
                    dx_mm,
                    dy_mm,
                    dz_mm,
                    coord_basis,
                    robot_direction_map
                )

                new_desired = (
                    robot_reference[:]
                )

                new_desired[0] += offset[0]
                new_desired[1] += offset[1]
                new_desired[2] += offset[2]

                # Az XYZ-t a kamera adja. Az orientacio lent,
                # minden servo ciklusban az ESP32 quaternionbol frissul.
                new_desired[3] = robot_reference[3]
                new_desired[4] = robot_reference[4]
                new_desired[5] = robot_reference[5]

                if desired_pose is None:
                    desired_pose = (
                        new_desired
                    )

                else:
                    if (
                        vector_distance_xyz(
                            new_desired,
                            desired_pose
                        )
                        >= TARGET_CHANGE_DEADBAND_M
                    ):
                        desired_pose[0] = (
                            new_desired[0]
                        )
                        desired_pose[1] = (
                            new_desired[1]
                        )
                        desired_pose[2] = (
                            new_desired[2]
                        )

                    desired_pose[3] = robot_reference[3]
                    desired_pose[4] = robot_reference[4]
                    desired_pose[5] = robot_reference[5]

            if (
                calibrated
                and tracking
                and desired_pose is not None
            ):
                # ------------------------------------------------
                # V15: direkt, V8-szeru XYZ servo kovetes.
                # Nincs tracking kozbeni IK szamitas.
                # ------------------------------------------------
                resume_ramp_factor = 1.0
                if resume_ramp_start_time is not None:
                    ramp_t = max(0.0, now - resume_ramp_start_time)
                    if ramp_t < REACQUIRE_RESUME_RAMP_S:
                        alpha = ramp_t / REACQUIRE_RESUME_RAMP_S
                        resume_ramp_factor = (
                            REACQUIRE_RAMP_START_FACTOR
                            + (1.0 - REACQUIRE_RAMP_START_FACTOR) * alpha
                        )
                    else:
                        resume_ramp_start_time = None
                        resume_ramp_reason = ""
                        resume_ramp_factor = 1.0

                rate_limit_xyz(
                    commanded_pose,
                    desired_pose,
                    MAX_TARGET_SPEED_M_S
                    * resume_ramp_factor
                    * DT
                )

                # ------------------------------------------------
                # IMU quaternion orientacio - robot LOCAL tengelyeken.
                # A szenzor X altal vezerelt robot local Z max 20 deg/s,
                # a masik ket local tengely 10 deg/s.
                # ------------------------------------------------
                if (
                    imu_fresh
                    and robot_reference_R is not None
                ):
                    commanded_imu_robot_local_rv = (
                        rate_limit_robot_local_orientation(
                            commanded_imu_robot_local_rv,
                            desired_imu_robot_local_rv,
                            DT
                        )
                    )

                relative_robot_R = rotvec_to_matrix(
                    commanded_imu_robot_local_rv
                )

                target_robot_R = (
                    robot_reference_R
                    @ relative_robot_R
                )

                target_robot_rotvec = matrix_to_rotvec(
                    target_robot_R
                )

                commanded_pose[3] = float(
                    target_robot_rotvec[0]
                )
                commanded_pose[4] = float(
                    target_robot_rotvec[1]
                )
                commanded_pose[5] = float(
                    target_robot_rotvec[2]
                )

                rtde_c.servoL(
                    commanded_pose,
                    VELOCITY,
                    ACCELERATION,
                    DT,
                    LOOKAHEAD_TIME,
                    GAIN
                )

                servo_active = True

            pot_cal_ok = (
                pot_calibration_valid(
                    pot_closed_raw,
                    pot_open_raw
                )
            )

            latest_pot_gripper_width = None

            if (
                pot_cal_ok
                and pot_fresh
                and filtered_pot_raw is not None
            ):
                try:
                    latest_pot_gripper_width = (
                        pot_raw_to_gripper_width(
                            filtered_pot_raw,
                            pot_closed_raw,
                            pot_open_raw
                        )
                    )

                    if (
                        gripper_tracking
                        and gripper_worker is not None
                    ):
                        gripper_worker.set_target(
                            latest_pot_gripper_width
                        )

                except Exception as e:
                    print()
                    print(
                        "[POT -> RG2 HIBA] "
                        + type(e).__name__
                        + ": "
                        + str(e)
                    )

            loss_elapsed_s = (
                0.0 if astra_control_valid else max(0.0, float(hand_age))
            )
            send_pickplace_status(
                status_sock,
                tracking,
                gripper_tracking,
                control_mode,
                control_source,
                reacquire_policy,
                measurement_logger.active_trial is not None,
                bool(top_metric_info.get("valid", False)),
                loss_elapsed_s,
            )

            # -------------------------------------------------
            # 50 Hz reszletes meresi logger - TDK + kesobbi szakdolgozat
            # -------------------------------------------------
            if (
                measurement_logger.active_trial is not None
                and now - last_measurement_log_time
                >= MEASUREMENT_LOG_PERIOD_S
            ):
                last_measurement_log_time = now

                measurement_logger.update_trial(
                    now - session_t0,
                    astra_control_valid,
                    control_source
                )

                try:
                    actual_tcp = rtde_r.getActualTCPPose()
                except Exception:
                    actual_tcp = None

                desired_tcp = pose_or_blank(desired_pose)
                commanded_tcp = pose_or_blank(commanded_pose)
                actual_tcp_vals = pose_or_blank(actual_tcp)

                tcp_xyz_error_mm = ""
                if (
                    commanded_pose is not None
                    and actual_tcp is not None
                ):
                    try:
                        tcp_xyz_error_mm = 1000.0 * math.sqrt(
                            (float(commanded_pose[0]) - float(actual_tcp[0])) ** 2
                            + (float(commanded_pose[1]) - float(actual_tcp[1])) ** 2
                            + (float(commanded_pose[2]) - float(actual_tcp[2])) ** 2
                        )
                    except Exception:
                        tcp_xyz_error_mm = ""

                if gripper_worker is not None:
                    (
                        rg2_sent_log,
                        rg2_grip_log,
                        rg2_error_log
                    ) = gripper_worker.get_status()
                    try:
                        rg2_di0_log = int(gripper_worker._tool_input(0))
                        rg2_di1_log = int(gripper_worker._tool_input(1))
                    except Exception:
                        rg2_di0_log = ""
                        rg2_di1_log = ""
                else:
                    rg2_sent_log = None
                    rg2_grip_log = False
                    rg2_error_log = ""
                    rg2_di0_log = ""
                    rg2_di1_log = ""

                imu = latest_imu or {}
                trial_id_log = (
                    ""
                    if measurement_logger.active_trial is None
                    else measurement_logger.active_trial["trial_id"]
                )

                row = {
                    "pc_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "mono_s": f"{now - session_t0:.6f}",
                    "trial_id": trial_id_log,
                    "trial_active": int(measurement_logger.active_trial is not None),
                    "mode": control_mode,
                    "control_source": control_source,
                    "tracking": int(tracking),
                    "calibrated": int(calibrated),
                    "servo_active": int(servo_active),
                    "astra_seq": "" if latest_sequence is None else int(latest_sequence),
                    "astra_packet_fresh": int(packet_fresh),
                    "astra_control_valid": int(astra_control_valid),
                    "latest_xyz_packet_valid": int(bool(latest_xyz_packet_valid)),
                    "astra_rgb_hand_valid": int(bool(astra_rgb_hand_valid)),
                    "astra_loss_reason": astra_loss_reason,
                    "pose_guard_good": int(demo_hand_pose_good),
                    "palm_camera_alignment": f"{palm_camera_alignment:.6f}",
                    "astra_hand_age_s": f"{hand_age:.6f}",
                    "astra_x_mm": "" if latest_hand is None else float(latest_hand[0]),
                    "astra_y_mm": "" if latest_hand is None else float(latest_hand[1]),
                    "astra_z_mm": "" if latest_hand is None else float(latest_hand[2]),
                    "top_seq": "" if latest_top_sequence is None else int(latest_top_sequence),
                    "top_packet_valid_raw": int(bool(latest_top_packet_valid_raw)),
                    "top_packet_age_s": f"{top_packet_age:.6f}",
                    "top_fresh": int(top_hand_fresh),
                    "top_hand_age_s": f"{top_hand_age:.6f}",
                    "top_u": "" if latest_top_palm is None else float(latest_top_palm[0]),
                    "top_v": "" if latest_top_palm is None else float(latest_top_palm[1]),
                    "top_width": "" if latest_top_palm is None else float(latest_top_palm[2]),
                    "top_pred_x_mm": "" if top_metric_info.get("x_mm") is None else float(top_metric_info["x_mm"]),
                    "top_pred_y_mm": "" if top_metric_info.get("y_mm") is None else float(top_metric_info["y_mm"]),
                    "top_pred_z_mm": "" if top_metric_info.get("z_mm") is None else float(top_metric_info["z_mm"]),
                    "top_gate_valid": int(bool(top_metric_info.get("valid", False))),
                    "top_gate_reason": str(top_metric_info.get("reason", "")),
                    "top_nearest_plane": str(top_metric_info.get("nearest_plane", "")),
                    "top_nearest_plane_y_dist_mm": "" if top_metric_info.get("nearest_y_dist_mm") is None else float(top_metric_info["nearest_y_dist_mm"]),
                    "top_interp_alpha": "" if top_metric_info.get("alpha") is None else float(top_metric_info["alpha"]),
                    "top_vs_astra_xz_error_mm": "" if top_metric_info.get("top_vs_astra_xz_error_mm") is None else float(top_metric_info["top_vs_astra_xz_error_mm"]),
                    "imu_fresh": int(imu_fresh),
                    "imu_qw": imu.get("qw", ""),
                    "imu_qx": imu.get("qx", ""),
                    "imu_qy": imu.get("qy", ""),
                    "imu_qz": imu.get("qz", ""),
                    "imu_roll_deg": imu.get("roll", ""),
                    "imu_pitch_deg": imu.get("pitch", ""),
                    "imu_yaw_deg": imu.get("yaw", ""),
                    "imu_des_rvx": float(desired_imu_robot_local_rv[0]),
                    "imu_des_rvy": float(desired_imu_robot_local_rv[1]),
                    "imu_des_rvz": float(desired_imu_robot_local_rv[2]),
                    "imu_cmd_rvx": float(commanded_imu_robot_local_rv[0]),
                    "imu_cmd_rvy": float(commanded_imu_robot_local_rv[1]),
                    "imu_cmd_rvz": float(commanded_imu_robot_local_rv[2]),
                    "pot_fresh": int(pot_fresh),
                    "pot_raw": "" if latest_pot_raw is None else float(latest_pot_raw),
                    "pot_filtered": "" if filtered_pot_raw is None else float(filtered_pot_raw),
                    "rg2_tracking": int(gripper_tracking),
                    "rg2_mapped_width_mm": "" if latest_pot_gripper_width is None else float(latest_pot_gripper_width),
                    "rg2_last_sent_target_mm": "" if rg2_sent_log is None else float(rg2_sent_log),
                    "rg2_grip_detected": int(bool(rg2_grip_log)),
                    "rg2_ready_di1": rg2_di1_log,
                    "rg2_grip_di0": rg2_di0_log,
                    "rg2_error": rg2_error_log,
                    "desired_tcp_x": desired_tcp[0],
                    "desired_tcp_y": desired_tcp[1],
                    "desired_tcp_z": desired_tcp[2],
                    "desired_tcp_rx": desired_tcp[3],
                    "desired_tcp_ry": desired_tcp[4],
                    "desired_tcp_rz": desired_tcp[5],
                    "commanded_tcp_x": commanded_tcp[0],
                    "commanded_tcp_y": commanded_tcp[1],
                    "commanded_tcp_z": commanded_tcp[2],
                    "commanded_tcp_rx": commanded_tcp[3],
                    "commanded_tcp_ry": commanded_tcp[4],
                    "commanded_tcp_rz": commanded_tcp[5],
                    "actual_tcp_x": actual_tcp_vals[0],
                    "actual_tcp_y": actual_tcp_vals[1],
                    "actual_tcp_z": actual_tcp_vals[2],
                    "actual_tcp_rx": actual_tcp_vals[3],
                    "actual_tcp_ry": actual_tcp_vals[4],
                    "actual_tcp_rz": actual_tcp_vals[5],
                    "tcp_xyz_error_mm": tcp_xyz_error_mm,
                    "reacquire_policy": reacquire_policy,
                    "reacquire_window_s": (FLOW_REACQUIRE_WINDOW_S if reacquire_policy == "FLOW" else HAND_LOSS_GRACE_S),
                    "auto_reacquire_wait": int(bool(auto_reacquire_wait)),
                    "resume_ramp_active": int(resume_ramp_start_time is not None),
                    "resume_ramp_factor": f"{float(resume_ramp_factor):.6f}",
                    "loss_elapsed_s": f"{float(loss_elapsed_s):.6f}",
                    "shadow_astra_seq": "" if latest_astra_shadow_seq is None else int(latest_astra_shadow_seq),
                    "shadow_astra_valid": int(bool(latest_astra_shadow_valid)),
                    "shadow_astra_u": "" if latest_astra_shadow_uv is None else float(latest_astra_shadow_uv[0]),
                    "shadow_astra_v": "" if latest_astra_shadow_uv is None else float(latest_astra_shadow_uv[1]),
                    "shadow_astra_age_s": (
                        "" if last_astra_shadow_packet_time is None
                        else f"{max(0.0, now-last_astra_shadow_packet_time):.6f}"
                    ),
                    "shadow_pair_skew_ms": (
                        "" if last_astra_shadow_packet_time is None or last_top_packet_time is None
                        else f"{1000.0*abs(last_astra_shadow_packet_time-last_top_packet_time):.3f}"
                    ),
                    "shadow_pair_ready": int(
                        latest_astra_shadow_valid
                        and top_hand_fresh
                        and last_astra_shadow_packet_time is not None
                        and last_top_packet_time is not None
                        and abs(last_astra_shadow_packet_time-last_top_packet_time) <= SHADOW_PAIR_MAX_SKEW_S
                    ),
                    "shadow_stereo_status": (
                        "PAIR_2D_READY_CALIBRATION_REQUIRED"
                        if (latest_astra_shadow_valid and top_hand_fresh)
                        else "PAIR_NOT_READY"
                    ),
                }

                measurement_logger.write_sample(row)

            if (
                TERMINAL_LIVE_STATUS
                and now - last_print
                >= 0.25
            ):
                last_print = now

                hand_state = (
                    "OK"
                    if (
                        packet_fresh
                        and hand_age <= 0.12
                    )
                    else (
                        "HOLD"
                        if (
                            packet_fresh
                            and hand_age
                            <= HAND_LOSS_GRACE_S
                        )
                        else "LOST"
                    )
                )

                limit_text = (
                    "".join(
                        axis
                        for axis, active
                        in zip(
                            "XYZ",
                            limits
                        )
                        if active
                    )
                    or "-"
                )

                if latest_hand is None:
                    xyz_text = "nincs"
                else:
                    xyz_text = (
                        f"{latest_hand[0]:+.0f},"
                        f"{latest_hand[1]:+.0f},"
                        f"{latest_hand[2]:+.0f}"
                    )

                (
                    op_right,
                    op_up,
                    op_forward
                ) = operator_components_mm

                if (
                    hand_orientation_reference_R is not None
                    and latest_hand_R is not None
                ):
                    (
                        ori_roll,
                        ori_pitch,
                        ori_yaw
                    ) = matrix_to_euler_zyx(
                        hand_orientation_reference_R.T
                        @ latest_hand_R
                    )
                else:
                    (
                        ori_roll,
                        ori_pitch,
                        ori_yaw
                    ) = (
                        0.0,
                        0.0,
                        0.0
                    )

                if gripper_worker is not None:
                    (
                        rg2_sent,
                        rg2_grip,
                        rg2_error
                    ) = gripper_worker.get_status()
                else:
                    (
                        rg2_sent,
                        rg2_grip,
                        rg2_error
                    ) = (
                        None,
                        False,
                        ""
                    )

                grip_target_text = (
                    "--"
                    if latest_pot_gripper_width is None
                    else f"{latest_pot_gripper_width:.0f}"
                )

                pot_raw_text = (
                    "--"
                    if filtered_pot_raw is None
                    else f"{filtered_pot_raw:.0f}"
                )

                top_text = (
                    "OK"
                    if top_hand_fresh
                    else "LOST"
                )

                print(
                    "\r"
                    f"seq={str(latest_sequence):>8} | "
                    f"hand={hand_state:4s} | "
                    f"cmap={'Y' if coord_basis is not None else 'N'} | "
                    f"rmap={'Y' if robot_direction_map is not None else 'N'} | "
                    f"ref={'Y' if calibrated else 'N'} | "
                    f"track={'ON ' if tracking else 'OFF'} | "
                    f"mode={control_mode} | src={control_source} | reacq={reacquire_policy} | ramp={resume_ramp_factor:.2f} | "
                    f"IMU={'OK' if imu_fresh else 'LOST'} | "
                    f"LIMIT={limit_text:3s} | "
                    f"R/U/F={op_right:+.0f}/"
                    f"{op_up:+.0f}/"
                    f"{op_forward:+.0f}mm | "
                    f"POSE={'OK' if demo_hand_pose_good else 'HOLD'}:"
                    f"{palm_camera_alignment:.2f} | "
                    f"HOME={'Y' if robot_home is not None else 'N'} | "
                    f"REC={'Y' if recovery_home is not None else 'N'} | "
                    f"TOP={top_text} | "
                    f"jump={last_rejected_hand_jump_mm:.0f}mm | "
                    f"POT={pot_raw_text} | "
                    f"RG2={'ON' if gripper_tracking else 'OFF'}:"
                    f"{grip_target_text}mm | "
                    f"ojump={last_rejected_orientation_jump_deg:.0f}deg | "
                    f"XYZmm={xyz_text:>18s}",
                    end="",
                    flush=True
                )

            rtde_c.waitPeriod(
                cycle_start
            )

    except KeyboardInterrupt:
        print()
        print(
            "Ctrl+C -> kilepes"
        )

    except Exception as e:
        print()
        print()
        print("=" * 60)
        print("HIBA")
        print("=" * 60)
        print(
            type(e).__name__ + ":",
            e
        )

    finally:
        print()
        print()
        print(
            "Teljes rendszer leallitasa..."
        )

        if rtde_c is not None:
            try:
                if servo_active:
                    rtde_c.servoStop()
            except Exception:
                pass

            try:
                rtde_c.stopScript()
            except Exception:
                pass

        if gripper_worker is not None:
            try:
                gripper_worker.stop()
            except Exception:
                pass

        stop_child(
            "RGB",
            rgb_proc
        )

        stop_child(
            "Depth",
            depth_proc
        )

        stop_child(
            "Top camera",
            top_proc
        )

        if imu_ser is not None:
            try:
                imu_ser.close()
            except Exception:
                pass

        try:
            xyz_sock.close()
        except Exception:
            pass

        try:
            feature_sock.close()
        except Exception:
            pass

        try:
            top_sock.close()
            astra_shadow_sock.close()
        except Exception:
            pass

        try:
            status_sock.close()
        except Exception:
            pass

        try:
            measurement_logger.close(
                time.monotonic() - session_t0,
                control_mode
            )
        except Exception:
            pass

        print(
            "Minden process leallitva."
        )


if __name__ == "__main__":
    main()
