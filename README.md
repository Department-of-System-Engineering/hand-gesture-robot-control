# Kézmozgás-alapú teleoperációs rendszer UR5e robotkarhoz

A 2026-os TDK-dolgozatomhoz egy kézmozgással vezérelhető teleoperációs rendszert fejlesztettem Universal Robots UR5e robotkarhoz. Az elsődleges kézpozíciót Orbbec Astra Pro Plus RGB-D kamerával és MediaPipe-alapú kézdetektálással határozom meg. A kéz orientációját ESP32-höz kapcsolt inerciális érzékelők, a megfogó célértékét pedig egy potenciométer szolgáltatja. A robot végpontját RTDE/servoL vezérléssel, az OnRobot RG2 megfogót a robot I/O-kapcsolatán keresztül irányítom.

A vizsgálatokhoz egy második, felülnézeti Eternico ET201 kamerát is alkalmaztam. A kamera eltérő nézőpontból egészíti ki az Astra kézmegfigyelését. A háromsíkú kalibrációból becsült XZ-pozíciót az elsődleges kamera kiesésekor csak meghatározott érvényességi feltételek mellett használom.

## Forráskódok

| Fájl | Feladat |
| --- | --- |
| `01_rgb_mediapipe_full_sender_v5_pickplace_ui.py` | Astra RGB-kép és kézdetektálás |
| `02_openni_depth_xyz_bridge_v5_adaptive_robust.py` | Mélységfeldolgozás és 3D pozíció |
| `03_top_camera_hand_sender_v5_pickplace_ui.py` | Felső kamera kézmegfigyelése |
| `01_rgb_mediapipe_calibration_sender_v6_guided.py` | Astra kézdetektálása a kalibrációhoz |
| `02_openni_depth_xyz_bridge_v4_calibration_diag.py` | Kalibrációs mélységfeldolgozás |
| `03_top_camera_hand_sender_v5_adaptive_guide.py` | Felső kamera kalibrációs segédprogramja |
| `04_top_to_astra_multiplane_calibration_v4_quality.py` | Háromsíkú homográfiás kalibráció |
| `top_astra_multiplane_calibration_v4.json` | A vizsgálatokhoz használt kalibrációs modell |
| `two_camera_blindspot_tdk_test_v5.py` | Kétkamerás detektálási vizsgálat |
| `esp32_imu_quaternion_pot_v5_pickplace.ino` | IMU és potenciométer firmware |
| `ur5e_tdk_pick_place_full_v1_fix1.py` | Robotvezérlés szigorú érvényességi feltételekkel |
| `ur5e_tdk_pick_place_relaxed_gate_v1.py` | Robotvezérlés kiterjesztett érvényességi feltételekkel |

A tárolóba a fejlesztés szempontjából legfontosabb programváltozatokat válogattam ki, nem a teljes fejlesztési előzményt. A nyers mérési naplókat nem tettem közzé.

## Futtatási feltételek

A rendszer fejlesztéséhez Windows környezetet és Python 3.10-et használtam. A szükséges Python-csomagokat a `requirements.txt`, a hardveres és szoftveres előfeltételeket a [`docs/INSTALLATION.md`](docs/INSTALLATION.md) foglalja össze. A MediaPipe `hand_landmarker.task` modellfájlt külön kell beszerezni.

## Demonstráció

A rendszer működését a `UR5e_hand_gesture_teleoperation_demo.mp4` című videó szemlélteti. A videót külön, a tároló **Releases** részében teszem elérhetővé.

A forráskódok a saját laboratóriumi összeállításomhoz készültek. A fizikai robot mozgatása megfelelő kalibrációt és biztonságos, felügyelt környezetet igényel.
