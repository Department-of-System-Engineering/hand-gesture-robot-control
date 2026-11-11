# Telepítési és indítási tudnivalók

A programokat Windows környezetben, Python 3.10 használatával fejlesztettem. A mélységi adatok feldolgozásához OpenNI2-t, a kézdetektáláshoz MediaPipe Hand Landmarkert alkalmaztam.

## Szükséges szoftverek

- Python 3.10 és a `requirements.txt` fájlban felsorolt Python-csomagok
- OpenNI2 runtime az Orbbec Astra Pro Plus mélységi adatfolyamához
- MediaPipe `hand_landmarker.task` modellfájl, a Python-programokkal azonos mappában
- ESP32-höz megfelelő Arduino fejlesztőkörnyezet az `.ino` firmware feltöltéséhez
- UR5e robotkapcsolat és a használt RTDE-kommunikáció beállításai

A Python-függőségek telepítésének alapja:

```powershell
py -3.10 -m venv openni_env
.\openni_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Az OpenNI2 telepítése külön szükséges. Ha a runtime könyvtárát a Python nem találja, a `OPENNI2_REDIST` környezeti változóval adható meg az elérési útja. A MediaPipe modell a [hivatalos dokumentációból](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) szerezhető be.

## Programok indítása

A normál robotvezérléshez a három fő kamerás program és a kiválasztott UR5e-program ugyanabban a mappában maradjon. A robotvezérlő a kamerás segédfolyamatokat automatikusan indítja. A méréshez használt kalibrációs modell a `top_astra_multiplane_calibration_v4.json` fájlban található.

A két robotvezérlési változat:

```powershell
python ur5e_tdk_pick_place_full_v1_fix1.py
```

```powershell
python ur5e_tdk_pick_place_relaxed_gate_v1.py
```

A háromsíkú kalibráció külön indítható, ehhez a három kalibrációs segédprogramot is a főprogram mellé kell helyezni:

```powershell
python 04_top_to_astra_multiplane_calibration_v4_quality.py
```

A kétkamerás detektálási mérőprogram robotkapcsolat nélkül is használható:

```powershell
python two_camera_blindspot_tdk_test_v5.py
```

Az ESP32 firmware 115200 baud soros adatátvitelt használ. Az UR5e, az RG2 megfogó, a kameraelhelyezés és a soros kapcsolat beállításai az adott laboratóriumi összeállításhoz igazítandók. Robotmozgatás előtt az összes biztonsági és kalibrációs feltételt ellenőrizni kell.
