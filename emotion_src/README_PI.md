# Affect-Pi — Raspberry Pi deployment (`emotion_src/`)

Drag-and-drop this whole `emotion_src/` folder onto your Raspberry Pi 5, then
follow the steps below. It runs: **Pi camera → emotion → adaptive robot eyes on
the two ST7789 LCD panels.**

Everything is self-contained — the two model files are already inside
(`models/face_landmarker.task`, `artifacts/emotion_tasks_model.joblib`).

---

## 0. What you need
- Raspberry Pi 5 (4 GB), Raspberry Pi OS **Bookworm** (64-bit).
- A camera: the **Pi Camera 2** (CSI ribbon) *or* a USB webcam.
- 2× 1.47" 172×320 ST7789 SPI LCD panels.

## 1. Wire the two panels (BCM pin numbers)
These are the defaults in `robot_eyes/config.py`. Both panels share the SPI bus,
the DC line and the RST line; each has its own chip-select and backlight.

| Signal | Panel 0 (left eye) | Panel 1 (right eye) |
|---|---|---|
| SPI bus / CS | SPI0, **CE0** (GPIO8) | SPI0, **CE1** (GPIO7) |
| SCLK | GPIO11 | GPIO11 (shared) |
| MOSI / SDA | GPIO10 | GPIO10 (shared) |
| DC | **GPIO25** | GPIO25 (shared) |
| RST | **GPIO27** | GPIO27 (shared) |
| BL (backlight) | **GPIO18** | **GPIO24** |
| VCC / GND | 3V3 / GND | 3V3 / GND |

> If your panels are wired differently, edit the `eye0` / `eye1` pins in
> `robot_eyes/config.py` to match — nothing else needs to change.

## 2. Transfer the folder to the Pi
From your PC (or just copy via a USB stick):
```bash
scp -r emotion_src  pi@<your-pi-ip>:~/
```

## 3. One-time setup (on the Pi, inside the folder)
```bash
cd ~/emotion_src
bash setup_pi.sh
sudo reboot          # so SPI is active
```
`setup_pi.sh` installs the system packages, enables SPI, makes a virtualenv, and
installs the Python dependencies.

> **MediaPipe note:** if `pip install mediapipe` fails on your OS image, install a
> version that has an aarch64 wheel for your Python (e.g. `pip install
> mediapipe==0.10.14`) — that's the one piece most sensitive to the OS/Python combo.

## 4. Calibrate the panels (once)
```bash
cd ~/emotion_src
source .venv/bin/activate
python3 run_on_pi.py --calibrate
```
You should see, on each panel: a **1-px white border** and **R/G/B corner blocks**
(red top-left, green top-right, blue bottom-left). Then:
- Border clipped/wrapped → adjust `col_offset` / `row_offset` in `robot_eyes/config.py`
  (try values near 34 for the 172-wide panel) and re-run.
- Red/blue swapped → set `bgr: bool = True` in `config.py`.
- Washed-out/inverted → toggle `invert` in `config.py`.
Press **Ctrl-C** to exit the pattern.

## 5. Run it
```bash
source .venv/bin/activate
python3 run_on_pi.py
```
The eyes appear on the panels and react to whoever is in front of the camera.
It silently learns your face and, over time, favours the emotes you respond best
to (saved in `artifacts/known_faces.txt` and `artifacts/emote_policy.txt`).

**Quit:** Ctrl-C.

### Useful options
| Command | What it does |
|---|---|
| `python3 run_on_pi.py --mirror` | mirror the camera (selfie view) |
| `python3 run_on_pi.py --display windows` | show OpenCV windows instead of panels (needs a monitor) |
| `python3 run_on_pi.py --display both` | panels **and** an on-screen preview |
| `python3 run_on_pi.py --min-conf 0.3` | react more readily (lower confidence gate) |
| `python3 run_on_pi.py --epsilon 0.3` | try new emotes more often (more variety) |
| `python3 run_on_pi.py --no-learn` | keep what it learned but stop updating |
| `python3 run_on_pi.py --camera 1` | pick a different USB webcam index |

## Refresh / improve the model — the `--retrain-on-PC` helper

The bundled emotion model is a baseline. To improve it, **retrain on your PC**
(not the Pi — training needs the image dataset and CPU), then re-deploy. From the
`affect_pi_base` repo on your PC (the one that contains `data/`):

```bash
python scripts/retrain_on_pc.py --max-per-class 400 --de-iterations 8
```

This runs the full trainer, then automatically refreshes
`emotion_src/artifacts/emotion_tasks_model.joblib` **and** `emotion_src.zip`.
(Quick low-data test of the flow: `python scripts/retrain_on_pc.py --max-per-class 60 --no-de`.)

Then re-deploy — copy the whole `emotion_src/` folder to the Pi again, **or** just
sync the one model file over the old one:

```bash
scp emotion_src/artifacts/emotion_tasks_model.joblib  pi@<pi-ip>:~/emotion_src/artifacts/
```

No need to re-run `setup_pi.sh` — only the model changed.

## 6. Run automatically on boot (optional)
```bash
sudo tee /etc/systemd/system/affect-eyes.service >/dev/null <<EOF
[Unit]
Description=Affect-Pi robot eyes
After=multi-user.target
[Service]
WorkingDirectory=/home/pi/emotion_src
ExecStart=/home/pi/emotion_src/.venv/bin/python3 /home/pi/emotion_src/run_on_pi.py
Restart=on-failure
User=pi
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now affect-eyes.service
# logs: journalctl -u affect-eyes -f
```

## Troubleshooting
| Symptom | Fix |
|---|---|
| `No camera found` | check the CSI ribbon / `libcamera-hello`, or pass `--camera 1` for USB |
| Panels stay black | SPI not enabled (`sudo raspi-config` → Interface → SPI), check wiring/backlight pin |
| `GPIO busy` / chip errors | reboot; ensure `python3-lgpio` installed; nothing else is using the pins |
| One eye only | a wrong CS/DC/BL pin — recheck panel 1 (CE1=GPIO7, BL=GPIO24) |
| `mediapipe` import error | install an aarch64-compatible version (see §3 note) |
| Image shifted/wrapped | wrong `col_offset` — recalibrate (§4) |

## Notes / honesty
- The bundled emotion model is a small baseline (~53% over 5 classes, trained on a
  face-dataset) — expect noticeable mistakes, especially under different lighting
  than the training data. It improves with retraining (done on a PC, not the Pi).
- No internet is needed at runtime. No images are stored — only compact numeric
  face signatures in a local text file (`artifacts/known_faces.txt`); delete it to
  forget everyone.
