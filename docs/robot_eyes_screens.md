# robot_eyes — dual-display pixelated robot eyes (Raspberry Pi 5 + ST7789)

Two animated, pixelated robot eyes — one per display — with emote moods
(neutral / happy / angry / tired / surprised), blinking and idle gaze.
Identical rendering code runs on the Pi hardware and in a desktop simulator,
so you can develop in VS Code and deploy unchanged.

## Target hardware (assumed — verify against yours)
- 2× Waveshare 1.47" LCD: **172×320**, **ST7789V3**, 4-wire SPI, RGB565.
  https://www.waveshare.com/wiki/1.47inch_LCD_Module
- Raspberry Pi 5 (4 GB). GPIO via **gpiozero/lgpio**, SPI via **spidev**.
  (`RPi.GPIO` does **not** work on Pi 5 — RP1 southbridge.)

If your panel/driver differs, change `config.py` and re-run `--calibrate`.

## Layout
```
robot_eyes/
  config.py      all tunables (geometry, colours, timing, pins, offsets)
  renderer.py    pure numpy eye renderer (no hardware) — unit-testable
  controller.py  blink/saccade/mood state machine (pure)
  st7789.py      ST7789V3 SPI driver (spidev + gpiozero) — hardware only
  backends.py    SpiBackend (panels) | SimBackend (pygame) — same interface
  main.py        entry point: auto-detect, demo loop, calibrate, RSS report
```

## Default wiring (both panels on SPI0, two chip-selects)
| Panel signal | Eye 0 (BCM)        | Eye 1 (BCM)        |
|--------------|--------------------|--------------------|
| SCLK         | GPIO11 (SPI0 SCLK) | GPIO11 (shared)    |
| MOSI / DIN   | GPIO10 (SPI0 MOSI) | GPIO10 (shared)    |
| CS           | GPIO8  (CE0)       | GPIO7  (CE1)       |
| DC           | GPIO25             | GPIO25 (shared)    |
| RST          | GPIO27             | GPIO27 (shared)    |
| BL           | GPIO18             | GPIO24             |
| VCC / GND    | 3V3 / GND          | 3V3 / GND          |

Change any of these in `config.py → HardwareConfig`. To roughly **double the
frame rate**, put each eye on its own bus (SPI0 + SPI1) with separate DC/RST.

## One-time Pi setup
```bash
# Enable SPI (CE0+CE1 give /dev/spidev0.0 and /dev/spidev0.1)
sudo raspi-config   # Interface Options -> SPI -> Enable      (or add dtparam=spi=on)
# For a 2nd hardware bus (optional, faster): add to /boot/firmware/config.txt:
#   dtoverlay=spi1-1cs
sudo apt update && sudo apt install -y python3-gpiozero python3-lgpio python3-spidev
sudo usermod -aG gpio,spi "$USER" && newgrp gpio   # avoid running as root
```
If you use a virtualenv, create it with access to the apt GPIO packages:
```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install numpy
```

## Desktop simulator (PC / VS Code)
```bash
pip install numpy pygame
python -m robot_eyes.main --sim
# keys: 1-5 mood | arrows look | c centre | space blink | q quit
```

## Run on the Pi
```bash
python -m robot_eyes.main                # auto-detects Pi 5 -> hardware
python -m robot_eyes.main --calibrate    # FIRST: verify orientation/offset/colour
python -m robot_eyes.main --rss          # print peak RAM once per second
```

## Calibration (do this first on real panels)
`--calibrate` draws a 1px white border with coloured corners:
- **Border clipped / image wrapped** → wrong RAM offset. The 172-wide panel sits
  in 240-wide controller RAM, so portrait needs `col_offset = (240-172)/2 = 34`.
  If you set `rotation` to 90/270 the offset moves to the row axis — try
  `col_offset=0, row_offset=34` instead. Tune in `config.py`.
- **Red/blue swapped** → set `hw.bgr = True`.
- **Washed out / inverted** → toggle `hw.invert`.

## Tuning the look
All in `config.py`:
- `screen.pixel_size` — block size (4 → 43×80 logical grid; lower = finer).
- `style.eye_color`, `style.bg_color`, `style.eye_w_frac`, `style.eye_h_frac`,
  `style.corner_radius_frac`.
- `anim.target_fps`, blink/saccade timings.
- `hw.rotation` — map the portrait buffer onto a physically rotated panel.
- `hw.eye0/eye1.eye_side` ("left"/"right") — flip if your angry/tired brows
  point the wrong way, or just swap the two panels physically.

## Performance (full-frame SPI bandwidth)
One frame = 172×320×2 = **110,080 bytes** (RGB565). Two eyes on one bus is
sequential. Theoretical (transfer-only) full-frame rates:

| SPI clock | 1 eye        | 2 eyes / 1 bus |
|-----------|--------------|----------------|
| 40 MHz    | ~44 fps      | ~22 fps        |
| 62.5 MHz  | ~71 fps      | ~35 fps        |

Real throughput is ~60–75% of that (CS toggling, command bytes, ioctl chunking),
so budget ~15–18 fps for two full frames at 40 MHz. Mitigations, in order of
impact: (1) **dirty-rectangle updates** (default on — only the changed bounding
box is transmitted; a blink/saccade touches a fraction of the screen), (2) two
separate SPI buses, (3) raise `hw.spi_hz` to 60–80 MHz on short clean wiring,
(4) raise the kernel SPI buffer (`spidev.bufsiz=65536` on the kernel cmdline).
CPU rendering itself is ~0.75 ms for both eyes (numpy), i.e. not the bottleneck.

## Memory
Measured peak RSS of python+numpy+renderer was **35.2 MB**; spidev (<1 MB) and
gpiozero+lgpio (~3–6 MB) bring the on-Pi process to roughly **40–50 MB** — well
under your 100 MB ceiling (confidence high, ≈0.9; exact figure depends on the
numpy build and OS page sharing). Verify on your unit with `--rss`. pygame is
never imported on the Pi (only the simulator loads it).
