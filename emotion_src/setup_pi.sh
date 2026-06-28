#!/usr/bin/env bash
# One-time setup for Affect-Pi on a Raspberry Pi 5 (Raspberry Pi OS Bookworm).
# Run from inside the emotion_src/ folder:   bash setup_pi.sh
set -e

echo ">> Updating apt and installing system packages..."
sudo apt update
sudo apt install -y python3-venv python3-pip python3-picamera2 python3-lgpio libcap-dev

echo ">> Enabling the SPI interface (needed for the LCD panels)..."
sudo raspi-config nonint do_spi 0   # 0 = enable

echo ">> Creating a virtual environment (with access to apt-installed picamera2/lgpio)..."
python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel

echo ">> Installing Python dependencies..."
pip install -r requirements-pi.txt

echo ""
echo ">> Done. A reboot is recommended so SPI is active:   sudo reboot"
echo ">> After reboot, from this folder:"
echo "     source .venv/bin/activate"
echo "     python3 run_on_pi.py --calibrate     # align the panels first"
echo "     python3 run_on_pi.py                 # run the eyes on the panels"
