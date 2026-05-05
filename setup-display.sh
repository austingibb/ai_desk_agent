#!/bin/bash
set -e
echo "=== AI E-Ink Roommate — Pi Zero 2W Display Server Setup ==="

echo "=== Enabling SPI ==="
if ! grep -q "^dtparam=spi=on" /boot/firmware/config.txt; then
    echo "dtparam=spi=on" | sudo tee -a /boot/firmware/config.txt
    echo "SPI enabled. A reboot will be needed."
    REBOOT_NEEDED=1
else
    echo "SPI already enabled."
fi

echo "=== Installing system packages ==="
sudo apt-get update
sudo apt-get install -y python3-pip fonts-dejavu python3-venv

echo "=== Creating virtual environment ==="
python3 -m venv venv

echo "=== Installing Python packages ==="
./venv/bin/pip install -r requirements-display.txt

echo "=== Installing systemd service ==="
sudo cp display-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable display-server

if [ -n "$REBOOT_NEEDED" ]; then
    echo ""
    echo "=== Reboot required for SPI ==="
    echo "Run: sudo reboot"
    echo "Then after reboot: sudo systemctl start display-server"
else
    echo "=== Done ==="
    echo "Start with: sudo systemctl start display-server"
fi
