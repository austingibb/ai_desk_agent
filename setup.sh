#!/bin/bash
set -e
echo "=== AI E-Ink Roommate — Pi 5 Camera + Orchestrator Setup ==="

echo "=== Installing system packages ==="
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

echo "=== Creating virtual environment ==="
python3 -m venv venv

echo "=== Installing Python packages ==="
./venv/bin/pip install -r requirements.txt

echo "=== Installing systemd service ==="
sudo cp ai-eink.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-eink

echo "=== Done ==="
echo "Start with: sudo systemctl start ai-eink"
echo "Logs: journalctl -u ai-eink -f"
