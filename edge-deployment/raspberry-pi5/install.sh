#!/bin/bash
# ============================================================
# ArduMedics - Raspberry Pi 5 One-Click Installation Script
# ============================================================
# Run this on a fresh Raspberry Pi 5 with Raspberry Pi OS (64-bit)
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# This script installs:
#   - System dependencies (OpenCV, NCNN, build tools)
#   - Python virtual environment with all packages
#   - ArduMedics source code (from git)
#   - GPIO support for alerts
#
# Model weights must be downloaded separately (see models/README.md)
# ============================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  ArduMedics - Raspberry Pi 5 Installation${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Raspberry Pi.${NC}"
    echo -e "${YELLOW}Continuing anyway, but some features may not work.${NC}"
fi

# Step 1: Update system
echo -e "${GREEN}[1/7] Updating system...${NC}"
sudo apt update && sudo apt upgrade -y

# Step 2: Install system dependencies
echo -e "${GREEN}[2/7] Installing system dependencies...${NC}"
sudo apt install -y \
    build-essential cmake git pkg-config \
    libopencv-dev python3-opencv \
    libprotobuf-dev protobuf-compiler \
    python3-pip python3-venv \
    libatlas-base-dev \
    libopenjp2-7 libtiff5 libjpeg-dev \
    v4l-utils \
    wget curl \
    libncnn-dev ncnn 2>/dev/null || {
    echo -e "${YELLOW}NCNN system package not found, will install via pip${NC}"
}

# Step 3: Enable camera
echo -e "${GREEN}[3/7] Enabling camera...${NC}"
sudo raspi-config nonint do_camera 0 2>/dev/null || {
    echo -e "${YELLOW}Could not enable camera via raspi-config. Enable manually if needed.${NC}"
}

# Step 4: Clone repository (if not already cloned)
echo -e "${GREEN}[4/7] Setting up ArduMedics repository...${NC}"
INSTALL_DIR="$HOME/Ardumedics"
if [ ! -d "$INSTALL_DIR" ]; then
    git clone https://github.com/smalakar-is-here/Ardumedics.git "$INSTALL_DIR"
else
    echo -e "${YELLOW}Repository already exists at $INSTALL_DIR, pulling latest...${NC}"
    cd "$INSTALL_DIR" && git pull
fi
cd "$INSTALL_DIR"

# Step 5: Create virtual environment
echo -e "${GREEN}[5/7] Creating Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Step 6: Install Python packages
echo -e "${GREEN}[6/7] Installing Python packages...${NC}"
pip install --upgrade pip setuptools wheel

# Core packages (required for fall detection)
pip install numpy opencv-python-headless ncnn

# GPIO packages (for buzzer/LED alerts)
pip install RPi.GPIO gpiozero 2>/dev/null || {
    echo -e "${YELLOW}GPIO packages not installed (not on Pi). Skipping.${NC}"
}

# YAML config support
pip install pyyaml

echo -e "${YELLOW}Core fall detection packages installed.${NC}"
echo -e "${YELLOW}For OCR support, run: pip install easyocr paddleocr paddlepaddle tesseract${NC}"

# Step 7: Verify installation
echo -e "${GREEN}[7/7] Verifying installation...${NC}"

# Check Python
python3 -c "import numpy; print(f'  NumPy: {numpy.__version__}')" && \
python3 -c "import cv2; print(f'  OpenCV: {cv2.__version__}')" && \
python3 -c "import ncnn; print(f'  NCNN: OK')" || {
    echo -e "${RED}Some packages failed to install. Check errors above.${NC}"
    exit 1
}

# Check camera
if ls /dev/video* 1>/dev/null 2>&1; then
    echo -e "  Camera: ${GREEN}Found${NC}"
else
    echo -e "  Camera: ${YELLOW}Not found${NC} (connect camera and reboot)"
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. Download model weights (see models/README.md)"
echo -e "  2. Test camera: libcamera-hello --list-cameras"
echo -e "  3. Run fall detection:"
echo -e "     ${YELLOW}source venv/bin/activate${NC}"
echo -e "     ${YELLOW}python3 edge-deployment/raspberry-pi5/scripts/run_fall_detection.py --model models/best_nano_ncnn/ --camera 0 --test-only${NC}"
echo ""
echo -e "Full setup guide: edge-deployment/raspberry-pi5/SETUP_GUIDE.md"
