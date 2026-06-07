# Raspberry Pi 5 Setup Guide — ArduMedics Fall Detection

This guide walks you through setting up the ArduMedics fall detection system on a **Raspberry Pi 5 (8GB)** from scratch.

---

## Hardware Requirements

| Component | Specification | Notes |
|-----------|--------------|-------|
| Raspberry Pi 5 | 8GB RAM model | 4GB model also works but limits ensemble |
| microSD Card | 32GB+ Class 10 / A2 | Samsung EVO Plus or SanDisk Extreme recommended |
| Camera | Pi Camera Module v3 OR USB webcam | v3 supports auto-focus |
| Power Supply | 5V/5A USB-C PD | Official Pi 5 PSU recommended |
| Cooling | Active cooler | Pi 5 runs hot under AI inference load |
| Optional: Buzzer | 3.3V active buzzer | Connected to GPIO for fall alert |
| Optional: LED | Red LED + 220 ohm resistor | Visual fall indicator |
| Optional: Arduino Nano | With MPU6050 IMU | For accelerometer-based fall detection fusion |

---

## Step 1: Flash Raspberry Pi OS

```bash
# On your computer, download Raspberry Pi Imager
# https://www.raspberrypi.com/software/

# Flash Raspberry Pi OS Lite (64-bit, Bookworm)
# - Choose: Raspberry Pi OS (Other) → Raspberry Pi OS Lite (64-bit)
# - Configure: Enable SSH, set username/password, configure WiFi
# - Write to microSD card
```

After flashing, insert the microSD into Pi 5 and boot up.

---

## Step 2: Initial Pi 5 Configuration

```bash
# SSH into your Pi (or use keyboard + monitor)
ssh your_username@raspberrypi.local

# Update the system
sudo apt update && sudo apt upgrade -y

# Enable camera
sudo raspi-config
# → Interface Options → Camera → Enable
# → Interface Options → Legacy Camera → Disable (we use libcamera)

# Increase GPU memory (helps with camera)
echo "gpu_mem=256" | sudo tee -a /boot/config.txt

# Reboot
sudo reboot
```

---

## Step 3: Install System Dependencies

```bash
# Install build tools and libraries
sudo apt install -y \
    build-essential cmake git pkg-config \
    libopencv-dev python3-opencv \
    libprotobuf-dev protobuf-compiler \
    libncnn-dev ncnn \
    python3-pip python3-venv \
    libatlas-base-dev \
    libopenjp2-7 libtiff5 libjpeg-dev \
    v4l-utils \
    libcamera-apps \
    wget curl

# Verify camera
libcamera-hello --list-cameras
# Should show your camera device
```

---

## Step 4: Clone Repository & Install Python Dependencies

```bash
# Install Git LFS first
sudo apt install -y git-lfs
git lfs install

# Clone the repo
cd ~
git clone https://github.com/smalakar-is-here/Ardumedics.git
cd Ardumedics

# Pull model weights via LFS
git lfs pull

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install numpy opencv-python-headless
pip install ncnn  # NCNN Python binding for ARM inference

# For OCR (optional — requires more disk space):
pip install easyocr paddleocr paddlepaddle tesseract
sudo apt install -y tesseract-ocr

# For GPIO (buzzer/LED):
pip install RPi.GPIO gpiozero
```

**Or use the one-click installer:**
```bash
chmod +x edge-deployment/raspberry-pi5/install.sh
./edge-deployment/raspberry-pi5/install.sh
```

---

## Step 5: Export NCNN Model (Required)

The NCNN model is **NOT included** in the Kaggle outputs. You need to export it from `best_nano.pt`.

### Option A: Export on your computer BEFORE pushing to GitHub

```bash
# On your computer (with GPU recommended):
cd Ardumedics/models/
pip install ultralytics
python3 export_ncnn.py

# This creates best_nano_ncnn/ with model.ncnn.param and model.ncnn.bin
# Then push to GitHub (LFS will handle .ncnn.bin)
```

### Option B: Export directly on Raspberry Pi 5

```bash
# On Pi 5, after git lfs pull:
cd ~/Ardumedics
source venv/bin/activate
pip install ultralytics  # Only needed for export, can uninstall after

cd models/
python3 export_ncnn.py

# Verify
ls -la best_nano_ncnn/
# Should show: model.ncnn.param  model.ncnn.bin
```

### Option C: If NCNN export is already in the repo (pushed via LFS)

```bash
# If you already exported and pushed to GitHub:
cd ~/Ardumedics
git lfs pull  # Downloads the .ncnn.bin file
ls -la models/best_nano_ncnn/
```

---

## Step 6: Test Camera & Model

```bash
cd ~/Ardumedics
source venv/bin/activate

# Test camera
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print(f'Camera OK: {frame.shape}')
else:
    print('Camera FAILED — check connection')
cap.release()
"

# Test model inference
python3 edge-deployment/raspberry-pi5/scripts/run_fall_detection.py \
    --model models/best_nano_ncnn/ \
    --camera 0 \
    --test-only \
    --tpc-window 30
```

---

## Step 7: Run Fall Detection

```bash
cd ~/Ardumedics
source venv/bin/activate

# Basic fall detection with camera
python3 edge-deployment/raspberry-pi5/scripts/run_fall_detection.py \
    --model models/best_nano_ncnn/ \
    --camera 0 \
    --tpc-window 30 \
    --confidence 0.25 \
    --display

# Headless mode (no display, for background service)
python3 edge-deployment/raspberry-pi5/scripts/run_fall_detection.py \
    --model models/best_nano_ncnn/ \
    --camera 0 \
    --tpc-window 30 \
    --headless \
    --alert-gpio 17 \
    --log-file /tmp/fall_detection.log
```

---

## Step 8: (Optional) Run as Systemd Service

Create a systemd service so fall detection starts on boot:

```bash
# Create service file
sudo tee /etc/systemd/system/ardumedics.service << 'EOF'
[Unit]
Description=ArduMedics Fall Detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Ardumedics
ExecStart=/home/pi/Ardumedics/venv/bin/python3 edge-deployment/raspberry-pi5/scripts/run_fall_detection.py --model models/best_nano_ncnn/ --camera 0 --tpc-window 30 --headless --alert-gpio 17
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable ardumedics
sudo systemctl start ardumedics

# Check status
sudo systemctl status ardumedics

# View logs
journalctl -u ardumedics -f
```

---

## Step 9: (Optional) GPIO Setup for Alerts

### Buzzer (Active Buzzer on GPIO 17)

```
Pi 5 GPIO 17 ──► Buzzer (+) ──► GND
```

### LED (Red LED on GPIO 27)

```
Pi 5 GPIO 27 ──► 220Ω Resistor ──► LED (+) ──► GND
```

---

## Step 10: (Optional) OCR Setup

If you want to run the OCR prescription analysis on Pi 5:

```bash
# Install OCR engines
sudo apt install -y tesseract-ocr
pip install easyocr paddleocr paddlepaddle

# Run OCR
python3 edge-deployment/raspberry-pi5/scripts/run_ocr.py \
    --image /path/to/prescription.jpg \
    --config edge-deployment/raspberry-pi5/configs/ocr.yaml
```

> **Note:** OCR engines require ~4GB additional disk space. EasyOCR and PaddleOCR are heavy. For Pi 5, consider running OCR on a server and sending images via API.

---

## Performance Expectations

| Metric | Expected on Pi 5 |
|--------|------------------|
| Inference FPS (NCNN) | ~1.5-3.1 FPS |
| Inference latency | ~320-640 ms per frame |
| RAM usage (Nano only) | ~400-600 MB |
| RAM usage (Ensemble) | **NOT RECOMMENDED** (~8GB, exceeds Pi 5) |
| Temperature (with cooler) | 55-70°C under load |
| Temperature (no cooler) | 75-85°C — **thermal throttle risk** |

### Optimization Tips

1. **Use NCNN format** — 2-3x faster than PyTorch on ARM
2. **Reduce input resolution** — Try `--imgsz 320` instead of 640 for 2-3x speedup (lower accuracy)
3. **Frame skipping** — Process every 2nd or 3rd frame: `--skip-frames 2`
4. **Close unnecessary services** — Free up RAM for inference
5. **Overclock (risky)** — Pi 5 can be overclocked to 3.0 GHz for ~10% speed boost

```bash
# Frame skipping example (process every 3rd frame = ~9 FPS effective)
python3 edge-deployment/raspberry-pi5/scripts/run_fall_detection.py \
    --model models/best_nano_ncnn/ \
    --camera 0 \
    --tpc-window 30 \
    --skip-frames 3
```

---

## Troubleshooting

### Camera not found
```bash
# Check if camera is detected
libcamera-hello --list-cameras
ls /dev/video*
# If no camera, check ribbon cable connection or USB device
```

### Out of Memory (OOM)
```bash
# Check memory usage
free -h
# Increase swap
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=2048
sudo systemctl restart dphys-swapfile
```

### NCNN model fails to load
```bash
# Verify NCNN files exist and are not corrupted
ls -la ~/Ardumedics/models/best_nano_ncnn/
# model.ncnn.param and model.ncnn.bin must both be present
file ~/Ardumedics/models/best_nano_ncnn/model.ncnn.bin
# Should show "data" (not "empty")

# If missing, re-export:
cd ~/Ardumedics/models/
python3 export_ncnn.py
```

### High temperature / thermal throttle
```bash
# Check temperature
vcgencmd measure_temp
# If > 80°C, add active cooling
# Heatsink + fan is mandatory for AI inference on Pi 5
```

### Low FPS
```bash
# Check CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# If "powersave", switch to "performance":
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```
