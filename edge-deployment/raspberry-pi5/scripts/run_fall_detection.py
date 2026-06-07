#!/usr/bin/env python3
"""
ArduMedics Fall Detection — Raspberry Pi 5 Edge Deployment

Runs YOLOv8n-Pose (NCNN format) on Raspberry Pi 5 camera feed
with Temporal Pose Consistency (TPC) for robust fall detection.

Usage:
    python3 run_fall_detection.py --model models/best_nano_ncnn/ --camera 0 --tpc-window 30
    python3 run_fall_detection.py --model models/best_nano_ncnn/ --camera 0 --headless --alert-gpio 17
"""

import argparse
import sys
import os
import time
import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# Try to import ncnn
try:
    import ncnn
    HAS_NCNN = True
except ImportError:
    HAS_NCNN = False
    print("WARNING: ncnn not installed. Falling back to PyTorch (slower on Pi 5).")

# Try to import ultralytics for PyTorch fallback
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

# Try GPIO for alerts
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False

# ============================================================
# Configuration
# ============================================================

# COCO 17 keypoints
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Keypoint connections for visualization
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),             # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Fall detection thresholds
FALL_TORSO_ANGLE_THRESHOLD = 60   # degrees from vertical (more horizontal = fall)
FALL_GROUND_PROXIMITY = 0.4       # hip/ankle ratio threshold
FALL_VELOCITY_THRESHOLD = 0.15    # normalized velocity threshold
FALL_TPC_CONFIRM_FRAMES = 5       # consecutive fall frames needed for TPC confirmation


# ============================================================
# NCNN Pose Estimator
# ============================================================

class NCNNPoseEstimator:
    """NCNN-based YOLOv8n-Pose inference for Raspberry Pi 5."""

    def __init__(self, model_dir, conf_threshold=0.25, nms_threshold=0.45, imgsz=640):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.imgsz = imgsz

        param_path = os.path.join(model_dir, 'model.ncnn.param')
        bin_path = os.path.join(model_dir, 'model.ncnn.bin')

        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            raise FileNotFoundError(f"NCNN model files not found in {model_dir}")

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False  # Pi 5 doesn't have Vulkan
        self.net.load_param(param_path)
        self.load_model_safe(bin_path)

        print(f"[ArduMedics] NCNN model loaded from {model_dir}")

    def load_model_safe(self, bin_path):
        """Load NCNN model with error handling."""
        try:
            self.net.load_model(bin_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load NCNN model: {e}. "
                             f"Ensure model was exported with correct NCNN version.")

    def preprocess(self, img):
        """Preprocess image for NCNN inference."""
        h, w = img.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize with letterbox
        resized = cv2.resize(img, (new_w, new_h))
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_x = (self.imgsz - new_w) // 2
        pad_y = (self.imgsz - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        # Convert to NCNN Mat
        mat = ncnn.Mat.from_numpy(padded.astype(np.float32))
        # Normalize
        mat = (mat - 128.0) / 128.0

        return mat, scale, pad_x, pad_y

    def detect(self, img):
        """
        Run pose detection on an image.

        Returns:
            list of dicts with keys: bbox, confidence, keypoints
        """
        mat_in, scale, pad_x, pad_y = self.preprocess(img)

        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        ret, mat_out = ex.extract("out0")

        if ret != 0 or mat_out.empty():
            return []

        # Parse YOLOv8 output
        detections = mat_out.numpy()
        results = []

        for i in range(detections.shape[1]):
            det = detections[0, i, :]
            confidence = det[4]

            if confidence < self.conf_threshold:
                continue

            # Bounding box (center format)
            cx = (det[0] - pad_x) / scale
            cy = (det[1] - pad_y) / scale
            bw = det[2] / scale
            bh = det[3] / scale

            x1 = max(0, int(cx - bw / 2))
            y1 = max(0, int(cy - bh / 2))
            x2 = min(img.shape[1], int(cx + bw / 2))
            y2 = min(img.shape[0], int(cy + bh / 2))

            # Keypoints (17 x 3: x, y, confidence)
            keypoints = []
            for k in range(17):
                kx = (det[5 + k*3] - pad_x) / scale
                ky = (det[5 + k*3 + 1] - pad_y) / scale
                kc = det[5 + k*3 + 2]
                keypoints.append([float(kx), float(ky), float(kc)])

            results.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'keypoints': keypoints
            })

        return results


# ============================================================
# PyTorch Pose Estimator (Fallback)
# ============================================================

class PyTorchPoseEstimator:
    """PyTorch-based YOLOv8n-Pose inference (slower on Pi 5)."""

    def __init__(self, model_path, conf_threshold=0.25, imgsz=640):
        if not HAS_ULTRALYTICS:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        print(f"[ArduMedics] PyTorch model loaded from {model_path}")

    def detect(self, img):
        """Run pose detection using PyTorch YOLO."""
        results = self.model(img, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False)

        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and r.keypoints is not None:
                for i in range(len(r.boxes)):
                    box = r.boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(r.boxes.conf[i].cpu().numpy())
                    kpts = r.keypoints[i].cpu().numpy()  # (17, 3)

                    keypoints = []
                    for k in range(17):
                        keypoints.append([float(kpts[k, 0]), float(kpts[k, 1]), float(kpts[k, 2])])

                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': conf,
                        'keypoints': keypoints
                    })

        return detections


# ============================================================
# Temporal Pose Consistency (TPC) Tracker
# ============================================================

class TPCTracker:
    """
    Temporal Pose Consistency tracker for fall detection.

    Tracks keypoint velocities over a sliding window and requires
    consecutive fall confirmation frames before triggering an alert.
    This reduces false positives from transient pose variations.
    """

    def __init__(self, window_size=30, confirm_frames=5):
        self.window_size = window_size
        self.confirm_frames = confirm_frames
        self.velocity_history = deque(maxlen=window_size)
        self.fall_streak = 0  # consecutive frames classified as fall

    def compute_torso_angle(self, keypoints):
        """
        Compute the angle of the torso from vertical.
        A standing person has ~0° angle, a fallen person has ~90° angle.

        Args:
            keypoints: List of [x, y, confidence] for 17 COCO keypoints

        Returns:
            angle in degrees (0 = upright, 90 = horizontal)
        """
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        # Use midpoints of shoulders and hips for robustness
        shoulder_conf = min(left_shoulder[2], right_shoulder[2])
        hip_conf = min(left_hip[2], right_hip[2])

        if shoulder_conf < 0.3 or hip_conf < 0.3:
            return None  # Insufficient confidence

        shoulder_mid = np.array([(left_shoulder[0] + right_shoulder[0]) / 2,
                                  (left_shoulder[1] + right_shoulder[1]) / 2])
        hip_mid = np.array([(left_hip[0] + right_hip[0]) / 2,
                             (left_hip[1] + right_hip[1]) / 2])

        torso_vec = hip_mid - shoulder_mid
        vertical = np.array([0, 1])  # Down is positive in image coords

        cos_angle = np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) * np.linalg.norm(vertical) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        return angle

    def compute_ground_proximity(self, keypoints):
        """
        Compute how close the person is to the bottom of the frame.
        A fallen person has hips close to ankles in y-coordinate.

        Returns:
            ratio of hip_y to frame_height (0 = top, 1 = bottom)
        """
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]

        hip_conf = min(left_hip[2], right_hip[2])
        ankle_conf = min(left_ankle[2], right_ankle[2])

        if hip_conf < 0.3 or ankle_conf < 0.3:
            return None

        hip_y = (left_hip[1] + right_hip[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2

        # If hips are very close to ankles, person is likely horizontal
        if ankle_y > 0:
            ratio = abs(hip_y - ankle_y) / max(hip_y, ankle_y, 1)
        else:
            ratio = 0

        return ratio

    def compute_velocity(self, keypoints, prev_keypoints):
        """
        Compute normalized velocity of hip midpoint between frames.
        Fast downward velocity indicates a fall.
        """
        if prev_keypoints is None:
            return 0.0

        left_hip = keypoints[11]
        right_hip = keypoints[12]
        prev_left_hip = prev_keypoints[11]
        prev_right_hip = prev_keypoints[12]

        hip_conf = min(left_hip[2], right_hip[2])
        prev_hip_conf = min(prev_left_hip[2], prev_right_hip[2])

        if hip_conf < 0.3 or prev_hip_conf < 0.3:
            return 0.0

        hip_y = (left_hip[1] + right_hip[1]) / 2
        prev_hip_y = (prev_left_hip[1] + prev_right_hip[1]) / 2

        velocity = (hip_y - prev_hip_y)  # Positive = downward in image coords
        return velocity

    def update(self, keypoints, frame_height=1.0):
        """
        Update TPC tracker with new keypoints.

        Returns:
            dict with: is_fall, torso_angle, ground_proximity, velocity,
                       fall_streak, tpc_confirmed
        """
        prev_keypoints = self.velocity_history[-1] if self.velocity_history else None

        # Compute metrics
        torso_angle = self.compute_torso_angle(keypoints)
        ground_proximity = self.compute_ground_proximity(keypoints)
        velocity = self.compute_velocity(keypoints, prev_keypoints)

        # Store for velocity computation
        self.velocity_history.append(keypoints)

        # Fall classification using Hybrid Rule-ML
        is_fall = False

        if torso_angle is not None and torso_angle > FALL_TORSO_ANGLE_THRESHOLD:
            # Torso is more horizontal than vertical → likely fall
            is_fall = True

        if ground_proximity is not None and ground_proximity < FALL_GROUND_PROXIMITY:
            # Hips very close to ankles → lying down
            is_fall = True

        if velocity > FALL_VELOCITY_THRESHOLD:
            # Fast downward velocity → falling
            is_fall = True

        # TPC confirmation
        if is_fall:
            self.fall_streak += 1
        else:
            self.fall_streak = max(0, self.fall_streak - 1)  # Gradual decay

        tpc_confirmed = self.fall_streak >= self.confirm_frames

        return {
            'is_fall': is_fall,
            'tpc_confirmed': tpc_confirmed,
            'torso_angle': torso_angle,
            'ground_proximity': ground_proximity,
            'velocity': velocity,
            'fall_streak': self.fall_streak
        }


# ============================================================
# Fall Detection Pipeline
# ============================================================

class FallDetectionPipeline:
    """Complete fall detection pipeline with pose estimation + TPC."""

    def __init__(self, model_path, use_ncnn=True, tpc_window=30,
                 tpc_confirm=5, conf_threshold=0.25, imgsz=640):
        self.tpc = TPCTracker(window_size=tpc_window, confirm_frames=tpc_confirm)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        # Load model
        if use_ncnn and HAS_NCNN:
            try:
                self.estimator = NCNNPoseEstimator(
                    model_path, conf_threshold=conf_threshold, imgsz=imgsz)
            except Exception as e:
                print(f"[ArduMedics] NCNN failed: {e}")
                print(f"[ArduMedics] Falling back to PyTorch...")
                self.estimator = PyTorchPoseEstimator(
                    model_path, conf_threshold=conf_threshold, imgsz=imgsz)
        else:
            self.estimator = PyTorchPoseEstimator(
                model_path, conf_threshold=conf_threshold, imgsz=imgsz)

    def process_frame(self, frame):
        """
        Process a single frame for fall detection.

        Returns:
            dict with: detections, fall_status, annotated_frame
        """
        # Run pose estimation
        detections = self.estimator.detect(frame)

        # Process each detection
        fall_status = {
            'fall_detected': False,
            'tpc_confirmed': False,
            'person_count': len(detections),
            'details': []
        }

        for det in detections:
            keypoints = det['keypoints']
            result = self.tpc.update(keypoints, frame.shape[0])

            det['fall_result'] = result

            if result['tpc_confirmed']:
                fall_status['fall_detected'] = True
                fall_status['tpc_confirmed'] = True

            fall_status['details'].append(result)

        return {
            'detections': detections,
            'fall_status': fall_status
        }

    def annotate_frame(self, frame, result):
        """Draw detections and fall status on frame."""
        annotated = frame.copy()

        for det in result['detections']:
            bbox = det['bbox']
            conf = det['confidence']
            keypoints = det['keypoints']
            fall_result = det.get('fall_result', {})

            # Draw bounding box
            color = (0, 0, 255) if fall_result.get('tpc_confirmed') else (0, 255, 0)
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Draw keypoints
            for kp in keypoints:
                if kp[2] > 0.3:  # Confidence threshold
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(annotated, (x, y), 3, (0, 255, 255), -1)

            # Draw skeleton
            for i, j in SKELETON:
                if keypoints[i][2] > 0.3 and keypoints[j][2] > 0.3:
                    pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                    pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                    cv2.line(annotated, pt1, pt2, (255, 255, 255), 1)

            # Draw fall status
            if fall_result.get('tpc_confirmed'):
                cv2.putText(annotated, "FALL DETECTED!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            elif fall_result.get('is_fall'):
                streak = fall_result.get('fall_streak', 0)
                cv2.putText(annotated, f"Fall suspected (streak: {streak})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        return annotated


# ============================================================
# Alert Handler
# ============================================================

class AlertHandler:
    """Handle fall alerts via GPIO, sound, or logging."""

    def __init__(self, gpio_pin=None, log_file=None):
        self.gpio_pin = gpio_pin
        self.log_file = log_file
        self.last_alert_time = 0
        self.alert_cooldown = 10  # seconds between alerts

        # Setup GPIO
        if gpio_pin is not None and HAS_GPIO:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(gpio_pin, GPIO.OUT)
            GPIO.output(gpio_pin, GPIO.LOW)
            print(f"[ArduMedics] Alert GPIO pin {gpio_pin} configured")

        # Setup logging
        if log_file:
            logging.basicConfig(filename=log_file, level=logging.INFO,
                              format='%(asctime)s - %(message)s')
            self.logger = logging.getLogger('ardumedics')

    def trigger_alert(self, fall_status):
        """Trigger fall alert."""
        current_time = time.time()

        # Cooldown check
        if current_time - self.last_alert_time < self.alert_cooldown:
            return

        self.last_alert_time = current_time

        # GPIO alert
        if self.gpio_pin is not None and HAS_GPIO:
            GPIO.output(self.gpio_pin, GPIO.HIGH)
            # Will be reset after 3 seconds (handled in main loop)

        # Log alert
        if self.log_file:
            self.logger.info(f"FALL ALERT: {fall_status}")

        # Console alert
        print(f"\a[ALERT] FALL DETECTED at {time.strftime('%H:%M:%S')}")

    def reset_alert(self):
        """Reset GPIO alert."""
        if self.gpio_pin is not None and HAS_GPIO:
            GPIO.output(self.gpio_pin, GPIO.LOW)

    def cleanup(self):
        """Cleanup GPIO."""
        if HAS_GPIO:
            GPIO.cleanup()


# ============================================================
# Main Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='ArduMedics Fall Detection on Raspberry Pi 5')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model directory (NCNN) or .pt file (PyTorch)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--tpc-window', type=int, default=30,
                       help='TPC sliding window size (default: 30)')
    parser.add_argument('--tpc-confirm', type=int, default=5,
                       help='TPC confirmation frames (default: 5)')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Detection confidence threshold (default: 0.25)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size (default: 640, use 320 for faster Pi5)')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='Process every Nth frame (default: 1, use 2-3 for speed)')
    parser.add_argument('--display', action='store_true',
                       help='Display annotated video (requires monitor)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without display (for background service)')
    parser.add_argument('--alert-gpio', type=int, default=None,
                       help='GPIO pin for fall alert buzzer/LED')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path for fall events')
    parser.add_argument('--test-only', action='store_true',
                       help='Run a single test inference and exit')
    parser.add_argument('--ncnn', action='store_true', default=True,
                       help='Use NCNN format (default: True)')
    parser.add_argument('--pytorch', action='store_true',
                       help='Force PyTorch format instead of NCNN')

    args = parser.parse_args()

    use_ncnn = not args.pytorch

    # Initialize pipeline
    print("[ArduMedics] Initializing fall detection pipeline...")
    pipeline = FallDetectionPipeline(
        model_path=args.model,
        use_ncnn=use_ncnn,
        tpc_window=args.tpc_window,
        tpc_confirm=args.tpc_confirm,
        conf_threshold=args.confidence,
        imgsz=args.imgsz
    )

    # Initialize alert handler
    alert = AlertHandler(gpio_pin=args.alert_gpio, log_file=args.log_file)

    # Initialize camera
    print(f"[ArduMedics] Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        print("  - Check camera connection")
        print("  - Try: libcamera-hello --list-cameras")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[ArduMedics] Fall detection running!")
    print(f"  Model: {args.model}")
    print(f"  TPC Window: {args.tpc_window}, Confirm: {args.tpc_confirm}")
    print(f"  Confidence: {args.confidence}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Skip frames: {args.skip_frames}")
    print("  Press 'q' to quit\n")

    frame_count = 0
    fps_timer = time.time()
    fps = 0
    alert_reset_timer = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame")
                time.sleep(0.1)
                continue

            # Skip frames for performance
            frame_count += 1
            if frame_count % args.skip_frames != 0:
                continue

            # Process frame
            start_time = time.time()
            result = pipeline.process_frame(frame)
            inference_time = (time.time() - start_time) * 1000

            # Handle fall alert
            if result['fall_status']['tpc_confirmed']:
                alert.trigger_alert(result['fall_status'])
                alert_reset_timer = time.time()

            # Reset GPIO after 3 seconds
            if alert_reset_timer and time.time() - alert_reset_timer > 3:
                alert.reset_alert()
                alert_reset_timer = 0

            # Calculate FPS
            if time.time() - fps_timer > 1.0:
                fps = frame_count / (time.time() - fps_timer)
                frame_count = 0
                fps_timer = time.time()

            # Display
            if args.display and not args.headless:
                annotated = pipeline.annotate_frame(frame, result)
                cv2.putText(annotated, f"FPS: {fps:.1f} | Inference: {inference_time:.0f}ms",
                           (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow('ArduMedics Fall Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Headless mode - just log
                if frame_count % 30 == 0:
                    fall_str = "FALL!" if result['fall_status']['tpc_confirmed'] else "OK"
                    print(f"[{time.strftime('%H:%M:%S')}] FPS: {fps:.1f} | "
                          f"Inf: {inference_time:.0f}ms | "
                          f"Persons: {result['fall_status']['person_count']} | "
                          f"Status: {fall_str}")

            # Test mode - single frame
            if args.test_only:
                print(f"\nTest inference complete:")
                print(f"  Inference time: {inference_time:.0f} ms")
                print(f"  Persons detected: {result['fall_status']['person_count']}")
                print(f"  Fall detected: {result['fall_status']['fall_detected']}")
                break

    except KeyboardInterrupt:
        print("\n[ArduMedics] Stopped by user")
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        alert.cleanup()
        print("[ArduMedics] Cleanup complete")


if __name__ == '__main__':
    main()
