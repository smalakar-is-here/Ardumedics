"""
ArduMedics Full Pipeline — Fall Detection + OCR

Orchestrates the complete ArduMedics pipeline:
1. Camera input → Pose Estimation (YOLOv8-Pose)
2. Keypoints → TPC Tracker (Temporal Pose Consistency)
3. TPC Output → Fall Classifier (Hybrid Rule-ML)
4. Fall Alert → GPIO/SMS/LED
5. Medicine Image → Multi-Engine OCR Fusion
"""

import argparse
import yaml
import time
import cv2

from ..fall_detection import PoseEstimator, TPCTracker, FallClassifier
from ..ocr import MultiEngineOCR, PrescriptionParser


class ArduMedicsPipeline:
    """
    Complete ArduMedics pipeline combining fall detection and OCR.

    Args:
        config_path: Path to YAML configuration file
    """

    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)

        # Initialize fall detection components
        fall_config = self.config.get('model', {})
        self.pose_estimator = PoseEstimator(
            model_path=fall_config.get('path', 'models/best_nano_ncnn'),
            backend=fall_config.get('format', 'ncnn'),
            conf_threshold=fall_config.get('confidence', 0.25),
            imgsz=fall_config.get('imgsz', 640)
        )

        tpc_config = self.config.get('tpc', {})
        self.tpc_tracker = TPCTracker(
            window_size=tpc_config.get('window_size', 30),
            confirm_frames=tpc_config.get('confirm_frames', 5)
        )

        self.fall_classifier = FallClassifier(
            torso_angle_threshold=tpc_config.get('torso_angle_threshold', 60),
            ground_proximity_threshold=tpc_config.get('ground_proximity_threshold', 0.4),
            velocity_threshold=tpc_config.get('velocity_threshold', 0.15)
        )

        # Initialize OCR (lazy — only when needed)
        self._ocr = None
        self._parser = None

    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = MultiEngineOCR(gpu=False)
        return self._ocr

    @property
    def parser(self):
        if self._parser is None:
            self._parser = PrescriptionParser()
        return self._parser

    def _load_config(self, config_path):
        """Load YAML configuration."""
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}

    def process_frame(self, frame):
        """Process a video frame for fall detection."""
        detections = self.pose_estimator.detect(frame)
        results = []

        for det in detections:
            tpc_result = self.tpc_tracker.update(det['keypoints'], frame.shape[0])
            velocity = tpc_result.get('velocity', 0)

            fall_result = self.fall_classifier.classify(det['keypoints'], velocity)
            fall_result['tpc_confirmed'] = tpc_result['tpc_confirmed']

            results.append({
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'keypoints': det['keypoints'],
                'fall': fall_result
            })

        return results

    def process_prescription(self, image):
        """Process a prescription image with OCR."""
        ocr_result = self.ocr.extract(image)
        parsed = self.parser.parse(ocr_result['text'])
        return {
            'ocr': ocr_result,
            'parsed': parsed
        }


def main():
    parser = argparse.ArgumentParser(description='ArduMedics Full Pipeline')
    parser.add_argument('--config', type=str,
                       default='edge-deployment/raspberry-pi5/configs/fall_detection.yaml',
                       help='Path to config YAML')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--image', type=str, help='Prescription image path')

    args = parser.parse_args()

    pipeline = ArduMedicsPipeline(config_path=args.config)

    if args.image:
        result = pipeline.process_prescription(args.image)
        print(f"OCR Text: {result['ocr']['text'][:200]}")
        print(f"Medicines: {result['parsed']['medicines']}")
    else:
        cap = cv2.VideoCapture(args.camera)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = pipeline.process_frame(frame)
            for r in results:
                if r['fall']['tpc_confirmed']:
                    print(f"[ALERT] FALL DETECTED!")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()


if __name__ == '__main__':
    main()
