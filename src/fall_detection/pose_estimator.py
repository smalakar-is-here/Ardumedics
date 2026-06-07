"""
ArduMedics Pose Estimator — YOLOv8-Pose Wrapper

Supports both NCNN (for Raspberry Pi 5) and PyTorch inference.
"""

import cv2
import numpy as np
from pathlib import Path


class PoseEstimator:
    """
    Unified pose estimator supporting NCNN and PyTorch backends.

    Args:
        model_path: Path to model directory (NCNN) or .pt file (PyTorch)
        backend: 'ncnn' or 'pytorch'
        conf_threshold: Detection confidence threshold
        imgsz: Input image size
    """

    def __init__(self, model_path, backend='ncnn', conf_threshold=0.25, imgsz=640):
        self.model_path = model_path
        self.backend = backend
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        if backend == 'ncnn':
            self._init_ncnn(model_path)
        else:
            self._init_pytorch(model_path)

    def _init_ncnn(self, model_dir):
        """Initialize NCNN model."""
        import ncnn
        param_path = str(Path(model_dir) / 'model.ncnn.param')
        bin_path = str(Path(model_dir) / 'model.ncnn.bin')

        self.net = ncnn.Net()
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

    def _init_pytorch(self, model_path):
        """Initialize PyTorch YOLO model."""
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Detect persons and estimate keypoints in an image.

        Args:
            image: numpy array (H, W, 3) BGR format

        Returns:
            list of dicts: [{bbox, confidence, keypoints}]
        """
        if self.backend == 'ncnn':
            return self._detect_ncnn(image)
        else:
            return self._detect_pytorch(image)

    def _detect_ncnn(self, image):
        """Run NCNN inference."""
        import ncnn
        h, w = image.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_x = (self.imgsz - new_w) // 2
        pad_y = (self.imgsz - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        mat = ncnn.Mat.from_numpy(padded.astype(np.float32))
        mat = (mat - 128.0) / 128.0

        ex = self.net.create_extractor()
        ex.input("in0", mat)
        ret, mat_out = ex.extract("out0")

        if ret != 0 or mat_out.empty():
            return []

        # Parse output (simplified — actual parsing depends on YOLOv8 output format)
        return self._parse_ncnn_output(mat_out.numpy(), scale, pad_x, pad_y, image.shape)

    def _parse_ncnn_output(self, detections, scale, pad_x, pad_y, img_shape):
        """Parse NCNN YOLOv8 output into detection format."""
        results = []
        if detections.ndim == 3:
            for i in range(detections.shape[1]):
                det = detections[0, i, :]
                confidence = float(det[4])

                if confidence < self.conf_threshold:
                    continue

                cx = (float(det[0]) - pad_x) / scale
                cy = (float(det[1]) - pad_y) / scale
                bw = float(det[2]) / scale
                bh = float(det[3]) / scale

                x1 = max(0, int(cx - bw / 2))
                y1 = max(0, int(cy - bh / 2))
                x2 = min(img_shape[1], int(cx + bw / 2))
                y2 = min(img_shape[0], int(cy + bh / 2))

                keypoints = []
                for k in range(17):
                    kx = (float(det[5 + k*3]) - pad_x) / scale
                    ky = (float(det[5 + k*3 + 1]) - pad_y) / scale
                    kc = float(det[5 + k*3 + 2])
                    keypoints.append([kx, ky, kc])

                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'keypoints': keypoints
                })

        return results

    def _detect_pytorch(self, image):
        """Run PyTorch YOLO inference."""
        results = self.model(image, conf=self.conf_threshold,
                           imgsz=self.imgsz, verbose=False)

        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and r.keypoints is not None:
                for i in range(len(r.boxes)):
                    box = r.boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(r.boxes.conf[i].cpu().numpy())
                    kpts = r.keypoints[i].cpu().numpy()

                    keypoints = [[float(kpts[k, 0]), float(kpts[k, 1]), float(kpts[k, 2])]
                                for k in range(17)]

                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': conf,
                        'keypoints': keypoints
                    })

        return detections
