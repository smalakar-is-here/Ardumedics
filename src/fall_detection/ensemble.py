"""
ArduMedics Pose Ensemble — Weighted Ensemble of YOLOv8-Pose Models

Combines predictions from YOLOv8n/s/m-Pose models using
IoU-based detection matching and confidence-weighted keypoint averaging.
"""

import numpy as np


class PoseEnsemble:
    """
    Weighted ensemble of YOLOv8-Pose models.

    For each image:
    1. Run inference with each model
    2. Match person detections using IoU
    3. Average keypoints from matched detections (weighted by model confidence)
    4. Apply NMS on bounding boxes
    5. Return fused predictions with ensemble confidence

    Args:
        models: Dict of model_name -> YOLO model instance
        weights: Dict of model_name -> reliability_weight
        conf_threshold: Minimum confidence for predictions
        iou_threshold: IoU threshold for matching detections
    """

    def __init__(self, models, weights=None, conf_threshold=0.25, iou_threshold=0.5):
        self.models = models
        self.weights = weights or {name: 1.0/len(models) for name in models}
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

    def predict(self, image, imgsz=640):
        """
        Run ensemble prediction on an image.

        Args:
            image: numpy array (H, W, 3) BGR
            imgsz: Input image size

        Returns:
            list of dicts: [{bbox, confidence, keypoints}]
        """
        all_detections = {}

        # Run each model
        for name, model in self.models.items():
            results = model(image, conf=self.conf_threshold, imgsz=imgsz, verbose=False)
            detections = self._parse_results(results)
            all_detections[name] = detections

        # Fuse detections
        return self._fuse_detections(all_detections)

    def _parse_results(self, results):
        """Parse YOLO results into detection format."""
        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and r.keypoints is not None:
                for i in range(len(r.boxes)):
                    box = r.boxes.xyxy[i].cpu().numpy()
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

    def _fuse_detections(self, all_detections):
        """Fuse detections from multiple models using IoU matching."""
        # Use the model with highest weight as reference
        ref_model = max(self.weights, key=self.weights.get)
        ref_dets = all_detections.get(ref_model, [])

        if not ref_dets:
            # Fall back to any model with detections
            for name, dets in all_detections.items():
                if dets:
                    ref_dets = dets
                    ref_model = name
                    break

        if not ref_dets:
            return []

        fused = []

        for ref_det in ref_dets:
            matched_kpts = [ref_det['keypoints']]
            matched_confs = [ref_det['confidence'] * self.weights.get(ref_model, 0.33)]
            matched_weights = [self.weights.get(ref_model, 0.33)]

            # Match with other models
            for name, dets in all_detections.items():
                if name == ref_model:
                    continue

                best_iou = 0
                best_match = None

                for det in dets:
                    iou = self._compute_iou(ref_det['bbox'], det['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = det

                if best_iou > self.iou_threshold and best_match:
                    matched_kpts.append(best_match['keypoints'])
                    matched_confs.append(best_match['confidence'] * self.weights.get(name, 0.33))
                    matched_weights.append(self.weights.get(name, 0.33))

            # Weighted average keypoints
            fused_kpts = self._weighted_average_keypoints(
                matched_kpts, matched_confs, matched_weights)

            # Ensemble confidence = max of weighted confidences
            fused_conf = sum(matched_confs) / sum(matched_weights)

            fused.append({
                'bbox': ref_det['bbox'],
                'confidence': fused_conf,
                'keypoints': fused_kpts,
                'num_models': len(matched_kpts)
            })

        return fused

    @staticmethod
    def _compute_iou(box1, box2):
        """Compute IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

        union = area1 + area2 - intersection
        return intersection / max(union, 1e-6)

    @staticmethod
    def _weighted_average_keypoints(kpts_list, confs, weights):
        """Compute confidence-weighted average of keypoints."""
        total_weight = sum(w * c for w, c in zip(weights, confs))
        if total_weight == 0:
            total_weight = 1.0

        fused = []
        for k in range(17):
            wx = sum(w * c * kpts[k][0] for w, c, kpts in zip(weights, confs, kpts_list))
            wy = sum(w * c * kpts[k][1] for w, c, kpts in zip(weights, confs, kpts_list))
            wc = sum(w * c * kpts[k][2] for w, c, kpts in zip(weights, confs, kpts_list))

            fused.append([wx/total_weight, wy/total_weight, wc/total_weight])

        return fused
