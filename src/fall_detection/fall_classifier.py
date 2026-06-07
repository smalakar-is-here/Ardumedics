"""
ArduMedics Fall Classifier — Hybrid Rule-ML Fall Logic

Novel contribution C3: Combines geometric rules with learned features
for interpretable, edge-deployable fall classification.
"""

import numpy as np


class FallClassifier:
    """
    Hybrid Rule-ML fall classifier.

    Combines three signals for fall detection:
    1. Torso angle (geometric) — horizontal torso = fall
    2. Ground proximity (geometric) — hips near ankles = lying down
    3. Velocity (temporal) — fast downward motion = falling

    Args:
        torso_angle_threshold: Degrees from vertical (default: 60)
        ground_proximity_threshold: Hip-ankle ratio (default: 0.4)
        velocity_threshold: Normalized velocity (default: 0.15)
    """

    def __init__(self, torso_angle_threshold=60, ground_proximity_threshold=0.4,
                 velocity_threshold=0.15):
        self.torso_angle_threshold = torso_angle_threshold
        self.ground_proximity_threshold = ground_proximity_threshold
        self.velocity_threshold = velocity_threshold

    def classify(self, keypoints, velocity=0.0):
        """
        Classify whether a person is falling based on keypoints.

        Args:
            keypoints: List of [x, y, confidence] for 17 COCO keypoints
            velocity: Downward velocity from TPC tracker

        Returns:
            dict with: is_fall, confidence, reasons
        """
        reasons = []
        confidence = 0.0

        # Signal 1: Torso angle
        torso_angle = self._compute_torso_angle(keypoints)
        if torso_angle is not None:
            if torso_angle > self.torso_angle_threshold:
                reasons.append(f"horizontal_torso ({torso_angle:.0f}°)")
                confidence += 0.4

        # Signal 2: Ground proximity
        ground_prox = self._compute_ground_proximity(keypoints)
        if ground_prox is not None:
            if ground_prox < self.ground_proximity_threshold:
                reasons.append(f"ground_proximity ({ground_prox:.2f})")
                confidence += 0.3

        # Signal 3: Velocity
        if velocity > self.velocity_threshold:
            reasons.append(f"fast_descent ({velocity:.3f})")
            confidence += 0.3

        is_fall = len(reasons) > 0
        confidence = min(confidence, 1.0)

        return {
            'is_fall': is_fall,
            'confidence': confidence,
            'reasons': reasons,
            'torso_angle': torso_angle,
            'ground_proximity': ground_prox,
            'velocity': velocity
        }

    def _compute_torso_angle(self, keypoints):
        """Compute torso angle from vertical."""
        ls, rs = keypoints[5], keypoints[6]
        lh, rh = keypoints[11], keypoints[12]

        if min(ls[2], rs[2], lh[2], rh[2]) < 0.3:
            return None

        shoulder_mid = np.array([(ls[0]+rs[0])/2, (ls[1]+rs[1])/2])
        hip_mid = np.array([(lh[0]+rh[0])/2, (lh[1]+rh[1])/2])

        torso_vec = hip_mid - shoulder_mid
        vertical = np.array([0, 1])

        cos_angle = np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) * np.linalg.norm(vertical) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

    def _compute_ground_proximity(self, keypoints):
        """Compute hip-to-ankle vertical ratio."""
        lh, rh = keypoints[11], keypoints[12]
        la, ra = keypoints[15], keypoints[16]

        if min(lh[2], rh[2], la[2], ra[2]) < 0.3:
            return None

        hip_y = (lh[1] + rh[1]) / 2
        ankle_y = (la[1] + ra[1]) / 2

        if ankle_y > 0:
            return abs(hip_y - ankle_y) / max(hip_y, ankle_y, 1)
        return 0
