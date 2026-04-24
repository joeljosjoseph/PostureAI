import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import mediapipe as mp
import math
from _mediapipe_runtime import ensure_mediapipe_runtime_compatible

ensure_mediapipe_runtime_compatible()

mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """
    Angle at point b, using points a (first joint), b (middle), c (end joint).
    Points are (x, y) in pixel coordinates.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    angle = math.degrees(
        math.atan2(cy - by, cx - bx) -
        math.atan2(ay - by, ax - bx)
    )

    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle


class PostureDetector:
    """Wraps MediaPipe Pose and provides joint angles."""

    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame):
        """
        Returns:
          landmarks (mediapipe object or None),
          frame (unchanged),
          angles: dict with elbow, knee, hip angles (or {})
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None, frame, {}

        h, w = frame.shape[:2]
        lm = results.pose_landmarks.landmark

        def get(idx):
            return int(lm[idx].x * w), int(lm[idx].y * h)

        # Use left side of the body (11–15–23–25–27)
        shoulder = get(11)
        elbow = get(13)
        wrist = get(15)
        hip = get(23)
        knee = get(25)
        ankle = get(27)

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)

        angles = {
            "elbow": elbow_angle,
            "knee": knee_angle,
            "hip":  hip_angle,
        }

        return results.pose_landmarks, frame, angles

    def close(self):
        self.pose.close()
