import cv2
import time
import math
import os
import warnings
from collections import deque

import numpy as np
import mediapipe as mp
import joblib

from _mediapipe_runtime import ensure_mediapipe_runtime_compatible

# -------------------------------------------------------------------
#  Warning filters (protobuf + sklearn spam)
# -------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ensure_mediapipe_runtime_compatible()

# -------------------------------------------------------------------
#  Paths & window constants
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "workout_pose_model.joblib")

WINDOW_NAME = "PostureAI"
TARGET_W = 1280
TARGET_H = 720

# -------------------------------------------------------------------
#  Mediapipe pose
# -------------------------------------------------------------------
mp_pose = mp.solutions.pose

# For smoother angles & FPS
ANGLE_HISTORY_LEN = 5

BODY_CONNECTIONS = [
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]

# -------------------------------------------------------------------
#  Workout options (menu)
# -------------------------------------------------------------------
WORKOUTS = [
    "Barbell Biceps Curl",   # 1
    "Bench Press",           # 2
    "Chest Fly Machine",     # 3
    "Deadlift",              # 4
    "Decline Bench Press",   # 5
    "Hammer Curl",           # 6
    "Hip Thrust",            # 7
    "Incline Bench Press",   # 8
    "Lat Pulldown",          # 9
    "Lateral Raise",         # 10
    "Leg Extension",         # 11
    "Leg Raises",            # 12
    "Plank",                 # 13
    "Pull Up",               # 14
    "Push Up",               # 15
    "Romanian Deadlift",     # 16
    "Russian Twist",         # 17
    "Shoulder Press",        # 18
    "Squat",                 # 19
    "T Bar Row",             # 20
    "Tricep Dips",           # 21
    "Tricep Pushdown",       # 22
    # The seven UCF-based ones (to match ML labels)
    "BenchPress",            # 23
    "BodyWeightSquats",      # 24
    "JumpingJack",           # 25
    "Lunges",                # 26
    "PullUps",               # 27
    "PushUps",               # 28
    "WallPushups",           # 29
]

AUTO_DETECT_OPTION = 30  # menu option for ML model

# -------------------------------------------------------------------
#  Utility functions
# -------------------------------------------------------------------
def compute_angle(a, b, c):
    """Returns angle at point b given three (x, y) points."""
    ax, ay = a
    bx, by = b
    cx, cy = c

    ab = np.array([ax - bx, ay - by])
    cb = np.array([cx - bx, cy - by])

    ab_norm = np.linalg.norm(ab)
    cb_norm = np.linalg.norm(cb)
    if ab_norm == 0 or cb_norm == 0:
        return 0.0

    cos_angle = np.dot(ab, cb) / (ab_norm * cb_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))
    return angle


def calories_from_reps(reps):
    """Very rough estimation: each rep ~0.25 kcal."""
    return round(reps * 0.25, 1)


def _landmark_to_point(landmarks, idx, width, height, visibility_threshold=0.5):
    lm = landmarks.landmark[idx]
    if getattr(lm, "visibility", 1.0) < visibility_threshold:
        return None
    return int(lm.x * width), int(lm.y * height)


def draw_body_without_face(frame, landmarks):
    """Draw pose landmarks for the body only, ending at a synthetic neck point."""
    h, w, _ = frame.shape
    points = {idx: _landmark_to_point(landmarks, idx, w, h) for idx in range(11, 33)}

    for start_idx, end_idx in BODY_CONNECTIONS:
        start = points.get(start_idx)
        end = points.get(end_idx)
        if start and end:
            cv2.line(frame, start, end, (0, 220, 255), 2, lineType=cv2.LINE_AA)

    left_shoulder = points.get(11)
    right_shoulder = points.get(12)
    neck = None
    if left_shoulder and right_shoulder:
        neck = (
            (left_shoulder[0] + right_shoulder[0]) // 2,
            (left_shoulder[1] + right_shoulder[1]) // 2,
        )
        cv2.line(frame, neck, left_shoulder, (0, 220, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(frame, neck, right_shoulder, (0, 220, 255), 2, lineType=cv2.LINE_AA)

    for point in points.values():
        if point:
            cv2.circle(frame, point, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, point, 6, (0, 140, 255), 1, lineType=cv2.LINE_AA)

    if neck:
        cv2.circle(frame, neck, 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, neck, 7, (0, 200, 255), 1, lineType=cv2.LINE_AA)


# -------------------------------------------------------------------
#  ML model: auto-detect from pose landmarks
# -------------------------------------------------------------------
_ML_MODEL = None
_ML_CLASS_NAMES = None
_FEATURE_INDEXES = [11, 13, 15, 23, 25, 27]  # shoulders, wrists, hips, knees


def _load_ml_model():
    global _ML_MODEL, _ML_CLASS_NAMES
    if _ML_MODEL is not None:
        return

    pack = joblib.load(MODEL_PATH)
    if isinstance(pack, dict):
        _ML_MODEL = pack.get("model", None)
        _ML_CLASS_NAMES = pack.get("class_names", None)
        if _ML_MODEL is None:
            raise RuntimeError("Model pack missing 'model'")
        if _ML_CLASS_NAMES is None:
            _ML_CLASS_NAMES = list(_ML_MODEL.classes_)
    else:
        _ML_MODEL = pack
        _ML_CLASS_NAMES = list(_ML_MODEL.classes_)


def ml_predict_from_landmarks(landmarks, width, height):
    """
    Build the same 12-dim feature vector that we used in training:
    [x11, y11, x13, y13, x15, y15, x23, y23, x25, y25, x27, y27]
    using normalized coordinates from Mediapipe.
    """
    if landmarks is None:
        return None

    _load_ml_model()

    feats = []
    for idx in _FEATURE_INDEXES:
        lm = landmarks.landmark[idx]
        feats.append(lm.x)
        feats.append(lm.y)

    X = np.array(feats, dtype=np.float32).reshape(1, -1)
    pred = _ML_MODEL.predict(X)[0]

    # If the model returns an index, map via class_names; if it returns a string, just use it
    if isinstance(pred, (int, np.integer)) and _ML_CLASS_NAMES is not None:
        label = _ML_CLASS_NAMES[int(pred)]
    else:
        label = str(pred)

    return label


# -------------------------------------------------------------------
#  Rep detection (generic angle-based)
# -------------------------------------------------------------------
class RepCounter:
    """
    Simple angle-based rep counter.
    Uses one main joint angle (knee or elbow) and state machine:
      - going_down when angle < down_threshold
      - rep counted when angle > up_threshold after going_down
    """

    def __init__(self, exercise_name: str):
        self.exercise_name = exercise_name
        self.reps = 0
        self.state = "up"  # 'up' or 'down'
        self.down_threshold, self.up_threshold = self._get_thresholds(exercise_name)
        self.angle_history = deque(maxlen=ANGLE_HISTORY_LEN)

    def _get_thresholds(self, name):
        name_low = name.lower()
        if "squat" in name_low or "lunge" in name_low or "leg" in name_low:
            return 80, 160   # knee angle
        if "push" in name_low or "dip" in name_low:
            return 70, 150   # elbow angle
        if "curl" in name_low or "row" in name_low:
            return 60, 150
        if "press" in name_low or "raise" in name_low:
            return 70, 150
        return 70, 160

    def update(self, angle: float):
        self.angle_history.append(angle)
        avg_angle = sum(self.angle_history) / max(1, len(self.angle_history))

        if self.state == "up":
            if avg_angle < self.down_threshold:
                self.state = "down"
        elif self.state == "down":
            if avg_angle > self.up_threshold:
                self.state = "up"
                self.reps += 1

        return self.reps


# -------------------------------------------------------------------
#  Drawing helpers
# -------------------------------------------------------------------
def draw_hud(frame, workout_name, mode_text, reps, target_reps,
             calories, elapsed_sec, fps, message):
    h, w, _ = frame.shape

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)

    # Left top: mode + workout
    cv2.putText(frame, f"PostureAI  |  {mode_text}",
                (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Workout: {workout_name}",
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Right top: time, fps, calories
    cv2.putText(frame, f"Time: {int(elapsed_sec)}s",
                (w - 260, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, f"FPS: {int(fps)}",
                (w - 260, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.putText(frame, f"Calories: {calories:.1f} kcal",
                (w - 260, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Left mid: reps
    rep_text = f"Reps: {reps}"
    if target_reps > 0:
        rep_text += f" / {target_reps}"
    cv2.putText(frame, rep_text, (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

    # Progress bar on right side if target set
    if target_reps > 0:
        bar_top = 140
        bar_bottom = h - 80
        bar_left = w - 60
        bar_right = w - 30

        cv2.rectangle(frame, (bar_left, bar_top),
                      (bar_right, bar_bottom), (255, 255, 255), 2)

        progress = min(1.0, reps / target_reps) if target_reps > 0 else 0.0
        filled_height = int((bar_bottom - bar_top) * progress)
        cv2.rectangle(frame,
                      (bar_left + 2, bar_bottom - filled_height),
                      (bar_right - 2, bar_bottom - 2),
                      (0, 255, 0), -1)

    # Bottom message area
    cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, message, (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Bottom-left hints
    cv2.putText(frame, "ESC / Q: Exit    R: +1 rep (manual)",
                (20, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)


def get_form_message(exercise_name, angle):
    name_low = exercise_name.lower()

    if "curl" in name_low:
        if angle > 160:
            return "Start with arms fully extended."
        elif angle < 50:
            return "Great! Squeeze at the top of the curl."
        else:
            return "Bring the weight higher to finish the curl."

    if "squat" in name_low or "lunge" in name_low or "leg" in name_low:
        if angle > 170:
            return "Stand tall to finish the rep."
        elif angle < 80:
            return "Good depth! Keep knees tracking over toes."
        else:
            return "Sit back and keep chest up."

    if "push" in name_low or "press" in name_low:
        if angle > 160:
            return "Lock out gently at the top, don't hyperextend."
        elif angle < 80:
            return "Control the lowering, keep elbows under control."
        else:
            return "Maintain controlled, smooth reps."

    return "Maintain controlled, smooth reps."


# -------------------------------------------------------------------
#  Menu & main loop
# -------------------------------------------------------------------
def print_menu():
    print("\n==============================")
    print("       Select Workout")
    print("==============================")
    for idx, name in enumerate(WORKOUTS, start=1):
        print(f" {idx:2d}) {name}")
    print(f" {AUTO_DETECT_OPTION:2d}) Auto-Detect (ML model)")
    print("==============================")


def ask_workout_choice():
    while True:
        try:
            choice = int(input(f"Enter option (1–{AUTO_DETECT_OPTION}): ").strip())
            if 1 <= choice <= AUTO_DETECT_OPTION:
                return choice
        except ValueError:
            pass
        print("❌ Invalid choice. Try again.")


def ask_target_reps():
    while True:
        try:
            reps = int(input("Set target repetitions (0 for no limit): ").strip())
            if reps >= 0:
                print(f"Target reps set to: {reps}")
                return reps
        except ValueError:
            pass
        print("❌ Please enter a valid integer.")


def main():
    print_menu()
    choice = ask_workout_choice()

    auto_mode = (choice == AUTO_DETECT_OPTION)

    if auto_mode:
        workout_name = "Auto-Detect"
        print("\n✔ You selected: Auto-Detect")
    else:
        workout_name = WORKOUTS[choice - 1]
        print(f"\n✔ You selected: {workout_name}")

    target_reps = ask_target_reps()

    # Rep counter
    rep_counter = RepCounter(workout_name)

    # Mediapipe pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, TARGET_W, TARGET_H)

    start_time = time.time()
    last_time = start_time
    fps = 0.0
    reps = 0
    calories = 0.0

    done_by_target = False
    detected_label = ""
    print("\nPostureAI running... Press ESC or Q to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera frame not received. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # FPS
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        landmarks = result.pose_landmarks

        angle = None

        if landmarks:
            draw_body_without_face(frame, landmarks)

            if auto_mode:
                detected_label = ml_predict_from_landmarks(landmarks, w, h) or ""
                workout_for_angle = detected_label if detected_label else workout_name
            else:
                workout_for_angle = workout_name

            lm = landmarks.landmark
            name_low = workout_for_angle.lower()

            if "squat" in name_low or "lunge" in name_low or "leg" in name_low:
                hip = (lm[24].x * w, lm[24].y * h)
                knee = (lm[26].x * w, lm[26].y * h)
                ankle = (lm[28].x * w, lm[28].y * h)
                angle = compute_angle(hip, knee, ankle)
            else:
                shoulder = (lm[12].x * w, lm[12].y * h)
                elbow = (lm[14].x * w, lm[14].y * h)
                wrist = (lm[16].x * w, lm[16].y * h)
                angle = compute_angle(shoulder, elbow, wrist)

            if angle is not None:
                reps = rep_counter.update(angle)
                calories = calories_from_reps(reps)

        if target_reps > 0 and reps >= target_reps:
            done_by_target = True
            print("\n✅ Target reps completed!")
            break

        elapsed_sec = now - start_time

        if auto_mode:
            mode_text = "Auto-Detect (ML)"
            display_workout = detected_label if detected_label else "Detecting..."
        else:
            mode_text = "Manual"
            display_workout = workout_name

        if angle is not None:
            message = get_form_message(display_workout, angle)
        else:
            message = "Make sure your full body is visible in the frame."

        draw_hud(
            frame,
            display_workout,
            mode_text,
            reps,
            target_reps,
            calories,
            elapsed_sec,
            fps,
            message,
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q") or key == ord("Q"):
            print("Exiting session…")
            break
        if key == ord("r") or key == ord("R"):
            reps += 1
            rep_counter.reps = reps
            calories = calories_from_reps(reps)

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    total_time = time.time() - start_time
    print("\n================ Session Summary ================")
    if auto_mode:
        print(f"Mode         : Auto-Detect (ML)")
    else:
        print(f"Workout      : {workout_name}")
    print(f"Final reps   : {reps}")
    if target_reps > 0:
        print(f"Target reps  : {target_reps} ({'Reached' if done_by_target else 'Not reached'})")
    print(f"Calories est.: {calories:.1f} kcal")
    print(f"Time elapsed : {int(total_time)} seconds")
    print("================================================\n")


if __name__ == "__main__":
    main()
