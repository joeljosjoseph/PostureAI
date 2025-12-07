# fastapi_server.py
import io
import os
import time
import math
import warnings
from collections import deque
from typing import Optional

import numpy as np
import cv2
import joblib
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

# Import diet predictor
from diet_plan_predictor import DietPlanPredictor
# Import hydration predictor
from hydration_predictor import HydrationPredictor

# -------------------------------------------------------------------
# config
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "workout_pose_model.joblib")
DIET_DATA_PATH = os.path.join(BASE_DIR, "diet_data.csv")
HYDRATION_MODEL_DIR = os.path.join(BASE_DIR, "model", "hydration_model.pkl")

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -------------------------------------------------------------------
# Mediapipe
# -------------------------------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# smoothing
ANGLE_HISTORY_LEN = 5

# -------------------------------------------------------------------
# model loader (optional)
# -------------------------------------------------------------------
_ML_MODEL = None
_ML_CLASS_NAMES = None
_FEATURE_INDEXES = [11, 13, 15, 23, 25, 27]  # shoulders, wrists, hips, knees


def _load_ml_model():
    global _ML_MODEL, _ML_CLASS_NAMES
    if _ML_MODEL is not None:
        return
    if not os.path.exists(MODEL_PATH):
        print("ML model not found; auto-detect disabled. Place joblib at:", MODEL_PATH)
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
        try:
            _ML_CLASS_NAMES = list(_ML_MODEL.classes_)
        except Exception:
            _ML_CLASS_NAMES = None
    print("ML model loaded. Class names:", _ML_CLASS_NAMES)


def ml_predict_from_landmarks(landmarks):
    if landmarks is None:
        return None
    _load_ml_model()
    if _ML_MODEL is None:
        return None

    feats = []
    for idx in _FEATURE_INDEXES:
        lm = landmarks.landmark[idx]
        feats.append(lm.x)
        feats.append(lm.y)

    X = np.array(feats, dtype=np.float32).reshape(1, -1)
    pred = _ML_MODEL.predict(X)[0]

    if isinstance(pred, (int, np.integer)) and _ML_CLASS_NAMES is not None:
        label = _ML_CLASS_NAMES[int(pred)]
    else:
        label = str(pred)
    return label


# -------------------------------------------------------------------
# Diet Predictor Loader
# -------------------------------------------------------------------
_DIET_PREDICTOR = None

def _load_diet_predictor():
    """Load and train the diet predictor model"""
    global _DIET_PREDICTOR
    if _DIET_PREDICTOR is not None:
        return
    if not os.path.exists(DIET_DATA_PATH):
        print("Diet data CSV not found at:", DIET_DATA_PATH)
        return
    try:
        print("Loading diet predictor model...")
        _DIET_PREDICTOR = DietPlanPredictor(DIET_DATA_PATH)
        _DIET_PREDICTOR.train_models()
        print("Diet predictor model loaded successfully!")
    except Exception as e:
        print(f"Error loading diet predictor: {e}")
        _DIET_PREDICTOR = None


# -------------------------------------------------------------------
# Hydration Predictor Loader
# -------------------------------------------------------------------
_HYDRATION_PREDICTOR = None

def _load_hydration_predictor():
    """Load the hydration predictor model"""
    global _HYDRATION_PREDICTOR
    if _HYDRATION_PREDICTOR is not None:
        return
    try:
        print("Loading hydration predictor model...")
        _HYDRATION_PREDICTOR = HydrationPredictor(HYDRATION_MODEL_DIR)
        print("Hydration predictor model loaded successfully!")
    except Exception as e:
        print(f"Error loading hydration predictor: {e}")
        _HYDRATION_PREDICTOR = None


# -------------------------------------------------------------------
# utilities (from your original code)
# -------------------------------------------------------------------
def compute_angle(a, b, c):
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
    return round(reps * 0.25, 1)


class RepCounter:
    def __init__(self, exercise_name: str):
        self.exercise_name = exercise_name
        self.reps = 0
        self.state = "up"
        self.down_threshold, self.up_threshold = self._get_thresholds(exercise_name)
        self.angle_history = deque(maxlen=ANGLE_HISTORY_LEN)

    def _get_thresholds(self, name):
        name_low = name.lower()
        if "squat" in name_low or "lunge" in name_low or "leg" in name_low:
            return 80, 160
        if "push" in name_low or "dip" in name_low:
            return 70, 150
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
# Pydantic Models for Diet Plan
# -------------------------------------------------------------------
class DietPlanRequest(BaseModel):
    gender: str
    goal: str
    weight_kg: float
    height_cm: float

class DietPlanResponse(BaseModel):
    gender: str
    goal: str
    weight_kg: float
    height_cm: float
    bmi: float
    bmi_category: str
    meal_plan: str


# -------------------------------------------------------------------
# Pydantic Models for Hydration
# -------------------------------------------------------------------
class HydrationRequest(BaseModel):
    age: int
    weight: float  # kg
    height: float  # cm
    humidity: float  # percentage
    temperature: float  # celsius
    workout_goal: str  # 'Build Muscle', 'Lose Weight', 'Get Fit', 'Improve Endurance'
    season: str  # 'Spring', 'Summer', 'Autumn', 'Winter'

class HydrationResponse(BaseModel):
    age: int
    weight: float
    height: float
    humidity: float
    temperature: float
    workout_goal: str
    season: str
    recommended_intake_ml: int
    recommended_intake_liters: float


# -------------------------------------------------------------------
# FastAPI app & session store
# -------------------------------------------------------------------
app = FastAPI(title="PostureAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fitness-assistant-ai.vercel.app",
        "https://*.vercel.app",  # For preview deployments
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# keep a simple in-memory session store for rep counters per client session
SESSIONS = {}  # session_id -> dict { rep_counter, workout_name, target_reps, last_time, pose }
# WARNING: in-memory store is ephemeral. Use redis for production.


# -------------------------------------------------------------------
# Startup Event
# -------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    _load_ml_model()
    _load_diet_predictor()
    _load_hydration_predictor()


# -------------------------------------------------------------------
# Root Endpoint
# -------------------------------------------------------------------
@app.get("/")
def root():
    """API root endpoint"""
    return {
        "message": "PostureAI API with Diet Plan and Hydration Predictor",
        "endpoints": {
            "posture": ["/create_session", "/analyze", "/reset_session"],
            "diet": ["/diet/info", "/diet/predict", "/diet/calculate-bmi"],
            "hydration": ["/hydration/info", "/hydration/predict"]
        }
    }


# -------------------------------------------------------------------
# Posture Detection Endpoints (Original)
# -------------------------------------------------------------------
@app.post("/create_session")
def create_session(workout_name: Optional[str] = Form("Manual"),
                   mode: Optional[str] = Form("manual"),
                   target_reps: Optional[int] = Form(0)):
    session_id = str(uuid.uuid4())
    repc = RepCounter(workout_name)
    # create a Mediapipe Pose instance per session (lightweight)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    SESSIONS[session_id] = {
        "rep_counter": repc,
        "workout_name": workout_name,
        "mode": mode,
        "target_reps": target_reps,
        "pose": pose,
        "last_time": time.time(),
        "fps": 0.0,
    }
    return {"session_id": session_id}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...),
                  session_id: Optional[str] = Form(None),
                  workout_name: Optional[str] = Form(None),
                  mode: Optional[str] = Form(None),
                  target_reps: Optional[int] = Form(None)):
    """
    Expects a multipart/form-data POST with:
      - file: image bytes (jpeg/png)
      - session_id: optional session id (string); if not provided we treat it stateless
      - workout_name: optional string (overrides session)
      - mode: "auto" or "manual"
      - target_reps: optional int
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Unable to decode image"}

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # select session
    session = None
    if session_id and session_id in SESSIONS:
        session = SESSIONS[session_id]
    elif session_id and session_id not in SESSIONS:
        # create session with defaults
        repc = RepCounter(workout_name or "Manual")
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        session = {
            "rep_counter": repc,
            "workout_name": workout_name or "Manual",
            "mode": mode or "manual",
            "target_reps": int(target_reps or 0),
            "pose": pose,
            "last_time": time.time(),
            "fps": 0.0,
        }
        SESSIONS[session_id] = session

    # fallback if no session
    if session is None:
        repc = RepCounter(workout_name or "Manual")
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        session = {
            "rep_counter": repc,
            "workout_name": workout_name or "Manual",
            "mode": mode or "manual",
            "target_reps": int(target_reps or 0),
            "pose": pose,
            "last_time": time.time(),
            "fps": 0.0,
        }

    # compute FPS (light smoothing)
    now = time.time()
    dt = now - session["last_time"] if "last_time" in session else 0.0
    session["last_time"] = now
    if dt > 0:
        session["fps"] = 0.9 * session.get("fps", 0.0) + 0.1 * (1.0 / dt) if session.get("fps", 0.0) > 0 else (1.0 / dt)

    pose = session["pose"]
    result = pose.process(rgb)
    landmarks = result.pose_landmarks

    detected_label = ""
    angle = None
    reps = session["rep_counter"].reps
    calories = calories_from_reps(reps)

    # First try ML auto-detect if requested
    use_auto = (mode == "auto") or (session.get("mode") == "auto")
    if landmarks:
        if use_auto:
            try:
                detected_label = ml_predict_from_landmarks(landmarks) or ""
            except Exception as e:
                detected_label = ""
        workout_for_angle = detected_label if detected_label else (workout_name or session.get("workout_name", "Manual"))
        lm = landmarks.landmark
        name_low = (workout_for_angle or "").lower()
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
            reps = session["rep_counter"].update(angle)
            calories = calories_from_reps(reps)

    message = get_form_message(detected_label if detected_label else (workout_name or session.get("workout_name", "Manual")), angle or 0)

    response = {
        "angle": angle,
        "reps": reps,
        "calories": calories,
        "detected_label": detected_label,
        "message": message,
        "fps": session.get("fps", 0.0),
        "session_id": session_id,
    }

    # check target
    if session.get("target_reps", 0) > 0 and reps >= session.get("target_reps", 0):
        response["done_by_target"] = True

    return response


@app.post("/reset_session")
def reset_session(session_id: str = Form(...)):
    if session_id in SESSIONS:
        # re-create rep counter & pose
        workout = SESSIONS[session_id].get("workout_name", "Manual")
        SESSIONS[session_id]["rep_counter"] = RepCounter(workout)
        SESSIONS[session_id]["last_time"] = time.time()
        return {"ok": True}
    return {"ok": False, "error": "session not found"}


# -------------------------------------------------------------------
# Diet Plan Endpoints
# -------------------------------------------------------------------
@app.get("/diet/info")
def get_diet_info():
    """Get available options for diet plan prediction"""
    if _DIET_PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Diet predictor not available")
    
    return {
        "genders": _DIET_PREDICTOR.get_valid_values(_DIET_PREDICTOR.gender_col),
        "goals": _DIET_PREDICTOR.get_valid_values(_DIET_PREDICTOR.goal_col),
        "bmi_categories": _DIET_PREDICTOR.get_valid_values(_DIET_PREDICTOR.bmi_col)
    }


@app.post("/diet/predict", response_model=DietPlanResponse)
async def predict_diet_plan(request: DietPlanRequest):
    """
    Predict diet plan based on user characteristics
    
    - **gender**: Male or Female
    - **goal**: Build Muscle, Lose Weight, Get Fit, or Improve Endurance
    - **weight_kg**: Weight in kilograms
    - **height_cm**: Height in centimeters
    """
    if _DIET_PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Diet predictor not available")
    
    try:
        # Calculate BMI
        bmi, bmi_category = _DIET_PREDICTOR.calculate_bmi(request.weight_kg, request.height_cm)
        
        # Get prediction
        meal_plan = _DIET_PREDICTOR.predict_diet_plan(
            gender=request.gender,
            goal=request.goal,
            bmi_category=bmi_category
        )
        
        return DietPlanResponse(
            gender=request.gender,
            goal=request.goal,
            weight_kg=request.weight_kg,
            height_cm=request.height_cm,
            bmi=round(bmi, 1),
            bmi_category=bmi_category,
            meal_plan=meal_plan
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/diet/calculate-bmi")
async def calculate_bmi(weight_kg: float = Form(...), height_cm: float = Form(...)):
    """Calculate BMI from weight and height"""
    if _DIET_PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Diet predictor not available")
    
    try:
        bmi, bmi_category = _DIET_PREDICTOR.calculate_bmi(weight_kg, height_cm)
        return {
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "bmi": round(bmi, 1),
            "bmi_category": bmi_category
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------------------------------------------
# Hydration Prediction Endpoints
# -------------------------------------------------------------------
@app.get("/hydration/info")
def get_hydration_info():
    """Get information about hydration predictor and valid input values"""
    if _HYDRATION_PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Hydration predictor not available")
    
    return _HYDRATION_PREDICTOR.get_hydration_info()


@app.post("/hydration/predict", response_model=HydrationResponse)
async def predict_hydration(request: HydrationRequest):
    """
    Predict optimal daily water intake based on user characteristics and environment
    
    - **age**: Age in years (18-65)
    - **weight**: Weight in kilograms (50-100)
    - **height**: Height in centimeters (150-195)
    - **humidity**: Humidity percentage (30-90)
    - **temperature**: Temperature in Celsius (10-40)
    - **workout_goal**: One of: Build Muscle, Lose Weight, Get Fit, Improve Endurance
    - **season**: One of: Spring, Summer, Autumn, Winter
    """
    if _HYDRATION_PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Hydration predictor not available")
    
    try:
        # Get prediction
        intake_ml = _HYDRATION_PREDICTOR.predict_hydration(
            age=request.age,
            weight=request.weight,
            height=request.height,
            humidity=request.humidity,
            temperature=request.temperature,
            workout_goal=request.workout_goal,
            season=request.season
        )
        
        intake_liters = round(intake_ml / 1000, 2)
        
        return HydrationResponse(
            age=request.age,
            weight=request.weight,
            height=request.height,
            humidity=request.humidity,
            temperature=request.temperature,
            workout_goal=request.workout_goal,
            season=request.season,
            recommended_intake_ml=intake_ml,
            recommended_intake_liters=intake_liters
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)