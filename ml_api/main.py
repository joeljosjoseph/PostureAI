"""Single FastAPI entrypoint for fitness ML and lightweight heuristic endpoints."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from diet_model import predict_gym_plan
from fridge_model import detect_items_from_upload

app = FastAPI(title="AI Fitness Assistant — local ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Posture: in-memory sessions ---
_sessions: dict[str, dict[str, Any]] = {}


def _bmi(weight_kg: float, height_cm: float) -> float:
    h_m = height_cm / 100.0
    if h_m <= 0:
        return 0.0
    return round(weight_kg / (h_m * h_m), 1)


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


@app.post("/create_session")
async def create_session(
    workout_name: str = Form(...),
    mode: str = Form(...),
    target_reps: str = Form(...),
) -> dict[str, str]:
    try:
        target = max(1, int(target_reps))
    except ValueError:
        target = 10
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "workout_name": workout_name,
        "mode": mode,
        "target_reps": target,
        "frames": 0,
        "reps": 0,
    }
    return {"session_id": sid}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    workout_name: str = Form(...),
    mode: str = Form(...),
    target_reps: str = Form(...),
) -> dict[str, Any]:
    _ = await file.read()
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    s["frames"] = s.get("frames", 0) + 1
    target = s.get("target_reps", 10)
    # Simulate a rep roughly every 12 frames (~2.4s at 5 FPS)
    if s["frames"] % 12 == 0 and s.get("reps", 0) < target:
        s["reps"] = s.get("reps", 0) + 1

    reps = s.get("reps", 0)
    done = reps >= target
    return {
        "reps": reps,
        "calories": round(reps * 0.45, 2),
        "angle": 78.0 + (s["frames"] % 15),
        "message": "Target reached — great work!" if done else "Keep your form steady",
        "fps": 5,
        "detected_label": workout_name,
        "done_by_target": done,
    }


@app.post("/reset_session")
async def reset_session(session_id: str = Form(...)) -> dict[str, str]:
    if session_id in _sessions:
        _sessions[session_id]["frames"] = 0
        _sessions[session_id]["reps"] = 0
    return {"status": "ok"}


# --- Diet ---
@app.get("/diet/info")
async def diet_info() -> dict[str, list[str]]:
    return {
        "genders": ["Male", "Female", "Other"],
        "goals": [
            "Lose Weight",
            "Build Muscle",
            "Get Fit",
            "Improve Endurance",
        ],
    }


class DietPredictIn(BaseModel):
    gender: str
    goal: str
    weight_kg: float = Field(gt=0)
    height_cm: float = Field(gt=0)


@app.post("/diet/predict")
async def diet_predict(body: DietPredictIn) -> dict[str, Any]:
    bmi = _bmi(body.weight_kg, body.height_cm)
    cat = _bmi_category(bmi)

    # Rough daily targets by goal (demo)
    if body.goal == "Lose Weight":
        mult, protein_g_per_kg = 22, 1.8
        plan_cat = "Moderate deficit, high protein"
    elif body.goal == "Build Muscle":
        mult, protein_g_per_kg = 32, 2.0
        plan_cat = "Lean bulk, performance fuel"
    elif body.goal == "Improve Endurance":
        mult, protein_g_per_kg = 30, 1.6
        plan_cat = "Carb-aware endurance support"
    else:
        mult, protein_g_per_kg = 28, 1.6
        plan_cat = "Balanced maintenance"

    calories = int(body.weight_kg * mult)
    protein = int(body.weight_kg * protein_g_per_kg)

    meal_plan_details = (
        f"Breakfast: Oats with yogurt and berries (~{int(calories * 0.25)} kcal) | "
        f"Lunch: Grilled chicken salad with whole grains (~{int(calories * 0.35)} kcal) | "
        f"Dinner: Fish or tofu, vegetables, rice (~{int(calories * 0.30)} kcal) | "
        f"Snack: Fruit and nuts (~{int(calories * 0.10)} kcal) | "
        f"Total: aim near {calories} kcal with {protein}g protein (adjust portions to hunger)"
    )

    gym = predict_gym_plan(
        gender=body.gender,
        goal=body.goal,
        weight_kg=body.weight_kg,
        height_cm=body.height_cm,
    )

    return {
        "gender": body.gender,
        "goal": body.goal,
        "bmi": bmi,
        "bmi_category": cat,
        "meal_plan_category": plan_cat,
        "calories": calories,
        "protein": protein,
        "meal_plan_details": meal_plan_details,
        "gym": gym,
    }


@app.post("/fridge/detect")
async def fridge_detect(
    image: UploadFile = File(...),
    conf: float = Form(0.25),
) -> dict[str, Any]:
    try:
        items = detect_items_from_upload(image, conf=conf)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await image.close()

    return {"items": items}


# --- Hydration ---
class HydrationIn(BaseModel):
    age: int
    weight: int
    height: int
    humidity: int
    temperature: int
    workout_goal: str
    season: str


@app.post("/hydration/predict")
async def hydration_predict(body: HydrationIn) -> dict[str, Any]:
    # Base: ml/kg (WHO-style guideline-ish, simplified)
    base_per_kg = 33
    if body.workout_goal in ("Build Muscle", "Improve Endurance"):
        base_per_kg += 4
    if body.temperature > 28:
        base_per_kg += int((body.temperature - 28) * 0.8)
    if body.humidity > 60:
        base_per_kg += int((body.humidity - 60) * 0.05)
    if body.season.lower() in ("summer", "hot"):
        base_per_kg += 3

    ml = int(body.weight * base_per_kg)
    ml = max(1500, min(5000, ml))
    liters = round(ml / 1000, 2)
    return {
        "recommended_intake_ml": ml,
        "recommended_intake_liters": liters,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
