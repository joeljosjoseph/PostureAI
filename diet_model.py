from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from _bmi import (
    bmi,
    bmi_category,
    bmi_category_to_csv,
    norm_gender,
    ui_goal_to_csv_goal,
)

MODEL_CANDIDATES = [
    Path(__file__).resolve().parent / "model" / "diet_rf_pipeline.joblib",
    Path(__file__).resolve().parent / "ml_api" / "artifacts" / "diet_rf_pipeline.joblib",
]


def _resolve_model_path() -> Path | None:
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def _gender_for_model(gender: str) -> str:
    """Training data only has Male/Female; map Other to Female for encoding."""
    normalized = norm_gender(gender)
    if normalized == "Other":
        return "Female"
    return normalized


def predict_gym_plan(
    *,
    gender: str,
    goal: str,
    weight_kg: float,
    height_cm: float,
) -> dict[str, Any] | None:
    model_path = _resolve_model_path()
    if model_path is None:
        return None

    bmi_value = bmi(weight_kg, height_cm)
    csv_bmi_category = bmi_category_to_csv(bmi_category(bmi_value))
    csv_goal = ui_goal_to_csv_goal(goal)

    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    label_names = bundle["label_names"]
    estimator = pipe.steps[-1][1] if getattr(pipe, "steps", None) else None
    if estimator is not None and hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1

    row = pd.DataFrame(
        [
            {
                "Gender": _gender_for_model(gender),
                "Goal": csv_goal,
                "BMI Category": csv_bmi_category,
                "bmi": bmi_value,
            }
        ]
    )

    pred_idx = int(pipe.predict(row)[0])
    if pred_idx < 0 or pred_idx >= len(label_names):
        return None

    label = label_names[pred_idx]
    if "|||" not in label:
        return None

    exercise_schedule, meal_plan_focus = label.split("|||", 1)
    return {
        "exercise_schedule": exercise_schedule,
        "meal_plan_focus": meal_plan_focus,
        "csv_goal": csv_goal,
        "csv_bmi_category": csv_bmi_category,
        "bmi_used": bmi_value,
    }
