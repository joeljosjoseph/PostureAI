"""Shared BMI helpers for training and inference."""


def bmi(weight_kg: float, height_cm: float) -> float:
    h_m = height_cm / 100.0
    if h_m <= 0:
        return 0.0
    return round(weight_kg / (h_m * h_m), 1)


def bmi_category(bmi_val: float) -> str:
    if bmi_val < 18.5:
        return "Underweight"
    if bmi_val < 25:
        return "Normal"
    if bmi_val < 30:
        return "Overweight"
    return "Obese"


def bmi_category_to_csv(cat: str) -> str:
    value = (cat or "").strip()
    if value == "Normal":
        return "Normal weight"
    if value == "Obese":
        return "Obesity"
    return value


def ui_goal_to_csv_goal(goal: str) -> str:
    value = (goal or "").strip()
    if value == "Lose Weight":
        return "fat_burn"
    if value in ("Get Fit", "Improve Endurance", "Build Muscle"):
        return "muscle_gain"
    return "muscle_gain"


def norm_gender(gender: str) -> str:
    value = (gender or "").strip().lower()
    if value == "male":
        return "Male"
    if value == "female":
        return "Female"
    if value == "other":
        return "Other"
    return "Female"
