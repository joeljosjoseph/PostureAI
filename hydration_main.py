import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic hydration dataset"""
    np.random.seed(42)
    
    workout_types = ['Build Muscle', 'Lose Weight', 'Get Fit', 'Improve Endurance']
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    data = {
        'age': np.random.randint(18, 65, n_samples),
        'weight': np.random.uniform(50, 100, n_samples),
        'height': np.random.uniform(150, 195, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'temperature': np.random.uniform(10, 40, n_samples),
        'workout_type': np.random.choice(workout_types, n_samples),
        'workout_duration': np.random.uniform(20, 120, n_samples),
        'season': np.random.choice(seasons, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate target based on realistic formula
    base_intake = 2000
    weight_factor = df['weight'] * 30
    temp_factor = df['temperature'] * 20
    humidity_factor = df['humidity'] * 5
    workout_factor = df['workout_duration'] * 10
    age_factor = (65 - df['age']) * 10
    
    df['total_intake'] = (
        base_intake + 
        weight_factor + 
        temp_factor + 
        humidity_factor + 
        workout_factor + 
        age_factor +
        np.random.normal(0, 200, n_samples)
    )
    
    df['total_intake'] = df['total_intake'].clip(1500, 6000)
    
    return df

def main():
    print("="*50)
    print("HYDRATION MODEL TRAINING PIPELINE")
    print("="*50)
    
    # Load dataset
    print("\n[1/11] Loading dataset from CSV...")
    df = pd.read_csv('hydration_data.csv')
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    
    # Preprocess data
    print("\n[2/11] Preprocessing data...")
    label_encoders = {}
    
    le_workout = LabelEncoder()
    df['workout_type_encoded'] = le_workout.fit_transform(df['workout_type'])
    label_encoders['workout_type'] = le_workout
    
    le_season = LabelEncoder()
    df['season_encoded'] = le_season.fit_transform(df['season'])
    label_encoders['season'] = le_season
    
    # Feature engineering
    print("\n[3/11] Engineering features...")
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['temp_humidity_index'] = df['temperature'] * df['humidity'] / 100
    df['workout_intensity'] = df['workout_duration'] * df['workout_type_encoded']
    
    # Prepare features
    print("\n[4/11] Preparing features and target...")
    feature_columns = [
        'age', 'weight', 'height', 'humidity', 'temperature',
        'workout_type_encoded', 'workout_duration', 'season_encoded',
        'bmi', 'temp_humidity_index', 'workout_intensity'
    ]
    
    X = df[feature_columns]
    y = df['total_intake']
    
    # Split data
    print("\n[5/11] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    print("\n[6/11] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n[7/11] Training models...")
    
    # Linear Regression
    print("\n→ Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    
    print(f"  RMSE: {lr_rmse:.2f} ml | MAE: {lr_mae:.2f} ml | R²: {lr_r2:.4f}")
    
    # Random Forest
    print("\n→ Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    print(f"  RMSE: {rf_rmse:.2f} ml | MAE: {rf_mae:.2f} ml | R²: {rf_r2:.4f}")
    
    # Feature importance
    print("\n[8/11] Analyzing feature importance...")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.to_string(index=False))
    
    # Save artifacts
    print("\n[9/11] Saving models and artifacts...")
    best_model = rf_model
    
    joblib.dump(best_model, 'hydration_model.pkl')
    print("✓ hydration_model.pkl")
    
    joblib.dump(scaler, 'hydration_scaler.pkl')
    print("✓ hydration_scaler.pkl")
    
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("✓ label_encoders.pkl")
    
    with open('feature_columns.json', 'w') as f:
        json.dump(feature_columns, f)
    print("✓ feature_columns.json")
    
    metadata = {
        'model_type': 'Random Forest',
        'rmse': float(rf_rmse),
        'mae': float(rf_mae),
        'r2_score': float(rf_r2),
        'feature_columns': feature_columns,
        'workout_type_mapping': dict(zip(le_workout.classes_, le_workout.transform(le_workout.classes_).tolist())),
        'season_mapping': dict(zip(le_season.classes_, le_season.transform(le_season.classes_).tolist()))
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✓ model_metadata.json")
    
    # Define prediction function
    print("\n[10/11] Setting up prediction function...")
    
    def predict_hydration(age, weight, height, humidity, temperature, 
                          workout_goal, season):
        workout_duration_map = {
            'Build Muscle': 65,
            'Lose Weight': 45,
            'Get Fit': 40,
            'Improve Endurance': 85
        }
        workout_duration = workout_duration_map.get(workout_goal, 50)
        
        workout_type_encoded = le_workout.transform([workout_goal])[0]
        season_encoded = le_season.transform([season])[0]
        
        bmi = weight / ((height / 100) ** 2)
        temp_humidity_index = temperature * humidity / 100
        workout_intensity = workout_duration * workout_type_encoded
        
        features = np.array([[
            age, weight, height, humidity, temperature,
            workout_type_encoded, workout_duration, season_encoded,
            bmi, temp_humidity_index, workout_intensity
        ]])
        
        features_scaled = scaler.transform(features)
        prediction = best_model.predict(features_scaled)[0]
        
        return round(prediction)
    
    print("\n[11/11] Training complete!")
    print("\n" + "="*50)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()