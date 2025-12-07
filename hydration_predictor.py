import numpy as np
import joblib
import json
import os

class HydrationPredictor:
    """Predict daily water intake based on user characteristics and environment"""
    
    def __init__(self, model_dir="."):
        """
        Initialize the hydration predictor
        
        Args:
            model_dir: Directory containing the trained model files
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        self.metadata = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all necessary model artifacts"""
        try:
            # Load model
            model_path = os.path.join(self.model_dir, 'hydration_model.pkl')
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'hydration_scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            
            # Load label encoders
            encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
            self.label_encoders = joblib.load(encoders_path)
            
            # Load feature columns
            features_path = os.path.join(self.model_dir, 'feature_columns.json')
            with open(features_path, 'r') as f:
                self.feature_columns = json.load(f)
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            print("✓ Hydration predictor models loaded successfully")
            print(f"  Model type: {self.metadata['model_type']}")
            print(f"  R² Score: {self.metadata['r2_score']:.4f}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load hydration models: {e}")
    
    def get_valid_workout_goals(self):
        """Get list of valid workout goals"""
        if self.metadata and 'workout_type_mapping' in self.metadata:
            return list(self.metadata['workout_type_mapping'].keys())
        return ['Build Muscle', 'Lose Weight', 'Get Fit', 'Improve Endurance']
    
    def get_valid_seasons(self):
        """Get list of valid seasons"""
        if self.metadata and 'season_mapping' in self.metadata:
            return list(self.metadata['season_mapping'].keys())
        return ['Spring', 'Summer', 'Autumn', 'Winter']
    
    def predict_hydration(self, age, weight, height, humidity, temperature, 
                          workout_goal, season):
        """
        Predict optimal daily water intake
        
        Args:
            age: int (years)
            weight: float (kg)
            height: float (cm)
            humidity: float (%)
            temperature: float (°C)
            workout_goal: str ('Build Muscle', 'Lose Weight', 'Get Fit', 'Improve Endurance')
            season: str ('Spring', 'Summer', 'Autumn', 'Winter')
        
        Returns:
            int: Predicted daily water intake in milliliters (ml)
        """
        # Validate inputs
        if workout_goal not in self.get_valid_workout_goals():
            raise ValueError(f"Invalid workout goal. Must be one of: {self.get_valid_workout_goals()}")
        
        if season not in self.get_valid_seasons():
            raise ValueError(f"Invalid season. Must be one of: {self.get_valid_seasons()}")
        
        # Map workout goal to duration
        workout_duration_map = {
            'Build Muscle': 65,
            'Lose Weight': 45,
            'Get Fit': 40,
            'Improve Endurance': 85
        }
        workout_duration = workout_duration_map.get(workout_goal, 50)
        
        # Encode categorical features
        le_workout = self.label_encoders['workout_type']
        le_season = self.label_encoders['season']
        
        workout_type_encoded = le_workout.transform([workout_goal])[0]
        season_encoded = le_season.transform([season])[0]
        
        # Calculate engineered features
        bmi = weight / ((height / 100) ** 2)
        temp_humidity_index = temperature * humidity / 100
        workout_intensity = workout_duration * workout_type_encoded
        
        # Create feature array in correct order
        features = np.array([[
            age, weight, height, humidity, temperature,
            workout_type_encoded, workout_duration, season_encoded,
            bmi, temp_humidity_index, workout_intensity
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        return int(round(prediction))
    
    def get_hydration_info(self):
        """Get information about the model and valid inputs"""
        return {
            "model_type": self.metadata.get('model_type', 'Unknown'),
            "r2_score": self.metadata.get('r2_score', 0),
            "rmse": self.metadata.get('rmse', 0),
            "mae": self.metadata.get('mae', 0),
            "valid_workout_goals": self.get_valid_workout_goals(),
            "valid_seasons": self.get_valid_seasons(),
            "input_ranges": {
                "age": "18-65 years",
                "weight": "50-100 kg",
                "height": "150-195 cm",
                "humidity": "30-90%",
                "temperature": "10-40°C"
            }
        }