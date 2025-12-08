import numpy as np
import joblib
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    
    def evaluate_model(self, X_test, y_test, is_scaled=False):
        """
        Evaluate model performance on test data
        
        Args:
            X_test: Test features (numpy array or list)
            y_test: True hydration values (numpy array or list)
            is_scaled: bool, whether X_test is already scaled
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Convert to numpy arrays if needed
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Debug: Print shapes
        print(f"   X_test shape: {X_test.shape}")
        print(f"   y_test shape: {y_test.shape}")
        print(f"   Expected features: {len(self.feature_columns)}")
        
        # Scale features if not already scaled
        if not is_scaled:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Debug: Print sample predictions
        print(f"   Sample actual values: {y_test[:5]}")
        print(f"   Sample predictions: {y_pred[:5]}")
        
        # Calculate basic metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Calculate additional metrics
        max_error = np.max(np.abs(y_test - y_pred))
        min_error = np.min(np.abs(y_test - y_pred))
        mean_error = np.mean(y_test - y_pred)
        std_error = np.std(y_test - y_pred)
        
        # Calculate percentage of predictions within certain thresholds
        within_5_percent = np.sum(np.abs((y_test - y_pred) / y_test) <= 0.05) / len(y_test) * 100
        within_10_percent = np.sum(np.abs((y_test - y_pred) / y_test) <= 0.10) / len(y_test) * 100
        within_15_percent = np.sum(np.abs((y_test - y_pred) / y_test) <= 0.15) / len(y_test) * 100
        
        # Compile all metrics
        metrics = {
            "mae": round(mae, 2),
            "mse": round(mse, 2),
            "rmse": round(rmse, 2),
            "r2_score": round(r2, 4),
            "mape": round(mape, 2),
            "max_error": round(max_error, 2),
            "min_error": round(min_error, 2),
            "mean_error": round(mean_error, 2),
            "std_error": round(std_error, 2),
            "within_5_percent": round(within_5_percent, 2),
            "within_10_percent": round(within_10_percent, 2),
            "within_15_percent": round(within_15_percent, 2),
            "num_samples": len(y_test)
        }
        
        return metrics
    
    def print_evaluation_metrics(self, metrics):
        """
        Pretty print evaluation metrics
        
        Args:
            metrics: Dictionary of metrics from evaluate_model()
        """
        print("\n" + "="*60)
        print("          HYDRATION MODEL EVALUATION METRICS")
        print("="*60)
        print(f"\n📊 Overall Performance:")
        print(f"   R² Score:                     {metrics['r2_score']:.4f}")
        print(f"   Mean Absolute Error (MAE):    {metrics['mae']:.2f} ml")
        print(f"   Root Mean Square Error (RMSE): {metrics['rmse']:.2f} ml")
        print(f"   Mean Square Error (MSE):      {metrics['mse']:.2f}")
        
        print(f"\n📈 Percentage-based Metrics:")
        print(f"   Mean Absolute % Error (MAPE): {metrics['mape']:.2f}%")
        print(f"   Predictions within  5%:       {metrics['within_5_percent']:.2f}%")
        print(f"   Predictions within 10%:       {metrics['within_10_percent']:.2f}%")
        print(f"   Predictions within 15%:       {metrics['within_15_percent']:.2f}%")
        
        print(f"\n🎯 Error Distribution:")
        print(f"   Maximum Error:                {metrics['max_error']:.2f} ml")
        print(f"   Minimum Error:                {metrics['min_error']:.2f} ml")
        print(f"   Mean Prediction Error:        {metrics['mean_error']:.2f} ml")
        print(f"   Std Dev of Errors:            {metrics['std_error']:.2f} ml")
        
        print(f"\n📝 Test Set Size:                {metrics['num_samples']} samples")
        print("="*60 + "\n")
    
    def compare_predictions(self, X_test, y_test, num_samples=10, is_scaled=False):
        """
        Compare predictions with actual values for a sample of test data
        
        Args:
            X_test: Test features
            y_test: True hydration values
            num_samples: Number of samples to display
            is_scaled: Whether X_test is already scaled
        
        Returns:
            None (prints comparison)
        """
        # Convert to numpy arrays
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Scale if needed
        if not is_scaled:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Select random samples
        indices = np.random.choice(len(y_test), min(num_samples, len(y_test)), replace=False)
        
        print("\n" + "="*70)
        print("          SAMPLE PREDICTIONS VS ACTUAL VALUES")
        print("="*70)
        print(f"{'Sample':<8} {'Actual (ml)':<15} {'Predicted (ml)':<18} {'Error (ml)':<15} {'Error %':<10}")
        print("-"*70)
        
        for i, idx in enumerate(indices, 1):
            actual = y_test[idx]
            predicted = y_pred[idx]
            error = predicted - actual
            error_pct = (abs(error) / actual) * 100
            
            print(f"{i:<8} {actual:<15.0f} {predicted:<18.0f} {error:<15.1f} {error_pct:<10.2f}%")
        
        print("="*70 + "\n")
    
    def save_evaluation_report(self, metrics, output_file="hydration_evaluation_report.json"):
        """
        Save evaluation metrics to a JSON file
        
        Args:
            metrics: Dictionary of metrics from evaluate_model()
            output_file: Path to output JSON file
        """
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"✓ Evaluation report saved to: {output_file}")
    
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


# Example usage and testing
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Initialize predictor
    predictor = HydrationPredictor(model_dir="model")
    
    # Example: Single prediction
    print("\n🔮 Single Prediction Example:")
    prediction = predictor.predict_hydration(
        age=30,
        weight=75,
        height=175,
        humidity=60,
        temperature=25,
        workout_goal='Build Muscle',
        season='Summer'
    )
    print(f"Predicted daily water intake: {prediction} ml")
    
    # Automatic Model Evaluation
    try:
        print("\n" + "="*70)
        print("         RUNNING AUTOMATIC MODEL EVALUATION")
        print("="*70)
        
        # Load the hydration data
        print("\n📊 Loading test data from hydration_data.csv...")
        df = pd.read_csv('hydration_data.csv')
        
        # Prepare features (using your CSV column names)
        feature_cols = ['age', 'weight', 'height', 'humidity', 
                        'temperature', 'workout_type', 'workout_duration', 
                        'season']
        
        X = df[feature_cols].copy()
        y = df['total_intake'].values
        
        # Split the data (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   Test set size: {len(y_test)} samples")
        
        # Prepare test features with encoding and feature engineering
        print("\n🔧 Preparing features for evaluation...")
        X_processed = X_test.copy()
        
        # Encode categorical features
        X_processed['workout_type_encoded'] = predictor.label_encoders['workout_type'].transform(X_test['workout_type'])
        X_processed['season_encoded'] = predictor.label_encoders['season'].transform(X_test['season'])
        
        # Calculate engineered features
        X_processed['bmi'] = X_test['weight'] / ((X_test['height'] / 100) ** 2)
        X_processed['temp_humidity_index'] = X_test['temperature'] * X_test['humidity'] / 100
        X_processed['workout_intensity'] = X_processed['workout_duration'] * X_processed['workout_type_encoded']
        
        # Reorder columns to match training (using encoded columns)
        feature_order = ['age', 'weight', 'height', 'humidity', 'temperature',
                         'workout_type_encoded', 'workout_duration', 'season_encoded',
                         'bmi', 'temp_humidity_index', 'workout_intensity']
        
        X_test_final = X_processed[feature_order].values
        
        # Evaluate the model
        print("\n🧪 Evaluating model performance...")
        metrics = predictor.evaluate_model(X_test_final, y_test, is_scaled=False)
        
        # Print metrics
        predictor.print_evaluation_metrics(metrics)
        
        # Compare some predictions
        print("🔍 Comparing predictions with actual values...")
        predictor.compare_predictions(X_test_final, y_test, num_samples=15)
        
        # Save evaluation report
        predictor.save_evaluation_report(metrics, "hydration_evaluation_report.json")
        
        # Print summary interpretation
        print("\n" + "="*70)
        print("         EVALUATION SUMMARY & INTERPRETATION")
        print("="*70)
        
        if metrics['r2_score'] >= 0.90:
            print("✅ Excellent model performance (R² ≥ 0.90)")
            print("   The model explains >90% of variance in water intake.")
        elif metrics['r2_score'] >= 0.80:
            print("✅ Good model performance (R² ≥ 0.80)")
            print("   The model is reliable for most predictions.")
        elif metrics['r2_score'] >= 0.70:
            print("⚠️  Acceptable model performance (R² ≥ 0.70)")
            print("   The model is usable but could be improved.")
        else:
            print("❌ Poor model performance (R² < 0.70)")
            print("   Consider retraining with more data or features.")
        
        print(f"\n📏 Average prediction error: {metrics['mae']:.0f} ml ({metrics['mape']:.1f}%)")
        print(f"📊 {metrics['within_10_percent']:.1f}% of predictions within 10% accuracy")
        print(f"🎯 {metrics['within_5_percent']:.1f}% of predictions within 5% accuracy")
        
        print("\n" + "="*70)
        print("✅ Evaluation complete! Report saved to 'hydration_evaluation_report.json'")
        print("="*70 + "\n")
        
    except FileNotFoundError:
        print("\n⚠️  hydration_data.csv not found. Skipping automatic evaluation.")
        print("   To run evaluation, ensure hydration_data.csv is in the current directory.")
    except Exception as e:
        print(f"\n⚠️  Evaluation failed: {str(e)}")
        print("   You can still use the predictor for individual predictions.")