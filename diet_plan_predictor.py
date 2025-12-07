import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class DietPlanPredictor:
    def __init__(self, csv_file):
        """Initialize the Diet Plan Predictor"""
        self.df = pd.read_csv(csv_file)
        self.models = {}
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        
        # Auto-detect column names
        self.detect_columns()
        
    def detect_columns(self):
        """Automatically detect the correct column names"""
        columns = self.df.columns.tolist()
        
        self.gender_col = None
        self.goal_col = None
        self.bmi_col = None
        self.meal_col = None
        
        for col in columns:
            col_lower = col.lower().strip()
            if 'gender' in col_lower:
                self.gender_col = col
            elif 'goal' in col_lower:
                self.goal_col = col
            elif 'bmi' in col_lower:
                self.bmi_col = col
            elif 'meal' in col_lower or 'plan' in col_lower:
                self.meal_col = col
        
        if not all([self.gender_col, self.goal_col, self.bmi_col, self.meal_col]):
            raise ValueError("Missing required columns in CSV")
        
        self.feature_columns = [self.gender_col, self.goal_col, self.bmi_col]
        
    def preprocess_data(self):
        """Encode categorical variables"""
        # Encode features
        for column in self.feature_columns:
            le = LabelEncoder()
            self.df[f'{column}_encoded'] = le.fit_transform(self.df[column].astype(str))
            self.label_encoders[column] = le
        
        # Encode target variable
        self.df['target_encoded'] = self.target_encoder.fit_transform(self.df[self.meal_col].astype(str))
        
        # Prepare X and y
        X = self.df[[f'{col}_encoded' for col in self.feature_columns]]
        y = self.df['target_encoded']
        
        # Check if we have enough samples for stratification
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        
        # Use stratify only if we have at least 2 samples per class
        if min_count >= 2:
            return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_models(self):
        """Train multiple models and select the best one"""
        X_train, X_test, y_train, y_test = self.preprocess_data()
        
        # Define models
        models_to_train = {
            'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models_to_train.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.models[name] = model
            results[name] = accuracy
        
        # Select best model
        self.best_model_name = max(results, key=results.get)
        self.best_model = self.models[self.best_model_name]
        
        return results
    
    def get_valid_values(self, column):
        """Get valid values for a column"""
        return list(self.label_encoders[column].classes_)
    
    def calculate_bmi(self, weight_kg, height_cm):
        """Calculate BMI and return category"""
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 25:
            category = "Normal"
        elif 25 <= bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
        
        return bmi, category
    
    def predict_diet_plan(self, gender, goal, bmi_category):
        """Predict diet plan based on user inputs"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")

        # Encode input features
        encoded_features = []
        input_data = [gender, goal, bmi_category]

        for column, value in zip(self.feature_columns, input_data):
            encoded_value = self.label_encoders[column].transform([value])[0]
            encoded_features.append(encoded_value)

        # Make prediction
        X_input = np.array(encoded_features).reshape(1, -1)
        prediction_encoded = self.best_model.predict(X_input)[0]
        return self.target_encoder.inverse_transform([prediction_encoded])[0]