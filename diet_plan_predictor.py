import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import json
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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
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
            elif 'bmi' in col_lower or 'category' in col_lower:
                self.bmi_col = col
            elif 'meal' in col_lower or 'plan' in col_lower:
                self.meal_col = col
        
        if not all([self.gender_col, self.goal_col, self.bmi_col, self.meal_col]):
            # Manual fallback for exact column names
            if 'Gender' in columns:
                self.gender_col = 'Gender'
            if 'Goal' in columns:
                self.goal_col = 'Goal'
            if 'BMI Category' in columns:
                self.bmi_col = 'BMI Category'
            if 'Meal Plan' in columns:
                self.meal_col = 'Meal Plan'
        
        if not all([self.gender_col, self.goal_col, self.bmi_col, self.meal_col]):
            raise ValueError(f"Missing required columns in CSV. Found columns: {columns}")
        
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
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
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
        
        print("\n" + "="*70)
        print("         TRAINING DIET PLAN MODELS")
        print("="*70)
        
        # Train and evaluate each model
        for name, model in models_to_train.items():
            print(f"\n🔧 Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.models[name] = model
            results[name] = accuracy
            print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Select best model
        self.best_model_name = max(results, key=results.get)
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n✅ Best Model: {self.best_model_name}")
        print(f"   Best Accuracy: {results[self.best_model_name]:.4f}")
        print("="*70 + "\n")
        
        return results
    
    def evaluate_model(self):
        """
        Evaluate the best model with comprehensive metrics
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not available. Call train_models() first.")
        
        # Make predictions
        y_pred = self.best_model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # For multi-class, use weighted average
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Per-class accuracy
        class_names = self.target_encoder.classes_
        class_accuracies = {}
        
        for i, class_name in enumerate(class_names):
            mask = self.y_test == i
            if mask.sum() > 0:
                class_acc = accuracy_score(self.y_test[mask], y_pred[mask])
                class_accuracies[class_name] = round(class_acc * 100, 2)
        
        # Compile metrics
        metrics = {
            "model_name": self.best_model_name,
            "accuracy": round(accuracy, 4),
            "accuracy_percent": round(accuracy * 100, 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "num_classes": len(class_names),
            "class_names": class_names.tolist(),
            "class_accuracies": class_accuracies,
            "confusion_matrix": cm.tolist(),
            "num_test_samples": len(self.y_test),
            "num_train_samples": len(self.y_train)
        }
        
        return metrics
    
    def print_evaluation_metrics(self, metrics):
        """
        Pretty print evaluation metrics
        
        Args:
            metrics: Dictionary of metrics from evaluate_model()
        """
        print("\n" + "="*70)
        print("         DIET PLAN MODEL EVALUATION METRICS")
        print("="*70)
        
        print(f"\n🤖 Model Information:")
        print(f"   Best Model:                   {metrics['model_name']}")
        print(f"   Training Samples:             {metrics['num_train_samples']}")
        print(f"   Test Samples:                 {metrics['num_test_samples']}")
        print(f"   Number of Diet Plans:         {metrics['num_classes']}")
        
        print(f"\n📊 Overall Performance:")
        print(f"   Accuracy:                     {metrics['accuracy']:.4f} ({metrics['accuracy_percent']:.2f}%)")
        print(f"   Precision (weighted):         {metrics['precision']:.4f}")
        print(f"   Recall (weighted):            {metrics['recall']:.4f}")
        print(f"   F1-Score (weighted):          {metrics['f1_score']:.4f}")
        
        print(f"\n🎯 Per-Class Accuracy:")
        for diet_plan, acc in metrics['class_accuracies'].items():
            bar_length = int(acc / 5)  # Scale to 20 chars max
            bar = "█" * bar_length
            print(f"   {diet_plan:<25} {acc:>6.2f}% {bar}")
        
        print(f"\n📋 Available Diet Plans:")
        for i, plan in enumerate(metrics['class_names'], 1):
            print(f"   {i}. {plan}")
        
        print("="*70 + "\n")
    
    def print_confusion_matrix(self, metrics):
        """
        Print a formatted confusion matrix
        
        Args:
            metrics: Dictionary of metrics from evaluate_model()
        """
        cm = np.array(metrics['confusion_matrix'])
        class_names = metrics['class_names']
        
        # Only print confusion matrix if we have a reasonable number of classes
        if len(class_names) > 20:
            print("\n" + "="*70)
            print("         CONFUSION MATRIX")
            print("="*70)
            print(f"\n⚠️  Too many classes ({len(class_names)}) to display matrix.")
            print("   Confusion matrix saved to JSON report.")
            print("="*70 + "\n")
            return
        
        print("\n" + "="*70)
        print("         CONFUSION MATRIX")
        print("="*70)
        print("\nRows = Actual, Columns = Predicted\n")
        
        # Truncate class names for display
        display_names = [name[:15] + "..." if len(name) > 15 else name for name in class_names]
        
        # Print header
        print(f"{'Actual/Pred':<18}", end="")
        for name in display_names:
            print(f"{name:<18}", end="")
        print()
        print("-" * (18 + 18 * len(display_names)))
        
        # Print rows
        for i, name in enumerate(display_names):
            print(f"{name:<18}", end="")
            for j in range(len(class_names)):
                print(f"{cm[i][j]:<18}", end="")
            print()
        
        print("="*70 + "\n")
    
    def print_classification_report(self):
        """Print detailed classification report"""
        if self.best_model is None or self.X_test is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
        y_pred = self.best_model.predict(self.X_test)
        
        print("\n" + "="*70)
        print("         DETAILED CLASSIFICATION REPORT")
        print("="*70 + "\n")
        
        # Get unique classes that appear in test set
        unique_test_classes = np.unique(np.concatenate([self.y_test, y_pred]))
        test_class_names = self.target_encoder.inverse_transform(unique_test_classes)
        
        try:
            report = classification_report(
                self.y_test, 
                y_pred, 
                labels=unique_test_classes,
                target_names=test_class_names,
                zero_division=0
            )
            print(report)
        except Exception as e:
            print(f"⚠️  Unable to generate full classification report: {e}")
            print("\n   This typically happens when there are too many classes")
            print("   or classes that don't appear in the test set.")
        
        print("="*70 + "\n")
    
    def compare_predictions(self, num_samples=10):
        """
        Compare predictions with actual values for a sample of test data
        
        Args:
            num_samples: Number of samples to display
        """
        if self.best_model is None or self.X_test is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
        # Make predictions
        y_pred = self.best_model.predict(self.X_test)
        
        # Get actual class names
        y_test_names = self.target_encoder.inverse_transform(self.y_test)
        y_pred_names = self.target_encoder.inverse_transform(y_pred)
        
        # Select random samples
        indices = np.random.choice(len(self.y_test), min(num_samples, len(self.y_test)), replace=False)
        
        print("\n" + "="*70)
        print("         SAMPLE PREDICTIONS VS ACTUAL VALUES")
        print("="*70)
        print(f"{'Sample':<8} {'Actual Diet Plan':<25} {'Predicted Diet Plan':<25} {'Match':<8}")
        print("-"*70)
        
        for i, idx in enumerate(indices, 1):
            actual = y_test_names[idx]
            predicted = y_pred_names[idx]
            match = "✓" if actual == predicted else "✗"
            
            print(f"{i:<8} {actual:<25} {predicted:<25} {match:<8}")
        
        print("="*70 + "\n")
    
    def save_evaluation_report(self, metrics, output_file="diet_plan_evaluation_report.json"):
        """
        Save evaluation metrics to a JSON file
        
        Args:
            metrics: Dictionary of metrics from evaluate_model()
            output_file: Path to output JSON file
        """
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"✓ Evaluation report saved to: {output_file}")
    
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


# Example usage and automatic evaluation
if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("         DIET PLAN PREDICTOR - TRAINING & EVALUATION")
        print("="*70)
        
        # Initialize predictor
        print("\n📦 Loading data from diet_data.csv...")
        predictor = DietPlanPredictor('diet_data.csv')
        
        print(f"   Dataset size: {len(predictor.df)} samples")
        print(f"   Detected columns:")
        print(f"      Gender: {predictor.gender_col}")
        print(f"      Goal: {predictor.goal_col}")
        print(f"      BMI: {predictor.bmi_col}")
        print(f"      Meal Plan: {predictor.meal_col}")
        
        # Train models
        results = predictor.train_models()
        
        # Evaluate the best model
        print("\n🧪 Evaluating best model...")
        metrics = predictor.evaluate_model()
        
        # Print all evaluation metrics
        predictor.print_evaluation_metrics(metrics)
        
        # Print confusion matrix
        predictor.print_confusion_matrix(metrics)
        
        # Print detailed classification report
        predictor.print_classification_report()
        
        # Compare predictions
        print("🔍 Comparing predictions with actual values...")
        predictor.compare_predictions(num_samples=15)
        
        # Save evaluation report
        print("\n💾 Saving evaluation report...")
        predictor.save_evaluation_report(metrics, "diet_plan_evaluation_report.json")
        
        # Print summary interpretation
        print("\n" + "="*70)
        print("         EVALUATION SUMMARY & INTERPRETATION")
        print("="*70)
        
        accuracy = metrics['accuracy_percent']
        num_classes = metrics['num_classes']
        num_samples = metrics['num_test_samples']
        
        # Check for data quality issues
        if num_classes == num_samples or num_classes > num_samples * 0.8:
            print("\n⚠️  DATA QUALITY WARNING!")
            print("="*70)
            print(f"   Number of unique diet plans: {num_classes}")
            print(f"   Total samples in dataset: {len(predictor.df)}")
            print(f"   Training samples: {metrics['num_train_samples']}")
            print(f"   Test samples: {num_samples}")
            print("\n❌ PROBLEM: Too many unique diet plans for too few samples!")
            print("\n   Your dataset has {num_classes} different diet plans but only")
            print(f"   {len(predictor.df)} total samples. This means most diet plans appear")
            print("   only once, so the model cannot learn patterns.")
            print("\n💡 SOLUTIONS:")
            print("   1. RECOMMENDED: Simplify your meal plans into categories")
            print("      (e.g., 'High Protein', 'Low Carb', 'Balanced', 'High Calorie')")
            print("      instead of having 48 different specific meal plans.")
            print("\n   2. Collect more data - you need at least 10-20 examples per")
            print("      diet plan category for the model to learn effectively.")
            print("\n   3. Group similar meal plans together based on:")
            print("      - Calorie range (1200-1500, 1500-2000, 2000-2500, etc.)")
            print("      - Protein content (Low, Medium, High)")
            print("      - Goal alignment (Weight Loss, Muscle Gain, Maintenance)")
            print("="*70)
        elif accuracy >= 90:
            print("✅ Excellent model performance (Accuracy ≥ 90%)")
            print("   The model is highly reliable for diet plan recommendations.")
        elif accuracy >= 80:
            print("✅ Good model performance (Accuracy ≥ 80%)")
            print("   The model performs well for most predictions.")
        elif accuracy >= 70:
            print("⚠️  Acceptable model performance (Accuracy ≥ 70%)")
            print("   The model is usable but could benefit from more training data.")
        else:
            print("❌ Poor model performance (Accuracy < 70%)")
            print("   Consider collecting more diverse training data.")
        
        if accuracy > 0:
            print(f"\n📊 The model correctly predicts {accuracy:.1f}% of diet plans")
            print(f"🤖 Using {metrics['model_name']} as the best model")
            print(f"📋 Can recommend from {metrics['num_classes']} different diet plans")
        
        # Example prediction
        print("\n" + "="*70)
        print("         EXAMPLE PREDICTION")
        print("="*70)
        
        example_gender = predictor.get_valid_values(predictor.gender_col)[0]
        example_goal = predictor.get_valid_values(predictor.goal_col)[0]
        example_bmi = predictor.get_valid_values(predictor.bmi_col)[0]
        
        print(f"\n🔮 Input:")
        print(f"   Gender: {example_gender}")
        print(f"   Goal: {example_goal}")
        print(f"   BMI Category: {example_bmi}")
        
        diet_plan = predictor.predict_diet_plan(example_gender, example_goal, example_bmi)
        print(f"\n🍽️  Recommended Diet Plan: {diet_plan}")
        
        print("\n" + "="*70)
        print("✅ Evaluation complete! Report saved to 'diet_plan_evaluation_report.json'")
        print("="*70 + "\n")
        
    except FileNotFoundError:
        print("\n⚠️  diet_data.csv not found.")
        print("   Please ensure the CSV file is in the current directory.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()