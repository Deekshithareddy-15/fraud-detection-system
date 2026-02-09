
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import os
import sys

# Add src to path if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import load_data, clean_data, preprocess_features, split_data

# Constants
DATA_PATH = "data/raw/creditcard.csv"
MODEL_SAVE_PATH = "models/saved_models/"
METRICS_PATH = "models/metrics/"

def train_models():
    # 1. Load Data
    print("Loading data...")
    df = load_data(DATA_PATH)
    if df is None:
        print("Data loading failed. Exiting.")
        return

    # 2. Preprocessing
    print("Preprocessing data...")
    df = clean_data(df)
    
    # Feature scaling (Amount, Time)
    # Note: In a real pipeline, we fit the scaler on train only to prevent leakage.
    # However, for simplicity and since scaler is robust, we'll demonstrate a fit_transform on full 
    # and then split. Typically, fit on X_train, transform X_test is better.
    # Let's do it correctly: split first, then scale.
    
    # 3. Train/Test Split
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features properly (fit on train, transform test)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    
    # Scale 'Amount' and 'Time'
    # Assuming columns are consistent. RobustScaler handles outliers.
    cols_to_scale = ['Amount', 'Time']
    
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    # 4. Handle Imbalance with SMOTE (on training data only)
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Ensure DataFrame to keep column names for XGBoost consistency
    if not isinstance(X_train_resampled, pd.DataFrame):
        X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
        
    print(f"Training data shape after SMOTE: {X_train_resampled.shape}")
    
    # 5. Model Definitions
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_model = None
    best_f1 = 0.0
    best_name = ""
    
    os.makedirs(METRICS_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # 6. Train and Evaluate
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        precision_recall_auc = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        
        print(f"{name} Results:")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  PR AUC: {precision_recall_auc:.4f}")
        print("-" * 30)
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
    
    print(f"\nBest Model: {best_name} with F1-Score: {best_f1:.4f}")
    
    # 7. Save Best Model
    model_filename = os.path.join(MODEL_SAVE_PATH, "best_fraud_model.pkl")
    scaler_filename = os.path.join(MODEL_SAVE_PATH, "scaler.pkl")
    
    joblib.dump(best_model, model_filename)
    joblib.dump(scaler, scaler_filename)
    
    print(f"Model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")

    # 8. Save Baseline Stats for Drift Detection
    print("Saving baseline statistics...")
    baseline_stats = X_train_resampled.describe()
    baseline_stats.to_json(os.path.join(MODEL_SAVE_PATH, "baseline_stats.json"))
    print("Baseline statistics saved.")

if __name__ == "__main__":
    train_models()
