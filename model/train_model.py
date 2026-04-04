# model/train_model.py
# Train and save the Random Forest phishing detection model

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

# Allow importing from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.feature_extraction import extract_features, get_feature_names

# Paths
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'phishing_data.csv')
MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'phishing_model.pkl')


def load_and_prepare_data(csv_path: str):
    """
    Load dataset from CSV and extract features for each URL.
    Returns feature matrix X and label vector y.
    """
    print(f"[INFO] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}\n")

    X, y = [], []
    skipped = 0

    for idx, row in df.iterrows():
        url   = str(row['url']).strip()
        label = int(row['label'])
        try:
            features = extract_features(url)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"[WARN] Skipping row {idx} ({url}): {e}")
            skipped += 1

    print(f"[INFO] Processed {len(X)} URLs ({skipped} skipped)\n")
    return np.array(X), np.array(y)


def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    Returns the trained model.
    """
    print("[INFO] Splitting data into train/test sets (80/20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Training Random Forest Classifier ...")
    model = RandomForestClassifier(
        n_estimators=200,       # number of trees
        max_depth=None,         # grow trees until leaves are pure
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1               # use all CPU cores
    )
    model.fit(X_train, y_train)

    # ---- Evaluation ----
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    print("=" * 55)
    print("           MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Test Accuracy      : {accuracy * 100:.2f}%")
    print(f"  ROC-AUC Score      : {roc_auc:.4f}")
    print(f"  5-Fold CV Accuracy : {cv_scores.mean() * 100:.2f}% "
          f"(± {cv_scores.std() * 100:.2f}%)")
    print("-" * 55)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print("=" * 55)

    # Feature importance
    feature_names = get_feature_names()
    importances   = model.feature_importances_
    sorted_idx    = np.argsort(importances)[::-1]

    print("\nTop 10 Most Important Features:")
    for i in range(min(10, len(feature_names))):
        idx = sorted_idx[i]
        print(f"  {i+1:2}. {feature_names[idx]:<30} {importances[idx]:.4f}")
    print()

    return model


def save_model(model: RandomForestClassifier, path: str):
    """Persist the trained model to disk using joblib."""
    joblib.dump(model, path)
    print(f"[INFO] Model saved to: {path}")


def main():
    print("\n" + "=" * 55)
    print("   AI-Based Phishing Website Detection — Training")
    print("=" * 55 + "\n")

    # 1. Load data
    X, y = load_and_prepare_data(DATASET_PATH)

    # 2. Train model
    model = train_model(X, y)

    # 3. Save model
    save_model(model, MODEL_PATH)

    print("\n[DONE] Training complete. Model is ready for inference.\n")


if __name__ == '__main__':
    main()
