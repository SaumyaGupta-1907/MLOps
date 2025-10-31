import pytest
import os
import sys
import joblib
import json
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# Add src to path if needed
sys.path.insert(0, os.path.abspath('../src'))

def test_wine_data_loaded():
    data = load_wine()
    X, y = data.data, data.target
    assert X.shape[0] > 0
    assert X.shape[1] == 13
    assert len(y) == X.shape[0]

def test_model_training_creates_file(tmp_path):
    timestamp = "pytest"
    model_filename = tmp_path / f"model_{timestamp}_dt_model.joblib"

    # Load Wine dataset and split
    data = load_wine()
    X_train, _, y_train, _ = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

    # Train XGBClassifier with production params
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_filename)
    assert os.path.exists(model_filename)

def test_evaluate_metrics_file(tmp_path):
    timestamp = "pytest"
    model_filename = tmp_path / f"model_{timestamp}_dt_model.joblib"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metrics_file = metrics_dir / f"{timestamp}_metrics.json"

    # Load Wine dataset and split
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

    # Train model
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    joblib.dump(model, model_filename)

    # Simulate evaluation on test set
    loaded_model = joblib.load(model_filename)
    y_pred = loaded_model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1_Score": f1_score(y_test, y_pred, average="weighted")
    }

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    assert metrics_file.exists()
    with open(metrics_file) as f:
        loaded_metrics = json.load(f)
    assert "Accuracy" in loaded_metrics
    assert "F1_Score" in loaded_metrics
    assert 0 <= loaded_metrics["F1_Score"] <= 1
    assert 0 <= loaded_metrics["Accuracy"] <= 1
