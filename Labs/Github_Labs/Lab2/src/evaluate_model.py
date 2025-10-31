import os, json
from sklearn.metrics import accuracy_score, f1_score
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    timestamp = args.timestamp

    # Load model from models folder
    model_path = os.path.join("models", f"model_{timestamp}_dt_model.joblib")
    if not os.path.exists(model_path):
        raise ValueError(f'Model not found at {model_path}')
    model = joblib.load(model_path)
    
    # Load Wine dataset and split
    data = load_wine()
    _, X_test, _, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
    
    # Predict
    y_predict = model.predict(X_test)
    
    # Compute metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_predict),
        "F1_Score": f1_score(y_test, y_predict, average="weighted")
    }
    
    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    metrics_file_path = os.path.join("metrics", f"{timestamp}_metrics.json")
    with open(metrics_file_path, 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    print(f"Metrics saved to {metrics_file_path}")
