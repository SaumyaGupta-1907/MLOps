import mlflow, datetime, os
from joblib import dump
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")
    
    # Load Wine dataset and split
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )
    
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Wine Dataset"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"    
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=dataset_name):
        
        params = {
            "dataset_name": dataset_name,
            "number of datapoints": X_train.shape[0] + X_test.shape[0],
            "number of features": X_train.shape[1]
        }
        mlflow.log_params(params)
        
        # Train XGBClassifier
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
        
        # Evaluate
        y_predict = model.predict(X_test)
        mlflow.log_metrics({
            'Accuracy': accuracy_score(y_test, y_predict),
            'F1 Score': f1_score(y_test, y_predict, average='weighted')
        })
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model_filename = f"models/model_{timestamp}_dt_model.joblib"
        dump(model, model_filename)
        print(f"Model saved to {model_filename}")
