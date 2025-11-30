# Breast Cancer Classification with MLflow Lab

## Whats new that I have added?
This project trains and evaluates multiple machine learning models on the Breast Cancer dataset while tracking everything through MLflow.
* Loaded and Used the Breast Cancer dataset from sklearn
* Trained four models: Random Forest, Gradient Boosting, SVM, Logistic Regression
* Tuned hyperparameters using GridSearchCV
* Logged all metrics, parameters, and artifacts with MLflow
* Selected the best model based on evaluation metrics
* Generated evaluation reports and visualizations and stored them reports/ folder
* Saved the best model for production use
* Served the model through a simple REST API
* Generated and tracked plots for different model performance and stored in the plots/ directory

## Dataset
* **Source:** sklearn.datasets.load_breast_cancer
* **Samples:** 569 (455 training, 114 test)
* **Features:** 30 numeric features
* **Classes:** 2 (Malignant, Benign)
* **Preprocessing:** StandardScaler normalization

## Models Trained
1. **Random Forest** - GridSearchCV over 3×4×3 = 36 configurations
2. **Gradient Boosting** - GridSearchCV over 3×3×3 = 27 configurations
3. **SVM** - GridSearchCV over 3×2×2 = 12 configurations
4. **Logistic Regression** - GridSearchCV over 3×1×2 = 6 configurations

## MLflow Tracking Features

### Metrics Logged
* Training accuracy
* Test accuracy, precision, recall, F1 score
* Per-class accuracy
* Prediction confidence scores

### Parameters Logged
* All hyperparameters from GridSearchCV
* Model type
* Best parameter combinations

### Artifacts Logged (for each model)
* Confusion matrix plots
* Feature importance plots (for tree-based models)
* Prediction distribution charts
* Classification reports (text)
* Trained model objects

### Comparison Artifacts
* Model performance comparison grid (2×2 metrics)
* Grouped bar chart (all models, all metrics)
* Best model detailed analysis
* Error analysis visualization

We have Automated model training and comparison with MLFlow. All code and respective results can be seen in mlflow_labs.ipynb file , with reports in reports/ directory , plots in plots/ directory and screenshots of mlflow UI can also be found in the same directory. In  mlflow_labs.ipynb file we can also see results of the api exposed by serving mlflow. Results of different runs can be seen in mlruns directory.
