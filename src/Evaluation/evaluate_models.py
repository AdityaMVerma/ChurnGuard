# evaluate_model.py
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
accuracy_score, balanced_accuracy_score, recall_score,
precision_score, f1_score
)
import pandas as pd

def get_evaluation_results():
    # Load test data
    X_test, y_test = joblib.load("models/trained/test_data.pkl")

    # Define model names and file paths
    model_info = {
        "KNN": "models/trained/knn_model.pkl",
        "Logistic Regression": "models/trained/logistic_regression_model.pkl",
        "SVM": "models/trained/svm_model.pkl",
        "XGBoost": "models/trained/xgboost_model.pkl"
    }

    results = []

    for model_name, model_path in model_info.items():
        model = joblib.load(model_path)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        bal_acc = balanced_accuracy_score(y_test, preds)
        recall = recall_score(y_test, preds)
        precision = precision_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "Balanced Accuracy": bal_acc,
            "Recall": recall,
            "Precision": precision,
            "F1 Score": f1
        })

    return pd.DataFrame(results)
