import joblib
import pandas as pd

# Load preprocessing info
preprocessor = joblib.load("models/trained/preprocessor.pkl")
columns = preprocessor["columns"]
scaler = preprocessor["scaler"]

# Model paths
model_paths = {
    "KNN": "models/trained/knn_model.pkl",
    "Logistic Regression": "models/trained/logistic_regression_model.pkl",
    "SVM": "models/trained/svm_model.pkl",
    "XGBoost": "models/trained/xgboost_model.pkl"
}

def preprocess_input(user_input_dict):
    df = pd.DataFrame([user_input_dict])

    # Binary mapping
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    # One-hot encoding
    multi_class_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_class_cols)

    # Add missing columns
    for col in columns:
        if col not in df:
            df[col] = 0

    # Reorder columns
    df = df[columns]

    # Scale
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    return df

def predict_churn(user_input_dict, model_name):
    model_path = model_paths.get(model_name)
    if not model_path:
        raise ValueError(f"Unknown model: {model_name}")

    model = joblib.load(model_path)
    processed = preprocess_input(user_input_dict)
    pred = model.predict(processed)
    return int(pred[0])
