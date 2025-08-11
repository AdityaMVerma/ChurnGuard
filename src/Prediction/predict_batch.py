import pandas as pd
import joblib

# Load model paths
model_paths = {
    "KNN": "models/trained/knn_model.pkl",
    "Logistic Regression": "models/trained/logistic_regression_model.pkl",
    "SVM": "models/trained/svm_model.pkl",
    "XGBoost": "models/trained/xgboost_model.pkl"
}

# Binary mappings
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}

# Categorical columns to one-hot encode
multi_class_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

def preprocess_batch(df_input, preprocessor):
    try:
        # Drop customerID if present
        if "customerID" in df_input.columns:
            df_input.drop("customerID", axis=1, inplace=True)

        # Convert TotalCharges to numeric
        df_input['TotalCharges'] = pd.to_numeric(df_input['TotalCharges'], errors='coerce')
        df_input['TotalCharges'].fillna(df_input['TotalCharges'].median(), inplace=True)

        # Apply binary mapping
        for col in binary_cols:
            if col in df_input.columns:
                df_input[col] = df_input[col].map(binary_map)

        # One-hot encoding
        df_input = pd.get_dummies(df_input, columns=multi_class_cols)

        # Align columns with training data
        expected_columns = preprocessor['columns']
        for col in expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0  # Add missing column with 0

        df_input = df_input[expected_columns]  # Reorder

        # Apply scaler
        scaler = preprocessor['scaler']
        df_input[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
            df_input[['tenure', 'MonthlyCharges', 'TotalCharges']]
        )

        return df_input

    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return None

def predict_batch(file, model_name):
    try:
        # Load input file
        df_original = pd.read_csv(file)

        # Load preprocessor
        preprocessor = joblib.load("models/trained/preprocessor.pkl")

        # Preprocess
        X_processed = preprocess_batch(df_original.copy(), preprocessor)
        if X_processed is None:
            return None

        # Load model
        model_path = model_paths.get(model_name)
        if model_path is None:
            raise ValueError("Invalid model selected.")

        model = joblib.load(model_path)

        # Predict
        predictions = model.predict(X_processed)
        prediction_labels = ["Churn" if p == 1 else "No Churn" for p in predictions]

        # Add predictions to original DataFrame
        df_original["Prediction"] = prediction_labels

        return df_original

    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return None
