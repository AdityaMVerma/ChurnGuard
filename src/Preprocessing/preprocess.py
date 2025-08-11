import pandas as pd
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
import joblib



#load the Data
df=pd.read_csv('../../data/telco_churn.csv')

#Pre-processing of Data
df.drop("customerID", axis=1, inplace=True)

# Convert to numeric and coerce errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Handle missing values
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

#binary classification of yes/no/male/female
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling', 'Churn']

for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

#multiple class classification categorcial
multi_class_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=multi_class_cols)

#standard scaling
scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
    df[['tenure', 'MonthlyCharges', 'TotalCharges']])

#all in int/float format
df = df.astype({col: 'int' for col in df.select_dtypes('bool').columns})

#data splitting
X = df.drop('Churn', axis=1)
y = df['Churn']

df.to_csv("../../data/processed_telco_churn.csv", index=False)


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)
joblib.dump((X_test, y_test), "../../models/trained/test_data.pkl")


# Save column names and scaler
joblib.dump({
    "columns": X_train.columns.tolist(),
    "scaler": scaler  # StandardScaler used on numerical features
}, "../../models/trained/preprocessor.pkl")

'''from models.scripts.LR import train_logistic_regression
lr_model = train_logistic_regression(X_train, y_train, X_test, y_test)

from models.scripts.KNN import train_knn_model
knn_model = train_knn_model(X_train, y_train, X_test, y_test)

from models.scripts.SVM  import train_svm_model
svm_model = train_svm_model(X_train, y_train, X_test, y_test)

from models.scripts.XGBOOST import train_xgboost_model
xgb_model = train_xgboost_model(X_train, y_train, X_test, y_test) '''





