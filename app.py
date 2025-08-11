import streamlit as st
from src.Prediction.predict import predict_churn

st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("📞 Telecom Churn Prediction")

# Tabs for navigation
tabs = st.tabs(["📊 EDA", "🤖 Predict", "📁 Batch Prediction","Model Evaluation"])

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=0.8)  # Smaller font size

@st.cache_data
def load_data():
    return pd.read_csv("data/telco_churn.csv")

df = load_data()

with tabs[0]:
    st.title("📊 Exploratory Data Analysis")

    with st.expander("💡 Monthly Charges vs. Churn", expanded=True):
        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=80)
        sns.histplot(data=df, x="MonthlyCharges", hue="Churn", kde=True, bins=30, palette="Set2", ax=ax)
        st.pyplot(fig, bbox_inches="tight")

    with st.expander("📈 Tenure Distribution by Churn", expanded=False):
        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=80)
        sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, palette="coolwarm", common_norm=False, alpha=0.5, ax=ax)
        st.pyplot(fig, bbox_inches="tight")

    with st.expander("📬 Churn by Payment Method", expanded=False):
        fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
        sns.countplot(data=df, y="PaymentMethod", hue="Churn", palette="Set3", ax=ax)
        st.pyplot(fig, bbox_inches="tight")

    with st.expander("📞 Phone vs Internet Service", expanded=False):
        fig, ax = plt.subplots(figsize=(4.5, 2.2), dpi=80)
        sns.countplot(data=df, x="PhoneService", hue="InternetService", palette="Set1", ax=ax)
        st.pyplot(fig, bbox_inches="tight")

    with st.expander("📊 Churn Rate by Contract Type", expanded=False):
        contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index')
        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=80)
        contract_churn.plot(kind='bar', stacked=True, ax=ax, colormap="Accent", edgecolor="black")
        ax.set_ylabel("Proportion")
        ax.set_title("Contract Type vs Churn")
        st.pyplot(fig, bbox_inches="tight")


with tabs[1]:
    st.title("Predict single customer churn")
    # Model choice
    model_choice = st.selectbox(
        "Select a Model",
        ["KNN", "Logistic Regression", "SVM", "XGBoost"]
    )

    # User input fields
    user_input = {}
    user_input['gender'] = st.selectbox("Gender", ['Male', 'Female'])
    user_input['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1])
    user_input['Partner'] = st.selectbox("Partner", ['Yes', 'No'])
    user_input['Dependents'] = st.selectbox("Dependents", ['Yes', 'No'])
    user_input['PhoneService'] = st.selectbox("Phone Service", ['Yes', 'No'])
    user_input['MultipleLines'] = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
    user_input['InternetService'] = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    user_input['OnlineSecurity'] = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    user_input['OnlineBackup'] = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    user_input['DeviceProtection'] = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    user_input['TechSupport'] = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
    user_input['StreamingTV'] = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    user_input['StreamingMovies'] = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
    user_input['Contract'] = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    user_input['PaperlessBilling'] = st.selectbox("Paperless Billing", ['Yes', 'No'])
    user_input['PaymentMethod'] = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    user_input['tenure'] = st.number_input("Tenure (months)", min_value=0, max_value=100, value=10)
    user_input['MonthlyCharges'] = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    user_input['TotalCharges'] = st.number_input("Total Charges", min_value=0.0, value=600.0)


    if st.button("Predict Churn"):
        result = predict_churn(user_input, model_choice)
        if result == 1:
            st.error("⚠️ This customer is likely to churn.")
        else:
            st.success("✅ This customer is not likely to churn.")

import pandas as pd
import streamlit as st
from src.Prediction.predict_batch import predict_batch

with tabs[2]:
    st.subheader("📁 Batch Churn Prediction")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    batch_model_choice = st.selectbox(
        "Select a Model for Batch Prediction",
        ["KNN", "Logistic Regression", "SVM", "XGBoost"]
    )

    if uploaded_file is not None:
        if st.button("Run Batch Prediction"):
            st.info("Processing predictions...")
            result_df = predict_batch(uploaded_file, batch_model_choice)

            if result_df is not None:
                st.success("✅ Predictions completed!")
                st.write("Here are the predictions:")
                st.dataframe(result_df.head(), use_container_width=True)

                # Provide download link
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime='text/csv'
                )
            else:
                st.error("❌ Something went wrong while processing the file.")

from src.Evaluation.evaluate_models import get_evaluation_results
import matplotlib.pyplot as plt
import streamlit as st

with tabs[3]:
    st.header("📊 Model Evaluation")

    df = get_evaluation_results()  # Call immediately when tab is active

    st.dataframe(df.style.format({col: "{:.2f}" for col in df.columns if col != "Model"}))

    metrics = ["Accuracy", "Balanced Accuracy", "Recall", "Precision", "F1 Score"]

    for metric in metrics:
        st.subheader(f"{metric} Comparison")
        fig, ax = plt.subplots()
        ax.bar(df["Model"], df[metric], width=0.5)
        for i, val in enumerate(df[metric]):
            ax.text(i, val + 0.01, f"{val:.2f}", ha="center", va="bottom")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(fig)

