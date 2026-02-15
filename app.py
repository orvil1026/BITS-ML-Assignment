import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="Online Shopper Prediction", layout="wide")

st.title("Online Shoppers Purchasing Intention Prediction")

st.write("""
Upload a test CSV file to evaluate different trained machine learning models.
Select a model from the dropdown and view performance metrics.
""")

# -------------------------------
# Load Models and Preprocessing
# -------------------------------

@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl"),
    }
    scaler = joblib.load("model/scaler.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    return models, scaler, feature_columns

models, scaler, feature_columns = load_models()



# -------------------------------
# Download Sample Test CSV
# -------------------------------

st.subheader("Download Sample Test File")

if os.path.exists("test.csv"):
    with open("test.csv", "rb") as file:
        st.download_button(
            label="Download Sample test.csv",
            data=file,
            file_name="test.csv",
            mime="text/csv"
        )
else:
    st.warning("Sample test.csv not found in project directory.")
# -------------------------------
# File Upload
# -------------------------------

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.write(data.head())

    if "Revenue" not in data.columns:
        st.error("Uploaded file must contain 'Revenue' column.")
        st.stop()

    X = data.drop("Revenue", axis=1)
    y = data["Revenue"]

    # -------------------------------
    # Align Columns with Training Data
    # -------------------------------

    X = pd.get_dummies(X)

    # Add missing columns
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    # Ensure correct order
    X = X[feature_columns]

    # -------------------------------
    # Model Selection
    # -------------------------------

    model_choice = st.selectbox("Select Model", list(models.keys()))
    model = models[model_choice]

    # Scale only for specific models
    if model_choice in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_processed = scaler.transform(X)
    else:
        X_processed = X

    # -------------------------------
    # Prediction
    # -------------------------------

    y_pred = model.predict(X_processed)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_processed)[:, 1]
    else:
        y_prob = y_pred

    # -------------------------------
    # Metrics
    # -------------------------------

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy, 3))
    col2.metric("AUC", round(auc, 3))
    col3.metric("Precision", round(precision, 3))

    col1.metric("Recall", round(recall, 3))
    col2.metric("F1 Score", round(f1, 3))
    col3.metric("MCC", round(mcc, 3))

    # -------------------------------
    # Confusion Matrix
    # -------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)
