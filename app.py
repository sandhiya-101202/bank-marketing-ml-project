"""
Bank Marketing Campaign Prediction
Interactive Machine Learning Evaluation Dashboard

"""

import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_PATH = os.path.join(
    BASE_DIR, "results", "model_comparison.csv"
)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

# --------------------------------------------------
# 2. APPLICATION HEADER
# --------------------------------------------------

st.title("Bank Marketing Campaign Analysis")

st.write(
    "An interactive machine learning application "
    "for predicting customer term deposit subscriptions."
)

st.markdown("---")

# --------------------------------------------------
# 3. DATASET UPLOAD
# --------------------------------------------------

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your Bank Marketing test data (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload the test_data.csv file to begin.")
    st.stop()

try:
    test_data = pd.read_csv(uploaded_file)

except Exception as error:
    st.error(f"Unable to read the uploaded file: {error}")
    st.stop()

if "y" not in test_data.columns:
    st.error("The uploaded CSV must contain the target column 'y'.")
    st.stop()

if test_data.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

# Validate target values
test_data["y"] = test_data["y"].astype(str).str.strip().str.lower()

if not test_data["y"].isin(["yes", "no"]).all():
    st.error("Target column 'y' must contain only yes or no.")
    st.stop()

X_test = test_data.drop(columns=["y"])
y_test = test_data["y"].map({"no": 0, "yes": 1})

st.success("Dataset uploaded and validated successfully.")

# --------------------------------------------------
# 4. DATASET INFORMATION
# --------------------------------------------------

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{len(test_data):,}")
col2.metric("Input Features", X_test.shape[1])
col3.metric("Successful Subscriptions", int(y_test.sum()))

with st.expander("Preview Uploaded Dataset"):
    st.dataframe(test_data.head(10), use_container_width=True)

# --------------------------------------------------
# 5. MODEL SELECTION
# --------------------------------------------------

st.markdown("---")
st.subheader("Machine Learning Model Selection")

selected_model = st.selectbox(
    "Choose a classification algorithm",
    list(MODEL_FILES.keys())
)

model_path = os.path.join(
    MODEL_DIR,
    MODEL_FILES[selected_model]
)

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

try:
    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        prediction_scores = model.predict_proba(X_test)[:, 1]
    else:
        prediction_scores = model.decision_function(X_test)

except Exception as error:
    st.error(f"Model execution failed: {error}")
    st.stop()

# --------------------------------------------------
# 6. EVALUATION METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

auc = roc_auc_score(y_test, prediction_scores)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

mcc = matthews_corrcoef(y_test, predictions)

st.markdown("---")
st.subheader("Model Evaluation Results")

st.write(f"Selected Model: **{selected_model}**")

metric_columns = st.columns(3)

metrics = [
    ("Accuracy", accuracy),
    ("AUC Score", auc),
    ("Precision", precision),
    ("Recall", recall),
    ("F1 Score", f1),
    ("MCC Score", mcc)
]

for index, (metric_name, metric_value) in enumerate(metrics):
    with metric_columns[index % 3]:
        st.metric(metric_name, f"{metric_value:.4f}")

# --------------------------------------------------
# 7. CONFUSION MATRIX
# --------------------------------------------------

st.markdown("---")
st.subheader("Confusion Matrix")

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

figure, axis = plt.subplots(figsize=(6, 4))

image = axis.imshow(matrix, cmap="Blues")

axis.set_title(f"Confusion Matrix - {selected_model}")
axis.set_xlabel("Predicted Class")
axis.set_ylabel("Actual Class")

axis.set_xticks([0, 1])
axis.set_yticks([0, 1])

axis.set_xticklabels(["No", "Yes"])
axis.set_yticklabels(["No", "Yes"])

for row in range(2):
    for column in range(2):
        axis.text(
            column,
            row,
            matrix[row, column],
            ha="center",
            va="center"
        )

figure.colorbar(image, ax=axis)
figure.tight_layout()

st.pyplot(figure)
plt.close(figure)

# --------------------------------------------------
# 8. CLASSIFICATION REPORT
# --------------------------------------------------

st.markdown("---")
st.subheader("Classification Report")

report = classification_report(
    y_test,
    predictions,
    target_names=["No", "Yes"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)

# --------------------------------------------------
# 9. MODEL COMPARISON
# --------------------------------------------------

st.markdown("---")
st.subheader("Comparison of All Five Models")

if os.path.exists(RESULT_PATH):

    comparison_data = pd.read_csv(RESULT_PATH)

    st.dataframe(
        comparison_data.round(4),
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning("Model comparison file is not available.")

# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

st.markdown("---")

st.subheader("Prediction Summary")

predicted_yes = int((predictions == 1).sum())
predicted_no = int((predictions == 0).sum())

summary_col1, summary_col2 = st.columns(2)

summary_col1.metric("Predicted Subscriptions", predicted_yes)
summary_col2.metric("Predicted Non-Subscriptions", predicted_no)

st.caption(
    "Developed as part of the BITS Pilani M.Tech "
    "Data Science and Engineering Machine Learning Assignment 2."
)