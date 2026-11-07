"""
Bank Marketing Campaign Prediction
Machine Learning Model Training and Evaluation

"""

import os
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

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

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

RANDOM_STATE = 42

TRAIN_PATH = "data/processed/train_data.csv"
TEST_PATH = "test_data.csv"

MODEL_DIRECTORY = "models"
RESULT_DIRECTORY = "results"

os.makedirs(MODEL_DIRECTORY, exist_ok=True)
os.makedirs(RESULT_DIRECTORY, exist_ok=True)

# --------------------------------------------------
# 2. LOAD TRAINING AND TESTING DATA
# --------------------------------------------------

print("=" * 65)
print("BANK MARKETING - MACHINE LEARNING MODEL DEVELOPMENT")
print("=" * 65)

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

X_train = train_data.drop(columns=["y"])
y_train = train_data["y"].map({"no": 0, "yes": 1})

X_test = test_data.drop(columns=["y"])
y_test = test_data["y"].map({"no": 0, "yes": 1})

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Input features:", X_train.shape[1])

# --------------------------------------------------
# 3. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# --------------------------------------------------

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

# --------------------------------------------------
# 4. DATA PREPROCESSING PIPELINE
# --------------------------------------------------

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

# --------------------------------------------------
# 5. DEFINE CLASSIFICATION MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=10,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Support Vector Machine": LinearSVC(
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE
    )
}

# --------------------------------------------------
# 6. MODEL TRAINING AND EVALUATION
# --------------------------------------------------

evaluation_results = []

for model_name, classifier in models.items():

    print("\n" + "=" * 65)
    print("Training:", model_name)
    print("=" * 65)

    # Build complete machine learning pipeline
    model_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Generate predictions
    predictions = model_pipeline.predict(X_test)

    # Calculate prediction scores for AUC
    if hasattr(model_pipeline, "predict_proba"):
        prediction_scores = model_pipeline.predict_proba(X_test)[:, 1]
    else:
        prediction_scores = model_pipeline.decision_function(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    auc = roc_auc_score(
        y_test,
        prediction_scores
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        predictions
    )

    # Store evaluation results
    evaluation_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    # Display results
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"AUC Score : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"MCC Score : {mcc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["No", "Yes"],
        zero_division=0
    ))

    # Generate confusion matrix
    matrix = confusion_matrix(y_test, predictions)

    print("Confusion Matrix:")
    print(matrix)

    # Save confusion matrix image
    plt.figure(figsize=(6, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.yticks([0, 1], ["No", "Yes"])

    for row in range(2):
        for col in range(2):
            plt.text(
                col,
                row,
                matrix[row, col],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    image_name = model_name.lower().replace(" ", "_")
    plt.savefig(
        f"{RESULT_DIRECTORY}/{image_name}_confusion_matrix.png"
    )
    plt.close()

    # Save trained model
    model_filename = image_name + ".joblib"

    joblib.dump(
        model_pipeline,
        f"{MODEL_DIRECTORY}/{model_filename}"
    )

    print("Model saved:", model_filename)

# --------------------------------------------------
# 7. MODEL COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(evaluation_results)

print("\n\n" + "=" * 100)
print("FINAL MODEL PERFORMANCE COMPARISON")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        formatters={
            col: "{:.4f}".format
            for col in results_df.columns
            if col != "Model"
        }
    )
)

# Save comparison table
results_df.to_csv(
    f"{RESULT_DIRECTORY}/model_comparison.csv",
    index=False
)

# --------------------------------------------------
# 8. IDENTIFY BEST MODEL
# --------------------------------------------------

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\nBEST MODEL BASED ON F1 SCORE")
print("Model:", best_model["Model"])
print("F1 Score:", round(best_model["F1 Score"], 4))

print("\nModel training and evaluation completed.")
print("All trained models and evaluation results saved.")