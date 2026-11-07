"""
Bank Marketing Campaign Prediction
Data Preprocessing and Dataset Splitting
"""

import os
import pandas as pd

from sklearn.model_selection import train_test_split

# Project configuration
DATA_PATH = "data/raw/bank-full.csv"
PROCESSED_PATH = "data/processed"

os.makedirs(PROCESSED_PATH, exist_ok=True)

# Load dataset
bank_data = pd.read_csv(DATA_PATH, sep=";")

print("Original dataset shape:", bank_data.shape)

# Remove duration to prevent information leakage
bank_data = bank_data.drop(columns=["duration"])

# Separate features and target
X = bank_data.drop(columns=["y"])
y = bank_data["y"]

print("\nInput feature count:", X.shape[1])
print("Target variable:", y.name)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Combine features and target for saving
train_data = X_train.copy()
train_data["y"] = y_train

test_data = X_test.copy()
test_data["y"] = y_test

# Save processed datasets
train_data.to_csv(
    os.path.join(PROCESSED_PATH, "train_data.csv"),
    index=False
)

test_data.to_csv("test_data.csv", index=False)

# Display results
print("\nDATA SPLITTING RESULTS")
print("-" * 40)

print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))

print("\nFiles saved successfully!")