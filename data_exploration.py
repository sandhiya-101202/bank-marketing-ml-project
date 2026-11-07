"""
Bank Marketing Campaign Prediction
Exploratory Data Analysis
"""

import pandas as pd

# Load the original dataset
dataset_path = "data/raw/bank-full.csv"

bank_data = pd.read_csv(dataset_path, sep=";")

# Display basic dataset information
print("=" * 60)
print("BANK MARKETING DATASET - OVERVIEW")
print("=" * 60)

print("\nDataset Dimensions:")
print(bank_data.shape)

print("\nColumn Names:")
print(bank_data.columns.tolist())

print("\nFirst Five Records:")
print(bank_data.head())

print("\nDataset Information:")
bank_data.info()

print("\nStatistical Summary:")
print(bank_data.describe())

print("\nMissing Values:")
print(bank_data.isnull().sum())

print("\nTarget Variable Distribution:")
print(bank_data["y"].value_counts())