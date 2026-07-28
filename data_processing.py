# ==========================================
# TITANIC DATA PREPROCESSING
# ==========================================

import pandas as pd
import numpy as np

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("titanic_optimized_5000_rows.csv")

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# DATASET INFO
# ==========================================

print("\nDataset Information:")
print(df.info())

# ==========================================
# MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values if any exist

if df["Age"].isnull().sum() > 0:
    df["Age"] = df["Age"].fillna(df["Age"].median())

if df["Fare"].isnull().sum() > 0:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

if df["Embarked"].isnull().sum() > 0:
    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

# ==========================================
# REMOVE DUPLICATES
# ==========================================

print("\nDuplicates Before:")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Duplicates After:")
print(df.duplicated().sum())

# ==========================================
# NOTE: Sex and Embarked are kept as strings
# (e.g. "male"/"female", "S"/"C"/"Q") so that
# the OneHotEncoder in train_models.py can
# learn the correct categories.  This ensures
# app.py (which also provides string values)
# matches the training data exactly.
# ==========================================

# ==========================================
# HANDLE OUTLIERS
# ==========================================

numeric_columns = [
    "Age",
    "Fare",
    "FamilySize"
]

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)

    df[column] = np.where(
        df[column] < lower,
        lower,
        df[column]
    )

    df[column] = np.where(
        df[column] > upper,
        upper,
        df[column]
    )

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("Survived", axis=1)
y = df["Survived"]

# ==========================================
# FINAL DATASET
# ==========================================

processed_df = pd.concat(
    [X, y],
    axis=1
)

# ==========================================
# SAVE PREPROCESSED DATA
# ==========================================

processed_df.to_csv(
    "preprocessed_titanic_data.csv",
    index=False
)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("\nFinal Shape:")
print(processed_df.shape)

print("\nColumns:")
print(processed_df.columns.tolist())

print("\nFirst 5 Rows:")
print(processed_df.head())

print("\nSaved As:")
print("preprocessed_titanic_data.csv")
