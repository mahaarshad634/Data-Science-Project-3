# ==========================================
# FEATURE ENGINEERING
# ==========================================

import pandas as pd
import numpy as np

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("preprocessed_titanic_data.csv")

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# ==========================================
# FAMILY SIZE
# ==========================================

df["FamilySize"] = (
    df["SibSp"] +
    df["Parch"] +
    1
)

print("FamilySize Created")

# ==========================================
# IS ALONE
# ==========================================

df["IsAlone"] = np.where(
    df["FamilySize"] == 1,
    1,
    0
)

print("IsAlone Created")

# ==========================================
# AGE GROUP
# ==========================================

def age_group(age):

    if age <= 12:
        return "Child"

    elif age <= 19:
        return "Teen"

    elif age <= 35:
        return "Young Adult"

    elif age <= 60:
        return "Adult"

    else:
        return "Senior"

df["AgeGroup"] = df["Age"].apply(age_group)

print("AgeGroup Created")

# ==========================================
# FARE CATEGORY
# ==========================================

def fare_category(fare):

    if fare < 25:
        return "Low"

    elif fare < 80:
        return "Medium"

    elif fare < 150:
        return "High"

    else:
        return "Very High"

df["FareCategory"] = df["Fare"].apply(
    fare_category
)

print("FareCategory Created")

# ==========================================
# SOCIO ECONOMIC STATUS
# ==========================================

def social_status(pclass):

    if pclass == 1:
        return "Upper"

    elif pclass == 2:
        return "Middle"

    else:
        return "Lower"

df["SocialStatus"] = df["Pclass"].apply(
    social_status
)

print("SocialStatus Created")

# ==========================================
# CHILD OR ADULT
# ==========================================

df["Child"] = np.where(
    df["Age"] < 16,
    1,
    0
)

print("Child Feature Created")

# ==========================================
# FARE PER PERSON
# ==========================================

df["FarePerPerson"] = (
    df["Fare"] /
    df["FamilySize"]
)

print("FarePerPerson Created")

# ==========================================
# FAMILY TYPE
# ==========================================

def family_type(size):

    if size == 1:
        return "Single"

    elif size <= 4:
        return "Small Family"

    else:
        return "Large Family"

df["FamilyType"] = df["FamilySize"].apply(
    family_type
)

print("FamilyType Created")

# ==========================================
# DISPLAY NEW FEATURES
# ==========================================

print("\nNew Features Added:")

new_features = [

    "FamilySize",
    "IsAlone",
    "AgeGroup",
    "FareCategory",
    "SocialStatus",
    "Child",
    "FarePerPerson",
    "FamilyType"
]

print(new_features)

# ==========================================
# SAVE DATASET
# ==========================================

df.to_csv(
    "featured_titanic_data.csv",
    index=False
)

print("\nDataset Saved Successfully")

print(
    "\nFile Name: featured_titanic_data.csv"
)

print(
    "\nFinal Shape:",
    df.shape
)

print("\nFirst 5 Rows:")
print(df.head())