# ==========================================
# FEATURE IMPORTANCE ANALYSIS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("featured_titanic_data.csv")

print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# ==========================================
# ENCODE CATEGORICAL FEATURES
# ==========================================

categorical_columns = [
    "Sex",
    "Embarked",
    "AgeGroup",
    "FareCategory",
    "SocialStatus",
    "FamilyType"
]

encoder = LabelEncoder()

for col in categorical_columns:

    if col in df.columns:

        df[col] = encoder.fit_transform(
            df[col]
        )

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop(
    "Survived",
    axis=1
)

y = df["Survived"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\nFeature Importance Ranking\n")
print(importance)

# ==========================================
# SAVE RESULTS
# ==========================================

importance.to_csv(
    "Feature importance/feature_importance.csv",
    index=False
)

print("\nFeature importance/feature_importance.csv saved successfully")

# ==========================================
# BAR CHART
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title(
    "Feature Importance Analysis"
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Importance Score"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "Feature importance/feature_importance.png"
)

plt.show()

print(
    "\nFeature importance/feature_importance.png saved successfully"
)

# ==========================================
# TOP 5 FACTORS
# ==========================================

print("\nTop 5 Most Important Factors:\n")
print(
    importance.head(5)
)

print("\nAnalysis Completed Successfully")
