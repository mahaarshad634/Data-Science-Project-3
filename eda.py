# ==========================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("preprocessed_titanic_data.csv")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ==========================================
# BASIC INFORMATION
# ==========================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# SURVIVAL DISTRIBUTION
# ==========================================

survival_counts = df["Survived"].value_counts()

print("\nSurvival Distribution:")
print(survival_counts)

plt.figure(figsize=(6,5))
survival_counts.plot(kind="bar")
plt.title("Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("EDA images/survival_distribution.png")
plt.show()

# ==========================================
# GENDER VS SURVIVAL
# ==========================================

gender_survival = pd.crosstab(
    df["Sex"],
    df["Survived"]
)

print("\nGender vs Survival:")
print(gender_survival)

gender_survival.plot(
    kind="bar",
    figsize=(7,5)
)

plt.title("Gender vs Survival")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("EDA images/gender_survival.png")
plt.show()

# ==========================================
# PASSENGER CLASS VS SURVIVAL
# ==========================================

class_survival = pd.crosstab(
    df["Pclass"],
    df["Survived"]
)

print("\nPassenger Class vs Survival:")
print(class_survival)

class_survival.plot(
    kind="bar",
    figsize=(7,5)
)

plt.title("Passenger Class vs Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("EDA images/class_survival.png")
plt.show()

# ==========================================
# AGE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8,5))

plt.hist(
    df["Age"],
    bins=20
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("EDA images/age_distribution.png")
plt.show()

# ==========================================
# FARE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8,5))

plt.hist(
    df["Fare"],
    bins=20
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("EDA images/fare_distribution.png")
plt.show()

# ==========================================
# AGE VS SURVIVAL
# ==========================================

survived_age = df[df["Survived"] == 1]["Age"]
not_survived_age = df[df["Survived"] == 0]["Age"]

plt.figure(figsize=(8,5))

plt.hist(
    survived_age,
    alpha=0.7,
    label="Survived"
)

plt.hist(
    not_survived_age,
    alpha=0.7,
    label="Not Survived"
)

plt.title("Age vs Survival")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.savefig("EDA images/age_vs_survival.png")
plt.show()

# ==========================================
# FARE VS SURVIVAL
# ==========================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Fare"],
    df["Survived"]
)

plt.title("Fare vs Survival")
plt.xlabel("Fare")
plt.ylabel("Survived")

plt.tight_layout()
plt.savefig("EDA images/fare_vs_survival.png")
plt.show()

# ==========================================
# FAMILY SIZE VS SURVIVAL
# ==========================================

family_survival = pd.crosstab(
    df["FamilySize"],
    df["Survived"]
)

family_survival.plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Family Size vs Survival")
plt.xlabel("Family Size")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("EDA images/family_size_survival.png")
plt.show()

# ==========================================
# CORRELATION MATRIX
# ==========================================

correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(10,8))

plt.imshow(
    correlation,
    cmap="coolwarm",
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.savefig("EDA images/correlation_matrix.png")
plt.show()

# ==========================================
# MOST IMPORTANT FEATURES
# ==========================================

survival_corr = correlation["Survived"].sort_values(
    ascending=False
)

print("\nFeature Correlation with Survival:")
print(survival_corr)

print("\nTop Factors Affecting Survival:")
print(survival_corr.head(10))

print("\nEDA Completed Successfully")