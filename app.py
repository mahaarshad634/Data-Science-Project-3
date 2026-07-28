import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load trained model (cached for fast reruns)
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")

model = load_model()


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)


st.title("🚢 Titanic Survival Prediction System")
st.write("Enter passenger details to predict survival probability")


# -----------------------------
# User Inputs
# -----------------------------

col1, col2 = st.columns(2)


with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    age = st.slider(
        "Age",
        1,
        80,
        25
    )

    sibsp = st.number_input(
        "Siblings / Spouse",
        min_value=0,
        max_value=10,
        value=0
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=50.0
    )


with col2:

    parch = st.number_input(
        "Parents / Children",
        min_value=0,
        max_value=10,
        value=0
    )

    embarked = st.selectbox(
        "Embarked",
        ["S", "C", "Q"]
    )

    title = st.selectbox(
        "Title",
        [
            "Mr",
            "Mrs",
            "Miss",
            "Master",
            "Rare"
        ]
    )


# -----------------------------
# Prediction Button
# -----------------------------

if st.button("Predict Survival"):

    # =============================
    # Feature Engineering
    # =============================

    family_size = sibsp + parch + 1

    is_alone = 1 if family_size == 1 else 0

    # Age Group (must match featureengineering.py)

    if age <= 12:
        age_group = "Child"

    elif age <= 19:
        age_group = "Teen"

    elif age <= 35:
        age_group = "Young Adult"

    elif age <= 60:
        age_group = "Adult"

    else:
        age_group = "Senior"

    # Child Feature

    child = 1 if age < 16 else 0

    # Family Type (must match featureengineering.py)

    if family_size == 1:
        family_type = "Single"

    elif family_size <= 4:
        family_type = "Small Family"

    else:
        family_type = "Large Family"

    # Fare Category (must match featureengineering.py)

    if fare < 25:
        fare_category = "Low"

    elif fare < 80:
        fare_category = "Medium"

    elif fare < 150:
        fare_category = "High"

    else:
        fare_category = "Very High"

    # Fare per person

    fare_per_person = fare / family_size

    # Social Status

    if pclass == 1:
        social_status = "Upper"

    elif pclass == 2:
        social_status = "Middle"

    else:
        social_status = "Lower"

    # =============================
    # Create DataFrame
    # =============================

    data = pd.DataFrame({
        "PassengerId": [1],
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked],
        "FamilySize": [family_size],
        "IsAlone": [is_alone],
        "Title": [title],
        "AgeGroup": [age_group],
        "Child": [child],
        "FamilyType": [family_type],
        "FareCategory": [fare_category],
        "FarePerPerson": [fare_per_person],
        "SocialStatus": [social_status]
    })

    # =============================
    # Prediction
    # =============================
    # The model is a full sklearn Pipeline
    # (ColumnTransformer + estimator), so we
    # can pass the raw DataFrame directly.
    # The pipeline handles all preprocessing
    # (scaling, one-hot encoding) internally.
    # =============================

    try:

        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)[0][1]

        if prediction == 1:

            st.success(
                "🎉 Passenger Survived"
            )

        else:

            st.error(
                "😢 Passenger Did Not Survive"
            )

        st.info(
            f"Survival Probability: {probability*100:.2f}%"
        )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

        st.write(
            "Model expects the following feature columns:"
        )

        st.write(
            list(model.feature_names_in_)
            if hasattr(model, "feature_names_in_")
            else "Feature names not available"
        )

        st.write(
            "Provided features:"
        )

        st.write(
            list(data.columns)
        )
