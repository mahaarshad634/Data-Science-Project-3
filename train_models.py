# =====================================================
# TITANIC SURVIVAL MODEL TRAINING SYSTEM
# COMPLETE PIPELINE VERSION
# =====================================================


import pandas as pd
import joblib


from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)


from sklearn.pipeline import Pipeline


from sklearn.compose import ColumnTransformer


from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


from sklearn.calibration import CalibratedClassifierCV


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Models

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB


from xgboost import XGBClassifier



# =====================================================
# LOAD DATASET
# =====================================================


df = pd.read_csv(
    "featured_titanic_data.csv"
)


print("\n")
print("="*70)
print(" TITANIC SURVIVAL MODEL TRAINING ")
print("="*70)



# =====================================================
# FEATURES AND TARGET
# =====================================================


X = df.drop(["Survived","PassengerId"], axis=1)


y = df["Survived"]




# =====================================================
# FEATURE IDENTIFICATION
# =====================================================


categorical_features = [

    "Sex",

    "Embarked",

    "AgeGroup",

    "FareCategory",

    "SocialStatus",

    "FamilyType"

]



# only existing columns

categorical_features = [

    col

    for col in categorical_features

    if col in X.columns

]



numeric_features = [

    col

    for col in X.columns

    if col not in categorical_features

]



print("\nCategorical Features:")
print(categorical_features)


print("\nNumeric Features:")
print(numeric_features)



# =====================================================
# PREPROCESSING
# =====================================================


preprocessor = ColumnTransformer(

    transformers=[


        (

            "numeric",

            Pipeline([

                (

                    "scaler",

                    StandardScaler()

                )

            ]),

            numeric_features

        ),



        (

            "categorical",

            Pipeline([

                (

                    "encoder",

                    OneHotEncoder(

                        handle_unknown="ignore"

                    )

                )

            ]),

            categorical_features

        )

    ]

)



# =====================================================
# TRAIN TEST SPLIT
# =====================================================


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)




# =====================================================
# MODELS
# =====================================================


models = {



"Logistic Regression":

LogisticRegression(

    max_iter=3000

),




"Decision Tree":

DecisionTreeClassifier(

    max_depth=8,

    random_state=42

),




"Random Forest":

RandomForestClassifier(

    n_estimators=500,

    max_depth=12,

    random_state=42

),




"Extra Trees":

ExtraTreesClassifier(

    n_estimators=500,

    random_state=42

),




"Gradient Boosting":

GradientBoostingClassifier(

    n_estimators=300,

    learning_rate=0.05,

    random_state=42

),




"AdaBoost":

AdaBoostClassifier(

    n_estimators=300,

    random_state=42

),




"KNN":

KNeighborsClassifier(

    n_neighbors=7

),




"SVM":

CalibratedClassifierCV(

    estimator=SVC(),

    ensemble=False

),




"Naive Bayes":

GaussianNB(),




"XGBoost":

XGBClassifier(

    n_estimators=500,

    learning_rate=0.05,

    max_depth=5,

    random_state=42,

    eval_metric="logloss"

)

}





# =====================================================
# TRAINING ALL MODELS
# =====================================================


results=[]


best_accuracy = 0


best_model = None


best_model_name = ""



print("\nTraining Models...\n")



for name, model in models.items():


    print(
        "Training:",
        name
    )


    pipeline = Pipeline(

        steps=[

            (

                "preprocessor",

                preprocessor

            ),



            (

                "model",

                model

            )

        ]

    )



    # Train

    pipeline.fit(

        X_train,

        y_train

    )



    # Prediction

    prediction = pipeline.predict(

        X_test

    )



    # Metrics


    accuracy = accuracy_score(

        y_test,

        prediction

    )


    precision = precision_score(

        y_test,

        prediction

    )


    recall = recall_score(

        y_test,

        prediction

    )


    f1 = f1_score(

        y_test,

        prediction

    )



    cv = cross_val_score(

        pipeline,

        X,

        y,

        cv=5,

        scoring="accuracy"

    ).mean()



    results.append([


        name,

        accuracy,

        precision,

        recall,

        f1,

        cv


    ])




    if accuracy > best_accuracy:


        best_accuracy = accuracy

        best_model = pipeline

        best_model_name = name





# =====================================================
# RESULTS
# =====================================================


results_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "Cross Validation"

    ]

)



results_df.sort_values(

    by="Accuracy",

    ascending=False,

    inplace=True

)



print("\n")

print("="*70)

print("MODEL COMPARISON")

print("="*70)


print(results_df)



# Save comparison

results_df.to_csv(

    "model_comparison_results.csv",

    index=False

)




# =====================================================
# SAVE BEST PIPELINE
# =====================================================


joblib.dump(

    best_model,

    "titanic_model.pkl"

)



print("\n")

print("="*70)

print("BEST MODEL")

print("="*70)



print(

    "Model Name:",

    best_model_name

)



print(

    "Accuracy:",

    round(best_accuracy,4)

)




# =====================================================
# FINAL EVALUATION
# =====================================================


final_prediction = best_model.predict(

    X_test

)



print("\nConfusion Matrix")

print(

    confusion_matrix(

        y_test,

        final_prediction

    )

)



print("\nClassification Report")

print(

    classification_report(

        y_test,

        final_prediction

    )

)




print("\n")

print("="*70)

print("TRAINING COMPLETED")

print("titanic_model.pkl CREATED SUCCESSFULLY")

print("="*70)