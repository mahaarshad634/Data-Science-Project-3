# Titanic Survival Prediction System

A complete machine learning pipeline that predicts passenger survival on the Titanic using historical data.

## Quick Start

```bash
# Run the full pipeline
python data_processing.py
python featureengineering.py
python train_models.py

# Run the web app
python -m streamlit run app.py --server.port 8501
```

## Project Structure

- **app.py** - Streamlit web application
- **data_processing.py** - Data preprocessing (cleaning, outlier handling)
- **featureengineering.py** - Creates 8 new features (FamilySize, AgeGroup, FareCategory, etc.)
- **train_models.py** - Trains 10 models, selects best (Gradient Boosting, 90.6% accuracy)
- **eda.py** - Exploratory data analysis with visualizations
- **Feature importance/feature_importance.py** - Feature importance analysis

## Model Performance

- **Best Model**: Gradient Boosting Classifier
- **Accuracy**: 90.6%
- **Cross-Validation**: 88.6%

## Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Sex | 31.6% |
| 2 | Fare | 17.1% |
| 3 | FarePerPerson | 9.2% |
| 4 | Pclass | 8.3% |
| 5 | SocialStatus | 8.1% |

## How to Run

1. **Full pipeline**: Run data_processing.py, featureengineering.py, train_models.py in order
2. **Web app**: `python -m streamlit run app.py --server.port 8501`
3. **EDA**: `python eda.py`
4. **Feature importance**: `python "Feature importance/feature_importance.py"`

## Requirements

pandas, numpy, scikit-learn, matplotlib, joblib, streamlit, xgboost
