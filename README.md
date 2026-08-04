# Multiple Disease Prediction from Blood Samples

A machine learning app that predicts disease risk (Anemia, Diabetes, Healthy, Heart Disease, Thalassemia, Thrombocytopenia) from 24 blood test parameters, deployed with Streamlit.

**Live app:** https://multiple-diseases-prediction-5uxofpwyxy5wjesufqexbi.streamlit.app/

## Overview

This project trains a stacked ensemble classifier on a blood samples dataset to predict one of six disease classes from routine blood test values (Glucose, Cholesterol, Hemoglobin, Platelets, White Blood Cells, and 19 other markers). The trained model is bundled with a label encoder and served through an interactive Streamlit web app.

## Model

- **Pipeline:** `StandardScaler` → `StackingClassifier`
- **Base estimators:** `GradientBoostingClassifier`, `LogisticRegression`
- **Final estimator:** `RandomForestClassifier`
- **Hyperparameter tuning:** `RandomizedSearchCV` (5-fold CV), with an Optuna-based tuning pass also explored
- **Test accuracy:** ~96%
- **Evaluation:** classification report, confusion matrix, and multiclass ROC AUC (one-vs-rest)

### Disease classes
`Anemia`, `Diabetes`, `Healthy`, `Heart Disease`, `Thalassemia`, `Thrombocytopenia`

## Project structure

```
├── Disease_Predictor_Notebook.ipynb   # Data cleaning, model training, tuning, evaluation
├── disease_app.py                     # Streamlit app
├── blood_model.pkl                    # Saved model bundle (model, label encoder, feature ranges)
└── README.md
```

## How it works

1. Raw blood test values are entered in the app.
2. Each feature is scaled into the [0, 1] range using stored real-world min/max reference ranges (since the model was trained on min-max normalized data).
3. The scaled features are passed through the trained pipeline to produce a predicted disease class and class probabilities.

## Running locally

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run disease_app.py
```

## Tech stack

- Python, scikit-learn, Optuna
- pandas, numpy
- Streamlit (deployment)
- joblib (model persistence)

## Disclaimer

This tool is for educational and demonstration purposes only. It is **not a medical diagnostic device** and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
