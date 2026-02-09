# Multiple Diseases Prediction

A machine learning project that classifies diseases based on blood sample parameters using ensemble methods and hyperparameter optimization.

## Overview

This project implements a stacking classifier ensemble that combines Gradient Boosting, Logistic Regression, and Random Forest algorithms to predict diseases from blood test results. The model achieves 94% accuracy on the test set.

## Dataset

The dataset contains 486 blood samples with 24 biomarker features including:
- **Hematological markers**: Glucose, Cholesterol, Hemoglobin, Platelets, White/Red Blood Cells
- **Blood chemistry**: Insulin, BMI, Blood Pressure (Systolic/Diastolic), Triglycerides
- **Specialized markers**: HbA1c, LDL/HDL Cholesterol, ALT, AST, Heart Rate
- **Additional parameters**: Creatinine, Troponin, C-reactive Protein, Hematocrit, Mean Corpuscular Volume/Hemoglobin

**Target Classes**: The model predicts 6 different disease conditions (encoded as 0-5).

## Model Architecture

### Ensemble Method: Stacking Classifier
- **Base Estimators**:
  - Gradient Boosting Classifier
  - Logistic Regression
- **Meta-learner**: Random Forest Classifier
- **Preprocessing**: Standard Scaler for feature normalization

### Hyperparameter Optimization
The project employs two optimization strategies:

1. **Optuna Optimization** (20 trials)
   - Bayesian optimization for efficient hyperparameter search
   - Cross-validation scoring for robust evaluation

2. **Randomized Search CV**
   - Traditional grid search with 2-fold cross-validation
   - Parameters tuned across all ensemble components

## Key Features

- **Data Preprocessing**: Automated missing value handling and feature scaling
- **Model Pipeline**: Integrated preprocessing and modeling pipeline
- **Hyperparameter Tuning**: Dual optimization approach for best performance
- **Performance Metrics**: Comprehensive classification report with precision, recall, and F1-scores

## Results

- **Test Accuracy**: 93.88%
- **Cross-validation**: Optimized through hyperparameter search
- **Best Parameters** (from Optuna):
  - Gradient Boosting learning rate: 0.1
  - Random Forest: 50 estimators, log2 max features, depth 20
  - Logistic Regression: C=2.0, L2 regularization
## Live Link
https://multiple-diseases-prediction-5uxofpwyxy5wjesufqexbi.streamlit.app/
