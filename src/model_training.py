"""
Machine Learning Training and Model Benchmarking Pipeline
Project: Student Performance Analysis and Prediction
College: Goel Institute of Technology & Management (AIML)
Student: Prakhar Rai (Roll: 2303601530038)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
import xgboost as xgb

def train_exam_score_models(df: pd.DataFrame, models_dir: str = "models", reports_dir: str = "reports") -> dict:
    print("=" * 60)
    print("Training ML Models on 30k Student Exam Scores Dataset")
    print("=" * 60)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Feature definitions
    feature_cols = [
        'Gender', 'EthnicGroup', 'ParentEduc', 'LunchType', 'TestPrep',
        'ParentMaritalStatus', 'PracticeSport', 'IsFirstChild', 'NrSiblings',
        'TransportMeans', 'WklyStudyHours'
    ]
    target_reg = 'AverageScore'
    target_clf = 'RiskCategory'

    X = df[feature_cols].copy()
    y_reg = df[target_reg].values
    y_clf = df[target_clf].values

    cat_cols = [
        'Gender', 'EthnicGroup', 'ParentEduc', 'LunchType', 'TestPrep',
        'ParentMaritalStatus', 'PracticeSport', 'IsFirstChild', 'TransportMeans', 'WklyStudyHours'
    ]
    num_cols = ['NrSiblings']

    # Train/Test Split (80/20)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42
    )

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # Fit preprocessor
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor_exam_30k.pkl'))

    # Regression Models
    regression_models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42, n_jobs=-1)
    }

    reg_results = {}
    best_reg_r2 = -float('inf')
    best_reg_name = None
    best_reg_pipeline = None

    for name, model in regression_models.items():
        print(f" -> Training Regressor: {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_reg_train)
        
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        r2_test = r2_score(y_reg_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_reg_test, y_test_pred))
        mae_test = mean_absolute_error(y_reg_test, y_test_pred)

        r2_train = r2_score(y_reg_train, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_reg_train, y_train_pred))
        mae_train = mean_absolute_error(y_reg_train, y_train_pred)

        reg_results[name] = {
            'test_r2': float(np.round(r2_test, 4)),
            'test_rmse': float(np.round(rmse_test, 4)),
            'test_mae': float(np.round(mae_test, 4)),
            'train_r2': float(np.round(r2_train, 4)),
            'train_rmse': float(np.round(rmse_train, 4)),
            'train_mae': float(np.round(mae_train, 4))
        }

        # Save individual model pipeline
        model_filename = f"model_exam_{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(pipeline, os.path.join(models_dir, model_filename))

        if r2_test > best_reg_r2:
            best_reg_r2 = r2_test
            best_reg_name = name
            best_reg_pipeline = pipeline

    joblib.dump(best_reg_pipeline, os.path.join(models_dir, 'best_exam_regressor.pkl'))

    # Extract Feature Importances from Random Forest
    rf_reg = regression_models['Random Forest']
    feature_names = (
        num_cols + 
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    )
    importances = rf_reg.feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    feat_imp.to_csv(os.path.join(reports_dir, 'feature_importance_exam.csv'), index=False)

    # Classification Models for Risk Category
    classification_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost Classifier': xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42, n_jobs=-1)
    }

    # Map labels to numeric for XGBoost
    label_mapping = {'Low Risk': 0, 'Moderate Risk': 1, 'High Risk': 2}
    y_clf_train_encoded = np.array([label_mapping[val] for val in y_clf_train])
    y_clf_test_encoded = np.array([label_mapping[val] for val in y_clf_test])

    clf_results = {}
    for name, model in classification_models.items():
        print(f" -> Training Classifier: {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_clf_train_encoded)
        y_test_pred_clf = pipeline.predict(X_clf_test := X_test)

        acc = accuracy_score(y_clf_test_encoded, y_test_pred_clf)
        prec = precision_score(y_clf_test_encoded, y_test_pred_clf, average='weighted', zero_division=0)
        rec = recall_score(y_clf_test_encoded, y_test_pred_clf, average='weighted', zero_division=0)
        f1 = f1_score(y_clf_test_encoded, y_test_pred_clf, average='weighted', zero_division=0)

        clf_results[name] = {
            'accuracy': float(np.round(acc, 4)),
            'precision': float(np.round(prec, 4)),
            'recall': float(np.round(rec, 4)),
            'f1_score': float(np.round(f1, 4))
        }

    joblib.dump(label_mapping, os.path.join(models_dir, 'risk_label_mapping.pkl'))

    return {
        'regression_benchmarks': reg_results,
        'classification_benchmarks': clf_results,
        'best_regression_model': best_reg_name,
        'features_used': feature_cols
    }

def train_factors_models(df: pd.DataFrame, models_dir: str = "models", reports_dir: str = "reports") -> dict:
    print("=" * 60)
    print("Training ML Models on Behavioral Factors Dataset (Attendance, Tutoring, etc.)")
    print("=" * 60)

    features = [
        'Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
        'Extracurricular_Activities', 'Sleep_Hours', 'Previous_Scores', 'Motivation_Level',
        'Internet_Access', 'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Physical_Activity', 'Learning_Disabilities',
        'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]
    target = 'Exam_Score'

    X = df[features].copy()
    y = df[target].values

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor_factors.pkl'))

    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42, n_jobs=-1)
    }

    results = {}
    best_r2 = -float('inf')
    best_name = None
    best_pipe = None

    for name, model in models.items():
        print(f" -> Factors Regressor: {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results[name] = {
            'r2': float(np.round(r2, 4)),
            'rmse': float(np.round(rmse, 4)),
            'mae': float(np.round(mae, 4))
        }

        joblib.dump(pipeline, os.path.join(models_dir, f"model_factors_{name.lower().replace(' ', '_')}.pkl"))

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipe = pipeline

    joblib.dump(best_pipe, os.path.join(models_dir, 'best_factors_regressor.pkl'))

    # Feature Importance for Factors
    rf = models['Random Forest']
    feature_names = num_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
    feat_imp.to_csv(os.path.join(reports_dir, 'feature_importance_factors.csv'), index=False)

    return {
        'benchmarks': results,
        'best_model': best_name,
        'features': features
    }

if __name__ == "__main__":
    df_exam = pd.read_csv("data/processed/cleaned_student_exam_scores.csv")
    df_factors = pd.read_csv("data/processed/cleaned_student_factors.csv")

    exam_res = train_exam_score_models(df_exam)
    factors_res = train_factors_models(df_factors)

    all_benchmarks = {
        'exam_dataset_benchmarks': exam_res,
        'factors_dataset_benchmarks': factors_res
    }

    with open("reports/model_benchmarks.json", "w") as f:
        json.dump(all_benchmarks, f, indent=4)

    print("\nAll models trained, evaluated, and serialized successfully!")
