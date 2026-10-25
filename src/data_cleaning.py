"""
Data Cleaning and Preprocessing Pipeline
Project: Student Performance Analysis and Prediction Using Data Analytics
College: Goel Institute of Technology & Management (AIML)
Student: Prakhar Rai (Roll: 2303601530038)
"""

import os
import json
import numpy as np
import pandas as pd

def clean_exam_scores_dataset(raw_path: str, output_path: str) -> dict:
    print(f"[1/2] Cleaning 30k Exam Scores Dataset from: {raw_path}")
    df = pd.read_csv(raw_path)
    initial_shape = df.shape
    initial_nulls = df.isnull().sum().to_dict()

    # Drop index artifact column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Standardize string formatting
    str_cols = df.select_dtypes(include=['object']).columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c] == 'nan', c] = np.nan

    # Categorical Imputations based on domain baseline & mode
    df['EthnicGroup'] = df['EthnicGroup'].fillna('Unknown')
    df['ParentEduc'] = df['ParentEduc'].fillna('Unknown')
    df['TestPrep'] = df['TestPrep'].fillna('none')  # Default baseline is no prep
    df['ParentMaritalStatus'] = df['ParentMaritalStatus'].fillna(df['ParentMaritalStatus'].mode()[0])
    df['PracticeSport'] = df['PracticeSport'].fillna(df['PracticeSport'].mode()[0])
    df['IsFirstChild'] = df['IsFirstChild'].fillna(df['IsFirstChild'].mode()[0])
    df['TransportMeans'] = df['TransportMeans'].fillna(df['TransportMeans'].mode()[0])
    df['WklyStudyHours'] = df['WklyStudyHours'].fillna('5 - 10')  # Mode study hours

    # Numeric Imputations
    median_siblings = float(df['NrSiblings'].median())
    df['NrSiblings'] = df['NrSiblings'].fillna(median_siblings).astype(int)

    # Sanity checks on exam scores
    for score_col in ['MathScore', 'ReadingScore', 'WritingScore']:
        df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
        df[score_col] = df[score_col].clip(lower=0, upper=100)

    # Feature Engineering
    df['TotalScore'] = df['MathScore'] + df['ReadingScore'] + df['WritingScore']
    df['AverageScore'] = np.round(df['TotalScore'] / 3.0, 2)

    # Academic Grade Mapping
    def assign_grade(score):
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'F'

    df['Grade'] = df['AverageScore'].apply(assign_grade)

    # Risk Category (Educational early warning)
    def assign_risk(score):
        if score < 50:
            return 'High Risk'
        elif score < 70:
            return 'Moderate Risk'
        else:
            return 'Low Risk'

    df['RiskCategory'] = df['AverageScore'].apply(assign_risk)

    # Pass/Fail Status (Passing threshold: 50% average and at least 40% in each subject)
    df['Passed'] = (
        (df['AverageScore'] >= 50) &
        (df['MathScore'] >= 40) &
        (df['ReadingScore'] >= 40) &
        (df['WritingScore'] >= 40)
    ).map({True: 'Pass', False: 'Fail'})

    # Numeric representation for study hours
    study_map = {'< 5': 3.5, '5 - 10': 7.5, '> 10': 12.5}
    df['WklyStudyHours_Num'] = df['WklyStudyHours'].map(study_map).fillna(7.5)

    final_shape = df.shape
    remaining_nulls = df.isnull().sum().to_dict()

    # Save cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" -> Cleaned dataset saved to {output_path} with shape {final_shape}")

    return {
        "dataset_name": "Students Exam Scores (30k)",
        "initial_shape": initial_shape,
        "final_shape": final_shape,
        "initial_nulls": initial_nulls,
        "remaining_nulls": {k: v for k, v in remaining_nulls.items() if v > 0},
        "engineered_features": [
            "TotalScore", "AverageScore", "Grade", "RiskCategory", 
            "Passed", "WklyStudyHours_Num"
        ]
    }

def clean_performance_factors_dataset(raw_path: str, output_path: str) -> dict:
    print(f"[2/2] Cleaning Performance Factors Dataset from: {raw_path}")
    df = pd.read_csv(raw_path)
    initial_shape = df.shape
    initial_nulls = df.isnull().sum().to_dict()

    # String stripping
    str_cols = df.select_dtypes(include=['object']).columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c] == 'nan', c] = np.nan

    # Categorical Imputations
    df['Teacher_Quality'] = df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0])
    df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0])
    df['Distance_from_Home'] = df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0])

    # Outlier clipping (e.g. Exam_Score had max 101)
    df['Exam_Score'] = df['Exam_Score'].clip(lower=0, upper=100)
    df['Attendance'] = df['Attendance'].clip(lower=0, upper=100)

    # Feature Engineering
    def assign_factor_grade(score):
        if score >= 85:
            return 'A'
        elif score >= 75:
            return 'B'
        elif score >= 65:
            return 'C'
        elif score >= 55:
            return 'D'
        else:
            return 'F'

    df['Grade'] = df['Exam_Score'].apply(assign_factor_grade)

    def assign_factor_risk(score):
        if score < 60:
            return 'High Risk'
        elif score < 70:
            return 'Moderate Risk'
        else:
            return 'Low Risk'

    df['RiskCategory'] = df['Exam_Score'].apply(assign_factor_risk)

    df['AttendanceCategory'] = pd.cut(
        df['Attendance'], 
        bins=[0, 74.9, 89.9, 100], 
        labels=['Critical (<75%)', 'Average (75-89%)', 'Excellent (>=90%)']
    )

    df['StudyLifeBalanceHours'] = df['Hours_Studied'] + df['Sleep_Hours']

    final_shape = df.shape
    remaining_nulls = df.isnull().sum().to_dict()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" -> Cleaned factors dataset saved to {output_path} with shape {final_shape}")

    return {
        "dataset_name": "Student Performance Factors",
        "initial_shape": initial_shape,
        "final_shape": final_shape,
        "initial_nulls": initial_nulls,
        "remaining_nulls": {k: v for k, v in remaining_nulls.items() if v > 0},
        "engineered_features": [
            "Grade", "RiskCategory", "AttendanceCategory", "StudyLifeBalanceHours"
        ]
    }

if __name__ == "__main__":
    report = {}
    report['exam_scores'] = clean_exam_scores_dataset(
        raw_path="data/raw/student_exam_scores_30k.csv",
        output_path="data/processed/cleaned_student_exam_scores.csv"
    )
    report['performance_factors'] = clean_performance_factors_dataset(
        raw_path="data/raw/student_performance_factors.csv",
        output_path="data/processed/cleaned_student_factors.csv"
    )

    os.makedirs("reports", exist_ok=True)
    with open("reports/data_cleaning_audit.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\nData Cleaning Pipeline finished successfully! Audit report saved to reports/data_cleaning_audit.json")
