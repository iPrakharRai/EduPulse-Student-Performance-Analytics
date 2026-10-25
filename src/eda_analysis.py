"""
Exploratory Data Analysis and Statistical Analysis Pipeline
Project: Student Performance Analysis and Prediction
College: Goel Institute of Technology & Management (AIML)
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.sans-serif'] = 'Arial'

def perform_statistical_analysis(df_exam: pd.DataFrame, df_factors: pd.DataFrame, output_dir: str = "reports/figures") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    summary = {}

    print("--- 1. Statistical Hypothesis Testing on 30k Exam Scores ---")
    
    # 1.1 T-Test: Test Preparation Course Impact on Average Score
    prep_completed = df_exam[df_exam['TestPrep'] == 'completed']['AverageScore']
    prep_none = df_exam[df_exam['TestPrep'] == 'none']['AverageScore']
    t_stat_prep, p_val_prep = stats.ttest_ind(prep_completed, prep_none, equal_var=False)
    
    summary['ttest_test_preparation'] = {
        'factor': 'TestPrep (completed vs none)',
        't_statistic': float(np.round(t_stat_prep, 4)),
        'p_value': float(p_val_prep),
        'mean_completed': float(np.round(prep_completed.mean(), 2)),
        'mean_none': float(np.round(prep_none.mean(), 2)),
        'difference': float(np.round(prep_completed.mean() - prep_none.mean(), 2)),
        'significant': bool(p_val_prep < 0.05)
    }

    # 1.2 T-Test: Lunch Type (Standard vs Free/Reduced) Impact
    lunch_std = df_exam[df_exam['LunchType'] == 'standard']['AverageScore']
    lunch_free = df_exam[df_exam['LunchType'] == 'free/reduced']['AverageScore']
    t_stat_lunch, p_val_lunch = stats.ttest_ind(lunch_std, lunch_free, equal_var=False)

    summary['ttest_lunch_type'] = {
        'factor': 'LunchType (standard vs free/reduced)',
        't_statistic': float(np.round(t_stat_lunch, 4)),
        'p_value': float(p_val_lunch),
        'mean_standard': float(np.round(lunch_std.mean(), 2)),
        'mean_free_reduced': float(np.round(lunch_free.mean(), 2)),
        'difference': float(np.round(lunch_std.mean() - lunch_free.mean(), 2)),
        'significant': bool(p_val_lunch < 0.05)
    }

    # 1.3 ANOVA: Parental Education Level on Average Score
    educ_groups = [group['AverageScore'].values for _, group in df_exam.groupby('ParentEduc')]
    f_stat_educ, p_val_educ = stats.f_oneway(*educ_groups)
    
    summary['anova_parental_education'] = {
        'factor': 'Parental Education Level',
        'f_statistic': float(np.round(f_stat_educ, 4)),
        'p_value': float(p_val_educ),
        'significant': bool(p_val_educ < 0.05)
    }

    # 1.4 ANOVA: Weekly Study Hours on Average Score
    study_groups = [group['AverageScore'].values for _, group in df_exam.groupby('WklyStudyHours')]
    f_stat_study, p_val_study = stats.f_oneway(*study_groups)

    summary['anova_weekly_study_hours'] = {
        'factor': 'Weekly Study Hours (<5, 5-10, >10)',
        'f_statistic': float(np.round(f_stat_study, 4)),
        'p_value': float(p_val_study),
        'significant': bool(p_val_study < 0.05)
    }

    print("--- 2. Correlations and Factor Analysis on Factors Dataset ---")
    corr_attendance = stats.pearsonr(df_factors['Attendance'], df_factors['Exam_Score'])
    corr_study_hours = stats.pearsonr(df_factors['Hours_Studied'], df_factors['Exam_Score'])
    corr_prior_scores = stats.pearsonr(df_factors['Previous_Scores'], df_factors['Exam_Score'])
    corr_tutoring = stats.pearsonr(df_factors['Tutoring_Sessions'], df_factors['Exam_Score'])

    summary['factors_correlations'] = {
        'attendance_vs_exam': {'pearson_r': float(np.round(corr_attendance[0], 4)), 'p_value': float(corr_attendance[1])},
        'study_hours_vs_exam': {'pearson_r': float(np.round(corr_study_hours[0], 4)), 'p_value': float(corr_study_hours[1])},
        'previous_scores_vs_exam': {'pearson_r': float(np.round(corr_prior_scores[0], 4)), 'p_value': float(corr_prior_scores[1])},
        'tutoring_vs_exam': {'pearson_r': float(np.round(corr_tutoring[0], 4)), 'p_value': float(corr_tutoring[1])}
    }

    print("--- 3. Generating Analytical Figures ---")
    
    # Figure 1: Score Distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, col in enumerate(['MathScore', 'ReadingScore', 'WritingScore']):
        sns.histplot(df_exam[col], kde=True, ax=axes[i], color='#3b82f6', bins=30)
        axes[i].set_title(f'Distribution of {col}\nMean: {df_exam[col].mean():.1f} | Std: {df_exam[col].std():.1f}')
        axes[i].set_xlabel('Score (0 - 100)')
        axes[i].axvline(df_exam[col].mean(), color='red', linestyle='--', label='Mean')
        axes[i].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'figure1_score_distributions.png'))
    plt.close()

    # Figure 2: Impact of Study Hours & Test Prep
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order_study = ['< 5', '5 - 10', '> 10']
    sns.boxplot(x='WklyStudyHours', y='AverageScore', data=df_exam, order=order_study, palette='Blues_d', ax=axes[0])
    axes[0].set_title('Average Score by Weekly Study Hours')
    axes[0].set_xlabel('Weekly Study Hours')
    axes[0].set_ylabel('Average Score')

    sns.barplot(x='TestPrep', y='AverageScore', hue='LunchType', data=df_exam, palette='Set2', ax=axes[1], errorbar='ci')
    axes[1].set_title('Average Score by Test Prep & Lunch Type')
    axes[1].set_xlabel('Test Preparation Course')
    axes[1].set_ylabel('Average Score')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'figure2_study_and_testprep.png'))
    plt.close()

    # Figure 3: Correlation Matrix of 30k Scores
    fig, ax = plt.subplots(figsize=(7, 6))
    score_corr = df_exam[['MathScore', 'ReadingScore', 'WritingScore', 'TotalScore', 'AverageScore', 'WklyStudyHours_Num', 'NrSiblings']].corr()
    sns.heatmap(score_corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Pearson Correlation Heatmap (Exam Scores Dataset)')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'figure3_correlation_exam_scores.png'))
    plt.close()

    # Figure 4: Factors Analysis (Attendance & Study Hours vs Exam Score)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.regplot(x='Hours_Studied', y='Exam_Score', data=df_factors, ax=axes[0], 
                scatter_kws={'alpha': 0.2, 'color': '#2563eb'}, line_kws={'color': '#dc2626'})
    axes[0].set_title(f'Hours Studied vs Exam Score (r = {corr_study_hours[0]:.2f})')
    axes[0].set_xlabel('Hours Studied per Week')
    axes[0].set_ylabel('Exam Score')

    sns.regplot(x='Attendance', y='Exam_Score', data=df_factors, ax=axes[1],
                scatter_kws={'alpha': 0.2, 'color': '#059669'}, line_kws={'color': '#dc2626'})
    axes[1].set_title(f'Attendance (%) vs Exam Score (r = {corr_attendance[0]:.2f})')
    axes[1].set_xlabel('Attendance Rate (%)')
    axes[1].set_ylabel('Exam Score')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'figure4_factors_scatter_regression.png'))
    plt.close()

    print(f"Figures saved to {output_dir}/")
    return summary

if __name__ == "__main__":
    df1 = pd.read_csv("data/processed/cleaned_student_exam_scores.csv")
    df2 = pd.read_csv("data/processed/cleaned_student_factors.csv")
    
    analysis_res = perform_statistical_analysis(df1, df2)
    with open("reports/statistical_analysis_summary.json", "w") as f:
        json.dump(analysis_res, f, indent=4)
    print("Statistical Analysis completed and report exported successfully!")
