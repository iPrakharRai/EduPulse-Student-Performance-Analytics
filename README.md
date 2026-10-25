# EduPulse: Student Performance Analytics & Predictive Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **AI-powered student academic performance analytics and predictive machine learning dashboard using Streamlit.**

---

## 📌 Project Overview
**EduPulse** is an end-to-end Educational Data Mining (EDM) and Predictive Machine Learning web platform designed to analyze multidimensional student data and predict academic outcomes proactively.

Developed as a major college project for the **Department of Computer Science & Engineering (AIML)** at **Goel Institute of Technology & Management**, the system evaluates both demographic/socioeconomic parameters (**30,641 records**) and granular behavioral factors (**6,607 records: attendance rate, weekly study hours, tutoring sessions, and past scores**).

---

## 🚀 Live Deployment on Streamlit Cloud

Deploying this repository to **Streamlit Community Cloud** takes less than 2 minutes:

1. **Fork or Push** this repository to your GitHub account.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your repository: `EduPulse-Student-Performance-Analytics`.
5. Set:
   - **Branch:** `main`
   - **Main file path:** `app.py`
6. Click **"Deploy!"**.

Streamlit Cloud will automatically detect `requirements.txt` and launch your live application.

---

## 💻 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/EduPulse-Student-Performance-Analytics.git
cd EduPulse-Student-Performance-Analytics
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📂 Repository Directory Structure

```
.
├── app.py                                      # Streamlit Web Application (Main Entry Point)
├── requirements.txt                            # Precise dependencies for Streamlit Cloud
├── .gitignore                                  # Git exclusion rules
├── README.md                                   # Project documentation
├── REVISED_PRE_SYNOPSIS.md                     # College Major Project Pre-Synopsis
│
├── data/
│   ├── raw/
│   │   ├── student_exam_scores_30k.csv         # 30,641-row benchmark dataset
│   │   └── student_performance_factors.csv     # 6,607-row behavioral dataset
│   └── processed/
│       ├── cleaned_student_exam_scores.csv     # Zero-null cleaned 30k dataset
│       └── cleaned_student_factors.csv         # Cleaned behavioral factors dataset
│
├── models/
│   ├── best_exam_regressor.pkl                 # Best regressor pipeline (30k dataset)
│   ├── best_factors_regressor.pkl              # Best regressor pipeline (Factors dataset)
│   ├── model_exam_random_forest.pkl            # Trained Random Forest pipeline
│   ├── model_exam_xgboost.pkl                  # Trained XGBoost pipeline
│   ├── model_exam_ridge_regression.pkl         # Trained Ridge Regression pipeline
│   ├── preprocessor_exam_30k.pkl               # Scikit-learn ColumnTransformer
│   └── risk_label_mapping.pkl                  # Label encoder mapping
│
├── src/
│   ├── data_cleaning.py                        # Automated data cleaning pipeline
│   ├── eda_analysis.py                         # EDA and statistical hypothesis tests
│   └── model_training.py                       # ML model training and benchmarking
│
└── reports/
    ├── figures/                                # High-res generated figures (PNG)
    ├── data_cleaning_audit.json                # Imputation audit trail
    ├── statistical_analysis_summary.json       # Hypothesis test statistics (t, F, p)
    ├── model_benchmarks.json                   # Comprehensive ML evaluation metrics
    ├── feature_importance_exam.csv             # Global feature rankings (30k)
    ├── feature_importance_factors.csv          # Global feature rankings (factors)
    └── REVISED_PRE_SYNOPSIS.md                 # Pre-synopsis copy
```

---

## 📊 Key Analytical Findings & Model Benchmarks

### 1. Statistical Hypothesis Testing
* **Test Preparation Course (Two-Sample Welch's $t$-test):**  
  Students who completed test preparation scored an average of **+6.73 points higher** ($72.66$ vs $65.93$, $t = 39.48$, $p < 0.0001$).
* **Socioeconomic Status (Two-Sample $t$-test on Lunch Subsidies):**  
  Standard lunch students scored **+9.57 points higher** than subsidized students ($71.47$ vs $61.90$, $t = 58.02$, $p < 0.0001$).
* **Parental Education (One-Way ANOVA):**  
  $F = 250.89$ ($p < 10^{-300}$), confirming significant correlation with student achievement.
* **Weekly Study Hours (One-Way ANOVA):**  
  $F = 75.66$ ($p < 10^{-32}$), confirming significant score progression with dedicated study time.

### 2. Behavioral Feature Importance (Tree MDI Rankings)
1. **Attendance Rate:** **39.6%** importance ($r = 0.58$)
2. **Weekly Study Hours:** **24.9%** importance ($r = 0.45$)
3. **Previous Academic Scores:** **8.1%** importance
4. **Tutoring Sessions:** **2.9%** importance
5. **Sleep Hours:** **2.3%** importance

### 3. Machine Learning Benchmark Comparison
| Model | Dataset | $R^2$ Variance Explained | RMSE | MAE |
| :--- | :--- | :---: | :---: | :---: |
| **Ridge Regression** | Behavioral Factors | **76.97%** | **1.80** | **0.45** |
| **Gradient Boosting** | Behavioral Factors | **73.78%** | **1.93** | **0.76** |
| **XGBoost Regressor** | Behavioral Factors | **71.86%** | **2.01** | **0.92** |
| **Random Forest Regressor** | Behavioral Factors | **66.21%** | **2.19** | **1.15** |
| **Decision Tree Regressor** | Behavioral Factors | **49.39%** | **2.67** | **1.64** |
| **Ridge Regression** | 30k Exam Scores | **25.89%** | **12.36** | **10.01** |

---

## 🎯 Key Application Features
1. **⚡ Executive Overview & Search:** High-level KPI metric cards and interactive student profile search.
2. **🧹 Data Cleaning & Audit:** Complete visibility into initial null distributions and automated imputation strategies.
3. **📈 Exploratory Data Mining:** Interactive Plotly histograms, box plots, and correlation heatmaps.
4. **🎯 Behavioral Factors Studio:** Scatter regressions evaluating the direct impact of attendance and study habits.
5. **🤖 Predictive AI Models:** Side-by-side ML benchmarking charts ($R^2$, RMSE, MAE) and tree feature importances.
6. **🔮 Live Simulator & What-If:** Real-time form interface predicting a student's final score, letter grade, and risk tier, with AI-recommended interventions.

---

## 👨‍💻 Author & Academic Attribution
* **Project Lead:** Prakhar Rai
* **Roll No:** `2303601530038`
* **Department:** Department of Computer Science & Engineering (AIML)
* **Institution:** Goel Institute of Technology & Management, Lucknow
* **Email:** `raiprakhar0123@gmail.com`

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
