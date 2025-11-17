# Interpretable Credit Risk Scoring with SHAP and LIME

## Project Overview
This project builds a high-performance machine learning model to predict loan default risk using LightGBM. It emphasizes interpretability through SHAP and LIME, enabling transparent decision-making for loan officers and stakeholders.

## Objectives
- Train a robust classifier for credit risk prediction
- Provide global and local model interpretability using SHAP
- Generate case-specific explanations using LIME
- Compare SHAP and LIME for interpretability effectiveness
- Deliver actionable insights for business stakeholders

## Model Summary
- Algorithm: LightGBM Classifier
- Target: `loan_status` (0 = No Default, 1 = Default)
- Performance: AUC Score = 0.9522

## Interpretability Techniques
### SHAP
- Global feature importance ranking
- Local force plots for individual predictions
- Top 5 risk drivers identified

### LIME
- Local explanations for selected loan cases:
  - Low-risk applicant
  - High-risk applicant
  - Borderline case

## Repository Structure
credit-risk-ml/ ├── data/                  # Raw dataset │   └── credit_risk_dataset.csv ├── notebooks/             # Jupyter notebook │   └── credit_risk_model.ipynb ├── reports/               # Analysis and insights │   ├── shap_vs_lime_analysis.md │   └── top5_shap_insights.txt ├── outputs/               # Visualizations and model artifacts │   ├── shap_summary_plot.png │   ├── shap_force_low_risk.html │   ├── lime_explanation_high_risk.html │   ├── bestmodel.pkl │   └── ... ├── README.md              # Project overview └── requirements.txt       # Python dependencies

## Key Deliverables
- Clean and reproducible notebook
- SHAP summary and force plots
- LIME HTML explanations
- Comparative analysis of SHAP vs LIME
- Top 5 SHAP insights for loan officers
- Exported model (`bestmodel.pkl`)
- Optional fairness audit using SHAP grouped by protected attributes

## Setup Instructions

# Clone the repository
git clone https://github.com/your-username/credit-risk-ml.git
cd credit-risk-ml

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
