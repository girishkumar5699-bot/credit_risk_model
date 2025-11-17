Credit Risk Scoring with SHAP and LIME

Project Report

1. Introduction

Credit risk modeling is a critical task in financial services, enabling institutions to assess the likelihood of loan default. This project builds a high-performance, interpretable machine learning pipeline using LightGBM, with SHAP and LIME for model transparency. The goal is to support ethical, data-driven decision-making for loan approvals.

2. Dataset Overview

Source: Synthetic credit risk dataset

Size: ~32,000 records

Target Variable: loan_status (0 = No Default, 1 = Default)

Key Features:

person_income, person_age, person_home_ownership

loan_intent, loan_amount, loan_percent_income

cb_person_default_on_file, loan_int_rate

3. Modeling Approach

Algorithm: LightGBM Classifier

Train/Test Split: 80/20

Performance Metrics:

AUC Score: 0.9522

Accuracy: 91%

Precision: 89%

Recall: 87%

F1 Score: 88%

Model was tuned using early stopping and class balancing to handle default skew.

4. Interpretability with SHAP

4.1 Global Feature Importance

SHAP summary plot revealed the top 5 risk drivers:

person_income

loan_percent_income

loan_intent_VENTURE

loan_int_rate

person_home_ownership_OWN

These features consistently influenced predictions across the dataset.

4.2 Local Explanations

SHAP force plots were generated for three representative cases:

Low-Risk: High income, low loan burden

High-Risk: High interest rate, short employment

Borderline: Mixed signals across credit history and loan intent

Plots are saved as HTML files for stakeholder review.

5. Interpretability with LIME

LIME was used to generate local surrogate models for the same three cases. Each explanation highlights the top 10 features influencing the prediction. HTML outputs were saved for interactive review.

6. SHAP vs LIME: Comparative Analysis

Aspect

SHAP

LIME

Scope

Global + Local

Local only

Method

Shapley values

Local surrogate model

Stability

High

Medium (sampling-dependent)

Feature Interactions

Captures interactions

Assumes linearity

Use Case

Risk profiling, fairness, compliance

Case-by-case justification

SHAP was more consistent and robust, while LIME offered intuitive explanations for individual decisions.

7. Fairness Audit

SHAP values were grouped by person_gender and person_age to assess bias.

Findings:

SHAP values for loan_percent_income showed slightly higher impact on female applicants.

person_income SHAP values were more variable for younger age groups.

No significant bias found in loan_intent or loan_grade.

Recommendation:

Monitor model drift across demographic segments.

Consider fairness-aware retraining if bias persists.

8. Deliverables

All outputs are organized in the repository:

credit-risk-ml/
├── data/                  # Dataset
├── notebooks/             # Jupyter notebook
├── reports/               # This report + analysis
├── outputs/               # SHAP & LIME visuals + model
├── README.md              # Project overview
└── requirements.txt       # Dependencies

Artifacts include:

bestmodel.pkl

SHAP summary and force plots

LIME HTML explanations

Top 5 SHAP insights (top5_shap_insights.txt)

Comparative analysis (shap_vs_lime_analysis.md)

Fairness audit plots

9. Conclusion

This project delivers a complete, interpretable credit risk scoring pipeline. It balances predictive performance with transparency, enabling ethical and explainable loan decisions. SHAP and LIME together provide a robust framework for stakeholder trust and regulatory compliance.
