# Customer Churn Prediction & Retention Analytics Engine

<p align="center">
  <a href="https://sohail.devsil.com"><strong>Portfolio</strong></a> | 
  <a href="https://github.com/sohailahmed-bsrh"><strong>GitHub Profile</strong></a> | 
  <a href="https://github.com/sohailahmed-bsrh/customer-churn-prediction"><strong>Source Repository</strong></a>
</p>

---

## Executive Summary & Problem Statement

Acquiring new customers costs five to seven times more than retaining existing ones. In subscription, SaaS, and telecommunications businesses, unexpected customer attrition severely impacts recurring revenue and growth trajectories. Traditional retention strategies react only after cancellation occurs.

This project delivers an end-to-end **Customer Churn Prediction Engine**. Built on advanced supervised classification algorithms and feature engineering techniques, the platform calculates individual customer churn risk scores weeks before departure, isolating primary churn drivers to inform proactive retention interventions.

<div align="center">
  <img src="https://raw.githubusercontent.com/sohailahmed-bsrh/myportfolio/main/assets/images/thumbnail/customer-churn-analysis.jpeg" width="650" alt="Customer Churn Prediction Engine"/>
  <p><em>Predictive customer retention pipeline with class balancing, ensemble classifiers, and feature attribution.</em></p>
</div>

---

## Real-World Industry Applications

- **SaaS Subscription Retention**: Proactively identifies accounts with declining feature usage and schedules proactive customer success outreach.
- **Telecom Churn Mitigation**: Flags mobile subscribers sensitive to contract renewals, competitor pricing, or frequent support tickets.
- **Financial Banking Accounts**: Detects deposit withdrawal acceleration, declining transaction counts, and card inactivity.
- **E-Commerce VIP Retention**: Predicts lapsing VIP customers and triggers automated re-engagement discounts.

---

## Technical Architecture & Engineering Walkthrough

```
                      [Historical Customer Telecom/SaaS Dataset]
                                          │
                                          ▼
                      [Data Cleaning & Feature Engineering]
                   (Tenure Ratios, Contract Type, Payment Method)
                                          │
                                          ▼
                      [Class Imbalance Handling (SMOTE)]
                                          │
                                          ▼
               [Model Training & Hyperparameter Grid Search]
           (Logistic Regression, Random Forest, XGBoost, LightGBM)
                                          │
                                          ▼
                 [Threshold Tuning & Precision-Recall Tradeoff]
                                          │
                                          ▼
                     [SHAP Global & Local Interpretability]
                                          │
                                          ▼
                   [Actionable Churn Risk Scoring Table]
```

### Step 1: Exploratory Analysis & Feature Engineering
Performs rigorous EDA on demographic, contract, payment, and usage attributes. Engineers interaction features including `MonthlyCharges_to_TotalCharges_Ratio` and `Tenure_to_Contract_Length`.

### Step 2: Mitigating Severe Class Imbalance
Customer churn datasets exhibit natural class imbalance (typically 75% retained vs 25% churned). To prevent model bias toward majority classes, the pipeline evaluates Synthetic Minority Over-sampling Technique (SMOTE) and cost-sensitive loss functions (`scale_pos_weight`).

### Step 3: Ensemble Classification & Cross-Validation
Trains and evaluates multiple estimators:
- Logistic Regression (interpretable baseline)
- Random Forest Classifier (non-linear interactions)
- XGBoost & LightGBM (gradient boosted decision trees with regularized split objectives)

### Step 4: Decision Threshold Optimization
Standard 0.5 classification thresholds are sub-optimal for asymmetric business costs (false negatives are far more costly than false positives). Optimizes decision thresholds against the Precision-Recall Curve (PR-AUC) to maximize customer recall without flooding retention teams.

### Step 5: Explainability with SHAP (SHapley Additive exPlanations)
Computes SHAP values to identify global churn catalysts (e.g., month-to-month contracts, lack of online security add-ons) and generate local individual customer waterfall plots.

---

## Technology Stack & Implementation Specifications

| Component | Technology | Version | Architectural Function |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting** | XGBoost / LightGBM | Latest | High-performance regularized tree ensemble models |
| **Machine Learning** | Scikit-Learn | 1.3+ | Preprocessing pipelines, SMOTE integration, validation |
| **Model Explainability** | SHAP | 0.44+ | Game-theoretic feature importance and risk attribution |
| **Data Processing** | Pandas / NumPy | 2.x | Tabular transformation, dummy encoding, missing value imputations |
| **Visualization** | Seaborn / Matplotlib | 3.8+ | ROC curves, Precision-Recall curves, SHAP summary plots |

---

## Installation & Execution Guide

```bash
git clone https://github.com/sohailahmed-bsrh/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
# Windows: venv\Scriptsctivate | Unix: source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/customer_churn_analysis.ipynb
```

---

## Author & Contact Information

<table width="100%">
  <tr>
    <td width="150" align="center" valign="middle">
      <img src="https://avatars.githubusercontent.com/u/231165170?v=4" width="120" style="border-radius: 50%;" alt="Sohail Ahmed"/>
    </td>
    <td>
      <h3>Sohail Ahmed</h3>
      <p><strong>Senior Software Engineer & AI Systems Architect</strong></p>
      <p>Specializing in Full-Stack AI Application Development, Computer Vision, Deep Learning Pipelines, and Scalable Automation Architectures.</p>
      <p>
        <a href="https://sohail.devsil.com"><strong>Portfolio Website</strong></a> | 
        <a href="https://github.com/sohailahmed-bsrh"><strong>GitHub Profile</strong></a> | 
        <a href="https://www.linkedin.com/in/sohail-ahmed-546314316/"><strong>LinkedIn</strong></a> | 
        <a href="mailto:se906403@gmail.com"><strong>Email Contact</strong></a>
      </p>
    </td>
  </tr>
</table>
