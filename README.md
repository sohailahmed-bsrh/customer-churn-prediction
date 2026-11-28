# 📉 Customer Churn Intelligence & Retention Portal

An enterprise-ready, interactive customer attrition analytics and machine learning application built with Python and Streamlit. This project analyzes over 5,000 telecommunications subscriber profiles to identify churn drivers, calculate subscriber attrition probabilities, and recommend automated customer retention strategies.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Key Features

- **Executive Churn Dashboard:** Tracks cohort size, baseline churn rate, average monthly revenue, and average subscriber lifetime.
- **Contract & Financial Risk Mapping:** Detailed breakdown of attrition risks comparing month-to-month contracts vs 1-year and 2-year commitments.
- **Service & Stickiness Analysis:** Quantitative analysis of how value-add services like Tech Support and Online Security reduce subscriber churn.
- **Interactive Churn Risk Calculator:** Real-time risk probability gauge assessing customer profile parameters with automated prescriptive retention playbooks.
- **Model Evaluation Suite:** Random Forest Classifier diagnostics featuring ROC-AUC score, confusion matrix, and feature importance rankings.

---

## 📊 Analytical Methodology

1. **Cohort Segmentation:** Evaluated 5,043 subscriber records across billing methods, contract models, and product categories.
2. **Key Insight:** Month-to-month subscribers exhibit an attrition rate of 42.7%, while 2-year contracted subscribers demonstrate under 3% churn.
3. **Machine Learning Pipeline:** Stratified Random Forest Classifier achieving strong discrimination capability with high ROC-AUC validation score.

---

## 💻 Local Setup & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sohailahmed-bsrh/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 🚀 Deployment on Streamlit Community Cloud

1. Fork or push this repository to your GitHub account (`https://github.com/sohailahmed-bsrh/customer-churn-prediction`).
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**.
4. Select repository: `sohailahmed-bsrh/customer-churn-prediction`
5. Branch: `main`
6. Main file path: `app.py`
7. Click **Deploy!**

---

## 👤 Author

**Sohail Ahmed**
- **GitHub:** [@sohailahmed-bsrh](https://github.com/sohailahmed-bsrh)
- **Portfolio:** [https://sohail.devsil.com](https://sohail.devsil.com)
