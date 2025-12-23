# 👨‍💼 Employee Retention Prediction – Job Change Likelihood

A machine learning–based web application that predicts whether an employee is likely to look for a job change.  
The system helps HR teams and organizations take **data-driven retention decisions** by identifying employees at risk of leaving.

The model is deployed as a **real-time Streamlit web application**.

---

## 🚀 Live Application
👉 **Streamlit App:**  
https://employee-retention-prediction-gdrjfnto6dxghsnxhfaonn.streamlit.app/

---

## 📌 Problem Statement
Employee attrition is a major challenge for organizations. Losing skilled employees increases hiring costs and impacts productivity.

This project aims to:
- Predict **job change likelihood (Yes / No)**
- Analyze employee demographics, experience, training, and company-related factors
- Support proactive employee retention strategies

---

## 🎯 Project Objectives
- Build a robust **classification model** to predict job change likelihood
- Apply **feature engineering** to improve predictive performance
- Handle **imbalanced data** using SMOTE
- Deploy the model as a **real-time web application**
- Ensure the system is **scalable, stable, and cloud-ready**

---

## 🧠 Machine Learning Approach

### ✔ Algorithms Used
- **Random Forest Classifier** (Primary Model)
- Logistic Regression for comparison

### ✔ Techniques Applied
- Feature Engineering
- One-Hot Encoding for categorical variables
- Scaling for numerical features
- **SMOTE** for class imbalance
- ROC-AUC for model evaluation

---

## 🧪 Features Used
- City
- City Development Index
- Training Hours
- Experience (Years)
- Company Size
- Gender
- Relevant Experience
- Enrolled University
- Education Level
- Major Discipline
- Company Type
- Engineered Features:
  - `experience_years`
  - `company_size_num`
  - `last_new_job_num`
  - `training_per_year`

---

## 📊 Model Evaluation
- **Metric Used:** ROC-AUC Score
- Handles class imbalance effectively
- Ensures consistent feature alignment during training and inference

---

## 🌐 Web Application Features
- Interactive input form for employee details
- Dropdown-based city selection (from training data)
- Real-time job change probability prediction
- Clear decision output:
  - **Looking for Job Change**
  - **Not Looking for Job Change**
- Cloud-deployable via Streamlit

---


