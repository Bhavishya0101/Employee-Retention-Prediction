# app.py — Streamlit Cloud Ready (Clean Version)

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Employee Job Change Prediction",
    layout="centered"
)

st.title("👨‍💼 Employee Job Change Prediction")
st.write(
    "Predict whether an employee is likely to look for a job change using Machine Learning."
)

DATA_PATH = "aug_train.csv"
MODEL_PATH = "job_change_model.joblib"

# =====================================================
# FEATURE ENGINEERING
# =====================================================
def clean_experience(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if x == ">20":
        return 21
    if x == "<1":
        return 0
    return float(x)

def clean_company_size(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if "-" in x:
        a, b = x.split("-")
        return (int(a) + int(b)) / 2
    if "10000+" in x:
        return 10000
    if "<10" in x:
        return 5
    return np.nan

def clean_last_new_job(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if x == "never":
        return 0
    if x == ">4":
        return 5
    return float(x)

def feature_engineering(df):
    df = df.copy()
    df["experience_years"] = df["experience"].apply(clean_experience)
    df["company_size_num"] = df["company_size"].apply(clean_company_size)
    df["last_new_job_num"] = df["last_new_job"].apply(clean_last_new_job)
    df["training_per_year"] = df["training_hours"] / (1 + df["experience_years"].fillna(0))
    return df

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()
df = feature_engineering(df)

# Drop ID columns
for col in ["enrollee_id", "id"]:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

X = df.drop(columns=["target"])
y = df["target"].astype(int)

# =====================================================
# TRAIN MODEL (ONCE)
# =====================================================
@st.cache_resource
def train_and_save_model(X, y):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )

    model = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_and_save_model(X, y)

# =====================================================
# REAL-TIME PREDICTION UI
# =====================================================
st.subheader("🔮 Predict Job Change Likelihood")
city = st.text_input("City", "city_1")
city_development_index = st.slider("City Development Index", 0.0, 1.0, 0.6)
training_hours = st.number_input("Training Hours", 0, 1000, 40)
experience = st.number_input("Experience (years)", 0.0, 50.0, 3.0)
last_new_job = st.selectbox("Years Since Last Job Change", ["never", "1", "2", "3", "4", ">4"])
company_size = st.selectbox("Company Size", ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"])
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
relevent_experience = st.selectbox("Relevant Experience", ["Has relevant experience", "No relevant experience"])
enrolled_university = st.selectbox("University Enrollment", ["no_enrollment", "Full time course", "Part time course"])
education_level = st.selectbox("Education Level", ["Graduate", "Masters", "High School", "Phd"])
major_discipline = st.selectbox("Major Discipline", ["STEM", "Arts", "Business", "Humanities", "Other"])
company_type = st.selectbox("Company Type", ["Pvt Ltd", "Public Sector", "Startup", "NGO", "Other"])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "city": city,
        "city_development_index": city_development_index,
        "training_hours": training_hours,
        "experience": experience,
        "last_new_job": last_new_job,
        "company_size": company_size,
        "gender": gender,
        "relevent_experience": relevent_experience,
        "enrolled_university": enrolled_university,
        "education_level": education_level,
        "major_discipline": major_discipline,
        "company_type": company_type
    }])

    input_df = feature_engineering(input_df)

    # 🔐 CRITICAL FIX: ALIGN COLUMNS
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[X.columns]

    prob = model.predict_proba(input_df)[0][1]

    st.metric("Job Change Probability", f"{prob:.2%}")
    st.write(
        "Prediction:",
        "Looking for Job Change" if prob > 0.5 else "Not Looking"
    )


st.markdown("---")
st.caption("Model: Random Forest + SMOTE | Deployment: Streamlit Cloud")


