import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = r"C:\Users\Subbiah Santhosh .RM\Downloads\Project 3\loans_granting_model_dataset.csv"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

EMP_LENGTH_MAPPING = {
    '< 1 year': 0.5,
    '1 year': 1.0,
    '2 years': 2.0,
    '3 years': 3.0,
    '4 years': 4.0,
    '5 years': 5.0,
    '6 years': 6.0,
    '7 years': 7.0,
    '8 years': 8.0,
    '9 years': 9.0,
    '10+ years': 10.0,
    'NI': 0.0
}

TOP_PURPOSES = [
    'debt_consolidation',
    'credit_card',
    'home_improvement',
    'small_business',
    'major_purchase'
]

def load_and_preprocess_data(file_path=DATA_PATH, sample_size=None, random_state=42):
    """
    Load raw loan dataset, clean missing values, handle outliers,
    engineer features, and encode categorical variables.
    """
    print(f"Loading data from: {file_path}")
    if sample_size:
        print(f"Sampling {sample_size:,} records for high-speed modeling...")
        df_raw = pd.read_csv(file_path, low_memory=False)
        df = df_raw.sample(n=sample_size, random_state=random_state).copy()
    else:
        df = pd.read_csv(file_path, low_memory=False)
    
    print(f"Dataset shape loaded: {df.shape}")
    
    # 1. Parse Temporal Features
    df['issue_d'] = df['issue_d'].astype(str)
    # issue_d format is typically 'Dec-2015' or 'Apr-2008'
    date_parts = df['issue_d'].str.split('-', expand=True)
    if date_parts.shape[1] == 2:
        df['issue_month'] = date_parts[0]
        df['issue_year'] = pd.to_numeric(date_parts[1], errors='coerce').fillna(2015).astype(int)
    else:
        df['issue_month'] = 'Unknown'
        df['issue_year'] = 2015

    # 2. Map Employment Length to continuous numerical experience
    df['emp_years'] = df['emp_length'].map(EMP_LENGTH_MAPPING).fillna(0.0)

    # 3. Handle Numerical Missing Values & Outliers
    # Impute missing revenue, dti, fico, loan_amnt if any with median
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['dti_n'] = pd.to_numeric(df['dti_n'], errors='coerce')
    df['loan_amnt'] = pd.to_numeric(df['loan_amnt'], errors='coerce')
    df['fico_n'] = pd.to_numeric(df['fico_n'], errors='coerce')
    
    df['revenue'] = df['revenue'].fillna(df['revenue'].median())
    df['dti_n'] = df['dti_n'].fillna(df['dti_n'].median())
    df['loan_amnt'] = df['loan_amnt'].fillna(df['loan_amnt'].median())
    df['fico_n'] = df['fico_n'].fillna(df['fico_n'].median())

    # Capping outliers to prevent distortion during distance calculations
    # Revenue capped at 99th percentile ($250k) to prevent multi-millionaire skew
    df['revenue_clean'] = np.clip(df['revenue'], 1000, 250000)
    df['log_revenue'] = np.log1p(df['revenue_clean'])
    
    # DTI capped between 0 and 50 (values > 50 represent extreme insolvency or data errors)
    df['dti_clean'] = np.clip(df['dti_n'], 0, 50)
    
    # FICO score naturally between 600 and 850
    df['fico_clean'] = np.clip(df['fico_n'], 600, 850)
    
    # Loan amount naturally bounded between 500 and 40,000
    df['loan_clean'] = np.clip(df['loan_amnt'], 500, 40000)

    # Feature: Loan to Income Ratio
    df['loan_to_income'] = df['loan_clean'] / (df['revenue_clean'] + 1.0)
    df['loan_to_income'] = np.clip(df['loan_to_income'], 0, 1.5)

    # 4. Encode Categorical Variables
    # Clean Home Ownership
    df['home_ownership_clean'] = df['home_ownership_n'].fillna('OTHER').astype(str)
    for category in ['MORTGAGE', 'RENT', 'OWN']:
        df[f'home_{category}'] = (df['home_ownership_clean'] == category).astype(float)
    df['home_OTHER'] = (~df['home_ownership_clean'].isin(['MORTGAGE', 'RENT', 'OWN'])).astype(float)

    # Clean Purpose
    df['purpose_clean'] = df['purpose'].fillna('other').astype(str)
    for p in TOP_PURPOSES:
        df[f'purpose_{p}'] = (df['purpose_clean'] == p).astype(float)
    df['purpose_other'] = (~df['purpose_clean'].isin(TOP_PURPOSES)).astype(float)

    # 5. Define Feature Set for Clustering
    # Exclude ID, Default, issue_d, title, desc, addr_state, zip_code
    clustering_features = [
        'log_revenue',
        'dti_clean',
        'loan_clean',
        'fico_clean',
        'emp_years',
        'loan_to_income',
        'purpose_debt_consolidation',
        'purpose_credit_card',
        'purpose_home_improvement',
        'purpose_small_business',
        'purpose_major_purchase',
        'purpose_other',
        'home_MORTGAGE',
        'home_RENT',
        'home_OWN'
    ]

    X = df[clustering_features].copy()
    
    # 6. Fit Standard Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler and feature names
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(clustering_features, os.path.join(MODEL_DIR, "clustering_features.pkl"))
    print("StandardScaler and feature names saved to models directory.")

    return df, X, X_scaled, clustering_features

if __name__ == "__main__":
    df, X, X_scaled, features = load_and_preprocess_data(sample_size=10000)
    print("Sample preprocessing successful.")
    print(f"Features: {features}")
    print(f"Scaled shape: {X_scaled.shape}")
