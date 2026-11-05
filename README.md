# Credit Portfolio Segmentation & Risk Pattern Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Domain-Retail%20Banking%20%26%20Credit%20Risk-emerald.svg)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Algorithms](https://img.shields.io/badge/Algorithms-K--Means%20%7C%20Ward%20%7C%20DBSCAN-orange.svg)]()
[![Manifold Learning](https://img.shields.io/badge/Projections-PCA%20%7C%20t--SNE%20%7C%20UMAP-purple.svg)]()

> **HCL GUVI Capstone Project**  
> **Author:** Subbiah Santhosh .RM  
> **Domain:** Finance & Banking — Loan Analytics and Credit Risk Profiling

---

## 📌 Executive Summary
Financial institutions manage massive credit portfolios with substantial heterogeneity across borrower incomes, credit scores, debt-to-income (DTI) ratios, and loan intentions. Traditional credit scoring frameworks often rely on rigid, univariate thresholds that fail to detect subtle, multi-dimensional risk patterns.

This project implements an **unsupervised machine learning framework** on a historical Lending Club portfolio of **1,347,681 loan records ($19.41 Billion gross capital)**. The pipeline groups borrowers based purely on financial and credit characteristics, validates repayment outcomes post-clustering without target leakage, flags high-risk anomalies, and provides an interactive Streamlit intelligence dashboard with a live underwriting simulator.

---

## 🏗️ System Architecture

```text
Raw Lending Club Portfolio (1.35M Loans)
       │
       ▼
Data Cleaning & Preprocessing (preprocess.py)
├── Missing Value Imputation & Outlier Capping (Revenue clipped at $250k, DTI at 50%)
├── Skewness Mitigation (log1p transform on Annual Revenue)
├── Categorical One-Hot Encoding (Home Ownership, Stated Purpose)
├── Derived Feature Engineering (Loan-to-Income Leverage Ratio)
└── Feature Scaling (StandardScaler on 15 active features)
       │
       ├───► Dimensionality Reduction
       │     ├── PCA Scree Analysis (80%+ variance retained, 2D/3D embeddings)
       │     ├── t-SNE Manifold (Perplexity=30, local micro-clusters)
       │     └── UMAP Topology (n_neighbors=30, min_dist=0.3)
       │
       ├───► Unsupervised Clustering & Benchmarking
       │     ├── K-Means (Optimal k=4 via Elbow & Silhouette)
       │     ├── Hierarchical Clustering (Agglomerative with Ward Linkage)
       │     └── DBSCAN Density Filtering (eps=2.2, min_samples=25)
       │
       ├───► Empirical Risk Cross-Validation
       │     └── Historical Default Rate Comparison (17.34% to 22.80% risk spread)
       │
       ├───► Density Anomaly Detection
       │     └── DBSCAN Outlier Profiling (4.17% flagged, 23.71% default rate)
       │
       └───► Interactive Deployment (app.py)
             └── Streamlit Dashboard + Real-Time Applicant Underwriting Simulator
```

---

## 📊 Key Results & Discovered Borrower Segments

The optimal 4-cluster K-Means model revealed distinct, actionable credit cohorts:

| Segment ID | Persona Name | Portfolio Share | Median Income | Median FICO | Median DTI | Default Rate | Risk Multiplier | Underwriting Policy Action |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **Prime Homeowner Consolidators** | 6.43% | $75,000 | 697 | 14.60% | **18.43%** | 0.92x | **Auto-Approve**, prime APR (6.99%–9.99%), cross-sell HELOCs |
| **1** | **Subprime Renter High-DTI** | 44.61% | $75,000 | 692 | 17.96% | **17.34%** | 0.87x | Standard approval with digital bank payroll verification; $25k cap |
| **2** | **Prime Revolving Refinancers** | 9.53% | $60,000 | 692 | 18.44% | **21.33%** | 1.07x | **Direct Creditor Payout** (disburse to card issuers to prevent dual debt) |
| **3** | **High-Exposure Major Borrowers** | 39.42% | $55,500 | 687 | 17.47% | **22.80%** | **1.14x** | **Manual Underwriting Required**, APR risk premium (+250 bps) |

### Algorithmic Performance Benchmarks

| Algorithm | Clusters | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Noise / Anomaly Ratio | Production Scalability |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **K-Means Clustering (Selected)** | **4** | **0.2103** | **1.7430** | **1,034.1** | **0.0%** | **High ($O(k \cdot n \cdot d)$)** – Real-time enterprise scoring |
| **Hierarchical (Ward)** | 4 | 0.2168 | 1.7941 | 961.8 | 0.0% | **Low ($O(n^2)$ memory)** – Subsampling required |
| **DBSCAN (Density-Based)** | 14 | 0.2820 (core) | 1.3549 | 829.7 | 4.17% | **High Precision** – Outlier & fraud screening |

### Anomaly Detection Highlights
- **DBSCAN Outlier Cohort:** Flagged **4.17%** of the applicant pool as spatial noise anomalies.
- **Empirical Default Risk:** Anomalies defaulted at **23.71% vs. 19.52%** for regular borrowers (**+4.19% risk surge**).
- **Primary Anomalous Signatures:** Excessive Loan-to-Income (> 0.85) and discordant FICO scores paired with top-tier loan requests.

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/subbiahsanthosh90250/credit-portfolio-segmentation-.git
cd credit-portfolio-segmentation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Scripts
```bash
# 1. Exploratory Data Analysis & Plot Generation
python src/eda.py

# 2. Dimensionality Reduction & Model Training
python src/train_clustering.py

# 3. Launch Interactive Streamlit Dashboard
python -m streamlit run app.py
```

---

## 🖥️ Streamlit Dashboard Modules (`app.py`)
1. **Executive Overview & KPIs:** Key portfolio metrics ($19.41B capital, 19.98% baseline default rate).
2. **Exploratory Data Analysis:** Interactive distributions, correlation heatmap, loan purpose default breakdown.
3. **Dimensionality Reduction:** WebGL-accelerated interactive 3D PCA scatter plot, t-SNE, and UMAP embeddings.
4. **Algorithm Benchmarking:** Side-by-side performance comparison tables, Elbow curve, and Scree plots.
5. **Discovered Borrower Segments:** In-depth persona profiles, radar charts, and behavioral descriptions.
6. **Historical Default Analysis:** Cross-validation of repayment rates across clusters without target leakage.
7. **Anomaly Detection:** DBSCAN noise cluster visualization and comparative risk statistics.
8. **Real-Time Underwriting Simulator:** Live applicant scoring form outputting cluster assignment, expected default rate, and recommended underwriting action (`Auto-Approve`, `Verify`, `Manual Review`).
9. **Strategic Business Policy:** Recommendations for risk-based pricing grids, collateral rules, and balance sheet exposure caps.

---

## 📁 Repository Structure
```text
credit_portfolio_segmentation/
├── src/
│   ├── preprocess.py                          # Data cleaning, scaling, and feature encoding
│   ├── eda.py                                 # Statistical distributions, correlations, and trend plots
│   └── train_clustering.py                    # PCA, t-SNE, UMAP, K-Means, Hierarchical, DBSCAN pipeline
├── models/                                    # Serialized models & sample pre-computed embeddings
│   ├── scaler.pkl
│   ├── clustering_features.pkl
│   ├── pca_model.pkl
│   ├── kmeans_model.pkl
│   ├── clustering_algorithm_comparison.csv
│   ├── cluster_profiles.csv
│   ├── anomalies_summary.csv
│   └── clustered_sample.csv
├── reports/                                   # High-resolution publication charts
│   ├── eda_target_distribution.png
│   ├── eda_numerical_distributions.png
│   ├── eda_default_by_purpose.png
│   ├── eda_default_by_home_ownership.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_default_trend_by_year.png
│   ├── pca_variance_plot.png
│   ├── cluster_elbow_silhouette.png
│   ├── algorithm_metrics_comparison.png
│   ├── projection_visualizations.png
│   └── cluster_default_rates.png
├── app.py                                     # Interactive Streamlit Web Application
├── create_presentation.py                     # Python script to generate PowerPoint presentation
├── Credit_Portfolio_Segmentation_Presentation.pptx # 14-slide visual presentation deck
├── requirements.txt                           # Python dependencies
├── HOW_TO_RUN.md                              # Detailed execution guide
├── PROJECT_DOCUMENTATION.md                   # Full part-by-part technical capstone report
├── PRESENTATION_SCRIPT.md                     # Verbal script for GUVI live evaluation
└── README.md                                  # Repository overview
```

---

## 📜 Academic Integrity & Contact
- **Course:** HCL GUVI Data Science & Machine Learning Program
- **Author:** Subbiah Santhosh .RM
- **Submission Date:** November 2026
