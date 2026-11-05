# Comprehensive Project Documentation
# Credit Portfolio Segmentation and Risk Pattern Analysis

**Domain:** Banking, Financial Services, and Credit Risk Analytics  
**Course:** HCL GUVI Advanced Data Science & Machine Learning Capstone  
**Author:** Subbiah Santhosh .RM  
**Dataset:** Lending Club Historical Loan Portfolio (`loans_granting_model_dataset.csv`)  
**Evaluation Scope:** Unsupervised Learning, Dimensionality Reduction, Anomaly Detection, Risk Pattern Analysis, and Streamlit Dashboard Deployment

---

## Table of Contents
1. Executive Summary & Problem Statement
2. Dataset Architecture & Variable Dictionary
3. Data Cleaning, Preprocessing & Feature Engineering
4. Exploratory Data Analysis (EDA) & Empirical Findings
5. Dimensionality Reduction (PCA, t-SNE, UMAP)
6. Clustering Algorithm Implementation (K-Means, Hierarchical, DBSCAN)
7. Quantitative Evaluation & Benchmark Metrics
8. Discovered Borrower Segments & Financial Persona Profiles
9. Historical Default Pattern Analysis (Post-Clustering Cross-Validation)
10. Anomaly & Outlier Risk Detection (DBSCAN Profiling)
11. Interactive Streamlit Dashboard Architecture & Simulator
12. Strategic Business Recommendations & Credit Policy Design
13. Limitations, Ethical Considerations & Future Enhancements

---

## 1. Executive Summary & Problem Statement

### 1.1 Business Context
Retail commercial banks and non-bank financial institutions (NBFIs) face continuous challenges in managing diversified credit portfolios. When extending consumer and commercial credit, institutions deal with applicant pools characterized by heterogeneous income streams, divergent credit histories (FICO scores), varied employment tenures, differing loan purposes, and collateral statuses. 

Traditional credit scoring frameworks rely primarily on rigid, univariate score cut-offs (such as minimum FICO thresholds). While these models capture general creditworthiness, they frequently fail to detect latent, multi-dimensional borrower behavioral patterns. For instance, a borrower with a modest FICO score but high income and home equity backstops may exhibit lower true default risk than a prime-score applicant overleveraged with high debt-to-income (DTI) and multiple unsecured credit lines.

### 1.2 Objective & What Was Asked
As specified in the project guidelines, this study employs **unsupervised machine learning techniques** to:
1. Segment borrowers into homogeneous credit cohorts based purely on financial, employment, and loan characteristics.
2. Apply linear (PCA) and non-linear (t-SNE, UMAP) dimensionality reduction to visualize high-dimensional customer structures.
3. Compare the mathematical clustering performance of **K-Means**, **Hierarchical Clustering (Agglomerative with Ward linkage)**, and **DBSCAN**.
4. Evaluate clustering quality quantitatively using **Silhouette Scores**, **Davies-Bouldin Index**, and **Calinski-Harabasz Index**.
5. Conduct post-clustering **Historical Default Pattern Analysis** by comparing default rates across clusters without using the `Default` status during cluster formation (eliminating data leakage).
6. Detect and profile density-based **anomalies and outlier applicants** using DBSCAN.
7. Deploy an enterprise-grade **Streamlit Interactive Dashboard** featuring an underwriting risk simulator.

---

## 2. Dataset Architecture & Variable Dictionary

The dataset consists of **1,347,681 historical loan applicants** spanning loan originations from **April 2008 to September 2018**, representing over **$19.41 Billion in gross originated debt**.

| Feature Name | Raw Data Type | Financial Interpretation | Role in Project Pipeline |
| :--- | :--- | :--- | :--- |
| `id` | Integer | Unique applicant / loan identifier | Dropped (Identifier) |
| `issue_d` | String | Loan origination date (e.g., 'Dec-2015') | Parsed into `issue_year` & `issue_month` for vintage trend analysis |
| `revenue` | Float | Declared annual income of the borrower | Continuous feature; log-transformed & outlier-capped |
| `dti_n` | Float | Debt-to-Income ratio (percentage) | Continuous feature; bounded between [0, 50%] |
| `loan_amnt` | Integer | Total principal amount borrowed ($500 – $40,000) | Continuous clustering feature |
| `fico_n` | Float | Borrower FICO credit score (612 – 847.5) | Continuous clustering feature |
| `experience_c` | Integer | Binary employment experience flag (1 vs 0) | Dropped (Near-zero variance: 1.347M ones vs 25 zeros) |
| `emp_length` | String | Employment tenure ('10+ years', '3 years', 'NI') | Mapped to continuous years `emp_years` [0 – 10] |
| `purpose` | String | Stated purpose of loan (14 distinct categories) | One-hot encoded for top categories |
| `home_ownership_n` | String | Home ownership status (MORTGAGE, RENT, OWN, OTHER) | One-hot encoded into binary dummy features |
| `addr_state` | String | US State of borrower residence | Retained for geographic distribution analysis |
| `zip_code` | String | Truncated zip code of borrower | Dropped (High cardinality identifier) |
| `Default` | Integer | Historical repayment status (0 = Non-default, 1 = Default) | **Strictly excluded from clustering**; used for post-hoc validation |
| `title` | String | Borrower-entered loan title | Dropped (Free-text duplicate of purpose) |
| `desc` | String | Loan description text (91.16% missing) | Dropped (High missingness & unstructured text) |

---

## 3. Data Cleaning, Preprocessing & Feature Engineering

### 3.1 Handling Missing Values & Deduplication
- Core financial features (`revenue`, `loan_amnt`, `fico_n`, `dti_n`) exhibited complete data integrity with zero missing values.
- Unstructured text attributes (`desc` with 91.16% nulls, `title` with 1.24% nulls) were removed from clustering considerations to avoid high-sparsity noise.
- Single missing record in `zip_code` was safely bypassed.
- No duplicate loan application IDs existed in the dataset.

### 3.2 Mitigation of Skewness & Outlier Treatment
1. **Annual Revenue (`revenue`):**  
   The income distribution exhibited extreme positive right-skewness (mean = $77,370, median = $65,000, max = $10,999,200). Unscaled Euclidean distances would cause ultra-wealthy applicants to pull cluster centroids toward extreme boundaries.  
   *Remedy:* Capped revenue at the 99th percentile ($250,000) and computed `log_revenue = np.log1p(revenue_clean)`.
2. **Debt-to-Income (`dti_n`):**  
   DTI values spanned from 0% to an erroneous maximum of 999%. While DTI values up to 40% are standard in consumer underwriting, ratios exceeding 50% indicate either extreme insolvency or data entry anomalies.  
   *Remedy:* Bounded between 0% and 50% using `np.clip(dti_n, 0, 50)`.
3. **Credit Score (`fico_n`):**  
   Standardized between 600 and 850 points.
4. **Loan-to-Income (LTI) Ratio Engineering:**  
   Engineered a dynamic leverage ratio: $\text{LTI} = \frac{\text{loan\_amnt}}{\text{revenue} + 1.0}$, providing the model with a direct metric of capital exposure relative to cash flow capacity.

### 3.3 Categorical Encoding & Dimensional Standardization
- **Employment Length:** Mapped categorical string levels (`< 1 year` ➔ 0.5, `1 year` ➔ 1.0, ..., `10+ years` ➔ 10.0, and `NI` [Not Indicated] ➔ 0.0) into a continuous numerical variable `emp_years`.
- **One-Hot Encoding:**  
  - `home_ownership_n` encoded into `home_MORTGAGE`, `home_RENT`, `home_OWN`.
  - `purpose` encoded into top business classes: `purpose_debt_consolidation`, `purpose_credit_card`, `purpose_home_improvement`, `purpose_small_business`, `purpose_major_purchase`, and `purpose_other`.
- **Feature Scaling:**  
  Applied `StandardScaler` ($\mu = 0, \sigma = 1$) across all 15 active clustering features to ensure variance parity across bounded metrics, percentages, and log-transformed amounts.

---

## 4. Exploratory Data Analysis (EDA) & Empirical Findings

### 4.1 Target Variable Distribution (`Default`)
- **Total Portfolio Size:** 1,347,681 loans.
- **Fully Paid / Current (Class 0):** 1,078,462 loans (**80.02%**).
- **Default / Charged Off (Class 1):** 269,219 loans (**19.98%**).
- **Portfolio Baseline Default Rate:** **19.98%** (approximately 1 in every 5 originated loans defaulted).

### 4.2 Loan Purpose Risk Analysis
Historical default rates exhibited striking divergence across loan purposes:
- **Small Business Loans:** **29.86% Default Rate** (Highest risk category; nearly 1.5x portfolio baseline).
- **Renewable Energy:** 23.72% Default Rate.
- **Moving & Relocation:** 23.41% Default Rate.
- **Debt Consolidation:** 21.15% Default Rate (Largest volume: 781,206 loans).
- **Credit Card Refinancing:** **16.93% Default Rate** (Significantly safer than general consolidation).
- **Automobile / Car Purchase:** 14.70% Default Rate.
- **Wedding Loans:** 12.43% Default Rate (Lowest risk purpose).

### 4.3 Collateral & Home Ownership Risk Analysis
- **Mortgage Borrowers:** **17.23% Default Rate** (666,594 loans).
- **Homeowners (Own outright):** **20.63% Default Rate** (144,985 loans).
- **Renters:** **23.23% Default Rate** (535,585 loans).  
*Key Takeaway:* Mortgaged borrowers possess verifiable asset equity and established payment histories, demonstrating significantly higher credit resilience than renters.

### 4.4 Feature Correlation Findings
- FICO score shows the strongest negative linear correlation with default (-0.13), demonstrating that credit tier is a consistent driver of solvency.
- Debt-to-Income and Loan Amount exhibit positive correlations with default probability.
- Low cross-correlation among underlying features confirmed absence of multi-collinearity issues.

---

## 5. Dimensionality Reduction (PCA, t-SNE, UMAP)

High-dimensional data in clustering causes distance metrics to degenerate (the "curse of dimensionality"). We implemented three complementary reduction techniques:

### 5.1 Principal Component Analysis (PCA)
- Fitted full PCA decomposition to determine the eigenvalue spectrum and scree plot.
- **Component 1 (PC1):** Heavily aligned with borrower size, loading on `log_revenue`, `loan_amnt`, and `emp_years`.
- **Component 2 (PC2):** Aligned with credit solvency, contrasting `fico_n` against `dti_clean` and `loan_to_income`.
- **Cumulative Variance:** Over 80% of total variance was retained across the leading components, verifying that a 3D PCA projection preserves the principal geometric structures for interactive dashboard rendering.

### 5.2 Non-Linear Manifold Learning (t-SNE vs. UMAP)
- **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Tuned with perplexity = 30 and 1,000 iterations. Preserved local probabilistic neighborhoods, proving the emergence of dense micro-clusters of borrowers with identical debt consolidation profiles.
- **UMAP (Uniform Manifold Approximation and Projection):** Tuned with $n\_neighbors = 30$ and $min\_dist = 0.3$. Preserved both local cluster compactness and global topological distances, successfully separating revolving credit users from large unsecured borrowers with higher computational efficiency.

---

## 6. Clustering Algorithm Implementation (K-Means, Hierarchical, DBSCAN)

### 6.1 K-Means Clustering
- Tested $k \in [2, 7]$ using within-cluster sum of squares (Inertia Elbow Method) and the Silhouette Score.
- **Inertia Trajectory:** $k=2$ (653,203) ➔ $k=3$ (598,935) ➔ $k=4$ (543,108) ➔ $k=5$ (488,755).
- **Silhouette Trajectory:** Monotonically increased from 0.158 ($k=2$) to 0.2104 ($k=4$).
- **Selected Optimal $k = 4$:** Yielded distinct, actionable credit personas with clear business boundaries.

### 6.2 Hierarchical Agglomerative Clustering
- Implemented using **Ward's minimum variance criterion** ($linkage = 'ward'$).
- Evaluated on a representative 8,000-sample matrix to prevent $O(N^2)$ memory exhaustion.
- Achieved a Silhouette Score of **0.2168**, confirming hierarchical consistency with K-Means partitions.

### 6.3 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Configured with $\epsilon = 2.2$ and $min\_samples = 25$.
- Discovered **14 dense core micro-clusters** and successfully flagged **4.17% of records as noise/outliers (`label = -1`)**.
- While not suitable for coarse-grained portfolio segmentation due to variable density, DBSCAN served as a potent **anomaly detection filter**.

---

## 7. Quantitative Evaluation & Benchmark Metrics

The three algorithms were evaluated on identical representative validation matrices using standard cluster cohesion and separation indices:

| Clustering Algorithm | Clusters Identified | Silhouette Score (Higher $\uparrow$) | Davies-Bouldin Index (Lower $\downarrow$) | Calinski-Harabasz Index (Higher $\uparrow$) | Noise / Anomaly Ratio | Production Scalability |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **K-Means Clustering (Selected)** | **4** | **0.2103** | **1.7430** | **1,034.1** | **0.0%** | **High ($O(k \cdot n \cdot d)$)** – Real-time enterprise inference |
| **Hierarchical (Ward)** | 4 | 0.2168 | 1.7941 | 961.8 | 0.0% | **Low ($O(n^2)$ memory)** – Subsampling required |
| **DBSCAN (Density-Based)** | 14 | 0.2820 (core) | 1.3549 | 829.7 | 4.17% | **Medium** – Highly effective for anomaly detection |

---

## 8. Discovered Borrower Segments & Financial Persona Profiles

The 4-cluster K-Means model produced distinct borrower segments with clear underwriting implications:

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Cluster 0: Prime Homeowner Consolidators (6.43% Portfolio Share | Default Rate: 18.43%)        │
│ • Median Revenue: $75,000 | Median FICO: 697 | Median DTI: 14.60% | Collateral: Mortgage       │
│ • Profile: Established homeowners with strong cash flow and minimal non-mortgage debt.        │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Cluster 1: Subprime Renter High-DTI (44.61% Portfolio Share | Default Rate: 17.34%)            │
│ • Median Revenue: $75,000 | Median FICO: 692 | Median DTI: 17.96% | Collateral: Rent           │
│ • Profile: High-earning urban renters with healthy cash flows but lacking physical collateral. │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Cluster 2: Prime Revolving Credit Refinancers (9.53% Portfolio Share | Default Rate: 21.33%)   │
│ • Median Revenue: $60,000 | Median FICO: 692 | Median DTI: 18.44% | Purpose: Credit Card       │
│ • Profile: Middle-income borrowers consolidating high-interest revolving card balances.        │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Cluster 3: High-Exposure Major Borrowers (39.42% Portfolio Share | Default Rate: 22.80%)       │
│ • Median Revenue: $55,500 | Median FICO: 687 | Median DTI: 17.47% | Purpose: Unsecured / Biz   │
│ • Profile: Moderate income facing high leverage; highest risk tier (1.14x baseline default).   │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Historical Default Pattern Analysis (Post-Clustering Cross-Validation)

### 9.1 Methodology & Data Leakage Prevention
To ensure methodological integrity, the target variable `Default` was strictly isolated during data preparation. The clustering algorithms operated exclusively on borrower financial, employment, and loan characteristics.

### 9.2 Post-Clustering Findings
When mapping historical repayment outcomes back to the unsupervised clusters:
- **Segment 1 (Subprime Renter High-DTI):** Default rate = **17.34%** (Risk index = 0.87x vs baseline).
- **Segment 0 (Prime Homeowner Consolidators):** Default rate = **18.43%** (Risk index = 0.92x vs baseline).
- **Segment 2 (Prime Revolving Refinancers):** Default rate = **21.33%** (Risk index = 1.07x vs baseline).
- **Segment 3 (High-Exposure Major Borrowers):** Default rate = **22.80%** (Risk index = **1.14x vs baseline**).

*Empirical Conclusion:* Unsupervised clustering succeeded in partitioning borrowers into statistically distinct risk cohorts. Segment 3, which accounts for nearly 40% of the loan portfolio, generates an outsized portion of total defaults, highlighting where tightened underwriting policies will have maximum loss-reduction impact.

---

## 10. Anomaly & Outlier Risk Detection (DBSCAN Profiling)

Using DBSCAN ($\epsilon = 2.4, min\_samples = 25$), **4.17% (291 samples in the representative sample)** were flagged as noise outliers (`label = -1`).

### Comparative Anomaly Profiling
| Borrower Group | Applicant Count | Mean Revenue ($) | Mean DTI (%) | Mean Loan ($) | Mean FICO | Historical Default Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regular Borrowers (Core Clusters)** | 3,709 | $77,530 | 18.31% | $14,576 | 697.0 | **19.52%** |
| **DBSCAN Anomalies (Outliers)** | 291 | $79,138 | 15.60% | $14,323 | 714.6 | **23.71% (+4.19% Surge)** |

### Anomaly Archetypes
1. **Severe Loan-to-Income Discrepancies:** Applicants requesting maximum loan limits ($35k - $40k) despite volatile or near-minimum declared incomes.
2. **Discordant Credit Metrics:** Borrowers with high FICO scores (> 740) exhibiting extreme DTI ratios (> 45%), signaling rapid balance run-ups prior to borrowing.
3. **Synthetic Identity Indicators:** Irregular combinations of tenure, purpose, and collateral that fall into sparse spatial zones of the feature manifold.

---

## 11. Interactive Streamlit Dashboard Architecture & Simulator

The Streamlit application (`app.py`) provides an intuitive, web-based intelligence platform organized into 9 modules:
1. **Executive Overview & KPIs:** Real-time KPI summary cards displaying portfolio size ($19.41B), default rate (19.98%), and segment share donut chart.
2. **Exploratory Data Analysis (EDA):** Interactive feature distribution viewers, purpose-level default rate comparisons, and correlation matrices.
3. **Dimensionality Reduction Explorer:** WebGL-powered **Interactive 3D PCA scatter plot**, side-by-side 2D PCA, t-SNE, and UMAP scatter views with dynamic color toggling (Cluster vs Default vs Anomaly).
4. **Algorithm Benchmarks:** Tabulated comparison of Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics alongside Elbow and Scree plots.
5. **Borrower Segment Profiles:** Persona cards with deep-dive financial metrics and policy guidelines.
6. **Historical Default Analysis:** Relative risk index bar charts and default rate divergence graphs.
7. **Anomaly Detection Engine:** Scatter plot highlighting density anomalies and comparative risk statistics.
8. **Real-Time Underwriting Simulator:** Interactive application form allowing loan officers to input applicant financial metrics (Income, Loan, FICO, DTI, Employment, Purpose, Home Ownership) and instantly receive:
   - Cluster classification & Persona Assignment.
   - Expected default probability.
   - Formal Policy Decision (`Auto-Approve`, `Verify Income`, `Direct Payout`, or `Manual Underwrite`).
9. **Business Strategy & Credit Policy:** Executive guidelines for capital allocation, risk-based pricing, and compliance.

---

## 12. Strategic Business Recommendations & Credit Policy Design

### 12.1 Risk-Based Pricing Grid (APR Spread Calibration)
- **Prime Segments (Segment 0 & 1):** Prime Pricing: **6.99% – 9.99% APR**. Auto-approval enabled via open banking API income verification.
- **Moderate Risk (Segment 2):** Standard Pricing: **11.99% – 14.99% APR** with credit utilization covenants.
- **High Risk (Segment 3):** Subprime Surcharge: **16.99% – 24.99% APR** (+200-350 bps) to build appropriate loan loss provisions.

### 12.2 Loan Product Tailoring
- **Segment 0 (Prime Homeowners):** Cross-sell secured Home Equity Lines of Credit (HELOC) with lower interest rates and higher loan limits ($50,000+).
- **Segment 2 (Credit Card Consolidators):** Enforce **Direct Creditor Disbursements**—where funds are wired directly to credit card issuers—eliminating borrower cash access and preventing dual-debt accumulation.

### 12.3 Portfolio Capital Allocation Limits
- Limit unsecured balance sheet exposure to **Segment 3 to a maximum of 15%** of deployed capital.
- Allocate **60% – 70%** of origination volume to Segments 0 and 1 to stabilize institutional default rates below 18%.

---

## 13. Limitations, Ethical Considerations & Future Enhancements

### 13.1 Limitations
- Dataset contains historical closed-loan outcomes; rejected loan applicants are not included, introducing survivor bias.
- Macroeconomic conditions (interest rate cycles, inflation shocks) during 2008–2018 influenced default rates and may differ in contemporary economic environments.

### 13.2 Ethical Data Usage & Fair Lending Compliance
- Excluded protected class proxies (demographic, gender, racial indicators) from clustering features to uphold Fair Housing Act (FHA) and Equal Credit Opportunity Act (ECOA) standards.
- Cluster assignments represent statistical cohort memberships, not definitive individual credit decisions.

### 13.3 Future Enhancements
- Integration of alternative credit data (utility bills, rental payment verification, open banking cash flow streams).
- Real-time online stream clustering (e.g., Mini-Batch K-Means) to adjust cluster centroids continuously as new macroeconomic vintages mature.
