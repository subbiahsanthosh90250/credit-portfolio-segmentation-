# Capstone Live Evaluation Presentation Script
## Project: Credit Portfolio Segmentation & Risk Pattern Analysis
**Domain:** Retail Banking, Lending Analytics & Credit Risk Management  
**Author / Presenter:** Subbiah Santhosh .RM  
**Audience:** HCL GUVI Technical Evaluation Panel & Evaluators  
**Companion Deck:** `Credit_Portfolio_Segmentation_Presentation.pptx` (14 Slides)

---

### Slide 1: Title Slide (Duration: 30 seconds)
> **Spoken Presentation:**
> "Good morning / afternoon, respected mentors and evaluators. My name is Subbiah Santhosh, and today I am pleased to present my Capstone Project: **'Credit Portfolio Segmentation and Risk Pattern Analysis'**. 
> In this project, I engineered an unsupervised machine learning architecture on over 1.34 million loan records to discover natural borrower segments, evaluate linear and non-linear manifold representations, identify credit risk patterns, detect high-risk anomalies, and deploy an interactive underwriting intelligence dashboard."

---

### Slide 2: Executive Summary & Project Objectives (Duration: 45 seconds)
> **Spoken Presentation:**
> "In consumer and commercial lending, financial institutions manage portfolios with millions of borrowers who possess vastly different income levels, credit scores, debt-to-income burdens, and repayment capabilities. Traditional rule-based scoring models often fail to capture subtle multi-dimensional borrower behaviors.
> 
> To solve this, my project achieved six core objectives:
> 1. Cleansed and standardized a 167 MB portfolio of 1.347 million loans.
> 2. Applied PCA, t-SNE, and UMAP for both global variance preservation and local manifold visualization.
> 3. Implemented and benchmarked three distinct clustering algorithms: K-Means, Hierarchical Agglomerative Clustering, and DBSCAN.
> 4. Validated historical default rates across discovered borrower segments without any target data leakage.
> 5. Designed a density-based anomaly detection filter that isolated high-risk borrowers exhibiting a 23.71% default rate.
> 6. Deployed an interactive Streamlit application equipped with a real-time underwriting simulator for loan officers."

---

### Slide 3: Dataset Architecture & Preprocessing Pipeline (Duration: 1 minute)
> **Spoken Presentation:**
> "Looking at the data pipeline on Slide 3, the Lending Club dataset contains 1,347,681 historical loans originating between 2008 and 2018, representing over $19.4 Billion in total loan volume. The baseline default rate of the entire portfolio is 19.98%, or approximately 269,000 charged-off loans.
> 
> To guarantee clean clustering, I followed several rigorous preprocessing steps:
> First, the target column `Default` was strictly isolated and removed from all clustering features so our unsupervised models had zero information leakage.
> Second, unique identifiers such as loan ID, zip codes, and unstructured text descriptions were removed.
> Third, I addressed severe right-tail skewness in annual revenue by clipping extreme multi-millionaire outliers at the 99th percentile ($250,000) and applying a natural log transformation.
> Fourth, debt-to-income (DTI) was bounded to 50% to prevent extreme ratios from distorting Euclidean distances.
> Fifth, categorical attributes—including home ownership and loan purpose—were one-hot encoded, and employment length was mapped to continuous years. Finally, all numerical features were scaled using StandardScaler."

---

### Slide 4: Exploratory Data Analysis & Feature Distributions (Duration: 45 seconds)
> **Spoken Presentation:**
> "Slide 4 highlights our core feature distributions:
> - Annual income centers around a median of $65,000, with the interquartile range spanning from $46,600 to $92,000.
> - The median loan amount is $12,000, bounded between $500 and $40,000.
> - FICO credit scores follow a bell curve centered at a median of 692, with an interquartile range of 672 to 712.
> - The debt-to-income distribution shows that the typical borrower spends 17.6% of their monthly income servicing non-mortgage debt, with standard industry risk limits around 35%."

---

### Slide 5: Risk Drivers: Loan Purpose & Collateral Impact (Duration: 45 seconds)
> **Spoken Presentation:**
> "On Slide 5, we analyze historical default rates across loan purposes and collateral ownership:
> - Notice that **Small Business Loans** exhibit by far the highest portfolio risk, with a default rate of **29.86%**—almost 10 percentage points higher than the portfolio baseline. In contrast, credit card refinancing defaults at only 16.93%, and car loans at 14.70%.
> - Looking at home ownership, borrowers with a **Mortgage** exhibit a significantly lower default rate of **17.23%**, whereas **Renters** default at **23.23%**. This confirms that property ownership acts as a strong financial and psychological anchor against default."

---

### Slide 6: Dimensionality Reduction: Principal Component Analysis (Duration: 45 seconds)
> **Spoken Presentation:**
> "To address multi-collinearity and enable low-dimensional visualization, I applied Principal Component Analysis.
> As displayed in the Scree plot on Slide 6:
> - Principal Component 1 captures borrower scale and leverage, heavily loading on annual revenue and loan amount.
> - Principal Component 2 captures creditworthiness and solvency capacity, contrasting FICO scores with DTI.
> - Over 80% of cumulative variance is retained across the leading components, verifying that dimensionality reduction preserves the critical underlying variance of the portfolio."

---

### Slide 7: Non-Linear Manifold Visualizations: PCA vs. t-SNE vs. UMAP (Duration: 1 minute)
> **Spoken Presentation:**
> "Slide 7 compares our three dimensionality reduction approaches:
> - **PCA** captures global linear spread, illustrating continuous transitions across income and loan size.
> - **t-SNE** preserves local probabilistic neighborhoods, revealing dense borrower sub-clusters.
> - **UMAP** achieves the best balance: it retains both local neighborhood cohesion and global topological distances, clearly demonstrating the separation between conservative debt consolidators and high-exposure borrowers."

---

### Slide 8 & 9: Clustering Optimization & Algorithm Benchmarking (Duration: 1 minute)
> **Spoken Presentation:**
> "Moving to Slides 8 and 9, I evaluated cluster cohesion and separation using the Elbow Method (within-cluster sum of squares) and the Silhouette Score across $k=2$ to $k=7$. The optimal inflection point emerged at $k=4$, which offered maximum financial interpretability and balanced silhouette performance.
> 
> In Slide 9, I systematically benchmarked three algorithms:
> 1. **K-Means Clustering** achieved a Silhouette Score of **0.2103**, a Davies-Bouldin Index of **1.7430**, and a Calinski-Harabasz score of **1,034.1**. Because of its linear computational complexity, it is the most suitable model for high-throughput enterprise retail banking.
> 2. **Hierarchical (Agglomerative with Ward Linkage)** achieved a slightly higher Silhouette Score of **0.2168**, but its quadratic memory footprint ($O(N^2)$) prevents it from scaling to 1.35 million records without subsampling.
> 3. **DBSCAN** discovered 14 core density clusters with a Silhouette Score of **0.2820** and isolated **4.17% of records as noise anomalies**."

---

### Slide 10: Discovered Borrower Segments & Personas (Duration: 1 minute 15 seconds)
> **Spoken Presentation:**
> "Slide 10 presents our four discovered borrower segments:
> - **Segment 0: Prime Homeowner Consolidators (6.4% of portfolio):** High median income ($75,000), top-tier FICO (697), lowest DTI (14.6%), backed by home equity. Recommended action: Fast-track auto-approval and prime APR pricing.
> - **Segment 1: Subprime Renter High-DTI (44.6% of portfolio):** High earning power ($75,000 median) but renting and carrying an 18.0% DTI. Historical default rate is modest (17.34%). Recommended action: Digital payroll verification and standard unsecured loans up to $25,000.
> - **Segment 2: Prime Revolving Refinancers (9.5% of portfolio):** Moderate income ($60,000) seeking credit card consolidation. Default rate is slightly elevated at 21.33%. Recommended action: Disburse funds directly to existing credit card accounts to prevent borrowers from racking up duplicate debt.
> - **Segment 3: High-Exposure Major Borrowers (39.4% of portfolio):** Lower median income ($55,500), lower FICO (687), and highest historical default rate (**22.80%**—1.14 times portfolio baseline). Recommended action: Strict manual underwriting and an APR risk premium of +250 basis points."

---

### Slide 11: Historical Default Pattern Cross-Validation (Duration: 45 seconds)
> **Spoken Presentation:**
> "Slide 11 shows the empirical validation of our unsupervised approach. 
> Even though our clustering models never saw the historical `Default` target, the segments naturally separated by risk:
> - Segment 1 and Segment 0 represent our safest borrower pools with default rates of 17.34% and 18.43%.
> - Segment 3 carries a 22.80% default rate. Because Segment 3 represents nearly 40% of all loans, targeted underwriting adjustments on this group can prevent the vast majority of future portfolio charge-offs."

---

### Slide 12: Anomaly & Fraud Detection via DBSCAN (Duration: 45 seconds)
> **Spoken Presentation:**
> "On Slide 12, our density-based DBSCAN model identified 4.17% of the applicant pool as noise outliers.
> When we cross-referenced these anomalies against historical outcomes, we discovered that **anomalous borrowers defaulted at 23.71%**, compared to 19.52% for regular borrowers—representing an alarming **+4.19 percentage point risk surge**.
> These profiles typically exhibit high requested loan amounts despite lower incomes, or high credit scores paired with extreme DTI ratios. In production, this DBSCAN filter serves as an automated anti-fraud gate."

---

### Slide 13: Interactive Streamlit Underwriting Dashboard (Duration: 45 seconds)
> **Spoken Presentation:**
> "To deliver maximum business value, I built a comprehensive Streamlit dashboard containing 9 modular analytical tabs:
> - It includes WebGL-accelerated 3D PCA scatter plots, correlation heatmaps, and side-by-side model benchmarks.
> - Crucially, it features a **Real-Time Underwriting Simulator** where a credit officer enters an applicant's financial attributes, and the system instantly executes scaling, predicts the segment, calculates expected default probability, and issues an actionable underwriting recommendation."

---

### Slide 14: Strategic Recommendations & Governance (Duration: 45 seconds)
> **Spoken Presentation:**
> "In conclusion, on Slide 14, I outline four strategic recommendations:
> 1. Implement risk-based APR pricing grids to build necessary loan loss reserves on higher-risk segments.
> 2. Enforce direct creditor disbursements for debt consolidation loans.
> 3. Limit portfolio exposure to Segment 3 to a maximum of 15% of total deployed balance sheet capital.
> 4. Ensure compliance with Basel II/III internal rating-based governance and maintain quarterly centroid drift monitoring.
> 
> Thank you for your time and guidance throughout this project. I am now open to any questions."
