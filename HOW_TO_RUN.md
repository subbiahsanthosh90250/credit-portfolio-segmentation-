# Project Setup & Execution Instructions
## Credit Portfolio Segmentation and Risk Pattern Analysis
**Course:** HCL GUVI Data Science & Machine Learning Program  
**Author:** Subbiah Santhosh .RM  
**Project Folder:** `C:\Users\Subbiah Santhosh .RM\Downloads\Project 3\credit_portfolio_segmentation`

---

## 1. Directory Structure Overview
Ensure your project files are arranged as follows:

```text
credit_portfolio_segmentation/
│
├── data/                                      # Reference dataset location
│   └── loans_granting_model_dataset.csv       (Located in parent Project 3 folder)
│
├── src/
│   ├── preprocess.py                          # Data cleaning, scaling, and feature encoding
│   ├── eda.py                                 # Statistical distributions, correlations, and trend plots
│   └── train_clustering.py                    # PCA, t-SNE, UMAP, K-Means, Hierarchical, DBSCAN pipeline
│
├── models/                                    # Serialized artifacts & pre-computed datasets
│   ├── scaler.pkl                             # Fitted StandardScaler
│   ├── clustering_features.pkl                # Active feature list
│   ├── pca_model.pkl                          # Fitted PCA model (3 components)
│   ├── kmeans_model.pkl                       # Fitted optimal K-Means model (k=4)
│   ├── clustering_algorithm_comparison.csv    # Benchmark metrics (Silhouette, Davies-Bouldin, etc.)
│   ├── cluster_profiles.csv                   # Financial characteristics of each discovered segment
│   ├── anomalies_summary.csv                  # Statistical comparison of regular vs. anomaly borrowers
│   └── clustered_sample.csv                   # Pre-computed sample with 2D/3D embeddings for instant UI load
│
├── reports/                                   # High-resolution publication plots
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
│
├── app.py                                     # Interactive Streamlit Web Dashboard
├── create_presentation.py                     # Automated PPTX slide deck generator
├── Credit_Portfolio_Segmentation_Presentation.pptx # 14-slide visual presentation deck
├── requirements.txt                           # Python package dependencies
├── HOW_TO_RUN.md                              # This execution guide
├── PROJECT_DOCUMENTATION.md                   # Complete part-by-part technical capstone report
└── PRESENTATION_SCRIPT.md                     # Verbal presentation script for GUVI Live Evaluation
```

---

## 2. Environment Setup & Dependency Installation

Open PowerShell or Command Prompt, navigate to the project directory, and install the required dependencies:

```powershell
cd "C:\Users\Subbiah Santhosh .RM\Downloads\Project 3\credit_portfolio_segmentation"
pip install -r requirements.txt
```

*Key Packages Installed:*
- `pandas`, `numpy`, `scipy` – Data manipulation & matrix math
- `scikit-learn` – Preprocessing, PCA, K-Means, Hierarchical Clustering, DBSCAN, metrics
- `umap-learn` – Non-linear manifold learning
- `matplotlib`, `seaborn`, `plotly` – Visualizations & interactive 3D charts
- `streamlit` – Web application framework
- `python-pptx` – Automated PowerPoint generation

---

## 3. Step-by-Step Execution Guide

### Step 1: Execute Exploratory Data Analysis (EDA)
To process the 1.347M records and generate all publication charts:
```powershell
python src/eda.py
```
*Output:* Creates 6 high-resolution figures in the `reports/` folder covering target class balance, numerical distributions, loan purpose default rates, home ownership default rates, correlation matrix, and vintage trends.

---

### Step 2: Run Preprocessing & Clustering Pipeline
To run PCA, evaluate K-Means across $k=2..7$, benchmark against Hierarchical (Ward) and DBSCAN, compute t-SNE and UMAP embeddings, profile segments, and save model artifacts:
```powershell
python src/train_clustering.py
```
*Output:*
- Generates `scaler.pkl`, `kmeans_model.pkl`, `pca_model.pkl` in `models/`.
- Generates benchmark tables and segment profiles in `models/`.
- Saves `clustered_sample.csv` (used for instant loading in the interactive dashboard).
- Saves Scree plot, Elbow/Silhouette curves, manifold scatter plots, and default rate comparison charts in `reports/`.

---

### Step 3: Launch the Interactive Streamlit Dashboard
To launch the 9-tab interactive dashboard in your web browser, run:
```powershell
python -m streamlit run app.py
```
*(Using `python -m streamlit` ensures Windows executes the Python module directly without requiring user Scripts path configuration).*

*Access:* The dashboard will automatically open in your default browser at `http://localhost:8501`.  
*Features:*
- Interactive 3D PCA scatter plot using WebGL.
- Real-time Underwriting Simulator with applicant input sliders.
- Interactive filtering by loan purpose, home ownership, and credit tiers.
- Side-by-side clustering metrics benchmark.

---

### Step 4: Re-generate or Update PowerPoint Presentation (Optional)
If you ever want to regenerate the 14-slide presentation deck:
```powershell
python create_presentation.py
```
*Output:* Writes `Credit_Portfolio_Segmentation_Presentation.pptx`.

---

## 4. Git & GitHub Project Submission Guidelines

To submit this project to GitHub as required by HCL GUVI:

1. Initialize Git in the project root:
```powershell
cd "C:\Users\Subbiah Santhosh .RM\Downloads\Project 3\credit_portfolio_segmentation"
git init
```

2. Add a `.gitignore` to avoid committing large raw datasets:
```powershell
echo "data/" >> .gitignore
echo "*.csv" >> .gitignore
echo "!models/clustering_algorithm_comparison.csv" >> .gitignore
echo "!models/cluster_profiles.csv" >> .gitignore
echo "!models/anomalies_summary.csv" >> .gitignore
echo "!models/clustered_sample.csv" >> .gitignore
```

3. Commit project files:
```powershell
git add .
git commit -m "Complete Credit Portfolio Segmentation & Risk Pattern Analysis Project"
```

4. Push to your GitHub repository:
```powershell
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/credit-portfolio-segmentation.git
git push -u origin main
```

5. Submit the repository link and live Streamlit link (via Streamlit Community Cloud) in the GUVI portal.
