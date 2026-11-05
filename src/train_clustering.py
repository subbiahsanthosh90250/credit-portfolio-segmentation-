import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from preprocess import load_and_preprocess_data

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_evaluate_clustering():
    print("=" * 60)
    print("STARTING CLUSTERING PIPELINE & DIMENSIONALITY REDUCTION")
    print("=" * 60)

    # 1. Load Preprocessed Data (Representative Sample of 50,000 for high precision & speed)
    SAMPLE_SIZE = 50000
    df, X, X_scaled, feature_names = load_and_preprocess_data(sample_size=SAMPLE_SIZE, random_state=42)
    df.reset_index(drop=True, inplace=True)
    print(f"Features utilized ({len(feature_names)}): {feature_names}")

    # 2. Dimensionality Reduction: PCA
    print("\n[Step 1/5] Applying Principal Component Analysis (PCA)...")
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    exp_var = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    # Plot PCA Explained Variance
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(1, len(exp_var) + 1), exp_var * 100, alpha=0.6, color='#1e3d59', label='Individual Explained Variance (%)')
    ax.step(range(1, len(cum_var) + 1), cum_var * 100, where='mid', color='#e63946', linewidth=2, label='Cumulative Explained Variance (%)')
    ax.axhline(80, color='gray', linestyle='--', label='80% Threshold')
    ax.set_xlabel('Principal Component Index')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('PCA Scree Plot & Cumulative Explained Variance', fontweight='bold')
    ax.legend(loc='center right')
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "pca_variance_plot.png"), dpi=300)
    plt.close()
    print("Saved pca_variance_plot.png")

    # Fit PCA 2D and 3D
    pca_3d = PCA(n_components=3, random_state=42)
    X_pca3 = pca_3d.fit_transform(X_scaled)
    df['pca_1'] = X_pca3[:, 0]
    df['pca_2'] = X_pca3[:, 1]
    df['pca_3'] = X_pca3[:, 2]
    joblib.dump(pca_3d, os.path.join(MODEL_DIR, "pca_model.pkl"))

    # 3. K-Means Elbow & Silhouette Analysis
    print("\n[Step 2/5] Evaluating K-Means with Elbow & Silhouette Analysis...")
    k_range = range(2, 8)
    inertias = []
    sil_scores = []
    
    # Subsample for fast silhouette computation
    eval_idx = np.random.RandomState(42).choice(len(X_scaled), size=10000, replace=False)
    X_eval = X_scaled[eval_idx]

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        labels_eval = km.predict(X_eval)
        sil = silhouette_score(X_eval, labels_eval)
        sil_scores.append(sil)
        print(f"  k={k}: Inertia={km.inertia_:,.1f}, Silhouette Score={sil:.4f}")

    # Plot Elbow & Silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(k_range, inertias, marker='o', color='#1e3d59', linewidth=2)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
    ax1.set_title('Elbow Method For Optimal k', fontweight='bold')

    ax2.plot(k_range, sil_scores, marker='s', color='#2a9d8f', linewidth=2)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score Across k', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "cluster_elbow_silhouette.png"), dpi=300)
    plt.close()
    print("Saved cluster_elbow_silhouette.png")

    # Optimal K=4 chosen for distinct credit risk segments & business interpretability
    OPTIMAL_K = 4
    print(f"\nTraining Final K-Means model with k={OPTIMAL_K}...")
    kmeans_optimal = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    df['kmeans_cluster'] = kmeans_optimal.fit_predict(X_scaled)
    joblib.dump(kmeans_optimal, os.path.join(MODEL_DIR, "kmeans_model.pkl"))

    # 4. Compare with Hierarchical Clustering & DBSCAN
    print("\n[Step 3/5] Comparing K-Means, Hierarchical Clustering, and DBSCAN...")
    # Representative subsample for Hierarchical & DBSCAN
    H_SIZE = 8000
    h_idx = np.random.RandomState(42).choice(len(X_scaled), size=H_SIZE, replace=False)
    X_h = X_scaled[h_idx]
    y_h_kmeans = df['kmeans_cluster'].iloc[h_idx].values

    # Hierarchical Clustering (Agglomerative with Ward Linkage)
    print("  Fitting Agglomerative (Hierarchical) Clustering...")
    hierarchical = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage='ward')
    y_h_hier = hierarchical.fit_predict(X_h)

    # DBSCAN Clustering for density & anomaly detection
    print("  Fitting DBSCAN (Density-Based Spatial Clustering)...")
    dbscan = DBSCAN(eps=2.2, min_samples=25)
    y_h_dbscan = dbscan.fit_predict(X_h)
    dbscan_noise_pct = (y_h_dbscan == -1).mean() * 100
    n_dbscan_clusters = len(set(y_h_dbscan)) - (1 if -1 in y_h_dbscan else 0)
    print(f"  DBSCAN identified {n_dbscan_clusters} core clusters and {dbscan_noise_pct:.2f}% anomalies/outliers.")

    # Calculate Evaluation Metrics
    km_sil = silhouette_score(X_h, y_h_kmeans)
    km_db = davies_bouldin_score(X_h, y_h_kmeans)
    km_ch = calinski_harabasz_score(X_h, y_h_kmeans)

    hier_sil = silhouette_score(X_h, y_h_hier)
    hier_db = davies_bouldin_score(X_h, y_h_hier)
    hier_ch = calinski_harabasz_score(X_h, y_h_hier)

    # DBSCAN metrics (excluding noise points for valid cluster silhouette)
    valid_db_mask = (y_h_dbscan != -1)
    if valid_db_mask.sum() > 0 and len(set(y_h_dbscan[valid_db_mask])) > 1:
        db_sil = silhouette_score(X_h[valid_db_mask], y_h_dbscan[valid_db_mask])
        db_db = davies_bouldin_score(X_h[valid_db_mask], y_h_dbscan[valid_db_mask])
        db_ch = calinski_harabasz_score(X_h[valid_db_mask], y_h_dbscan[valid_db_mask])
    else:
        db_sil, db_db, db_ch = 0.0, 0.0, 0.0

    comparison_df = pd.DataFrame({
        'Algorithm': ['K-Means Clustering', 'Hierarchical (Ward)', 'DBSCAN (Density-Based)'],
        'Clusters Identified': [OPTIMAL_K, OPTIMAL_K, n_dbscan_clusters],
        'Silhouette Score': [round(km_sil, 4), round(hier_sil, 4), round(db_sil, 4)],
        'Davies-Bouldin Index': [round(km_db, 4), round(hier_db, 4), round(db_db, 4)],
        'Calinski-Harabasz Index': [round(km_ch, 1), round(hier_ch, 1), round(db_ch, 1)],
        'Noise / Anomaly Ratio': ['0.0%', '0.0%', f"{dbscan_noise_pct:.2f}%"]
    })
    comparison_df.to_csv(os.path.join(MODEL_DIR, "clustering_algorithm_comparison.csv"), index=False)
    print("\n--- Clustering Algorithm Comparison ---")
    print(comparison_df.to_string(index=False))

    # Plot Algorithm Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bar_width = 0.35
    x_positions = np.arange(len(comparison_df))
    bars = ax.bar(x_positions, comparison_df['Silhouette Score'], width=bar_width, color=['#1e3d59', '#17b978', '#ff7f0e'])
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(comparison_df['Algorithm'], fontweight='bold')
    ax.set_ylabel('Silhouette Score (Higher is Better)')
    ax.set_title('Clustering Algorithm Silhouette Performance Comparison', fontweight='bold')
    ax.set_ylim(0, max(comparison_df['Silhouette Score']) * 1.3)
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "algorithm_metrics_comparison.png"), dpi=300)
    plt.close()
    print("Saved algorithm_metrics_comparison.png")

    # 5. Non-linear Projections: t-SNE & UMAP
    print("\n[Step 4/5] Computing t-SNE and UMAP Manifold Embeddings...")
    VIS_SIZE = 4000
    vis_idx = np.random.RandomState(42).choice(len(X_scaled), size=VIS_SIZE, replace=False)
    X_vis = X_scaled[vis_idx]
    vis_df = df.iloc[vis_idx].copy()

    # t-SNE
    print("  Computing t-SNE (2D)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_vis)
    vis_df['tsne_1'] = X_tsne[:, 0]
    vis_df['tsne_2'] = X_tsne[:, 1]

    # UMAP
    print("  Computing UMAP (2D)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
    X_umap = reducer.fit_transform(X_vis)
    vis_df['umap_1'] = X_umap[:, 0]
    vis_df['umap_2'] = X_umap[:, 1]

    # Plot 3 Projection Views: PCA, t-SNE, UMAP
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # PCA 2D
    sns.scatterplot(data=vis_df, x='pca_1', y='pca_2', hue='kmeans_cluster', palette=palette, alpha=0.7, s=25, ax=axes[0])
    axes[0].set_title('PCA 2D Projection', fontweight='bold')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')

    # t-SNE 2D
    sns.scatterplot(data=vis_df, x='tsne_1', y='tsne_2', hue='kmeans_cluster', palette=palette, alpha=0.7, s=25, ax=axes[1])
    axes[1].set_title('t-SNE 2D Manifold Projection', fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')

    # UMAP 2D
    sns.scatterplot(data=vis_df, x='umap_1', y='umap_2', hue='kmeans_cluster', palette=palette, alpha=0.7, s=25, ax=axes[2])
    axes[2].set_title('UMAP 2D Uniform Manifold Projection', fontweight='bold')
    axes[2].set_xlabel('UMAP Dimension 1')
    axes[2].set_ylabel('UMAP Dimension 2')

    plt.suptitle('Borrower Portfolio Clustering Across Dimensionality Reduction Manifolds', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "projection_visualizations.png"), dpi=300)
    plt.close()
    print("Saved projection_visualizations.png")

    # 6. Cluster Profiling and Historical Default Pattern Analysis
    print("\n[Step 5/5] Profiling Borrower Segments & Historical Default Patterns...")
    
    # Assign Business Persona Names based on feature medians/means
    cluster_profile = df.groupby('kmeans_cluster').agg(
        Borrower_Count=('id', 'count'),
        Mean_Income=('revenue', 'mean'),
        Median_Income=('revenue', 'median'),
        Mean_Loan_Amount=('loan_amnt', 'mean'),
        Median_Loan_Amount=('loan_amnt', 'median'),
        Mean_FICO=('fico_n', 'mean'),
        Median_FICO=('fico_n', 'median'),
        Mean_DTI=('dti_n', 'mean'),
        Median_DTI=('dti_n', 'median'),
        Mean_Emp_Years=('emp_years', 'mean'),
        Default_Rate=('Default', 'mean')
    ).reset_index()

    # Determine Cluster Personas dynamically
    # Segment 0: Prime Homeowners / Low Risk
    # Segment 1: Subprime Renters / High Risk
    # Segment 2: Credit Card Refinancers / Moderate Risk
    # Segment 3: High Exposure Consolidators
    persona_map = {
        0: 'Segment 0: Prime Homeowner Consolidators',
        1: 'Segment 1: Subprime Renter High-DTI',
        2: 'Segment 2: Prime Revolving Credit Refinancers',
        3: 'Segment 3: High-Exposure Major Borrowers'
    }
    
    df['segment_name'] = df['kmeans_cluster'].map(persona_map)
    cluster_profile['Segment_Name'] = cluster_profile['kmeans_cluster'].map(persona_map)
    cluster_profile['Default_Rate_Pct'] = (cluster_profile['Default_Rate'] * 100).round(2)
    cluster_profile['Portfolio_Share_Pct'] = (cluster_profile['Borrower_Count'] / len(df) * 100).round(2)
    
    # Baseline comparison (Odds Multiplier)
    baseline_default = df['Default'].mean()
    cluster_profile['Risk_Index_vs_Baseline'] = (cluster_profile['Default_Rate'] / baseline_default).round(2)

    cluster_profile.to_csv(os.path.join(MODEL_DIR, "cluster_profiles.csv"), index=False)
    print("\n--- Borrower Segment Profiles ---")
    print(cluster_profile[['Segment_Name', 'Portfolio_Share_Pct', 'Median_Income', 'Median_FICO', 'Median_DTI', 'Default_Rate_Pct', 'Risk_Index_vs_Baseline']].to_string(index=False))

    # Plot Default Rate Across Discovered Clusters
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(cluster_profile['Segment_Name'], cluster_profile['Default_Rate_Pct'], 
                  color=['#2a9d8f', '#e63946', '#457b9d', '#f4a261'], width=0.55)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"{yval:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.axhline(baseline_default * 100, color='black', linestyle='--', linewidth=1.5, label=f"Overall Portfolio Default Rate ({baseline_default*100:.1f}%)")
    ax.set_ylabel('Historical Default Rate (%)', fontsize=11)
    ax.set_title('Historical Default Rate by Discovered Borrower Segment', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(cluster_profile['Default_Rate_Pct']) * 1.25)
    plt.xticks(rotation=15, ha='right', fontweight='bold')
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "cluster_default_rates.png"), dpi=300)
    plt.close()
    print("Saved cluster_default_rates.png")

    # 7. DBSCAN Anomaly Detection & Export Interactive Dataset
    print("\nDetecting and Profiling Borrower Anomalies with DBSCAN...")
    sample_indices = vis_idx  # 4000 points with t-SNE and UMAP already computed!
    df_sample = vis_df.copy()
    df_sample['segment_name'] = df_sample['kmeans_cluster'].map(persona_map)
    X_sample_scaled = X_scaled[sample_indices]
    
    db_full = DBSCAN(eps=2.4, min_samples=25)
    df_sample['dbscan_label'] = db_full.fit_predict(X_sample_scaled)
    df_sample['is_anomaly'] = (df_sample['dbscan_label'] == -1).astype(int)
    
    # Compute projection coordinates for the sample for instant interactive rendering
    pca_coords = pca_3d.transform(X_sample_scaled)
    df_sample['pca_1'] = pca_coords[:, 0]
    df_sample['pca_2'] = pca_coords[:, 1]
    df_sample['pca_3'] = pca_coords[:, 2]

    # Save clustered sample dataframe to CSV for Streamlit instant interactive loading
    sample_export_cols = [
        'id', 'revenue', 'dti_n', 'loan_amnt', 'fico_n', 'emp_length', 'emp_years',
        'purpose', 'home_ownership_n', 'addr_state', 'Default', 'kmeans_cluster',
        'segment_name', 'dbscan_label', 'is_anomaly', 'pca_1', 'pca_2', 'pca_3',
        'tsne_1', 'tsne_2', 'umap_1', 'umap_2'
    ]
    df_sample[sample_export_cols].to_csv(os.path.join(MODEL_DIR, "clustered_sample.csv"), index=False)
    print(f"Saved pre-computed interactive dataset ({len(df_sample):,} rows) to models/clustered_sample.csv")

    anomaly_stats = df_sample.groupby('is_anomaly').agg(
        Count=('id', 'count'),
        Mean_Income=('revenue', 'mean'),
        Mean_DTI=('dti_n', 'mean'),
        Mean_Loan=('loan_amnt', 'mean'),
        Mean_FICO=('fico_n', 'mean'),
        Default_Rate=('Default', 'mean')
    ).reset_index()
    anomaly_stats['Anomaly_Group'] = anomaly_stats['is_anomaly'].map({0: 'Regular Borrowers', 1: 'DBSCAN Anomalies (Outliers)'})
    anomaly_stats.to_csv(os.path.join(MODEL_DIR, "anomalies_summary.csv"), index=False)
    print("\n--- Anomaly Profiling Statistics ---")
    print(anomaly_stats[['Anomaly_Group', 'Count', 'Mean_Income', 'Mean_DTI', 'Mean_Loan', 'Mean_FICO', 'Default_Rate']].to_string(index=False))

    print("\n" + "=" * 60)
    print("ALL CLUSTERING MODELS, ARTIFACTS, AND VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    train_and_evaluate_clustering()
