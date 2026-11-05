import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

DATA_PATH = r"C:\Users\Subbiah Santhosh .RM\Downloads\Project 3\loans_granting_model_dataset.csv"
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

def run_exploratory_data_analysis():
    print("Executing Comprehensive Exploratory Data Analysis...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Total dataset records: {len(df):,}")

    # Parse Issue Year
    df['issue_year'] = df['issue_d'].astype(str).str.split('-').str[1]
    df['issue_year'] = pd.to_numeric(df['issue_year'], errors='coerce')

    # Color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    primary_blue = "#1e3d59"
    risk_crimson = "#ff6e40"

    # 1. Target Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    default_counts = df['Default'].value_counts()
    default_pcts = df['Default'].value_counts(normalize=True) * 100
    bars = ax.bar(['Fully Paid / Current (0)', 'Default / Charged Off (1)'], 
                  default_counts.values, color=[primary_blue, risk_crimson], width=0.5)
    for bar, pct in zip(bars, default_pcts.values):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 15000, f'{yval:,} ({pct:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Portfolio Default Status Distribution (Total: 1,347,681)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Loans', fontsize=11)
    ax.set_ylim(0, max(default_counts.values) * 1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "eda_target_distribution.png"), dpi=300)
    plt.close()
    print("Saved eda_target_distribution.png")

    # 2. Key Numerical Feature Distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Revenue (Capped at 200k for visual clarity)
    sns.histplot(df['revenue'].clip(upper=200000), bins=40, kde=True, ax=axes[0, 0], color='#2b580c')
    axes[0, 0].set_title('Annual Revenue Distribution (Capped at $200k)', fontweight='bold')
    axes[0, 0].set_xlabel('Annual Income ($)')
    axes[0, 0].axvline(df['revenue'].median(), color='red', linestyle='--', label=f"Median: ${df['revenue'].median():,.0f}")
    axes[0, 0].legend()

    # Loan Amount
    sns.histplot(df['loan_amnt'], bins=40, kde=True, ax=axes[0, 1], color='#1e3d59')
    axes[0, 1].set_title('Loan Amount Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Loan Amount ($)')
    axes[0, 1].axvline(df['loan_amnt'].median(), color='red', linestyle='--', label=f"Median: ${df['loan_amnt'].median():,.0f}")
    axes[0, 1].legend()

    # FICO Score
    sns.histplot(df['fico_n'], bins=35, kde=True, ax=axes[1, 0], color='#438a5e')
    axes[1, 0].set_title('FICO Credit Score Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('FICO Score')
    axes[1, 0].axvline(df['fico_n'].median(), color='red', linestyle='--', label=f"Median: {df['fico_n'].median():.0f}")
    axes[1, 0].legend()

    # DTI Ratio (Capped at 40)
    sns.histplot(df['dti_n'].clip(upper=40), bins=40, kde=True, ax=axes[1, 1], color='#f58b54')
    axes[1, 1].set_title('Debt-to-Income (DTI) Distribution (Capped at 40%)', fontweight='bold')
    axes[1, 1].set_xlabel('DTI Ratio (%)')
    axes[1, 1].axvline(df['dti_n'].median(), color='red', linestyle='--', label=f"Median: {df['dti_n'].median():.1f}%")
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "eda_numerical_distributions.png"), dpi=300)
    plt.close()
    print("Saved eda_numerical_distributions.png")

    # 3. Default Rate by Loan Purpose
    purpose_stats = df.groupby('purpose')['Default'].agg(['count', 'mean']).reset_index()
    purpose_stats = purpose_stats[purpose_stats['count'] >= 1000].sort_values(by='mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(purpose_stats['purpose'], purpose_stats['mean'] * 100, color='#17b978')
    # Highlight high risk (> 20%)
    for i, bar in enumerate(bars):
        if purpose_stats['mean'].iloc[i] > 0.20:
            bar.set_color('#ff5722')
        val = purpose_stats['mean'].iloc[i] * 100
        ax.text(val + 0.4, bar.get_y() + bar.get_height()/2.0, f"{val:.1f}%", va='center', fontsize=10, fontweight='bold')
    ax.axvline(df['Default'].mean() * 100, color='black', linestyle='--', label=f"Baseline Portfolio Default ({df['Default'].mean()*100:.1f}%)")
    ax.set_title('Historical Default Rate by Loan Purpose', fontsize=13, fontweight='bold')
    ax.set_xlabel('Default Rate (%)', fontsize=11)
    ax.set_xlim(0, 35)
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "eda_default_by_purpose.png"), dpi=300)
    plt.close()
    print("Saved eda_default_by_purpose.png")

    # 4. Default Rate by Home Ownership
    home_stats = df.groupby('home_ownership_n')['Default'].agg(['count', 'mean']).reset_index()
    home_stats = home_stats.sort_values(by='mean', ascending=False)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(home_stats['home_ownership_n'], home_stats['mean'] * 100, color=['#e63946', '#f4a261', '#2a9d8f', '#457b9d'], width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(df['Default'].mean() * 100, color='black', linestyle='--', label=f"Overall Avg ({df['Default'].mean()*100:.1f}%)")
    ax.set_title('Default Rate by Home Ownership Status', fontsize=13, fontweight='bold')
    ax.set_ylabel('Default Rate (%)', fontsize=11)
    ax.set_ylim(0, 28)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "eda_default_by_home_ownership.png"), dpi=300)
    plt.close()
    print("Saved eda_default_by_home_ownership.png")

    # 5. Correlation Heatmap
    num_cols = ['revenue', 'dti_n', 'loan_amnt', 'fico_n', 'Default']
    corr_matrix = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=-0.3, vmax=0.5, linewidths=0.5, ax=ax)
    ax.set_title('Pearson Correlation Heatmap (Financial Features & Default)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "eda_correlation_heatmap.png"), dpi=300)
    plt.close()
    print("Saved eda_correlation_heatmap.png")

    # 6. Default Trend Over Loan Origination Years
    yearly = df.dropna(subset=['issue_year']).groupby('issue_year')['Default'].agg(['count', 'mean']).reset_index()
    yearly = yearly[(yearly['issue_year'] >= 2008) & (yearly['issue_year'] <= 2018)]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_bar = '#a8dadc'
    color_line = '#e63946'
    
    ax1.bar(yearly['issue_year'], yearly['count'], color=color_bar, alpha=0.8, label='Originated Loans (Volume)')
    ax1.set_xlabel('Origination Year', fontsize=11)
    ax1.set_ylabel('Loan Count', fontsize=11, color='#1d3557')
    ax1.tick_params(axis='y', labelcolor='#1d3557')
    
    ax2 = ax1.twinx()
    ax2.plot(yearly['issue_year'], yearly['mean'] * 100, color=color_line, marker='o', linewidth=2.5, label='Default Rate (%)')
    ax2.set_ylabel('Default Rate (%)', fontsize=11, color=color_line)
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.set_ylim(0, 30)
    
    plt.title('Origination Volume vs. Historical Default Rate by Vintage Year (2008-2018)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "eda_default_trend_by_year.png"), dpi=300)
    plt.close()
    print("Saved eda_default_trend_by_year.png")

    print("Exploratory Data Analysis complete. All publication figures saved to reports/")

if __name__ == "__main__":
    run_exploratory_data_analysis()
