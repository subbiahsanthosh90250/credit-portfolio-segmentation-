import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
OUTPUT_PPTX = os.path.join(BASE_DIR, "Credit_Portfolio_Segmentation_Presentation.pptx")

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme Colors
    c_navy = RGBColor(30, 61, 89)
    c_slate = RGBColor(70, 80, 95)
    c_gold = RGBColor(245, 139, 84)
    c_white = RGBColor(255, 255, 255)
    c_light_bg = RGBColor(248, 249, 250)
    c_crimson = RGBColor(214, 39, 40)
    c_emerald = RGBColor(23, 185, 120)

    def add_header(slide, title_text, category_text="HCL GUVI CAPSTONE PROJECT | RETAIL BANKING & CREDIT RISK ANALYTICS"):
        # Header banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = c_gold
        
        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = c_navy

    def add_card(slide, left, top, width, height, title, content_bullets, bg_color=c_light_bg, border_color=None):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1.5)
        else:
            shape.line.fill.background()

        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.25)
        tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.2)
        tf.margin_bottom = Inches(0.2)

        if title:
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(15)
            p.font.bold = True
            p.font.color.rgb = c_navy
            p.space_after = Pt(8)
            first_bullet = True
        else:
            first_bullet = False

        for b in content_bullets:
            p = tf.add_paragraph() if (title or not first_bullet) else tf.paragraphs[0]
            first_bullet = False
            p.text = "• " + b
            p.font.size = Pt(11)
            p.font.color.rgb = c_slate
            p.space_after = Pt(5)

    # ==========================================
    # SLIDE 1: TITLE SLIDE
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = c_navy
    bg1.line.fill.background()

    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(3.8))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "CREDIT PORTFOLIO SEGMENTATION &\nRISK PATTERN ANALYSIS"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = c_white
    p.space_after = Pt(15)

    p2 = tf1.add_paragraph()
    p2.text = "Unsupervised Machine Learning, Manifold Dimensionality Reduction & Historical Default Cross-Validation"
    p2.font.size = Pt(16)
    p2.font.color.rgb = c_gold
    p2.space_after = Pt(30)

    p3 = tf1.add_paragraph()
    p3.text = "HCL GUVI Class Final Project Submission | Domain: Banking, Financial Services & Credit Analytics"
    p3.font.size = Pt(13)
    p3.font.color.rgb = RGBColor(200, 215, 230)

    # ==========================================
    # SLIDE 2: PROBLEM STATEMENT & OBJECTIVES
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Executive Summary & Core Objectives")
    
    add_card(s2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Industry Problem Context", [
        "Financial institutions manage immense portfolios with diverse borrower income levels, credit scores, debt obligations, and employment tenure.",
        "One-size-fits-all underwriting and static scoring models overlook multidimensional risk nuances across borrower segments.",
        "Need: An automated, data-driven unsupervised framework to group similar borrowers without target label bias and uncover empirical repayment patterns."
    ], border_color=c_navy)

    add_card(s2, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "Key Project Deliverables Achieved", [
        "1. Cleansed and engineered 1,347,681 loan records across 15 financial features.",
        "2. Applied PCA, t-SNE, and UMAP for linear and non-linear manifold mapping.",
        "3. Evaluated K-Means, Hierarchical (Ward), and DBSCAN with rigorous clustering metrics.",
        "4. Profiled 4 distinct borrower personas and validated historical default divergence (17.3% to 22.8%).",
        "5. Built an anomaly detection pipeline isolating high-risk outliers (23.71% default rate).",
        "6. Deployed an interactive Streamlit underwriting and portfolio monitoring dashboard."
    ], border_color=c_gold)

    # ==========================================
    # SLIDE 3: DATASET UNDERSTANDING & PREPROCESSING
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Dataset Architecture & Preprocessing Pipeline")

    add_card(s3, Inches(0.8), Inches(1.8), Inches(3.7), Inches(5.0), "Dataset Dimensions", [
        "Total Records: 1,347,681 loans",
        "Origination Span: 2008 – 2018",
        "Total Portfolio Exposure: $19.41 Billion",
        "Raw Attributes: 15 columns",
        "Overall Default Rate: 19.98% (269,219 charged-off/defaulted loans)",
        "Zero missing values in core underwriting features (revenue, loan amount, FICO, DTI)."
    ])

    add_card(s3, Inches(4.8), Inches(1.8), Inches(3.7), Inches(5.0), "Cleansing & Outlier Treatment", [
        "Target Isolation: 'Default' column strictly excluded from clustering features.",
        "Identifier Removal: 'id', 'zip_code', 'title', 'desc' dropped from clustering space.",
        "Income Capping: Right-skewed revenue clipped at 99th percentile ($250k) and log-transformed.",
        "DTI Normalization: Extreme debt ratios (>50%) capped to prevent distance distortions.",
        "FICO & Loan Bounds: Standardized within natural financial boundaries."
    ])

    add_card(s3, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0), "Encoding & Feature Scaling", [
        "Employment Mapping: Categorical tenure strings mapped to continuous years (0.5 to 10+).",
        "Categorical One-Hot Encoding: Home ownership (Mortgage, Rent, Own) and top loan purposes.",
        "Derived Feature: Loan-to-Income (LTI) ratio engineered for cash flow leverage measurement.",
        "Feature Scaling: StandardScaler applied to ensure equal geometric variance across features."
    ])

    # ==========================================
    # SLIDE 4: EXPLORATORY DATA ANALYSIS (EDA)
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Exploratory Data Analysis: Key Feature Distributions")

    eda_num_img = os.path.join(REPORT_DIR, "eda_numerical_distributions.png")
    if os.path.exists(eda_num_img):
        s4.shapes.add_picture(eda_num_img, Inches(0.8), Inches(1.7), Inches(7.5), Inches(5.2))

    add_card(s4, Inches(8.6), Inches(1.7), Inches(4.0), Inches(5.2), "Distributional Insights", [
        "Annual Income: Median $65,000, IQR $46,600 to $92,000. Significant right-tail skew mitigated via log scaling.",
        "Loan Amount: Median $12,000, Mean $14,408. Bounded between $500 and $40,000.",
        "FICO Scores: Median 692, Mean 698.2. Bell-shaped distribution reflecting standard prime-to-near-prime retail underwriting.",
        "Debt-to-Income (DTI): Median 17.6%. Normal underwriting ceiling observed at ~35-40%."
    ], border_color=c_navy)

    # ==========================================
    # SLIDE 5: RISK DRIVERS: PURPOSE & COLLATERAL
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Risk Drivers: Loan Purpose & Collateral Impact")

    p_purp_img = os.path.join(REPORT_DIR, "eda_default_by_purpose.png")
    p_home_img = os.path.join(REPORT_DIR, "eda_default_by_home_ownership.png")
    if os.path.exists(p_purp_img):
        s5.shapes.add_picture(p_purp_img, Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.0))
    if os.path.exists(p_home_img):
        s5.shapes.add_picture(p_home_img, Inches(7.1), Inches(1.8), Inches(5.4), Inches(5.0))

    # ==========================================
    # SLIDE 6: DIMENSIONALITY REDUCTION: PCA
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Dimensionality Reduction: Principal Component Analysis (PCA)")

    pca_img = os.path.join(REPORT_DIR, "pca_variance_plot.png")
    if os.path.exists(pca_img):
        s6.shapes.add_picture(pca_img, Inches(0.8), Inches(1.8), Inches(6.8), Inches(4.8))

    add_card(s6, Inches(7.9), Inches(1.8), Inches(4.6), Inches(4.8), "PCA Scree & Variance Retained", [
        "Objective: Reduce feature collinearity while retaining maximal variance.",
        "Principal Component 1: Captures borrower scale and leverage (Income vs. Loan Amount).",
        "Principal Component 2: Captures credit quality and risk capacity (FICO vs. DTI).",
        "Cumulative Explained Variance: 80% threshold achieved smoothly across leading components.",
        "Utility: Transformed high-dimensional feature space into orthogonal coordinates for 3D interactive visualization."
    ], border_color=c_navy)

    # ==========================================
    # SLIDE 7: MANIFOLD VISUALIZATIONS: PCA vs t-SNE vs UMAP
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Non-Linear Manifold Learning: PCA vs. t-SNE vs. UMAP")

    proj_img = os.path.join(REPORT_DIR, "projection_visualizations.png")
    if os.path.exists(proj_img):
        s7.shapes.add_picture(proj_img, Inches(0.8), Inches(1.8), Inches(11.7), Inches(3.6))

    add_card(s7, Inches(0.8), Inches(5.6), Inches(11.7), Inches(1.5), "Manifold Projection Takeaways", [
        "PCA preserves global linear distances, revealing a continuous credit spectrum.",
        "t-SNE preserves local probabilistic neighborhoods, proving the existence of dense micro-clusters.",
        "UMAP successfully combines local neighborhood fidelity with global topological geometry, demonstrating clear separation between low-risk consolidators and high-exposure borrowers."
    ])

    # ==========================================
    # SLIDE 8: CLUSTERING OPTIMIZATION & BENCHMARKING
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Clustering Algorithm Optimization & Benchmarking")

    elbow_img = os.path.join(REPORT_DIR, "cluster_elbow_silhouette.png")
    bench_img = os.path.join(REPORT_DIR, "algorithm_metrics_comparison.png")
    if os.path.exists(elbow_img):
        s8.shapes.add_picture(elbow_img, Inches(0.8), Inches(1.8), Inches(6.0), Inches(4.8))
    if os.path.exists(bench_img):
        s8.shapes.add_picture(bench_img, Inches(7.0), Inches(1.8), Inches(5.5), Inches(4.8))

    # ==========================================
    # SLIDE 9: ALGORITHM COMPARISON TABLE
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Algorithmic Comparison: K-Means vs. Hierarchical vs. DBSCAN")

    add_card(s9, Inches(0.8), Inches(1.8), Inches(3.7), Inches(5.0), "K-Means (Selected)", [
        "Clusters: k = 4",
        "Silhouette Score: 0.2103",
        "Davies-Bouldin Index: 1.7430",
        "Calinski-Harabasz: 1,034.1",
        "Noise Points: 0.0%",
        "Computational Complexity: O(k·n·d) - Linear & highly scalable.",
        "Verdict: Optimal for enterprise production scoring pipelines."
    ], border_color=c_emerald)

    add_card(s9, Inches(4.8), Inches(1.8), Inches(3.7), Inches(5.0), "Hierarchical (Ward)", [
        "Clusters: k = 4",
        "Silhouette Score: 0.2168",
        "Davies-Bouldin Index: 1.7941",
        "Calinski-Harabasz: 961.8",
        "Noise Points: 0.0%",
        "Computational Complexity: O(n²) memory, O(n³) compute.",
        "Verdict: Excellent for dendrogram validation on subsamples; computationally infeasible for 1.35M rows."
    ])

    add_card(s9, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0), "DBSCAN (Anomaly Filter)", [
        "Core Clusters: 14 micro-groups",
        "Silhouette Score: 0.2820 (core)",
        "Davies-Bouldin Index: 1.3549",
        "Calinski-Harabasz: 829.7",
        "Noise / Anomaly Ratio: 4.17%",
        "Parameter Tuning: eps=2.2, min_samples=25.",
        "Verdict: Ideal density-based anomaly detector to flag non-standard applications."
    ], border_color=c_gold)

    # ==========================================
    # SLIDE 10: DISCOVERED BORROWER SEGMENT PERSONAS
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Discovered Borrower Segments & Financial Personas")

    add_card(s10, Inches(0.8), Inches(1.8), Inches(5.6), Inches(2.4), "Segment 0: Prime Homeowner Consolidators", [
        "Share: 6.4% | Default Rate: 18.43% (Risk Index: 0.92x)",
        "Median Income: $75,000 | FICO: 697 | DTI: 14.6%",
        "Collateral: Mortgage / High Home Equity",
        "Strategy: Fast-track auto-approval, lowest APR pricing."
    ], border_color=c_emerald)

    add_card(s10, Inches(6.8), Inches(1.8), Inches(5.7), Inches(2.4), "Segment 1: Subprime Renter High-DTI", [
        "Share: 44.6% | Default Rate: 17.34% (Risk Index: 0.87x)",
        "Median Income: $75,000 | FICO: 692 | DTI: 18.0%",
        "Collateral: Renter / Non-Homeowner",
        "Strategy: Standard approval with automated payroll verification."
    ], border_color=c_navy)

    add_card(s10, Inches(0.8), Inches(4.5), Inches(5.6), Inches(2.4), "Segment 2: Prime Revolving Refinancers", [
        "Share: 9.5% | Default Rate: 21.33% (Risk Index: 1.07x)",
        "Median Income: $60,000 | FICO: 692 | DTI: 18.4%",
        "Purpose: Credit Card Debt Refinancing",
        "Strategy: Direct payout to existing credit card accounts."
    ], border_color=c_gold)

    add_card(s10, Inches(6.8), Inches(4.5), Inches(5.7), Inches(2.4), "Segment 3: High-Exposure Major Borrowers", [
        "Share: 39.4% | Default Rate: 22.80% (Risk Index: 1.14x)",
        "Median Income: $55,500 | FICO: 687 | DTI: 17.5%",
        "Purpose: Unsecured Expansion / Small Business",
        "Strategy: Strict manual underwriting, APR surcharge (+250 bps)."
    ], border_color=c_crimson)

    # ==========================================
    # SLIDE 11: HISTORICAL DEFAULT PATTERN ANALYSIS
    # ==========================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Historical Default Pattern Cross-Validation")

    p_def_img = os.path.join(REPORT_DIR, "cluster_default_rates.png")
    if os.path.exists(p_def_img):
        s11.shapes.add_picture(p_def_img, Inches(0.8), Inches(1.8), Inches(7.0), Inches(5.0))

    add_card(s11, Inches(8.1), Inches(1.8), Inches(4.4), Inches(5.0), "Default Cross-Validation Findings", [
        "Strict Methodology: The 'Default' target was entirely excluded from clustering features — eliminating data leakage.",
        "Empirical Validation: Discovered clusters naturally bifurcated into distinct risk tiers without supervision.",
        "Lowest Risk: Segment 1 (17.34%) and Segment 0 (18.43%).",
        "Highest Risk: Segment 3 (22.80%) - representing 1.14x baseline portfolio default odds.",
        "Loss Mitigation: Tightening underwriting on Segment 3 prevents the majority of portfolio charge-offs."
    ], border_color=c_navy)

    # ==========================================
    # SLIDE 12: ANOMALY & FRAUD DETECTION (DBSCAN)
    # ==========================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Anomaly & Outlier Risk Detection via DBSCAN")

    add_card(s12, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Density-Based Outlier Profiling", [
        "Isolation Metric: Distance threshold eps = 2.4, min_samples = 25.",
        "Anomaly Volume: 4.17% of borrower pool flagged as noise/outliers.",
        "Empirical Default Rate: Anomalous group defaults at 23.71% vs. 19.52% for regular borrowers (+4.19% risk premium).",
        "Financial Signatures:",
        "  - Excessive Loan-to-Income (> 0.85)",
        "  - Mismatched debt burdens relative to declared income",
        "  - Irregular combinations of low credit score with top-tier loan requests."
    ], border_color=c_crimson)

    add_card(s12, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "Risk Control & Anti-Fraud Actions", [
        "1. Automated Red-Flag Gate: Real-time DBSCAN scoring triggers automated review for applicants falling outside core clusters.",
        "2. Fraud Prevention: Screen flagged applicants against synthetic identity and income inflation databases.",
        "3. Enhanced Collateral Requirements: Require verified bank statement feeds (Plaid / Open Banking API) rather than self-reported income.",
        "4. Risk Containment: Restrict maximum unsecured credit limit to $15,000 for all flagged anomaly applicants."
    ], border_color=c_gold)

    # ==========================================
    # SLIDE 13: INTERACTIVE STREAMLIT DASHBOARD
    # ==========================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Interactive Streamlit Underwriting & Risk Dashboard")

    add_card(s13, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Dashboard Architectural Highlights", [
        "Multi-Tab Navigation: 9 comprehensive modules covering KPIs, EDA, Projections, Clustering, Default Validation, and Policy.",
        "Interactive 3D Projections: WebGL-accelerated 3D PCA, t-SNE, and UMAP scatter plots with dynamic cluster/default toggles.",
        "Instant Load Latency: Pre-computed embeddings and pickled scikit-learn models allow sub-second responsiveness without recomputing 1.35M rows.",
        "Real-Time Application Simulator: Live input sliders for applicant income, loan amount, FICO, DTI, and purpose."
    ], border_color=c_navy)

    add_card(s13, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "Underwriting Simulator Workflow", [
        "1. Loan officer inputs borrower details into the Streamlit interface.",
        "2. Pipeline executes robust outlier capping, log transform, and standard scaling.",
        "3. K-Means model assigns the applicant to an empirical segment.",
        "4. System outputs: Assigned Persona, Expected Historical Default Rate, Risk Tier, and Concrete Decision Policy (Auto-Approve, Verify, or Manual Review)."
    ], border_color=c_emerald)

    # ==========================================
    # SLIDE 14: STRATEGIC BUSINESS RECOMMENDATIONS
    # ==========================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Strategic Recommendations & Policy Governance")

    add_card(s14, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Product & Pricing Strategies", [
        "1. Risk-Based APR Grid: Implement tiered APR margins (+200 to +350 bps) on Segment 3 to generate loss reserves.",
        "2. Direct Creditor Payouts: Mandate that debt consolidation disbursements go directly to credit card providers.",
        "3. Tailored Loan Products: Cross-sell home equity credit lines (HELOCs) to Segment 0 prime homeowners.",
        "4. Balanced Portfolio Allocation: Cap high-exposure loans at 15% of total deployed balance sheet capital."
    ], border_color=c_navy)

    add_card(s14, Inches(6.8), Inches(1.8), Inches(5.0), Inches(5.0), "Ethical AI & Compliance Governance", [
        "1. Fair Lending & Non-Discrimination: Ensure geographic and demographic variables do not introduce proxy bias.",
        "2. Model Explainability: Segment assignments are backed by transparent feature centroid distances.",
        "3. Continuous Monitoring: Quarterly drift monitoring on cluster centroids to capture macroeconomic shifts.",
        "4. Basel II / III Alignment: Model parameters calibrated against internal rating-based (IRB) capital adequacy standards."
    ], border_color=c_gold)

    prs.save(OUTPUT_PPTX)
    print(f"Presentation successfully created and saved to: {OUTPUT_PPTX}")

if __name__ == "__main__":
    create_deck()
