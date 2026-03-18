import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import warnings
import io

warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank | Loan Intelligence Hub",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy: #0a1628;
    --navy-mid: #112240;
    --gold: #c9a84c;
    --gold-light: #e8c97a;
    --cream: #f5f0e8;
    --teal: #0d7377;
    --red-soft: #c0392b;
    --green-soft: #1a7a4a;
    --text-main: #1a1a2e;
    --text-sub: #4a5568;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background: #f8f6f1; }

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0a1628 0%, #112240 50%, #0d4a6e 100%);
    padding: 2.5rem 3rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(201,168,76,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 900;
    color: #f5f0e8;
    margin: 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1rem;
    color: rgba(245,240,232,0.7);
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}
.hero-badge {
    display: inline-block;
    background: rgba(201,168,76,0.2);
    border: 1px solid rgba(201,168,76,0.5);
    color: #c9a84c;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    border-left: 4px solid var(--gold);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--navy);
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: var(--text-sub);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}
.metric-delta {
    font-size: 0.82rem;
    color: var(--teal);
    margin-top: 0.4rem;
    font-weight: 500;
}

/* Section headers */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--navy);
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--gold);
    margin-bottom: 0.4rem;
}
.section-sub {
    font-size: 0.85rem;
    color: var(--text-sub);
    margin-bottom: 1.5rem;
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #f0f7ff, #e8f4fd);
    border-left: 4px solid #2980b9;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.87rem;
    color: #1a3a5c;
    line-height: 1.6;
}
.insight-gold {
    background: linear-gradient(135deg, #fffbf0, #fff4d6);
    border-left: 4px solid var(--gold);
    color: #5a3e1b;
}
.insight-green {
    background: linear-gradient(135deg, #f0fff4, #d4edda);
    border-left: 4px solid var(--green-soft);
    color: #1a4a2e;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: white;
    padding: 6px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: var(--navy) !important;
    color: white !important;
}

/* Divider */
.gold-divider {
    height: 2px;
    background: linear-gradient(90deg, var(--gold), transparent);
    margin: 2rem 0;
    border-radius: 2px;
}

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: var(--navy) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
PALETTE = {
    "navy": "#0a1628",
    "gold": "#c9a84c",
    "teal": "#0d7377",
    "cream": "#f5f0e8",
    "soft_red": "#c0392b",
    "soft_green": "#1a7a4a",
    "mid_blue": "#2471a3",
    "purple": "#6c3483",
}
GRAD = ["#0a1628", "#0d4a6e", "#0d7377", "#c9a84c", "#e8c97a"]

def make_cmap():
    return LinearSegmentedColormap.from_list("ub", ["#f5f0e8", "#0a1628"])

def style_fig(fig, ax_list=None):
    fig.patch.set_facecolor("white")
    if ax_list:
        for ax in ax_list:
            ax.set_facecolor("#fdfcfa")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#ddd")
            ax.spines["bottom"].set_color("#ddd")
            ax.tick_params(colors="#555", labelsize=9)
            ax.title.set_fontsize(12)
            ax.title.set_fontweight("bold")
            ax.title.set_color(PALETTE["navy"])

@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")
    df["Experience"] = df["Experience"].clip(lower=0)
    df = df.drop(columns=["ID", "ZIP Code"], errors="ignore")
    df["Education_Label"] = df["Education"].map({1: "Undergrad", 2: "Graduate", 3: "Advanced"})
    return df

@st.cache_data
def train_models(df):
    feature_cols = ["Age", "Experience", "Income", "Family", "CCAvg",
                    "Education", "Mortgage", "Securities Account",
                    "CD Account", "Online", "CreditCard"]
    X = df[feature_cols]
    y = df["Personal Loan"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosted Tree": GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    }
    results, trained = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {
            "train_acc": accuracy_score(y_train, y_pred_train),
            "test_acc": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": auc(fpr, tpr),
            "fpr": fpr, "tpr": tpr,
            "cm": confusion_matrix(y_test, y_pred),
            "y_pred": y_pred, "y_test": y_test,
        }
    return results, trained, X_train, X_test, y_train, y_test, feature_cols

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data()
results, trained_models, X_train, X_test, y_train, y_test, feature_cols = train_models(df)
best_model_name = max(results, key=lambda k: results[k]["f1"])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-family:Playfair Display,serif; font-size:1.4rem; color:#c9a84c; font-weight:700;'>🏦 Universal Bank</div>
        <div style='font-size:0.72rem; color:rgba(245,240,232,0.5); letter-spacing:0.12em; text-transform:uppercase; margin-top:4px;'>Loan Intelligence Hub</div>
    </div>
    <hr style='border-color:rgba(201,168,76,0.3); margin: 0.8rem 0;'>
    """, unsafe_allow_html=True)

    nav = st.radio("Navigate", [
        "🏠 Executive Overview",
        "📊 Descriptive Analytics",
        "🔍 Diagnostic Analytics",
        "🤖 Predictive Models",
        "🎯 Prescriptive Analytics",
        "📁 Predict New Customers"
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:rgba(201,168,76,0.3); margin: 1rem 0;'>
    <div style='font-size:0.72rem; color:rgba(245,240,232,0.4); text-align:center; padding-bottom:1rem;'>
        Best Model: <span style='color:#c9a84c; font-weight:600;'>""" + best_model_name + """</span>
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-header">
  <div class="hero-badge">Marketing Intelligence Dashboard</div>
  <div class="hero-title">Personal Loan Campaign Hub</div>
  <div class="hero-subtitle">Universal Bank · AI-Powered Customer Analytics · Term 2026</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
if nav == "🏠 Executive Overview":
    st.markdown('<div class="section-header">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Key metrics across the customer base and model performance at a glance.</div>', unsafe_allow_html=True)

    total = len(df)
    loan_yes = df["Personal Loan"].sum()
    loan_pct = loan_yes / total * 100
    avg_income = df["Income"].mean()
    avg_ccavg = df["CCAvg"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, f"{total:,}", "Total Customers", "Full dataset"),
        (c2, f"{loan_yes:,}", "Loan Acceptors", f"{loan_pct:.1f}% of base"),
        (c3, f"${avg_income:.0f}K", "Avg. Annual Income", "Across all customers"),
        (c4, f"${avg_ccavg:.2f}K", "Avg. CC Spend/mo", "Monthly credit card avg"),
        (c5, f"{results[best_model_name]['auc']:.3f}", "Best AUC Score", best_model_name),
    ]
    for col, val, lbl, delta in cards:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
            <div class="metric-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1.9])
    with col1:
        st.markdown("**Campaign Conversion Summary**")
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        fig.patch.set_facecolor("white")
        sizes = [loan_yes, total - loan_yes]
        labels = ["Accepted\nLoan", "Did Not\nAccept"]
        colors = [PALETTE["gold"], "#dde3ec"]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=3)
        )
        for at in autotexts:
            at.set_fontsize(11); at.set_fontweight("bold"); at.set_color(PALETTE["navy"])
        for t in texts:
            t.set_fontsize(10); t.set_color(PALETTE["navy"])
        ax.set_title("Loan Acceptance Rate", fontsize=12, fontweight="bold",
                     color=PALETTE["navy"], pad=10)
        centre = plt.Circle((0,0), 0.35, color="white")
        ax.add_artist(centre)
        ax.text(0, 0, f"{loan_pct:.1f}%", ha="center", va="center",
                fontsize=16, fontweight="bold", color=PALETTE["navy"],
                fontfamily="serif")
        st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">Only <b>9.6%</b> of customers accepted the loan in the last campaign — a large untapped segment. Precision targeting can dramatically improve conversion economics.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("**Model Performance Summary**")
        rows = []
        for name, r in results.items():
            rows.append({
                "Model": name,
                "Train Acc.": f"{r['train_acc']*100:.1f}%",
                "Test Acc.": f"{r['test_acc']*100:.1f}%",
                "Precision": f"{r['precision']*100:.1f}%",
                "Recall": f"{r['recall']*100:.1f}%",
                "F1 Score": f"{r['f1']*100:.1f}%",
                "AUC": f"{r['auc']:.3f}",
            })
        perf_df = pd.DataFrame(rows)
        st.dataframe(perf_df.set_index("Model"), use_container_width=True)

        st.markdown(f"""
        <div class="insight-gold">
        🏆 <b>{best_model_name}</b> is the top performer with an F1 of <b>{results[best_model_name]['f1']*100:.1f}%</b>
        and AUC of <b>{results[best_model_name]['auc']:.3f}</b>, making it the recommended model for campaign targeting.
        </div>""", unsafe_allow_html=True)

        # Feature importance from best model
        model = trained_models[best_model_name]
        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(8)
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        style_fig(fig2, [ax2])
        bars = ax2.barh(fi.index, fi.values * 100,
                        color=[PALETTE["gold"] if v == fi.max() else "#8fa8c8" for v in fi.values],
                        height=0.6, edgecolor="none")
        for bar, val in zip(bars, fi.values):
            ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{val*100:.1f}%", va="center", fontsize=8.5, color=PALETTE["navy"], fontweight="600")
        ax2.set_xlabel("Importance (%)", fontsize=9)
        ax2.set_title(f"Top Predictors — {best_model_name}", fontsize=11,
                      fontweight="bold", color=PALETTE["navy"])
        ax2.set_xlim(0, fi.max()*120)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — DESCRIPTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════
elif nav == "📊 Descriptive Analytics":
    st.markdown('<div class="section-header">Descriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Understanding the customer base — demographics, financial behaviour, and product ownership patterns.</div>', unsafe_allow_html=True)

    # Row 1 — Age & Income distributions
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        style_fig(fig, [ax])
        loan_0 = df[df["Personal Loan"]==0]["Age"]
        loan_1 = df[df["Personal Loan"]==1]["Age"]
        ax.hist(loan_0, bins=25, alpha=0.65, color="#8fa8c8", label="No Loan", edgecolor="white")
        ax.hist(loan_1, bins=25, alpha=0.85, color=PALETTE["gold"], label="Accepted Loan", edgecolor="white")
        ax.set_xlabel("Age (years)", fontsize=9)
        ax.set_ylabel("Number of Customers", fontsize=9)
        ax.set_title("Age Distribution by Loan Acceptance", fontsize=11, fontweight="bold", color=PALETTE["navy"])
        ax.legend(fontsize=9, framealpha=0.8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">Loan acceptors span a <b>broad age range (30–55)</b>, with the peak around ages 40–50 — indicating working professionals at peak earning and borrowing capacity are the prime target.</div>', unsafe_allow_html=True)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        style_fig(fig, [ax])
        ax.hist(df[df["Personal Loan"]==0]["Income"], bins=30, alpha=0.65, color="#8fa8c8", label="No Loan", edgecolor="white")
        ax.hist(df[df["Personal Loan"]==1]["Income"], bins=30, alpha=0.85, color=PALETTE["gold"], label="Accepted Loan", edgecolor="white")
        ax.set_xlabel("Annual Income ($000s)", fontsize=9)
        ax.set_ylabel("Number of Customers", fontsize=9)
        ax.set_title("Income Distribution by Loan Acceptance", fontsize=11, fontweight="bold", color=PALETTE["navy"])
        ax.legend(fontsize=9, framealpha=0.8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box"><b>Higher income customers</b> (above $100K/year) show significantly greater loan acceptance — income is the strongest demographic signal for personalised targeting.</div>', unsafe_allow_html=True)

    # Row 2 — Education & Family
    col3, col4 = st.columns(2)
    with col3:
        edu_loan = df.groupby("Education_Label")["Personal Loan"].agg(["sum", "count"])
        edu_loan["rate"] = edu_loan["sum"] / edu_loan["count"] * 100
        edu_loan = edu_loan.reindex(["Undergrad", "Graduate", "Advanced"])
        fig, ax = plt.subplots(figsize=(6, 3.8))
        style_fig(fig, [ax])
        bars = ax.bar(edu_loan.index, edu_loan["rate"],
                      color=[PALETTE["navy"], PALETTE["teal"], PALETTE["gold"]],
                      width=0.5, edgecolor="white", linewidth=2)
        for bar, (_, row) in zip(bars, edu_loan.iterrows()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{row['rate']:.1f}%\n({int(row['sum'])} / {int(row['count'])})",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=PALETTE["navy"])
        ax.set_ylabel("Loan Acceptance Rate (%)", fontsize=9)
        ax.set_title("Loan Acceptance Rate by Education Level", fontsize=11, fontweight="bold", color=PALETTE["navy"])
        ax.set_ylim(0, edu_loan["rate"].max() * 1.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box"><b>Graduate and Advanced</b> degree holders accept loans at nearly 2× the rate of undergraduates — education level should be a priority segmentation variable in campaign targeting.</div>', unsafe_allow_html=True)

    with col4:
        fam_loan = df.groupby("Family")["Personal Loan"].agg(["sum", "count"])
        fam_loan["rate"] = fam_loan["sum"] / fam_loan["count"] * 100
        fig, ax = plt.subplots(figsize=(6, 3.8))
        style_fig(fig, [ax])
        x = fam_loan.index
        ax.bar(x, fam_loan["count"], color="#dde3ec", width=0.6, label="Total Customers", edgecolor="white")
        ax2b = ax.twinx()
        ax2b.plot(x, fam_loan["rate"], color=PALETTE["gold"], marker="o",
                  linewidth=2.5, markersize=8, markerfacecolor=PALETTE["navy"], label="Acceptance Rate (%)")
        ax2b.set_ylabel("Acceptance Rate (%)", color=PALETTE["gold"], fontsize=9)
        ax2b.tick_params(axis="y", colors=PALETTE["gold"])
        ax.set_xlabel("Family Size", fontsize=9)
        ax.set_ylabel("Number of Customers", fontsize=9)
        ax.set_title("Family Size vs Loan Acceptance Rate", fontsize=11, fontweight="bold", color=PALETTE["navy"])
        ax.set_xticks(x); ax.set_xticklabels([f"Size {i}" for i in x])
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="upper left")
        style_fig(fig, [ax])
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">Families of size <b>3 and 4</b> show the highest loan acceptance rates, suggesting that larger families with higher expenses are more likely to seek personal financing.</div>', unsafe_allow_html=True)

    # Row 3 — CC Spend & Product ownership
    col5, col6 = st.columns(2)
    with col5:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        style_fig(fig, [ax])
        bp = ax.boxplot(
            [df[df["Personal Loan"]==0]["CCAvg"], df[df["Personal Loan"]==1]["CCAvg"]],
            labels=["No Loan", "Accepted Loan"],
            patch_artist=True,
            boxprops=dict(linewidth=1.5),
            medianprops=dict(color=PALETTE["navy"], linewidth=2.5),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markersize=3, alpha=0.4)
        )
        bp["boxes"][0].set_facecolor("#c5d5e8")
        bp["boxes"][1].set_facecolor(PALETTE["gold"])
        ax.set_ylabel("Avg Monthly CC Spend ($000s)", fontsize=9)
        ax.set_title("Credit Card Spend vs Loan Acceptance", fontsize=11, fontweight="bold", color=PALETTE["navy"])
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">Customers who accepted loans show <b>significantly higher credit card spending</b> — high CCAvg customers are financially active and more open to credit products, making them prime loan campaign targets.</div>', unsafe_allow_html=True)

    with col6:
        products = ["Securities Account", "CD Account", "Online", "CreditCard"]
        prod_labels = ["Securities\nAccount", "CD Account", "Online\nBanking", "Credit\nCard"]
        rates_yes = [df[df["Personal Loan"]==1][p].mean()*100 for p in products]
        rates_no  = [df[df["Personal Loan"]==0][p].mean()*100 for p in products]
        x = np.arange(len(products))
        fig, ax = plt.subplots(figsize=(6, 3.8))
        style_fig(fig, [ax])
        w = 0.38
        ax.bar(x - w/2, rates_no,  width=w, label="No Loan",       color="#8fa8c8", edgecolor="white")
        ax.bar(x + w/2, rates_yes, width=w, label="Accepted Loan", color=PALETTE["gold"], edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(prod_labels, fontsize=9)
        ax.set_ylabel("% of Customers with Product", fontsize=9)
        ax.set_title("Product Ownership by Loan Status", fontsize=11, fontweight="bold", color=PALETTE["navy"])
        ax.legend(fontsize=9)
        for i, (r1, r2) in enumerate(zip(rates_no, rates_yes)):
            ax.text(i - w/2, r1 + 0.5, f"{r1:.0f}%", ha="center", fontsize=7.5, color=PALETTE["navy"])
            ax.text(i + w/2, r2 + 0.5, f"{r2:.0f}%", ha="center", fontsize=7.5, color=PALETTE["navy"])
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box"><b>CD Account holders</b> have a dramatically higher loan acceptance rate — customers with a CD account are deeply engaged with the bank and far more likely to accept additional financial products.</div>', unsafe_allow_html=True)

    # Heatmap
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Correlation Heatmap — All Features**")
    num_cols = ["Age", "Experience", "Income", "Family", "CCAvg", "Education",
                "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard", "Personal Loan"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    style_fig(fig, [ax])
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", ax=ax,
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                center=0, linewidths=0.5, linecolor="white",
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.7})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold",
                 color=PALETTE["navy"], pad=15)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">Income and CCAvg show the <b>strongest positive correlation</b> with Personal Loan acceptance. Age and Experience are highly correlated with each other — the model benefits from treating these as one effective signal.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — DIAGNOSTIC ANALYTICS
# ═══════════════════════════════════════════════════════════════════════
elif nav == "🔍 Diagnostic Analytics":
    st.markdown('<div class="section-header">Diagnostic Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Why do customers accept or reject loans? Deep-dive into segment-level patterns and cross-variable interactions.</div>', unsafe_allow_html=True)

    # Income vs CCAvg scatter
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        style_fig(fig, [ax])
        no  = df[df["Personal Loan"]==0]
        yes = df[df["Personal Loan"]==1]
        ax.scatter(no["Income"],  no["CCAvg"],  alpha=0.25, s=12, color="#8fa8c8", label="No Loan")
        ax.scatter(yes["Income"], yes["CCAvg"], alpha=0.65, s=18, color=PALETTE["gold"], label="Accepted Loan")
        ax.set_xlabel("Annual Income ($000s)", fontsize=9)
        ax.set_ylabel("Avg Monthly CC Spend ($000s)", fontsize=9)
        ax.set_title("Income vs CC Spend — Loan Status", fontsize=11, fontweight="bold", color=PALETTE["navy"])
        ax.legend(fontsize=9, markerscale=1.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">The <b>high-income + high CC spend quadrant</b> is dominated by loan acceptors. This dual-filter is the single most powerful segment for any targeted campaign.</div>', unsafe_allow_html=True)

    with col2:
        # Mortgage distribution
        fig, ax = plt.subplots(figsize=(6, 4.2))
        style_fig(fig, [ax])
        mort_no  = df[(df["Personal Loan"]==0) & (df["Mortgage"]>0)]["Mortgage"]
        mort_yes = df[(df["Personal Loan"]==1) & (df["Mortgage"]>0)]["Mortgage"]
        ax.hist(mort_no,  bins=25, alpha=0.65, color="#8fa8c8", label="No Loan (w/ Mortgage)",  edgecolor="white")
        ax.hist(mort_yes, bins=25, alpha=0.85, color=PALETTE["gold"], label="Accepted Loan (w/ Mortgage)", edgecolor="white")
        ax.set_xlabel("Mortgage Value ($000s)", fontsize=9)
        ax.set_ylabel("Number of Customers", fontsize=9)
        ax.set_title("Mortgage Distribution by Loan Status", fontsize=11, fontweight="bold", color=PALETTE["navy"])
        ax.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">Customers with <b>existing mortgages</b> still accept personal loans — indicating debt appetite is not a deterrent for higher-income segments and cross-selling to mortgage holders is viable.</div>', unsafe_allow_html=True)

    # Income buckets
    st.markdown("**Loan Acceptance Rate by Income Bracket**")
    df_diag = df.copy()
    df_diag["Income_Bracket"] = pd.cut(df["Income"],
        bins=[0, 50, 100, 150, 200, 300], labels=["<$50K","$50-100K","$100-150K","$150-200K","$200K+"])
    inc_grp = df_diag.groupby("Income_Bracket", observed=True)["Personal Loan"].agg(["sum","count"])
    inc_grp["rate"] = inc_grp["sum"] / inc_grp["count"] * 100

    col3, col4 = st.columns([2, 1])
    with col3:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        style_fig(fig, [ax])
        colors = [PALETTE["navy"], "#1a5276", PALETTE["teal"], "#c9a84c", PALETTE["gold"]]
        bars = ax.bar(inc_grp.index.astype(str), inc_grp["rate"],
                      color=colors, edgecolor="white", linewidth=2, width=0.55)
        for bar, (_, row) in zip(bars, inc_grp.iterrows()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{row['rate']:.1f}%\n(n={int(row['count'])})",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=PALETTE["navy"])
        ax.set_ylabel("Loan Acceptance Rate (%)", fontsize=9)
        ax.set_xlabel("Annual Income Bracket", fontsize=9)
        ax.set_title("Loan Acceptance Rate Across Income Brackets", fontsize=11, fontweight="bold", color=PALETTE["navy"])
        ax.set_ylim(0, inc_grp["rate"].max() * 1.35)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.markdown('<div class="insight-gold" style="margin-top:1rem;">Loan acceptance surges past <b>$100K income</b> — customers earning above $100K are the primary addressable segment for the next campaign. Concentrating budget here maximises ROI per contact.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-green">Customers earning <b>$150K+</b> have acceptance rates above <b>{inc_grp[inc_grp.index.astype(str).str.contains("150|200")]["rate"].max():.0f}%</b> — the highest-value micro-segment.</div>', unsafe_allow_html=True)

    # CD Account cross-tab
    st.markdown("**CD Account × Education × Loan Acceptance Heatmap**")
    pivot = df.pivot_table(values="Personal Loan", index="Education_Label",
                           columns="CD Account", aggfunc="mean") * 100
    pivot.columns = ["No CD Account", "Has CD Account"]
    pivot.index = ["Undergrad", "Graduate", "Advanced"]
    fig, ax = plt.subplots(figsize=(6, 3))
    style_fig(fig, [ax])
    sns.heatmap(pivot, annot=True, fmt=".1f", ax=ax,
                cmap=sns.light_palette(PALETTE["gold"], as_cmap=True),
                linewidths=1, linecolor="white",
                annot_kws={"size": 11, "fontweight": "bold"},
                cbar_kws={"label": "Acceptance Rate (%)","shrink": 0.8})
    ax.set_title("Loan Acceptance Rate (%) — Education × CD Account", fontsize=11, fontweight="bold", color=PALETTE["navy"])
    ax.set_xlabel("CD Account Status", fontsize=9)
    ax.set_ylabel("Education Level", fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">The intersection of <b>Advanced education + CD Account holder</b> is the single hottest segment — with acceptance rates many times the base rate. This is your highest-priority micro-segment for the next campaign.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIVE MODELS
# ═══════════════════════════════════════════════════════════════════════
elif nav == "🤖 Predictive Models":
    st.markdown('<div class="section-header">Predictive Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Decision Tree, Random Forest, and Gradient Boosted Tree — performance metrics, ROC curves, and confusion matrices.</div>', unsafe_allow_html=True)

    # ── Performance Table ──
    st.markdown("#### 📋 Comprehensive Model Performance Table")
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Train Accuracy": f"{r['train_acc']*100:.2f}%  ({r['train_acc']:.4f})",
            "Test Accuracy":  f"{r['test_acc']*100:.2f}%  ({r['test_acc']:.4f})",
            "Precision":      f"{r['precision']*100:.2f}%  ({r['precision']:.4f})",
            "Recall":         f"{r['recall']*100:.2f}%  ({r['recall']:.4f})",
            "F1 Score":       f"{r['f1']*100:.2f}%  ({r['f1']:.4f})",
            "AUC-ROC":        f"{r['auc']:.4f}",
        })
    perf_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(perf_df, use_container_width=True)
    st.markdown(f'<div class="insight-gold">🏆 <b>{best_model_name}</b> achieves the best F1 score (<b>{results[best_model_name]["f1"]*100:.1f}%</b>) — balancing both precision and recall, which is critical for imbalanced loan data. High AUC of <b>{results[best_model_name]["auc"]:.3f}</b> means the model effectively separates loan acceptors from non-acceptors.</div>', unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Single ROC Curve ──
    st.markdown("#### 📈 Combined ROC Curve — All Models")
    model_colors = {
        "Decision Tree": PALETTE["teal"],
        "Random Forest": PALETTE["gold"],
        "Gradient Boosted Tree": PALETTE["soft_red"],
    }
    fig, ax = plt.subplots(figsize=(7, 5.5))
    style_fig(fig, [ax])
    ax.plot([0,1],[0,1], color="#ccc", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    for name, r in results.items():
        ax.plot(r["fpr"], r["tpr"], lw=2.5, color=model_colors[name],
                label=f"{name}  (AUC = {r['auc']:.3f})")
    ax.fill_between(results[best_model_name]["fpr"], results[best_model_name]["tpr"],
                    alpha=0.08, color=model_colors[best_model_name])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    ax.set_title("ROC Curve Comparison — All Classification Models", fontsize=12,
                 fontweight="bold", color=PALETTE["navy"], pad=12)
    ax.legend(fontsize=9.5, loc="lower right",
              framealpha=0.9, edgecolor="#ddd")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.05)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">The <b>ROC curve</b> measures how well each model distinguishes loan acceptors from non-acceptors across all classification thresholds. A curve hugging the top-left corner (AUC near 1.0) means near-perfect discrimination. All three models outperform random chance significantly, with Gradient Boosted Tree leading.</div>', unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Confusion Matrices ──
    st.markdown("#### 🔲 Confusion Matrices — All Models")
    cols = st.columns(3)
    for (name, r), col in zip(results.items(), cols):
        with col:
            cm = r["cm"]
            total_cm = cm.sum()
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            fig.patch.set_facecolor("white")
            custom_cmap = LinearSegmentedColormap.from_list("ub_cm", ["#f5f0e8", PALETTE["navy"]])
            sns.heatmap(cm, annot=False, ax=ax, cmap=custom_cmap,
                        linewidths=2, linecolor="white",
                        xticklabels=["Predicted\nNo Loan", "Predicted\nLoan"],
                        yticklabels=["Actual\nNo Loan", "Actual\nLoan"],
                        cbar=False)
            # Annotate with values AND percentages
            for i in range(2):
                for j in range(2):
                    val = cm[i, j]
                    pct = val / total_cm * 100
                    color = "white" if cm[i,j] > cm.max()*0.4 else PALETTE["navy"]
                    ax.text(j+0.5, i+0.38, f"{val:,}", ha="center", va="center",
                            fontsize=14, fontweight="bold", color=color)
                    ax.text(j+0.5, i+0.65, f"({pct:.1f}%)", ha="center", va="center",
                            fontsize=9, color=color, alpha=0.85)
            labels_map = {(0,0):"TN", (0,1):"FP", (1,0):"FN", (1,1):"TP"}
            label_colors = {(0,0): PALETTE["teal"], (0,1): PALETTE["soft_red"],
                            (1,0): PALETTE["soft_red"], (1,1): PALETTE["gold"]}
            for (i,j), lbl in labels_map.items():
                ax.text(j+0.5, i+0.15, lbl, ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color=label_colors[(i,j)], alpha=0.9)
            ax.set_title(name, fontsize=10, fontweight="bold", color=PALETTE["navy"], pad=8)
            ax.tick_params(labelsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="insight-box"><b>TP</b> = Correctly predicted loan acceptors (marketing gold). <b>FP</b> = Predicted loan but customer won\'t accept (wasted campaign budget). <b>FN</b> = Missed loan acceptors (lost opportunity). <b>TN</b> = Correctly predicted non-acceptors. Minimising FP and FN simultaneously is the model\'s core challenge with imbalanced data.</div>', unsafe_allow_html=True)

    # ── Feature Importance Comparison ──
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 🔑 Feature Importance — All Models Compared")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    style_fig(fig, list(axes))
    for ax, (name, model) in zip(axes, trained_models.items()):
        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        bars = ax.barh(fi.index, fi.values * 100,
                       color=[PALETTE["gold"] if v == fi.max() else "#8fa8c8" for v in fi.values],
                       height=0.65, edgecolor="none")
        for bar, val in zip(bars, fi.values):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f"{val*100:.1f}%", va="center", fontsize=7.5, color=PALETTE["navy"])
        ax.set_title(name, fontsize=10, fontweight="bold", color=PALETTE["navy"])
        ax.set_xlabel("Importance (%)", fontsize=8)
        ax.set_xlim(0, fi.max() * 130)
    plt.suptitle("Feature Importance Across All Models", fontsize=12,
                 fontweight="bold", color=PALETTE["navy"], y=1.02)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box"><b>Income</b> consistently ranks as the #1 predictor across all three models, followed by CCAvg and Education. This cross-model agreement strongly validates that financial capacity is the primary driver of loan acceptance decisions.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 5 — PRESCRIPTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════
elif nav == "🎯 Prescriptive Analytics":
    st.markdown('<div class="section-header">Prescriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Actionable campaign recommendations — where to focus, who to target, and how to maximise ROI with a halved marketing budget.</div>', unsafe_allow_html=True)

    model = trained_models[best_model_name]
    X_all = df[feature_cols]
    df_pred = df.copy()
    df_pred["Predicted_Loan"] = model.predict(X_all)
    df_pred["Loan_Probability"] = model.predict_proba(X_all)[:, 1]
    df_pred["Priority_Tier"] = pd.cut(
        df_pred["Loan_Probability"],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=["Low (< 30%)", "Medium (30-60%)", "High (60-80%)", "Priority (> 80%)"]
    )

    # Tier summary
    tier_summary = df_pred.groupby("Priority_Tier", observed=True).agg(
        Customers=("Loan_Probability", "count"),
        Avg_Income=("Income", "mean"),
        Avg_CCSpend=("CCAvg", "mean"),
        Actual_Loan_Rate=("Personal Loan", "mean")
    ).round(2)
    tier_summary["Actual_Loan_Rate"] = (tier_summary["Actual_Loan_Rate"] * 100).map("{:.1f}%".format)

    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.markdown("**Customer Priority Tiers — Campaign Targeting Framework**")
        tier_colors = ["#dde3ec", "#8fa8c8", "#c9a84c", PALETTE["navy"]]
        tier_counts = df_pred["Priority_Tier"].value_counts(sort=False)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        style_fig(fig, [ax])
        bars = ax.barh(tier_counts.index.astype(str), tier_counts.values,
                       color=tier_colors, edgecolor="white", linewidth=2, height=0.55)
        for bar, val in zip(bars, tier_counts.values):
            ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
                    f"{val:,} customers ({val/len(df_pred)*100:.1f}%)",
                    va="center", fontsize=9, fontweight="600", color=PALETTE["navy"])
        ax.set_xlabel("Number of Customers", fontsize=9)
        ax.set_title(f"Campaign Priority Tiers — {best_model_name}", fontsize=11,
                     fontweight="bold", color=PALETTE["navy"])
        ax.set_xlim(0, tier_counts.max() * 1.4)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Tier Summary Statistics**")
        st.dataframe(tier_summary, use_container_width=True)

    st.markdown(f'<div class="insight-gold">🎯 With budget halved, focus <b>100% of campaign spend on Priority + High tiers</b> — just <b>{tier_counts.get("Priority (> 80%)", 0) + tier_counts.get("High (60-80%)", 0):,} customers</b> ({(tier_counts.get("Priority (> 80%)", 0) + tier_counts.get("High (60-80%)", 0))/len(df_pred)*100:.1f}% of base) but representing the bulk of expected conversions.</div>', unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # Ideal Customer Profile
    st.markdown("#### 🧩 Ideal Customer Profile (ICP) for Personal Loan Campaign")
    priority = df_pred[df_pred["Priority_Tier"] == "Priority (> 80%)"]
    if len(priority) == 0:
        priority = df_pred[df_pred["Loan_Probability"] >= 0.6]

    col3, col4 = st.columns(2)
    with col3:
        icp_stats = {
            "Avg Annual Income": f"${priority['Income'].mean():.0f}K",
            "Avg CC Spend/mo": f"${priority['CCAvg'].mean():.2f}K",
            "Has CD Account": f"{priority['CD Account'].mean()*100:.0f}%",
            "Graduate+ Education": f"{(priority['Education']>=2).mean()*100:.0f}%",
            "Family Size 3-4": f"{priority['Family'].isin([3,4]).mean()*100:.0f}%",
            "Has Mortgage": f"{(priority['Mortgage']>0).mean()*100:.0f}%",
            "Avg Age": f"{priority['Age'].mean():.0f} years",
            "Segment Size": f"{len(priority):,} customers",
        }
        for k, v in icp_stats.items():
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:7px 12px;
                background:{'#fffbf0' if list(icp_stats.keys()).index(k)%2==0 else 'white'};
                border-radius:6px; margin-bottom:3px;'>
                <span style='font-size:0.88rem; color:{PALETTE["text_sub"] if False else "#555"};'>{k}</span>
                <span style='font-size:0.95rem; font-weight:700; color:{PALETTE["navy"]};'>{v}</span>
            </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("**Top 3 Campaign Recommendations**")
        recs = [
            ("1. Target High-Income + High CC Spend", "Customers earning $100K+ with monthly CC spend above $2K represent your highest-converting segment. Personalise messaging around lifestyle and aspirational goals."),
            ("2. Cross-Sell to CD Account Holders", "CD Account customers show dramatically higher loan acceptance. Use existing relationship as a trust bridge — offer preferential rates to this engaged segment."),
            ("3. Graduate+ Education Segment", "Graduate and Advanced degree holders convert at 2× the rate of undergrads. Tailor messaging around career milestones, home renovation, or investment opportunities."),
        ]
        for title, body in recs:
            st.markdown(f"""
            <div style='background:white; border-radius:10px; padding:1rem 1.2rem;
                border-left:4px solid {PALETTE["gold"]}; margin-bottom:0.8rem;
                box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
                <div style='font-size:0.9rem; font-weight:700; color:{PALETTE["navy"]}; margin-bottom:5px;'>{title}</div>
                <div style='font-size:0.83rem; color:#555; line-height:1.6;'>{body}</div>
            </div>""", unsafe_allow_html=True)

    # ROI estimator
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 💰 Campaign ROI Estimator")
    c1, c2, c3 = st.columns(3)
    with c1:
        cost_per_contact = st.number_input("Cost per Contact ($)", value=10, min_value=1, step=1)
    with c2:
        revenue_per_loan = st.number_input("Revenue per Loan Accepted ($)", value=500, min_value=50, step=50)
    with c3:
        budget = st.number_input("Total Campaign Budget ($)", value=50000, min_value=1000, step=1000)

    max_contacts = budget // cost_per_contact
    priority_count = tier_counts.get("Priority (> 80%)", 0)
    high_count = tier_counts.get("High (60-80%)", 0)
    target_count = min(max_contacts, priority_count + high_count)
    expected_conversions = int(target_count * 0.30)
    total_revenue = expected_conversions * revenue_per_loan
    roi = ((total_revenue - budget) / budget * 100) if budget > 0 else 0

    r1, r2, r3, r4 = st.columns(4)
    for col, val, lbl in [
        (r1, f"{int(max_contacts):,}", "Max Contacts w/ Budget"),
        (r2, f"{int(target_count):,}", "AI-Targeted Contacts"),
        (r3, f"~{expected_conversions:,}", "Expected Conversions"),
        (r4, f"{roi:.0f}%", "Estimated Campaign ROI"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 6 — PREDICT NEW CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════
elif nav == "📁 Predict New Customers":
    st.markdown('<div class="section-header">Predict New Customers</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a CSV of new customers and download predictions with loan probability scores.</div>', unsafe_allow_html=True)

    col_info, col_up = st.columns([1, 1.4])
    with col_info:
        st.markdown("""
        <div style='background:white; border-radius:12px; padding:1.2rem 1.4rem;
            box-shadow:0 2px 10px rgba(0,0,0,0.06); border-top:3px solid #c9a84c;'>
        <div style='font-weight:700; color:#0a1628; font-size:0.95rem; margin-bottom:0.8rem;'>
            📋 Required CSV Columns
        </div>
        """, unsafe_allow_html=True)
        for fc in feature_cols:
            st.markdown(f"- `{fc}`")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box" style="margin-top:0.8rem;">
        The model will predict <b>Personal Loan (0/1)</b> and append a <b>Loan_Probability (%)</b>
        and <b>Priority_Tier</b> column for each customer. Download the enriched CSV for CRM integration.
        </div>""", unsafe_allow_html=True)

        # Download sample test file
        st.markdown("**📥 Download Sample Test File**")
        sample = df[feature_cols].head(20).copy()
        sample_csv = sample.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Sample Test Data (20 rows)",
            data=sample_csv,
            file_name="sample_test_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_up:
        uploaded = st.file_uploader("Upload Customer CSV for Prediction", type=["csv"])
        if uploaded:
            try:
                test_df = pd.read_csv(uploaded)
                missing = [c for c in feature_cols if c not in test_df.columns]
                if missing:
                    st.error(f"❌ Missing columns: {missing}")
                else:
                    X_new = test_df[feature_cols].copy()
                    X_new["Experience"] = X_new["Experience"].clip(lower=0)
                    model = trained_models[best_model_name]
                    preds = model.predict(X_new)
                    probs = model.predict_proba(X_new)[:, 1] * 100

                    out = test_df.copy()
                    out["Predicted_Personal_Loan"] = preds
                    out["Predicted_Personal_Loan_Label"] = out["Predicted_Personal_Loan"].map({0: "Will NOT Accept", 1: "Will Accept"})
                    out["Loan_Probability_%"] = probs.round(2)
                    out["Priority_Tier"] = pd.cut(
                        probs/100,
                        bins=[0, 0.3, 0.6, 0.8, 1.0],
                        labels=["Low", "Medium", "High", "Priority"]
                    ).astype(str)

                    st.success(f"✅ Predictions complete — {len(out):,} customers scored using **{best_model_name}**")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Will Accept Loan", f"{preds.sum():,} ({preds.mean()*100:.1f}%)")
                    c2.metric("Priority Tier (>80%)", f"{(probs>=80).sum():,}")
                    c3.metric("Avg Loan Probability", f"{probs.mean():.1f}%")

                    st.dataframe(out[["Predicted_Personal_Loan_Label", "Loan_Probability_%", "Priority_Tier"] + feature_cols].head(50),
                                 use_container_width=True)

                    csv_out = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Full Predictions CSV",
                        data=csv_out,
                        file_name="universal_bank_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.info("👆 Upload a CSV file with customer data to generate loan predictions.")

# Footer
st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem; color:#aaa; font-size:0.78rem;'>
    Universal Bank · Personal Loan Intelligence Hub · Built with Streamlit & scikit-learn
</div>""", unsafe_allow_html=True)
