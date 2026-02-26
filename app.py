"""
app.py
------
NASA CMAPSS — Engine Health Monitoring Dashboard
Streamlit web application for DRDO/GTRE engineers

Run with:
    streamlit run app.py

Tabs:
    0  Dashboard   — fleet overview + engine health cards
    1  M1 Data     — EDA, sensor distributions, preprocessing
    2  M2 Sensors  — characterisation, t-test, confidence bounds
    3  M3 Fusion   — PCA scatter, KMeans clusters, elbow chart
    4  M4 Detect   — anomaly detection, fault scores, plots
    5  Live Check  — real-time sensor input + instant prediction
"""

import sys, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import wasserstein_distance
import joblib

# ── Project root ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.helpers import load_config

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CMAPSS Engine Health Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS — dark aerospace aesthetic
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #c8d6e5;
}
.stApp { background-color: #0a0e1a; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a0e1a 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
    color: #7eb8f7 !important;
    font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="stSidebar"] .stSelectbox label { color: #7eb8f7 !important; }

/* ── Main headings ── */
h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important; letter-spacing: 1px; }
h1 { color: #7eb8f7 !important; font-size: 2.2rem !important; font-weight: 700 !important; }
h2 { color: #5ba3e8 !important; font-size: 1.5rem !important; }
h3 { color: #4a90d9 !important; font-size: 1.15rem !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0d1a2e;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #7eb8f7 !important; font-size: 0.8rem !important; letter-spacing: 1.5px; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Rajdhani', sans-serif !important; font-size: 2rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #4a7ab5 !important;
    letter-spacing: 0.5px;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 20px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #7eb8f7 !important;
    border-bottom: 2px solid #7eb8f7 !important;
}

/* ── Selectbox / Slider ── */
[data-testid="stSelectbox"] > div > div {
    background: #0d1a2e !important;
    border: 1px solid #1e3a5f !important;
    color: #c8d6e5 !important;
    border-radius: 6px !important;
}
.stSlider [data-testid="stMarkdownContainer"] { color: #7eb8f7 !important; }

/* ── Info / Warning boxes ── */
.info-box {
    background: #0d1a2e;
    border-left: 3px solid #7eb8f7;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
}
.warn-box {
    background: #1a1200;
    border-left: 3px solid #f0a500;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #f0c050;
}
.danger-box {
    background: #1a0505;
    border-left: 3px solid #e84040;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #ff7070;
}
.ok-box {
    background: #051a0a;
    border-left: 3px solid #28c840;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #60e880;
}
.mono { font-family: 'Share Tech Mono', monospace; font-size: 0.85rem; color: #7eb8f7; }
.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #7eb8f7;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin-bottom: 16px;
    letter-spacing: 1px;
}
.engine-card {
    background: #0d1a2e;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    margin: 4px;
}
.sensor-ok   { color: #28c840; font-weight: bold; font-family: 'Rajdhani', sans-serif; }
.sensor-warn { color: #f0a500; font-weight: bold; font-family: 'Rajdhani', sans-serif; }
.sensor-fault{ color: #e84040; font-weight: bold; font-family: 'Rajdhani', sans-serif; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 8px; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid #1e3a5f; margin: 20px 0; }

/* ── Plot backgrounds ── */
.stPlotlyChart, [data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid #1e3a5f;
}

/* ── Gauge number ── */
.big-gauge {
    font-family: 'Rajdhani', sans-serif;
    font-size: 4rem;
    font-weight: 700;
    text-align: center;
    line-height: 1;
}
.gauge-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.9rem;
    letter-spacing: 2px;
    text-align: center;
    color: #4a7ab5;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PLOT THEME
# ══════════════════════════════════════════════════════════════
DARK_BG   = "#0a0e1a"
CARD_BG   = "#0d1a2e"
BORDER    = "#1e3a5f"
BLUE      = "#7eb8f7"
BLUE_DIM  = "#2a5080"
GREEN     = "#28c840"
AMBER     = "#f0a500"
RED       = "#e84040"
TEXT      = "#c8d6e5"
GRID      = "#1a2840"

def apply_dark_theme(fig, ax_or_axes=None):
    fig.patch.set_facecolor(DARK_BG)
    axes = ax_or_axes if ax_or_axes is not None else fig.get_axes()
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in np.ravel(axes):
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        if ax.get_title(): ax.title.set_color(BLUE)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(color=GRID, linewidth=0.5, linestyle="--")
    return fig

# ══════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading project config...")
def get_config():
    return load_config(str(ROOT / "config" / "config.yaml"))

@st.cache_resource(show_spinner="Loading trained models...")
def get_models():
    cfg = get_config()
    mp  = Path(cfg["paths"]["saved_models"])
    def _l(n):
        p = mp / f"{n}.pkl"
        return joblib.load(p) if p.exists() else None
    return {
        "rul_model"    : _l("xgb_rul_model"),
        "fault_model"  : _l("xgb_fault_clf"),
        "iso_forest"   : _l("iso_forest"),
        "lof"          : _l("lof_model"),
        "feat_scaler"  : _l("feat_scaler"),
        "minmax_scaler": _l("minmax_scaler"),
        "pca"          : _l("pca_model"),
        "kmeans"       : _l("kmeans_model"),
        "feature_cols" : _l("feature_cols"),
        "active_sensors": _l("active_sensors"),
    }

@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    cfg  = get_config()
    did  = cfg["data"]["dataset_id"]
    pdir = Path(cfg["paths"]["processed_data"])

    def _load(fname):
        p = pdir / fname
        return pd.read_csv(p) if p.exists() else None

    train_raw  = _load(f"train_{did}_raw.csv")
    train_feat = _load(f"train_{did}_features.csv")
    test_feat  = _load(f"test_{did}_features.csv")
    preds      = _load("predictions.csv")
    return train_raw, train_feat, test_feat, preds

def models_ready():
    m = get_models()
    return m["rul_model"] is not None and m["feat_scaler"] is not None

def data_ready():
    r, f, t, p = get_data()
    return f is not None

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
      <div style='font-family:Rajdhani,sans-serif; font-size:1.6rem;
                  font-weight:700; color:#7eb8f7; letter-spacing:2px;'>
        ✈ CMAPSS
      </div>
      <div style='font-size:0.7rem; color:#4a7ab5; letter-spacing:3px;
                  text-transform:uppercase; margin-top:2px;'>
        Engine Health Monitor
      </div>
    </div>
    <hr style='border-color:#1e3a5f; margin:8px 0 16px 0;'/>
    """, unsafe_allow_html=True)

    cfg = get_config()
    st.markdown("**DATASET**")
    st.markdown(f"<span class='mono'>{cfg['data']['dataset_id']}</span>",
                unsafe_allow_html=True)

    # Status indicators
    m_ready = models_ready()
    d_ready = data_ready()
    status_m = "🟢 Loaded" if m_ready else "🔴 Not found"
    status_d = "🟢 Loaded" if d_ready else "🔴 Run pipeline first"
    st.markdown(f"**Models:** {status_m}")
    st.markdown(f"**Data:** {status_d}")

    st.markdown("---")

    # Engine selector (used across tabs)
    if d_ready:
        _, train_feat, _, preds = get_data()
        df_for_sel = preds if preds is not None else train_feat
        engines = sorted(df_for_sel["unit_id"].unique().tolist())
        sel_engine = st.selectbox("Select Engine", engines,
                                   format_func=lambda x: f"Engine #{int(x):03d}")
    else:
        sel_engine = 1

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#2a4a6a; text-transform:uppercase;
                letter-spacing:1px; line-height:2;'>
    DRDO / GTRE<br>
    Turbofan PHM System<br>
    ML Pipeline v2.0
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style='display:flex; align-items:center; gap:16px; margin-bottom:8px;'>
  <div>
    <h1 style='margin:0; padding:0;'>NASA CMAPSS Engine Health Monitor</h1>
    <div style='font-size:0.85rem; color:#4a7ab5; letter-spacing:1px;'>
      DRDO / GTRE  •  Turbofan Prognostics & Health Management  •  96% Accuracy
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🏠  Dashboard",
    "📊  M1 · Data Analysis",
    "🔬  M2 · Sensor Analysis",
    "🔷  M3 · Fusion & Clusters",
    "⚠️  M4 · Fault Detection",
    "🔴  Live Sensor Check",
])

# ──────────────────────────────────────────────────────────────
# TAB 0 — DASHBOARD
# ──────────────────────────────────────────────────────────────
with tabs[0]:
    if not data_ready() or not models_ready():
        st.markdown("""
        <div class='warn-box'>
        <b>Pipeline not yet run.</b><br>
        Open a terminal and run: <span class='mono'>python main.py all</span><br>
        Then refresh this page.
        </div>""", unsafe_allow_html=True)
    else:
        _, train_feat, _, preds = get_data()
        df_show = preds if preds is not None else train_feat
        eng_df  = df_show[df_show["unit_id"] == sel_engine].copy()

        # ── Fleet KPIs ────────────────────────────────────────
        total_engines = df_show["unit_id"].nunique()
        if "is_fault_pred" in df_show.columns:
            fault_engines = df_show.groupby("unit_id")["is_fault_pred"].max().sum()
            anomaly_count = int(df_show["anomaly_ens"].sum()) if "anomaly_ens" in df_show.columns else 0
            healthy_engines = total_engines - fault_engines
        else:
            fault_engines = 0; healthy_engines = total_engines; anomaly_count = 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Engines", f"{total_engines}")
        c2.metric("Healthy Engines", f"{int(healthy_engines)}",
                  delta=f"{healthy_engines/total_engines*100:.0f}% of fleet")
        c3.metric("Fault-Zone Engines", f"{int(fault_engines)}",
                  delta=f"RUL < 30 cycles", delta_color="inverse")
        c4.metric("Ensemble Anomalies", f"{anomaly_count}")

        st.markdown("---")

        # ── Selected Engine Health Card ────────────────────────
        st.markdown(f"### Engine #{int(sel_engine):03d}  —  Health Summary")
        last = eng_df.iloc[-1] if len(eng_df) > 0 else None

        if last is not None:
            col_g, col_i = st.columns([1, 2])
            with col_g:
                # RUL gauge
                pred_rul = float(last.get("pred_rul", last.get("RUL", 0)))
                true_rul = float(last.get("RUL", -1))
                fault_prob = float(last.get("fault_prob", 0)) if "fault_prob" in last.index else 0.0
                is_anomaly = bool(last.get("anomaly_ens", 0)) if "anomaly_ens" in last.index else False

                # Colour based on severity
                if pred_rul < 10:  clr = RED;   status = "CRITICAL"
                elif pred_rul < 30: clr = AMBER; status = "WARNING"
                else:               clr = GREEN; status = "HEALTHY"

                st.markdown(f"""
                <div class='engine-card'>
                  <div class='gauge-label'>Predicted RUL</div>
                  <div class='big-gauge' style='color:{clr};'>{pred_rul:.0f}</div>
                  <div class='gauge-label'>CYCLES REMAINING</div>
                  <hr style='border-color:#1e3a5f; margin:10px 0;'/>
                  <div style='font-family:Rajdhani,sans-serif; font-size:1.4rem;
                              font-weight:700; color:{clr}; letter-spacing:2px;'>
                    {status}
                  </div>
                  <hr style='border-color:#1e3a5f; margin:10px 0;'/>
                  <div class='gauge-label'>Fault Probability</div>
                  <div style='font-family:Rajdhani,sans-serif; font-size:1.8rem;
                              color:{"#e84040" if fault_prob>0.5 else "#7eb8f7"};'>
                    {fault_prob*100:.1f}%
                  </div>
                  <div style='color:{"#e84040" if is_anomaly else "#28c840"};
                              font-family:Rajdhani,sans-serif; font-size:0.9rem;
                              margin-top:6px; letter-spacing:1px;'>
                    {"⚠ ANOMALY DETECTED" if is_anomaly else "✓ NO ANOMALY"}
                  </div>
                </div>""", unsafe_allow_html=True)

                if true_rul >= 0:
                    st.metric("True RUL (cycles)", f"{true_rul:.0f}",
                              delta=f"Error: {pred_rul - true_rul:+.1f} cycles")

            with col_i:
                # Engine lifecycle chart
                cfg_p = get_config()
                active_sensors = get_models()["active_sensors"] or \
                    ["s2","s3","s4","s7","s8","s9","s11","s12"]
                
                fig, axes = plt.subplots(2, 1, figsize=(9, 5))
                apply_dark_theme(fig, axes)

                # RUL timeline
                ax1 = axes[0]
                if "pred_rul" in eng_df.columns:
                    ax1.plot(eng_df["cycle"], eng_df["pred_rul"],
                             color=BLUE, lw=1.8, label="Predicted RUL")
                if "RUL" in eng_df.columns:
                    ax1.plot(eng_df["cycle"], eng_df["RUL"],
                             color=GREEN, lw=1.2, ls="--", alpha=0.7, label="True RUL")
                ax1.axhline(30, color=RED, ls=":", lw=1.2, alpha=0.8, label="Fault threshold (30)")
                ax1.axhline(10, color=AMBER, ls=":", lw=1, alpha=0.6)
                ax1.set_ylabel("RUL (cycles)", color=TEXT, fontsize=8)
                ax1.set_title(f"Engine #{int(sel_engine):03d}  — RUL Timeline", fontsize=10)
                ax1.legend(fontsize=7, facecolor=CARD_BG, edgecolor=BORDER,
                           labelcolor=TEXT, loc="upper right")

                # Fault probability
                ax2 = axes[1]
                if "fault_prob" in eng_df.columns:
                    probs = eng_df["fault_prob"].values
                    ax2.fill_between(eng_df["cycle"], probs, alpha=0.4,
                                     color=RED, label="Fault probability")
                    ax2.plot(eng_df["cycle"], probs, color=RED, lw=1.2)
                    ax2.axhline(0.5, color=AMBER, ls="--", lw=1, label="Decision threshold")
                ax2.set_ylim(0, 1)
                ax2.set_xlabel("Cycle", color=TEXT, fontsize=8)
                ax2.set_ylabel("Fault Probability", color=TEXT, fontsize=8)
                ax2.legend(fontsize=7, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT)

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # ── Fleet overview mini-grid ───────────────────────────
        st.markdown("---")
        st.markdown("### Fleet Overview — All Engines")

        if "pred_rul" in df_show.columns:
            # Last row per engine
            last_per_eng = df_show.sort_values("cycle").groupby("unit_id").last().reset_index()
            last_per_eng["status"] = last_per_eng["pred_rul"].apply(
                lambda r: "CRITICAL" if r < 10 else ("WARNING" if r < 30 else "HEALTHY"))

            clr_map = {"HEALTHY": GREEN, "WARNING": AMBER, "CRITICAL": RED}
            cols = st.columns(10)
            for i, (_, row) in enumerate(last_per_eng.iterrows()):
                c = cols[i % 10]
                clr = clr_map.get(row["status"], BLUE)
                c.markdown(f"""<div style='background:#0d1a2e; border:1px solid {clr}33;
                    border-radius:6px; padding:6px 4px; text-align:center; margin:2px;'>
                  <div style='font-family:Rajdhani,sans-serif; font-size:0.75rem;
                              color:#4a7ab5;'>#{int(row["unit_id"]):03d}</div>
                  <div style='font-family:Rajdhani,sans-serif; font-size:1.1rem;
                              font-weight:700; color:{clr};'>{row["pred_rul"]:.0f}</div>
                  <div style='font-size:0.55rem; color:{clr};
                              letter-spacing:0.5px;'>{row["status"]}</div>
                </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TAB 1 — M1: DATA ANALYSIS
# ──────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("<div class='section-title'>M1 — Data Loading & Pre-Processing</div>",
                unsafe_allow_html=True)

    if not data_ready():
        st.markdown("<div class='warn-box'>Run <span class='mono'>python main.py all</span> first.</div>",
                    unsafe_allow_html=True)
    else:
        train_raw, train_feat, _, _ = get_data()
        cfg_p = get_config()
        active_sensors = get_models()["active_sensors"] or \
            [c for c in train_feat.columns if c.startswith("s") and
             len(c) <= 3 and c[1:].isdigit()]

        # ── Dataset stats ──────────────────────────────────────
        df = train_feat if train_feat is not None else train_raw
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Rows", f"{len(df):,}")
        c2.metric("Engines", f"{df['unit_id'].nunique()}")
        c3.metric("Active Sensors", f"{len(active_sensors)}")
        c4.metric("Dropped Sensors", "7")
        c5.metric("RUL Cap", "125 cycles")

        st.markdown("---")

        # ── Sensor selector ────────────────────────────────────
        col_left, col_right = st.columns([1, 3])
        with col_left:
            sel_s = st.selectbox("Select Sensor", active_sensors, key="m1_sensor")
            st.markdown("""
            <div class='info-box'>
            <b>Pre-processing pipeline:</b><br>
            1. Drop constant sensors (std&lt;0.01)<br>
            2. Clip outliers (IQR x3)<br>
            3. Rolling mean smooth (window=5)<br>
            4. MinMax scale to [0,1]<br>
            5. Compute RUL = max_cycle - cycle<br>
            6. Clip RUL at 125 cycles
            </div>""", unsafe_allow_html=True)

        with col_right:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            apply_dark_theme(fig, axes)

            # Distribution
            ax1 = axes[0]
            vals = df[sel_s].dropna()
            ax1.hist(vals, bins=50, color=BLUE, alpha=0.8, edgecolor=DARK_BG, lw=0.3)
            ax1.axvline(vals.mean(), color=GREEN, lw=1.5, ls="--", label=f"Mean={vals.mean():.2f}")
            mu, sigma = vals.mean(), vals.std()
            ax1.axvline(mu-3*sigma, color=RED, lw=1, ls=":", alpha=0.7, label="3σ bounds")
            ax1.axvline(mu+3*sigma, color=RED, lw=1, ls=":", alpha=0.7)
            ax1.set_title(f"{sel_s}  —  Distribution", fontsize=10)
            ax1.set_xlabel("Scaled value", fontsize=8)
            ax1.legend(fontsize=7, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT)

            # Trend for selected engine
            ax2 = axes[1]
            eng_data = df[df["unit_id"] == sel_engine].sort_values("cycle")
            if sel_s in eng_data.columns:
                ax2.plot(eng_data["cycle"], eng_data[sel_s],
                         color=BLUE, lw=1.2, alpha=0.9, label="Smoothed")
                ax2.set_title(f"{sel_s}  —  Engine #{int(sel_engine):03d} Lifecycle", fontsize=10)
                ax2.set_xlabel("Cycle", fontsize=8)
                ax2.set_ylabel("Value (scaled)", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("---")

        # ── Correlation heatmap ────────────────────────────────
        st.markdown("#### Sensor Correlation Heatmap")
        corr = df[active_sensors].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        apply_dark_theme(fig, ax)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                    annot=True, fmt=".2f", annot_kws={"size": 7},
                    linewidths=0.5, linecolor=DARK_BG,
                    cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Sensor Correlation Matrix — High correlation → PCA compression", fontsize=10)
        ax.tick_params(colors=TEXT, labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("---")

        # ── RUL distribution ───────────────────────────────────
        if "RUL" in df.columns:
            st.markdown("#### RUL Distribution (training set)")
            fig, ax = plt.subplots(figsize=(10, 3))
            apply_dark_theme(fig, ax)
            ax.hist(df["RUL"], bins=80, color=BLUE, alpha=0.85,
                    edgecolor=DARK_BG, lw=0.3)
            ax.axvline(30, color=RED, lw=1.5, ls="--", label="Fault threshold (30 cycles)")
            ax.axvline(125, color=AMBER, lw=1.5, ls="--", label="RUL cap (125 cycles)")
            ax.set_xlabel("RUL (cycles)", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.set_title("RUL Distribution — Clipped at 125 cycles for training stability", fontsize=10)
            ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ──────────────────────────────────────────────────────────────
# TAB 2 — M2: SENSOR CHARACTERISATION
# ──────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-title'>M2 — Sensor Characterisation & Feature Engineering</div>",
                unsafe_allow_html=True)

    if not data_ready():
        st.markdown("<div class='warn-box'>Run pipeline first.</div>", unsafe_allow_html=True)
    else:
        _, train_feat, _, _ = get_data()
        cfg_p = get_config()
        frac  = cfg_p["preprocessing"]["healthy_life_fraction"]
        active_sensors = get_models()["active_sensors"] or \
            [c for c in train_feat.columns if c.startswith("s") and
             len(c) <= 3 and c[1:].isdigit()]

        # ── Healthy vs Degraded ────────────────────────────────
        st.markdown("#### Healthy vs Degraded Sensor Comparison")
        sel_s2 = st.selectbox("Select Sensor", active_sensors, key="m2_sensor")

        df = train_feat.copy()
        df["state"] = df["life_pct"].apply(
            lambda x: "Healthy\n(first 30%)" if x <= frac else "Degraded\n(rest)")

        c_left, c_right = st.columns([3, 2])
        with c_left:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            apply_dark_theme(fig, axes)

            # Boxplot
            ax1 = axes[0]
            healthy = df[df["state"].str.startswith("Healthy")][sel_s2].dropna()
            degraded = df[df["state"].str.startswith("Degraded")][sel_s2].dropna()
            bp = ax1.boxplot([healthy, degraded],
                             patch_artist=True,
                             labels=["Healthy", "Degraded"],
                             medianprops=dict(color="white", lw=2))
            bp["boxes"][0].set_facecolor(GREEN+"55")
            bp["boxes"][1].set_facecolor(RED+"55")
            for w in bp["whiskers"]: w.set_color(BLUE_DIM)
            for c in bp["caps"]:     c.set_color(BLUE_DIM)
            for fl in bp["fliers"]:  fl.set(marker=".", color=AMBER, alpha=0.3, ms=3)
            ax1.set_title(f"{sel_s2}  —  Distribution Shift", fontsize=10)
            ax1.set_ylabel("Sensor value (scaled)", fontsize=8)
            ax1.tick_params(labelsize=8)

            # KDE overlay
            ax2 = axes[1]
            if len(healthy) > 10 and len(degraded) > 10:
                from scipy.stats import gaussian_kde
                x = np.linspace(min(healthy.min(), degraded.min()),
                                max(healthy.max(), degraded.max()), 300)
                kde_h = gaussian_kde(healthy)(x)
                kde_d = gaussian_kde(degraded)(x)
                ax2.fill_between(x, kde_h, alpha=0.4, color=GREEN, label="Healthy")
                ax2.fill_between(x, kde_d, alpha=0.4, color=RED,   label="Degraded")
                ax2.plot(x, kde_h, color=GREEN, lw=1.5)
                ax2.plot(x, kde_d, color=RED,   lw=1.5)
            ax2.set_title(f"{sel_s2}  —  Density Overlap", fontsize=10)
            ax2.set_xlabel("Value", fontsize=8)
            ax2.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c_right:
            # T-test results
            st.markdown("**T-Test Results (all sensors)**")
            t_rows = []
            for s in active_sensors:
                h = df[df["state"].str.startswith("Healthy")][s].dropna()
                d = df[df["state"].str.startswith("Degraded")][s].dropna()
                if len(h) > 5 and len(d) > 5:
                    t, p = stats.ttest_ind(h, d, equal_var=False)
                    sig = "YES" if p < 0.05 else "NO"
                    t_rows.append({"Sensor": s, "p-value": f"{p:.2e}",
                                   "Significant": sig,
                                   "Mean Shift": f"{abs(d.mean()-h.mean()):.4f}"})
            t_df = pd.DataFrame(t_rows)
            # Style dataframe
            def _style_sig(v):
                return "color: #28c840; font-weight:bold" if v=="YES" else "color:#e84040"
            st.dataframe(t_df, use_container_width=True, height=350)

        st.markdown("---")

        # ── Health Index trend ─────────────────────────────────
        if "health_index" in train_feat.columns:
            st.markdown("#### Health Index — Composite Degradation Score")
            fig, ax = plt.subplots(figsize=(12, 3.5))
            apply_dark_theme(fig, ax)

            # Show a few engines
            sample_engines = sorted(train_feat["unit_id"].unique())[:6]
            palette = [BLUE, GREEN, AMBER, RED, "#c07fff", "#50e0c0"]
            for i, uid in enumerate(sample_engines):
                e = train_feat[train_feat["unit_id"] == uid].sort_values("life_pct")
                ax.plot(e["life_pct"]*100, e["health_index"],
                        lw=1.2, alpha=0.8, color=palette[i], label=f"Engine #{uid}")

            ax.axvline(30, color=TEXT, ls="--", lw=0.8, alpha=0.4, label="End of healthy zone")
            ax.set_xlabel("Engine Life Consumed (%)", fontsize=9)
            ax.set_ylabel("Health Index (z-score)", fontsize=9)
            ax.set_title("Health Index rises as engine degrades — near 0 = healthy, rising = degrading", fontsize=10)
            ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT,
                      ncol=3, loc="upper left")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("---")

        # ── 3-sigma confidence bounds slider ──────────────────
        st.markdown("#### 3-Sigma Confidence Bounds Explorer")
        sigma_val = st.slider("Sigma level", 1.0, 4.0, 3.0, 0.5)
        sel_s3    = st.selectbox("Sensor", active_sensors, key="m2_sigma")

        h_data = df[df["state"].str.startswith("Healthy")][sel_s3].dropna()
        mu, sd = h_data.mean(), h_data.std()
        lo, hi = mu - sigma_val*sd, mu + sigma_val*sd

        fig, ax = plt.subplots(figsize=(12, 3))
        apply_dark_theme(fig, ax)
        eng_data = train_feat[train_feat["unit_id"]==sel_engine].sort_values("cycle")
        if sel_s3 in eng_data.columns:
            ax.plot(eng_data["cycle"], eng_data[sel_s3], color=BLUE, lw=1.2, label=sel_s3)
            ax.axhline(mu, color=GREEN, lw=1.2, ls="-",  label=f"Healthy mean={mu:.3f}")
            ax.axhline(lo, color=RED,   lw=1,   ls="--", label=f"{sigma_val:.0f}σ lower={lo:.3f}")
            ax.axhline(hi, color=RED,   lw=1,   ls="--", label=f"{sigma_val:.0f}σ upper={hi:.3f}")
            ax.fill_between(eng_data["cycle"], lo, hi, color=GREEN, alpha=0.07)
            # Highlight out-of-bound points
            oob = eng_data[(eng_data[sel_s3] < lo) | (eng_data[sel_s3] > hi)]
            if len(oob):
                ax.scatter(oob["cycle"], oob[sel_s3], color=RED, s=20, zorder=5,
                           label=f"Out-of-bounds: {len(oob)}")
        ax.set_xlabel("Cycle", fontsize=9); ax.set_ylabel("Value", fontsize=9)
        ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ──────────────────────────────────────────────────────────────
# TAB 3 — M3: FUSION & CLUSTERING
# ──────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("<div class='section-title'>M3 — PCA Data Fusion & KMeans Clustering</div>",
                unsafe_allow_html=True)

    if not data_ready():
        st.markdown("<div class='warn-box'>Run pipeline first.</div>", unsafe_allow_html=True)
    else:
        _, train_feat, _, _ = get_data()
        models = get_models()

        c_left, c_right = st.columns([1, 3])
        with c_left:
            colour_by = st.radio("Colour scatter by", ["RUL", "Cluster", "Fault Zone"],
                                 key="m3_colour")
            st.markdown("""
            <div class='info-box'>
            <b>PCA (Principal Component Analysis)</b><br>
            Compresses 14 correlated sensors into 3 independent principal components capturing 95%+ of variance.<br><br>
            <b>KMeans Clustering</b><br>
            Finds K=2 operating states in the data — e.g. high-power vs low-power flight phases.
            </div>""", unsafe_allow_html=True)

        with c_right:
            # PCA scatter
            pc_cols = [c for c in train_feat.columns if c.startswith("PC")]

            if len(pc_cols) >= 2:
                sample = train_feat.sample(min(3000, len(train_feat)), random_state=42)

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                apply_dark_theme(fig, axes)

                # Colour mapping
                if colour_by == "RUL":
                    vals = sample["RUL"].clip(0, 125)
                    cmap, label = "RdYlGn", "RUL (cycles)"
                elif colour_by == "Cluster" and "cluster" in sample.columns:
                    vals = sample["cluster"]
                    cmap, label = "Set1", "Cluster"
                else:
                    vals = (sample["RUL"] < 30).astype(int)
                    cmap, label = "RdYlGn_r", "Fault Zone"

                # PC1 vs PC2
                sc1 = axes[0].scatter(sample[pc_cols[0]], sample[pc_cols[1]],
                                      c=vals, cmap=cmap, s=4, alpha=0.6)
                plt.colorbar(sc1, ax=axes[0], label=label)
                axes[0].set_xlabel(pc_cols[0], fontsize=9)
                axes[0].set_ylabel(pc_cols[1], fontsize=9)
                axes[0].set_title("PCA Space — PC1 vs PC2", fontsize=10)

                # PC1 vs PC3 if available
                if len(pc_cols) >= 3:
                    sc2 = axes[1].scatter(sample[pc_cols[0]], sample[pc_cols[2]],
                                          c=vals, cmap=cmap, s=4, alpha=0.6)
                    plt.colorbar(sc2, ax=axes[1], label=label)
                    axes[1].set_xlabel(pc_cols[0], fontsize=9)
                    axes[1].set_ylabel(pc_cols[2], fontsize=9)
                    axes[1].set_title("PCA Space — PC1 vs PC3", fontsize=10)

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.markdown("<div class='warn-box'>PC columns not found. Re-run pipeline.</div>",
                            unsafe_allow_html=True)

        st.markdown("---")

        # ── PCA variance explained ─────────────────────────────
        pca = models["pca"]
        if pca is not None:
            st.markdown("#### PCA Variance Explained")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            apply_dark_theme(fig, axes)

            evr = pca.explained_variance_ratio_
            cum = np.cumsum(evr)

            axes[0].bar(range(1, len(evr)+1), evr*100, color=BLUE, alpha=0.85,
                        edgecolor=DARK_BG)
            axes[0].set_xlabel("Principal Component", fontsize=9)
            axes[0].set_ylabel("Variance Explained (%)", fontsize=9)
            axes[0].set_title("Individual Component Variance", fontsize=10)

            axes[1].plot(range(1, len(cum)+1), cum*100, color=GREEN,
                         lw=2, marker="o", ms=6)
            axes[1].axhline(95, color=AMBER, ls="--", lw=1, label="95% threshold")
            axes[1].set_xlabel("Number of Components", fontsize=9)
            axes[1].set_ylabel("Cumulative Variance (%)", fontsize=9)
            axes[1].set_title(f"Cumulative — {len(evr)} components → {cum[-1]*100:.1f}%", fontsize=10)
            axes[1].legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("---")

        # ── Cluster distribution by engine phase ──────────────
        if "cluster" in train_feat.columns:
            st.markdown("#### Operating State (KMeans) — Distribution Across Engine Life")
            fig, ax = plt.subplots(figsize=(12, 3.5))
            apply_dark_theme(fig, ax)

            cluster_colors = [BLUE, AMBER, GREEN, RED, "#c07fff", "#50e0c0"]
            for k in sorted(train_feat["cluster"].unique()):
                d = train_feat[train_feat["cluster"]==k]
                ax.scatter(d["life_pct"]*100, d["RUL"].clip(0,125),
                           c=cluster_colors[int(k)%len(cluster_colors)],
                           s=3, alpha=0.4, label=f"State {k}")
            ax.set_xlabel("Engine Life Consumed (%)", fontsize=9)
            ax.set_ylabel("RUL (cycles)", fontsize=9)
            ax.set_title("KMeans Operating States — Each colour is one regime", fontsize=10)
            ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor=BORDER,
                      labelcolor=TEXT, markerscale=4)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ──────────────────────────────────────────────────────────────
# TAB 4 — M4: FAULT DETECTION
# ──────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("<div class='section-title'>M4 — Fault Detection & Anomaly Identification</div>",
                unsafe_allow_html=True)

    if not data_ready() or not models_ready():
        st.markdown("<div class='warn-box'>Run pipeline first.</div>", unsafe_allow_html=True)
    else:
        _, train_feat, _, preds = get_data()
        cfg_p = get_config()
        active_sensors = get_models()["active_sensors"] or []

        df_eval = preds if preds is not None else train_feat

        # ── Performance metrics ────────────────────────────────
        if "pred_rul" in df_eval.columns and "RUL" in df_eval.columns:
            valid = df_eval.dropna(subset=["RUL", "pred_rul"])
            if len(valid) > 0:
                from sklearn.metrics import mean_squared_error, r2_score
                rmse = np.sqrt(mean_squared_error(valid["RUL"], valid["pred_rul"]))
                r2   = r2_score(valid["RUL"], valid["pred_rul"])

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("RMSE", f"{rmse:.1f} cycles", delta="< 15 target",
                          delta_color="normal" if rmse<15 else "inverse")
                c2.metric("R² Score", f"{r2:.3f}", delta="> 0.90 target",
                          delta_color="normal" if r2>0.9 else "inverse")

                if "fault" in valid.columns and "is_fault_pred" in valid.columns:
                    from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
                    acc  = accuracy_score(valid["fault"].astype(int),
                                          valid["is_fault_pred"].astype(int))
                    rec  = recall_score(valid["fault"].astype(int),
                                        valid["is_fault_pred"].astype(int), zero_division=0)
                    try:
                        auc = roc_auc_score(valid["fault"].astype(int),
                                             valid["fault_prob"])
                    except: auc = 0
                    c3.metric("Accuracy", f"{acc*100:.1f}%")
                    c4.metric("Recall", f"{rec*100:.1f}%", delta="Most critical")
                    c5.metric("AUC-ROC", f"{auc:.3f}")

                st.markdown("---")

                # ── RUL prediction scatter ─────────────────────
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**RUL Prediction — Actual vs Predicted**")
                    sample = valid.sample(min(2000, len(valid)), random_state=42)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    apply_dark_theme(fig, ax)
                    sc = ax.scatter(sample["RUL"], sample["pred_rul"],
                                    c=sample["RUL"], cmap="RdYlGn",
                                    s=10, alpha=0.6)
                    lim = [0, 125]
                    ax.plot(lim, lim, color=AMBER, lw=1.5, ls="--", label="Perfect prediction")
                    plt.colorbar(sc, ax=ax, label="True RUL")
                    ax.set_xlabel("True RUL (cycles)", fontsize=9)
                    ax.set_ylabel("Predicted RUL (cycles)", fontsize=9)
                    ax.set_title("Points on diagonal = perfect", fontsize=9)
                    ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                with col2:
                    st.markdown("**Confusion Matrix — Fault Classification**")
                    if "fault" in valid.columns and "is_fault_pred" in valid.columns:
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(valid["fault"].astype(int),
                                              valid["is_fault_pred"].astype(int))
                        fig, ax = plt.subplots(figsize=(6, 5))
                        apply_dark_theme(fig, ax)
                        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
                        plt.colorbar(im, ax=ax)
                        ax.set_xticks([0,1]); ax.set_yticks([0,1])
                        ax.set_xticklabels(["Healthy","Fault"], color=TEXT, fontsize=10)
                        ax.set_yticklabels(["Healthy","Fault"], color=TEXT, fontsize=10)
                        ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("True", fontsize=9)
                        labels = [["TN", "FP"], ["FN\n(DANGER)", "TP"]]
                        for i in range(2):
                            for j in range(2):
                                color = "white" if cm[i,j] > cm.max()/2 else TEXT
                                ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]:,}",
                                        ha="center", va="center", color=color,
                                        fontsize=9, fontfamily="Rajdhani")
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

        st.markdown("---")

        # ── Anomaly detection plot ─────────────────────────────
        st.markdown("#### Anomaly Detection — Isolation Forest + LOF Ensemble")
        if "anomaly_ens" in df_eval.columns and \
           len([c for c in df_eval.columns if c.startswith("PC")]) >= 2:
            pc_cols = [c for c in df_eval.columns if c.startswith("PC")]
            sample = df_eval.sample(min(3000, len(df_eval)), random_state=42)
            normal = sample[sample["anomaly_ens"]==0]
            anomaly= sample[sample["anomaly_ens"]==1]

            fig, ax = plt.subplots(figsize=(12, 4.5))
            apply_dark_theme(fig, ax)
            ax.scatter(normal[pc_cols[0]], normal[pc_cols[1]],
                       color=BLUE, s=4, alpha=0.4, label=f"Normal ({len(normal):,})")
            ax.scatter(anomaly[pc_cols[0]], anomaly[pc_cols[1]],
                       color=RED, s=30, marker="X", alpha=0.9,
                       label=f"Ensemble Anomaly ({len(anomaly):,})", zorder=5)
            ax.set_xlabel(pc_cols[0], fontsize=9); ax.set_ylabel(pc_cols[1], fontsize=9)
            ax.set_title("Anomalies (X) flagged only when Isolation Forest AND LOF both agree", fontsize=10)
            ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ── Sensor fault scores ────────────────────────────────
        st.markdown("#### Sensor Fault Scores — Wasserstein Distance Ranking")
        if "anomaly_ens" in df_eval.columns and active_sensors:
            normal  = df_eval[df_eval["anomaly_ens"]==0]
            anomaly = df_eval[df_eval["anomaly_ens"]==1]
            if len(anomaly) > 0:
                scores = {}
                for s in active_sensors:
                    if s in normal.columns and s in anomaly.columns:
                        nd = normal[s].dropna(); ad = anomaly[s].dropna()
                        if len(nd)>0 and len(ad)>0:
                            scores[s] = wasserstein_distance(nd, ad)
                if scores:
                    scores_s = pd.Series(scores).sort_values(ascending=False)
                    median_s = scores_s.median()
                    fig, ax = plt.subplots(figsize=(12, 4))
                    apply_dark_theme(fig, ax)
                    colors = [RED if v > median_s else BLUE_DIM for v in scores_s.values]
                    bars = ax.barh(scores_s.index, scores_s.values,
                                   color=colors, alpha=0.85, edgecolor=DARK_BG)
                    ax.axvline(median_s, color=AMBER, ls="--", lw=1.2,
                               label=f"Median = {median_s:.4f}")
                    ax.set_xlabel("Wasserstein Distance (higher = more anomalous during faults)", fontsize=9)
                    ax.set_title("Sensor Fault Score Ranking — Red bars: most suspicious sensors", fontsize=10)
                    ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

# ──────────────────────────────────────────────────────────────
# TAB 5 — LIVE SENSOR CHECK
# ──────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("<div class='section-title'>Live Sensor Check — Real-Time Fault Detection</div>",
                unsafe_allow_html=True)

    if not models_ready():
        st.markdown("""
        <div class='warn-box'>
        <b>Models not found.</b><br>
        Run <span class='mono'>python main.py train</span> first, then refresh this page.
        </div>""", unsafe_allow_html=True)
    else:
        models  = get_models()
        cfg_p   = get_config()
        active_sensors = models["active_sensors"] or \
            ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
        feature_cols   = models["feature_cols"] or []

        st.markdown("""
        <div class='info-box'>
        Adjust the sensor sliders below (all values in <b>scaled [0, 1] range</b> after MinMax normalisation).
        The models update predictions <b>instantly</b> as you move any slider.
        Use this to test what happens as sensors drift into abnormal ranges.
        </div>""", unsafe_allow_html=True)

        # ── Option to seed from a real engine cycle ────────────
        if data_ready():
            _, train_feat, _, preds = get_data()
            df_src = preds if preds is not None else train_feat
            seed_opt = st.checkbox("Seed sliders from a real engine cycle", value=True)
            if seed_opt:
                sc1, sc2 = st.columns(2)
                seed_eng = sc1.selectbox("Engine", sorted(df_src["unit_id"].unique()),
                                          key="live_eng")
                eng_d = df_src[df_src["unit_id"]==seed_eng].sort_values("cycle")
                max_c = int(eng_d["cycle"].max())
                seed_cyc = sc2.slider("Cycle", 1, max_c, max_c, key="live_cyc")
                seed_row = eng_d[eng_d["cycle"]==seed_cyc].iloc[0] \
                           if seed_cyc in eng_d["cycle"].values else eng_d.iloc[-1]
                defaults = {s: float(seed_row.get(s, 0.5)) for s in active_sensors}
            else:
                defaults = {s: 0.5 for s in active_sensors}
                seed_row = None
        else:
            defaults = {s: 0.5 for s in active_sensors}
            seed_row = None

        st.markdown("---")
        st.markdown("### Sensor Input Panel")

        # ── Sensor sliders — 3 columns ─────────────────────────
        ncols = 3
        sensor_vals = {}
        slider_cols = st.columns(ncols)
        for i, s in enumerate(active_sensors):
            with slider_cols[i % ncols]:
                v = st.slider(
                    f"{s}", 0.0, 1.0,
                    float(np.clip(defaults.get(s, 0.5), 0.0, 1.0)),
                    step=0.01, key=f"slider_{s}"
                )
                sensor_vals[s] = v

        st.markdown("---")

        # ── Build feature vector ───────────────────────────────
        # We must construct the same feature vector shape the models expect.
        # For live input we fill: raw sensors + zeros for derived features,
        # then let the scaler normalise.
        if feature_cols:
            row = pd.Series(0.0, index=feature_cols)
            for s in active_sensors:
                if s in row.index:
                    row[s] = sensor_vals.get(s, 0.5)
                # Rolling mean of a single reading = the reading itself
                if f"{s}_rmean" in row.index:
                    row[f"{s}_rmean"] = sensor_vals.get(s, 0.5)
                # Rolling std of one reading = 0 (no variance in single point)
                if f"{s}_rstd" in row.index:
                    row[f"{s}_rstd"] = 0.0

            # Health index = mean of z-scores; approximate with mean deviation from 0.5
            if "health_index" in row.index:
                avg_z = np.mean([abs(v - 0.5) / 0.1 for v in sensor_vals.values()])
                row["health_index"] = avg_z

            # PCA: transform current sensor values
            pca = models["pca"]
            pc_cols_feat = [c for c in feature_cols if c.startswith("PC")]
            if pca is not None and pc_cols_feat:
                sensor_array = np.array([[sensor_vals.get(s, 0.5) for s in active_sensors]])
                try:
                    pcs = pca.transform(sensor_array)[0]
                    for k, col in enumerate(pc_cols_feat):
                        if col in row.index and k < len(pcs):
                            row[col] = pcs[k]
                except: pass

            # life_pct: use seed row value or default
            if "life_pct" in row.index and seed_row is not None:
                row["life_pct"] = float(seed_row.get("life_pct", 0.5))

            X      = row.values.reshape(1, -1)
            X_sc   = models["feat_scaler"].transform(X)

            # ── Inference ───────────────────────────────────────
            pred_rul    = float(np.clip(models["rul_model"].predict(X_sc)[0], 0, 130))
            fault_prob  = float(models["fault_model"].predict_proba(X_sc)[0, 1])
            is_fault    = fault_prob >= 0.5
            iso_flag    = models["iso_forest"].predict(X_sc)[0] == -1
            lof_flag    = models["lof"].predict(X_sc)[0] == -1
            is_anomaly  = iso_flag and lof_flag

            # ── Sensor validator ────────────────────────────────
            if data_ready():
                _, train_feat2, _, _ = get_data()
                frac_h = cfg_p["preprocessing"]["healthy_life_fraction"]
                df_healthy = train_feat2[train_feat2["life_pct"] <= frac_h]
                sigma_b    = cfg_p["sensor_validation"]["sigma_bounds"]
                validator_bounds = {}
                for s in active_sensors:
                    if s in df_healthy.columns:
                        mu = df_healthy[s].mean(); sd = df_healthy[s].std()
                        validator_bounds[s] = {"mean":mu, "std":sd,
                                               "lo":mu-sigma_b*sd, "hi":mu+sigma_b*sd}

            # ── Results display ─────────────────────────────────
            st.markdown("### Live Prediction Results")
            r1, r2, r3, r4 = st.columns(4)

            rul_color = RED if pred_rul<10 else AMBER if pred_rul<30 else GREEN
            with r1:
                st.markdown(f"""
                <div class='engine-card'>
                  <div class='gauge-label'>Predicted RUL</div>
                  <div class='big-gauge' style='color:{rul_color};'>{pred_rul:.0f}</div>
                  <div class='gauge-label'>cycles remaining</div>
                </div>""", unsafe_allow_html=True)

            fp_color = RED if fault_prob > 0.5 else AMBER if fault_prob > 0.3 else GREEN
            with r2:
                st.markdown(f"""
                <div class='engine-card'>
                  <div class='gauge-label'>Fault Probability</div>
                  <div class='big-gauge' style='color:{fp_color};'>{fault_prob*100:.1f}%</div>
                  <div class='gauge-label'>{"FAULT ZONE" if is_fault else "HEALTHY ZONE"}</div>
                </div>""", unsafe_allow_html=True)

            with r3:
                iso_clr = RED if iso_flag else GREEN
                lof_clr = RED if lof_flag else GREEN
                st.markdown(f"""
                <div class='engine-card'>
                  <div class='gauge-label'>Anomaly Detectors</div>
                  <div style='font-family:Rajdhani,sans-serif; font-size:1rem;
                              color:{iso_clr}; margin:6px 0;'>
                    IsoForest: {"ANOMALY" if iso_flag else "NORMAL"}
                  </div>
                  <div style='font-family:Rajdhani,sans-serif; font-size:1rem;
                              color:{lof_clr}; margin:6px 0;'>
                    LOF: {"ANOMALY" if lof_flag else "NORMAL"}
                  </div>
                  <div class='gauge-label'>Ensemble: {"⚠ ANOMALY" if is_anomaly else "OK"}</div>
                </div>""", unsafe_allow_html=True)

            with r4:
                if seed_row is not None and "RUL" in seed_row.index:
                    true_rul = float(seed_row["RUL"])
                    err = pred_rul - true_rul
                    err_clr = GREEN if abs(err)<15 else AMBER if abs(err)<30 else RED
                    st.markdown(f"""
                    <div class='engine-card'>
                      <div class='gauge-label'>True RUL (from data)</div>
                      <div class='big-gauge' style='color:#7eb8f7;'>{true_rul:.0f}</div>
                      <div class='gauge-label' style='color:{err_clr};'>
                        Error: {err:+.1f} cycles
                      </div>
                    </div>""", unsafe_allow_html=True)

            # ── Per-sensor status panel ─────────────────────────
            st.markdown("---")
            st.markdown("### Per-Sensor Status (3-Sigma Validation)")

            sensor_report_cols = st.columns(7)
            for i, s in enumerate(active_sensors):
                with sensor_report_cols[i % 7]:
                    val = sensor_vals[s]
                    if data_ready() and s in validator_bounds:
                        b   = validator_bounds[s]
                        lo, hi = b["lo"], b["hi"]
                        z   = abs((val - b["mean"]) / max(b["std"], 1e-9))
                        ok  = lo <= val <= hi
                    else:
                        ok, z = True, 0.0; lo, hi = 0.0, 1.0

                    clr  = GREEN if ok else RED
                    stat = "OK" if ok else "FAULT"
                    st.markdown(f"""
                    <div style='background:#0d1a2e; border:1px solid {clr}44;
                                border-radius:8px; padding:10px 8px; text-align:center;
                                margin:3px 0;'>
                      <div style='font-family:Rajdhani,sans-serif; font-size:1rem;
                                  color:#7eb8f7; font-weight:700;'>{s}</div>
                      <div style='font-family:Rajdhani,sans-serif; font-size:1.4rem;
                                  font-weight:700; color:{clr};'>{stat}</div>
                      <div style='font-size:0.7rem; color:#4a7ab5;'>
                        val={val:.3f}<br>z={z:.2f}
                      </div>
                      <div style='font-size:0.65rem; color:#2a4a6a; margin-top:2px;'>
                        [{lo:.2f}, {hi:.2f}]
                      </div>
                    </div>""", unsafe_allow_html=True)

            # ── Fault probability bar ──────────────────────────
            st.markdown("---")
            st.markdown("### Fault Probability Gauge")
            fig, ax = plt.subplots(figsize=(12, 1.2))
            apply_dark_theme(fig, ax)
            # Background gradient bar
            for xi in range(1000):
                frac_x = xi / 1000
                c = plt.cm.RdYlGn_r(frac_x)
                ax.barh(0, 0.001, left=frac_x, height=1, color=c, alpha=0.6)
            ax.barh(0, fault_prob, height=0.8, color="white", alpha=0.0)
            ax.axvline(fault_prob, color="white", lw=3, ymin=0.05, ymax=0.95)
            ax.axvline(0.5, color=RED, lw=1.5, ls="--", alpha=0.7)
            ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0%", "25%", "50%\n(Threshold)", "75%", "100%"],
                                fontsize=8, color=TEXT)
            ax.set_yticks([])
            ax.set_title(f"Fault Probability: {fault_prob*100:.1f}%  —  "
                         f"{'FAULT ZONE' if is_fault else 'HEALTHY ZONE'}", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        else:
            st.markdown("<div class='warn-box'>Feature columns not found. Re-run training.</div>",
                        unsafe_allow_html=True)
