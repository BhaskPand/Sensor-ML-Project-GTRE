"""
src/visualization/visualize.py
──────────────────────────────
All plotting functions. Every figure is saved to reports/figures/.
Covers: EDA, statistical analysis, clustering, anomaly, model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA as _PCA
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.helpers import load_config, get_logger

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _save(fig, name: str, config: dict) -> str:
    d = Path(config["paths"]["figures"])
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def _pca2d(df: pd.DataFrame, sensors: list) -> np.ndarray:
    """Quick 2-component PCA for visualisation only."""
    p = _PCA(n_components=2)
    return p.fit_transform(df[sensors].values)


# ─────────────────────────────────────────────────────────────
# M1 — EDA PLOTS
# ─────────────────────────────────────────────────────────────

def plot_sensor_distributions(df: pd.DataFrame, sensors: list,
                               config: dict) -> str:
    """Histogram of each active sensor value distribution."""
    cols = 4
    rows = (len(sensors) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
    axes = axes.flatten()
    for i, s in enumerate(sensors):
        axes[i].hist(df[s], bins=50, color="steelblue",
                     alpha=0.75, edgecolor="white")
        axes[i].set_title(s, fontweight="bold")
        axes[i].set_xlabel("Value")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Sensor Value Distributions", fontsize=16,
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, "01_sensor_distributions", config)


def plot_sensor_trends(df: pd.DataFrame, sensors: list,
                       unit_id: int, config: dict) -> str:
    """Line plots of sensor values over the lifecycle of one engine."""
    eng  = df[df["unit_id"] == unit_id]
    n    = len(sensors)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(18, rows * 2.5))
    axes = axes.flatten()
    for i, s in enumerate(sensors):
        axes[i].plot(eng["cycle"], eng[s], color="coral", linewidth=1.2)
        axes[i].set_title(f"Engine {unit_id} — {s}", fontweight="bold")
        axes[i].set_xlabel("Cycle")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Sensor Trends Over Engine Life (Unit {unit_id})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"02_sensor_trends_u{unit_id}", config)


def plot_correlation_heatmap(df: pd.DataFrame, sensors: list,
                              config: dict) -> str:
    """Lower-triangular Pearson correlation heatmap."""
    corr = df[sensors].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Sensor Correlation Heatmap", fontsize=16,
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, "03_correlation_heatmap", config)


# ─────────────────────────────────────────────────────────────
# M2 — STATISTICAL ANALYSIS PLOTS
# ─────────────────────────────────────────────────────────────

def plot_healthy_vs_degraded(df: pd.DataFrame, sensors: list,
                              config: dict) -> str:
    """Boxplots comparing healthy (first 30%) vs degraded sensor values."""
    df = df.copy()
    df["state"] = df["life_pct"].apply(
        lambda x: "Healthy\n(0–30%)" if x <= 0.3 else "Degraded\n(30–100%)"
    )
    n, cols = min(len(sensors), 12), 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes = axes.flatten()
    pal = {"Healthy\n(0–30%)": "#2ecc71", "Degraded\n(30–100%)": "#e74c3c"}
    for i, s in enumerate(sensors[:n]):
        sns.boxplot(data=df, x="state", y=s, ax=axes[i], palette=pal)
        axes[i].set_title(s, fontweight="bold")
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Healthy vs Degraded Sensor Distributions",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_healthy_vs_degraded", config)


def plot_pca_variance(pca, config: dict) -> str:
    """Scree plot: individual + cumulative explained variance."""
    indiv = pca.explained_variance_ratio_ * 100
    cumul = np.cumsum(indiv)
    n     = len(indiv)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.bar(range(1, n + 1), indiv, color="steelblue", alpha=0.7)
    a1.set(xlabel="Component", ylabel="Variance Explained (%)",
           title="Individual Variance")
    a2.plot(range(1, n + 1), cumul, "ro-", linewidth=2)
    a2.axhline(95, color="red", linestyle="--", label="95% threshold")
    a2.set(xlabel="Components", ylabel="Cumulative Variance (%)",
           title="Cumulative Variance")
    a2.legend()
    fig.suptitle("PCA Explained Variance — Sensor Data Fusion",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "05_pca_variance", config)


# ─────────────────────────────────────────────────────────────
# M3 — CLUSTERING PLOTS
# ─────────────────────────────────────────────────────────────

def plot_pca_rul(df: pd.DataFrame, sensors: list, config: dict) -> str:
    """2-D PCA scatter coloured by RUL (green=healthy, red=failure)."""
    X = _pca2d(df, sensors)
    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(X[:, 0], X[:, 1], c=df["RUL"].values,
                    cmap="RdYlGn", alpha=0.4, s=5)
    plt.colorbar(sc, ax=ax, label="RUL (cycles)")
    ax.set(xlabel="PC1", ylabel="PC2",
           title="PCA 2D — Coloured by RUL\n"
                 "(Green = Healthy  |  Red = Near Failure)")
    fig.tight_layout()
    return _save(fig, "06_pca_2d_rul", config)


def plot_clusters(df: pd.DataFrame, sensors: list, config: dict) -> str:
    """2-D PCA scatter coloured by KMeans cluster label."""
    X      = _pca2d(df, sensors)
    labels = df["cluster"].values
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(X[m, 0], X[m, 1], label=f"Cluster {c}",
                   alpha=0.4, s=5)
    ax.legend(markerscale=5, title="Cluster")
    ax.set(xlabel="PC1", ylabel="PC2",
           title=f"KMeans Clusters (K={len(np.unique(labels))}) in PCA Space")
    fig.tight_layout()
    return _save(fig, "07_kmeans_clusters", config)


def plot_kmeans_elbow(df: pd.DataFrame, sensors: list,
                      config: dict) -> str:
    """Elbow + silhouette score plot for K selection."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score as ss
    cc      = config["clustering"]
    X       = df[sensors].values
    Ks      = range(cc["k_min"], cc["k_max"] + 1)
    inertia, sil = [], []
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=cc["random_state"],
                    n_init=cc["n_init"])
        lbl = km.fit_predict(X)
        inertia.append(km.inertia_)
        sil.append(ss(X, lbl, sample_size=min(5000, len(X))))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(Ks, inertia, "bo-")
    a1.set(xlabel="K", ylabel="Inertia", title="Elbow Method")
    a2.plot(Ks, sil, "ro-")
    a2.set(xlabel="K", ylabel="Silhouette Score",
           title="Silhouette Score")
    fig.suptitle("Optimal K Selection", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "08_kmeans_elbow", config)


# ─────────────────────────────────────────────────────────────
# M4 — ANOMALY & FAULT PLOTS
# ─────────────────────────────────────────────────────────────

def plot_anomalies(df: pd.DataFrame, sensors: list, config: dict) -> str:
    """PCA scatter highlighting anomalous points in red."""
    X   = _pca2d(df, sensors)
    ens = df["anomaly_ens"].values
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(X[ens == 0, 0], X[ens == 0, 1],
               c="steelblue", alpha=0.3, s=3, label="Normal")
    ax.scatter(X[ens == 1, 0], X[ens == 1, 1],
               c="red", alpha=0.8, s=20, marker="x",
               linewidths=1.5, label="Anomaly")
    ax.legend(markerscale=4)
    ax.set(xlabel="PC1", ylabel="PC2",
           title="Anomaly Detection — Ensemble (IsoForest + LOF)")
    fig.tight_layout()
    return _save(fig, "09_anomaly_detection", config)


def plot_sensor_fault_scores(scores: pd.Series, config: dict) -> str:
    """Bar chart of per-sensor Wasserstein fault scores."""
    med    = scores.median()
    colors = ["red" if v > med else "steelblue" for v in scores.values]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(scores.index, scores.values, color=colors)
    ax.axhline(med, color="orange", linestyle="--",
               label=f"Median = {med:.4f}")
    ax.set(xlabel="Sensor", ylabel="Wasserstein Distance",
           title="Per-Sensor Fault Score  (Red = Above Median → Likely Faulty)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    return _save(fig, "10_sensor_fault_scores", config)


# ─────────────────────────────────────────────────────────────
# MODEL PERFORMANCE PLOTS
# ─────────────────────────────────────────────────────────────

def plot_rul_prediction(y_true: np.ndarray, y_pred: np.ndarray,
                        config: dict) -> str:
    """Actual vs Predicted RUL scatter + residual histogram."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2   = float(1 - np.sum((y_true - y_pred)**2) /
                 np.sum((y_true - y_true.mean())**2))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    lim = max(y_true.max(), y_pred.max())
    a1.scatter(y_true, y_pred, alpha=0.3, s=5, color="steelblue")
    a1.plot([0, lim], [0, lim], "r--", linewidth=2, label="Perfect")
    a1.set(xlabel="Actual RUL", ylabel="Predicted RUL",
           title=f"Actual vs Predicted RUL\nRMSE={rmse:.2f}   R²={r2:.4f}")
    a1.legend()
    residuals = y_true - y_pred
    a2.hist(residuals, bins=60, color="coral", alpha=0.75,
            edgecolor="white")
    a2.axvline(0, color="red", linestyle="--")
    a2.set(xlabel="Residual (Actual − Predicted)", ylabel="Count",
           title="Residual Distribution")
    fig.suptitle("RUL Regression Performance", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, "11_rul_prediction", config)


def plot_confusion_roc(y_true: np.ndarray, y_pred: np.ndarray,
                       y_proba: np.ndarray, config: dict) -> str:
    """Confusion matrix + ROC curve side-by-side."""
    auc        = roc_auc_score(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    cm         = confusion_matrix(y_true, y_pred)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=a1,
                xticklabels=["Healthy", "Fault"],
                yticklabels=["Healthy", "Fault"],
                linewidths=0.5)
    a1.set(ylabel="Actual", xlabel="Predicted",
           title="Confusion Matrix")
    a2.plot(fpr, tpr, "b-", linewidth=2.5, label=f"AUC = {auc:.4f}")
    a2.fill_between(fpr, tpr, alpha=0.1, color="blue")
    a2.plot([0, 1], [0, 1], "r--", label="Random")
    a2.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve")
    a2.legend()
    fig.suptitle("Fault Classification Performance", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, "12_fault_classification", config)


def plot_shap(model, X: np.ndarray, feature_cols: list,
              config: dict) -> str:
    """SHAP beeswarm summary plot (feature importance + direction)."""
    try:
        import shap
        exp  = shap.TreeExplainer(model)
        vals = exp.shap_values(X[:2000])
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(vals, X[:2000],
                          feature_names=feature_cols, show=False)
        plt.title("SHAP Feature Importance — RUL Model",
                  fontweight="bold")
        plt.tight_layout()
        return _save(plt.gcf(), "13_shap_importance", config)
    except Exception as e:
        print(f"  SHAP skipped: {e}")
        return ""


# ─────────────────────────────────────────────────────────────
# GENERATE ALL PLOTS (called from main pipeline)
# ─────────────────────────────────────────────────────────────

def generate_all_plots(df_train: pd.DataFrame, df_pred: pd.DataFrame,
                       sensors: list, pca, split: dict,
                       config: dict) -> None:
    """Generate and save every plot in sequence."""
    log = get_logger(__name__, config)
    log.info("Generating all visualisations …")

    paths = []
    paths.append(plot_sensor_distributions(df_train, sensors, config))
    paths.append(plot_sensor_trends(df_train, sensors, 1, config))
    paths.append(plot_correlation_heatmap(df_train, sensors, config))
    paths.append(plot_healthy_vs_degraded(df_train, sensors, config))
    paths.append(plot_pca_variance(pca, config))
    paths.append(plot_pca_rul(df_train, sensors, config))
    paths.append(plot_clusters(df_train, sensors, config))
    paths.append(plot_kmeans_elbow(df_train, sensors, config))

    if "anomaly_ens" in df_pred.columns:
        paths.append(plot_anomalies(df_pred, sensors, config))

    y_true = split["y_rul_test"]
    n      = len(y_true)
    y_pred = df_pred["pred_rul"].values[:n]
    paths.append(plot_rul_prediction(y_true, y_pred, config))
    paths.append(plot_confusion_roc(
        split["y_fault_test"],
        df_pred["is_fault_pred"].values[:n],
        df_pred["fault_prob"].values[:n],
        config
    ))

    for p in paths:
        if p:
            log.info(f"  Saved → {p}")
    log.info("All plots saved ✓")
