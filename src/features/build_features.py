"""
src/features/build_features.py
───────────────────────────────
Feature engineering pipeline:
  • Rolling statistics (mean, std per sensor per engine)
  • Statistical significance testing (t-test healthy vs degraded)
  • Composite Health Index
  • PCA-based data fusion (sensor dimensionality reduction)
  • KMeans clustering (operating state identification)
  • Binary fault label (RUL < threshold → fault=1)

Milestones M2 (analysis) + M3 (clustering/fusion)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import zscore, ttest_ind
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.helpers import (load_config, get_logger, save_artifact,
                                timer, banner)


# ─────────────────────────────────────────────────────────────
# ROLLING FEATURES
# ─────────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, sensors: list,
                          window: int = 10) -> pd.DataFrame:
    """
    Add per-engine rolling mean and rolling std for each sensor.
    Rolling std captures increasing variability as engine degrades.
    """
    df = df.copy()
    for uid in df["unit_id"].unique():
        m = df["unit_id"] == uid
        for s in sensors:
            col       = df.loc[m, s]
            df.loc[m, f"{s}_rmean"] = col.rolling(window, min_periods=1).mean()
            df.loc[m, f"{s}_rstd"]  = (col.rolling(window, min_periods=1)
                                         .std().fillna(0))
    return df


# ─────────────────────────────────────────────────────────────
# SIGNIFICANT SENSORS (t-test)
# ─────────────────────────────────────────────────────────────

def find_significant_sensors(df: pd.DataFrame, sensors: list,
                              healthy_frac: float = 0.30,
                              alpha: float = 0.05) -> list:
    """
    Two-sample t-test: does a sensor change between healthy and degraded?
    Healthy  = first healthy_frac of each engine's life (life_pct ≤ 0.30)
    Degraded = rest of engine life

    Returns sensors where p-value < alpha (genuinely responsive to wear).
    """
    h = df[df["life_pct"] <= healthy_frac]
    d = df[df["life_pct"] >  healthy_frac]
    sig = []
    for s in sensors:
        _, p = ttest_ind(h[s].dropna(), d[s].dropna())
        if p < alpha:
            sig.append(s)
    return sig


# ─────────────────────────────────────────────────────────────
# HEALTH INDEX
# ─────────────────────────────────────────────────────────────

def compute_health_index(df: pd.DataFrame, sensors: list) -> pd.DataFrame:
    """
    Single-number engine health score.
    = mean z-score of significant sensors.
    Starts near 0 (healthy), drifts as engine degrades.
    """
    df = df.copy()
    if not sensors:
        df["health_index"] = 0.0
        return df
    z = df[sensors].apply(zscore, nan_policy="omit")
    df["health_index"] = z.mean(axis=1)
    return df


# ─────────────────────────────────────────────────────────────
# PCA — DATA FUSION
# ─────────────────────────────────────────────────────────────

def fit_pca(df_train: pd.DataFrame, sensors: list,
            variance: float = 0.95, config: dict = None) -> PCA:
    """
    Fit PCA retaining `variance` fraction of explained variance.
    Fuses redundant correlated sensors into independent components.
    """
    pca = PCA(n_components=variance, random_state=42)
    pca.fit(df_train[sensors].values)
    if config:
        save_artifact(pca, "pca_model", config)
    return pca


def apply_pca(df: pd.DataFrame, sensors: list, pca: PCA) -> pd.DataFrame:
    """Transform sensors with fitted PCA → add PC1, PC2, … columns."""
    X  = pca.transform(df[sensors].values)
    pc = pd.DataFrame(X,
                      columns=[f"PC{i+1}" for i in range(X.shape[1])],
                      index=df.index)
    return pd.concat([df, pc], axis=1)


# ─────────────────────────────────────────────────────────────
# CLUSTERING — OPERATING STATE DISCOVERY
# ─────────────────────────────────────────────────────────────

def find_best_k(X: np.ndarray, k_min: int = 2,
                k_max: int = 8, seed: int = 42) -> int:
    """
    Silhouette-score scan to find optimal K for KMeans.
    Silhouette ∈ [-1, 1]; higher = more separated clusters.
    """
    best_k, best_s = k_min, -1
    for k in range(k_min, k_max + 1):
        km     = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        s      = silhouette_score(X, labels,
                                  sample_size=min(5000, len(X)))
        if s > best_s:
            best_s, best_k = s, k
    return best_k


def fit_kmeans(df_train: pd.DataFrame, sensors: list,
               config: dict) -> KMeans:
    """Fit KMeans on training sensors with auto-selected K."""
    cc   = config["clustering"]
    X    = df_train[sensors].values
    k    = find_best_k(X, cc["k_min"], cc["k_max"], cc["random_state"])
    km   = KMeans(n_clusters=k, random_state=cc["random_state"],
                  n_init=cc["n_init"])
    km.fit(X)
    save_artifact(km, "kmeans_model", config)
    return km


def apply_kmeans(df: pd.DataFrame, sensors: list,
                 km: KMeans) -> pd.DataFrame:
    """Add cluster label column to DataFrame."""
    df = df.copy()
    df["cluster"] = km.predict(df[sensors].values)
    return df


# ─────────────────────────────────────────────────────────────
# FAULT LABEL
# ─────────────────────────────────────────────────────────────

def add_fault_label(df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    """Binary label: fault=1 if RUL < threshold cycles."""
    df = df.copy()
    df["fault"] = (df["RUL"] < threshold).astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

@timer
def build_features(df_train: pd.DataFrame, df_test: pd.DataFrame,
                   active_sensors: list, config: dict = None) -> dict:
    """
    Complete feature engineering pipeline.

    Returns
    -------
    dict with keys:
        train, test, feature_cols,
        significant_sensors, pca, kmeans
    """
    config = config or load_config()
    log    = get_logger(__name__, config)
    banner("STAGE 3 · BUILD FEATURES")

    pc      = config["preprocessing"]
    fc      = config["features"]
    did     = config["data"]["dataset_id"]
    pdir    = Path(config["paths"]["processed_data"])
    thresh  = config["models"]["fault_threshold_cycles"]

    # ── Rolling features ──────────────────────────────────
    w        = pc["feature_window"]
    df_train = add_rolling_features(df_train, active_sensors, w)
    df_test  = add_rolling_features(df_test,  active_sensors, w)
    log.info(f"  Rolling features added (window={w}) ✓")

    # ── Significant sensors ───────────────────────────────
    sig = find_significant_sensors(df_train, active_sensors,
                                    pc["healthy_life_fraction"])
    log.info(f"  Significant sensors ({len(sig)}): {sig}")
    save_artifact(sig, "significant_sensors", config)

    # ── Health Index ──────────────────────────────────────
    df_train = compute_health_index(df_train, sig)
    df_test  = compute_health_index(df_test,  sig)
    log.info("  Health Index computed ✓")

    # ── PCA ───────────────────────────────────────────────
    pca      = fit_pca(df_train, active_sensors,
                       fc["pca_variance"], config)
    df_train = apply_pca(df_train, active_sensors, pca)
    df_test  = apply_pca(df_test,  active_sensors, pca)
    n_pc     = pca.n_components_
    var_tot  = pca.explained_variance_ratio_.sum() * 100
    log.info(f"  PCA: {len(active_sensors)} sensors → {n_pc} components"
             f"  ({var_tot:.1f}% variance) ✓")

    # ── KMeans ────────────────────────────────────────────
    km       = fit_kmeans(df_train, active_sensors, config)
    df_train = apply_kmeans(df_train, active_sensors, km)
    df_test  = apply_kmeans(df_test,  active_sensors, km)
    log.info(f"  KMeans (K={km.n_clusters}) ✓")

    # ── Fault label ───────────────────────────────────────
    df_train = add_fault_label(df_train, thresh)
    df_test  = add_fault_label(df_test,  thresh)
    fault_pct = df_train["fault"].mean() * 100
    log.info(f"  Fault labels (threshold={thresh} cycles,"
             f" {fault_pct:.1f}% positive) ✓")

    # ── Build feature column list ─────────────────────────
    roll_cols  = ([f"{s}_rmean" for s in sig[:6]] +
                  [f"{s}_rstd"  for s in sig[:6]])
    pc_cols    = [f"PC{i+1}" for i in range(n_pc)]
    extra_cols = ["health_index", "cluster", "life_pct"]

    feature_cols = [c for c in
                    active_sensors + roll_cols + extra_cols + pc_cols
                    if c in df_train.columns]
    save_artifact(feature_cols, "feature_cols", config)
    log.info(f"  Total features: {len(feature_cols)}")

    # ── Save feature DataFrames ───────────────────────────
    df_train.to_csv(pdir / f"train_{did}_features.csv", index=False)
    df_test .to_csv(pdir / f"test_{did}_features.csv",  index=False)
    log.info("Feature engineering complete ✓")

    return {
        "train"               : df_train,
        "test"                : df_test,
        "feature_cols"        : feature_cols,
        "significant_sensors" : sig,
        "pca"                 : pca,
        "kmeans"              : km,
    }


if __name__ == "__main__":
    from src.data.make_dataset import make_dataset
    from src.data.preprocess   import preprocess
    cfg  = load_config()
    raw  = make_dataset(cfg)
    proc = preprocess(raw["train"], raw["test"], cfg)
    build_features(proc["train"], proc["test"], proc["active_sensors"], cfg)
