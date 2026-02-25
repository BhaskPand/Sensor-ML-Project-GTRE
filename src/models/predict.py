"""
src/models/predict.py
─────────────────────
Load saved models and run inference:
  • predict_batch()     — full test-set predictions
  • score_engine()      — real-time single-engine scoring
  • SensorValidator     — per-sensor 3-sigma health checker
  • sensor_fault_scores — Wasserstein-distance fault ranking
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wasserstein_distance
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.helpers import load_config, get_logger, load_artifact


# ─────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────

def load_all_models(config: dict) -> dict:
    """Load every saved artifact needed for inference."""
    def _l(name):
        return load_artifact(name, config)
    return {
        "rul_model"       : _l("xgb_rul_model"),
        "fault_model"     : _l("xgb_fault_clf"),
        "iso_forest"      : _l("iso_forest"),
        "lof"             : _l("lof_model"),
        "feat_scaler"     : _l("feat_scaler"),
        "minmax_scaler"   : _l("minmax_scaler"),
        "pca"             : _l("pca_model"),
        "kmeans"          : _l("kmeans_model"),
        "feature_cols"    : _l("feature_cols"),
        "active_sensors"  : _l("active_sensors"),
    }


# ─────────────────────────────────────────────────────────────
# BATCH PREDICTION
# ─────────────────────────────────────────────────────────────

def predict_batch(df: pd.DataFrame, models: dict,
                  config: dict, log=None) -> pd.DataFrame:
    """
    Run all models on a feature DataFrame.

    Adds columns
    ────────────
    pred_rul        — predicted remaining useful life (cycles)
    fault_prob      — probability of imminent failure [0–1]
    is_fault_pred   — binary fault flag (threshold 0.5)
    anomaly_iso     — Isolation Forest flag (1=anomaly)
    anomaly_lof     — LOF flag (1=anomaly)
    anomaly_ens     — ensemble flag (both must agree)
    """
    if log is None:
        log = get_logger(__name__, config)

    fc   = models["feature_cols"]
    X    = df[fc].fillna(0).values
    X_sc = models["feat_scaler"].transform(X)

    df = df.copy()

    # RUL
    df["pred_rul"]      = np.clip(models["rul_model"].predict(X_sc), 0, None)

    # Fault
    df["fault_prob"]    = models["fault_model"].predict_proba(X_sc)[:, 1]
    df["is_fault_pred"] = (df["fault_prob"] >= 0.5).astype(int)

    # Anomaly
    df["anomaly_iso"] = (models["iso_forest"].predict(X_sc) == -1).astype(int)
    df["anomaly_lof"] = (models["lof"].predict(X_sc)        == -1).astype(int)
    min_votes         = config["anomaly"]["ensemble_min_votes"]
    df["anomaly_ens"] = (
        (df["anomaly_iso"] + df["anomaly_lof"]) >= min_votes
    ).astype(int)

    log.info(f"  Predictions: {len(df):,} rows  |  "
             f"Anomalies: {df['anomaly_ens'].sum():,}")
    return df


# ─────────────────────────────────────────────────────────────
# SENSOR VALIDATOR
# ─────────────────────────────────────────────────────────────

class SensorValidator:
    """
    Real-time per-sensor health checker using 3-sigma bounds
    learned from healthy engine operation.

    Usage
    -----
        v = SensorValidator(df_healthy, active_sensors, config)
        report = v.validate({'s7': 553.4, 's12': 23.1, ...})
        print(report['OVERALL'])   # 'HEALTHY' or 'FAULT DETECTED'
    """

    def __init__(self, df_healthy: pd.DataFrame,
                 sensors: list, config: dict):
        sigma       = config["sensor_validation"]["sigma_bounds"]
        self.sensors = sensors
        self.bounds  = {}
        for s in sensors:
            mu  = df_healthy[s].mean()
            std = df_healthy[s].std()
            self.bounds[s] = {
                "mean"  : mu,
                "std"   : std,
                "lo"    : mu - sigma * std,
                "hi"    : mu + sigma * std,
            }

    def validate(self, reading: dict) -> dict:
        """
        Validate one set of sensor readings.

        Parameters
        ----------
        reading : dict  {sensor_name: float_value}

        Returns
        -------
        dict  per-sensor report + 'OVERALL' key
        """
        report    = {}
        any_fault = False

        for s in self.sensors:
            if s not in reading:
                continue
            val = float(reading[s])
            b   = self.bounds[s]
            z   = abs((val - b["mean"]) / (b["std"] + 1e-9))
            ok  = b["lo"] <= val <= b["hi"]
            if not ok:
                any_fault = True
            report[s] = {
                "value"  : round(val, 4),
                "z_score": round(z,   3),
                "lower"  : round(b["lo"], 4),
                "upper"  : round(b["hi"], 4),
                "status" : "OK" if ok else "FAULT",
            }

        report["OVERALL"] = "FAULT DETECTED" if any_fault else "HEALTHY"
        return report


# ─────────────────────────────────────────────────────────────
# SENSOR FAULT SCORES
# ─────────────────────────────────────────────────────────────

def sensor_fault_scores(df_normal: pd.DataFrame,
                         df_anomaly: pd.DataFrame,
                         sensors: list) -> pd.Series:
    """
    Rank sensors by how differently they behave during anomalies.
    Uses Wasserstein distance (earth-mover distance) between the
    normal and anomalous value distributions.
    Higher score → sensor is more likely responsible for the fault.
    """
    scores = {}
    for s in sensors:
        if s in df_normal.columns and s in df_anomaly.columns:
            scores[s] = wasserstein_distance(
                df_normal[s].dropna(),
                df_anomaly[s].dropna()
            )
    return pd.Series(scores).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────
# SINGLE ENGINE SCORING (GUI / API)
# ─────────────────────────────────────────────────────────────

def score_engine(unit_id: int, cycle: int,
                 df_pred: pd.DataFrame) -> dict:
    """
    Return health summary for one engine at one cycle.
    df_pred must already contain prediction columns from predict_batch().
    """
    mask = (df_pred["unit_id"] == unit_id) & (df_pred["cycle"] == cycle)
    if not mask.any():
        return {"error": f"No data for unit={unit_id} cycle={cycle}"}

    r          = df_pred[mask].iloc[0]
    pred_rul   = float(r.get("pred_rul",   -1))
    actual_rul = float(r.get("RUL",        -1)) if "RUL" in r.index else -1
    fault_prob = float(r.get("fault_prob",  0))
    anomaly    = bool (r.get("anomaly_ens", 0))

    return {
        "unit_id"   : unit_id,
        "cycle"     : cycle,
        "pred_rul"  : round(pred_rul,   1),
        "actual_rul": round(actual_rul, 1),
        "fault_pct" : round(fault_prob * 100, 1),
        "anomaly"   : anomaly,
        "status"    : "🚨 FAULT RISK" if fault_prob >= 0.5 else "✅ HEALTHY",
    }
