"""
src/models/evaluate.py
──────────────────────
Compute and report all evaluation metrics:
  • Regression  — RMSE, MAE, R², NASA score
  • Classification — Accuracy, Precision, Recall, F1, AUC-ROC
  • Anomaly     — detection summary statistics
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.helpers import load_config, get_logger, save_metrics, banner


# ─────────────────────────────────────────────────────────────
# REGRESSION — RUL
# ─────────────────────────────────────────────────────────────

def eval_rul(y_true: np.ndarray, y_pred: np.ndarray,
             config: dict, log) -> dict:
    """
    Evaluate RUL regression.

    NASA Score
    ──────────
    The official NASA scoring function penalises late predictions
    (underestimating remaining life) more than early ones:
        error < 0  →  exp(|error|/13) − 1   (engine might fail unexpectedly)
        error ≥ 0  →  exp(|error|/10) − 1   (unnecessary maintenance)
    Lower is better.
    """
    rmse       = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae        = float(mean_absolute_error(y_true, y_pred))
    r2         = float(r2_score(y_true, y_pred))
    errors     = y_pred - y_true
    nasa_score = float(np.sum(
        np.where(errors < 0,
                 np.exp(-errors / 13) - 1,
                 np.exp( errors / 10) - 1)
    ))

    ev  = config["evaluation"]
    log.info("─" * 50)
    log.info("RUL REGRESSION METRICS")
    log.info("─" * 50)
    log.info(f"  RMSE        : {rmse:.3f}  (target < {ev['rmse_target']})")
    log.info(f"  MAE         : {mae:.3f}  (target < {ev['mae_target']})")
    log.info(f"  R²          : {r2:.4f}  (target > {ev['r2_target']})")
    log.info(f"  NASA Score  : {nasa_score:.1f}  (lower = better)")
    grade = ("EXCELLENT" if r2 >= ev["r2_target"] else
             "GOOD"      if r2 >= 0.75 else "NEEDS IMPROVEMENT")
    log.info(f"  Grade       : {grade}")

    return {"rmse": rmse, "mae": mae, "r2": r2,
            "nasa_score": nasa_score, "grade": grade}


# ─────────────────────────────────────────────────────────────
# CLASSIFICATION — FAULT DETECTION
# ─────────────────────────────────────────────────────────────

def eval_fault(y_true: np.ndarray, y_pred: np.ndarray,
               y_proba: np.ndarray, config: dict, log) -> dict:
    """
    Evaluate binary fault classification.

    Key metric: Recall (sensitivity)
    ──────────────────────────────────
    In a safety-critical system, missing a real fault (False Negative)
    is far more dangerous than a false alarm (False Positive).
    Recall = TP / (TP + FN)  — fraction of real faults we caught.
    We always prioritise maximising Recall.
    """
    acc   = float(accuracy_score(y_true, y_pred))
    prec  = float(precision_score(y_true, y_pred, zero_division=0))
    rec   = float(recall_score(y_true, y_pred, zero_division=0))
    f1    = float(f1_score(y_true, y_pred, zero_division=0))
    auc   = float(roc_auc_score(y_true, y_proba))
    cm    = confusion_matrix(y_true, y_pred).tolist()

    ev  = config["evaluation"]
    log.info("─" * 50)
    log.info("FAULT CLASSIFICATION METRICS")
    log.info("─" * 50)
    log.info(f"  Accuracy    : {acc*100:.2f}%  (target > {ev['auc_target']*100:.0f}%)")
    log.info(f"  Precision   : {prec*100:.2f}%  (target > {ev['f1_target']*100:.0f}%)")
    log.info(f"  Recall      : {rec*100:.2f}%  ← most critical metric")
    log.info(f"  F1 Score    : {f1*100:.2f}%")
    log.info(f"  AUC-ROC     : {auc:.4f}  (target > {ev['auc_target']})")
    log.info(f"  Confusion   : TN={cm[0][0]}  FP={cm[0][1]}  "
             f"FN={cm[1][0]}  TP={cm[1][1]}")
    grade = ("EXCELLENT" if auc >= ev["auc_target"] else
             "GOOD"      if auc >= 0.90 else "NEEDS IMPROVEMENT")
    log.info(f"  Grade       : {grade}")
    log.info("\n" + classification_report(
        y_true, y_pred, target_names=["Healthy", "Fault"]))

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc, "confusion_matrix": cm, "grade": grade,
    }


# ─────────────────────────────────────────────────────────────
# ANOMALY SUMMARY
# ─────────────────────────────────────────────────────────────

def eval_anomaly(df: pd.DataFrame, log) -> dict:
    """
    Since anomaly detection is unsupervised (no ground-truth labels),
    we report detection rates rather than accuracy metrics.
    """
    n     = len(df)
    n_iso = int(df["anomaly_iso"].sum())
    n_lof = int(df["anomaly_lof"].sum())
    n_ens = int(df["anomaly_ens"].sum())

    log.info("─" * 50)
    log.info("ANOMALY DETECTION SUMMARY")
    log.info("─" * 50)
    log.info(f"  Total rows        : {n:,}")
    log.info(f"  Isolation Forest  : {n_iso:,}  ({n_iso/n*100:.2f}%)")
    log.info(f"  LOF               : {n_lof:,}  ({n_lof/n*100:.2f}%)")
    log.info(f"  Ensemble (agreed) : {n_ens:,}  ({n_ens/n*100:.2f}%)")

    return {
        "total": n, "iso": n_iso, "lof": n_lof, "ensemble": n_ens,
        "pct_ens": round(n_ens / n * 100, 2),
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def evaluate_all(df_pred: pd.DataFrame, split: dict,
                 config: dict = None) -> dict:
    """
    Run complete evaluation and save JSON results.

    Parameters
    ----------
    df_pred : DataFrame output from predict_batch()
    split   : dict from engine_split() with y_rul_test, y_fault_test
    config  : project config

    Returns
    -------
    dict: {regression, classification, anomaly}
    """
    config = config or load_config()
    log    = get_logger(__name__, config)
    banner("STAGE 5 · EVALUATE")

    # Align lengths
    n        = len(split["y_rul_test"])
    rows     = df_pred.head(n)
    y_rul_p  = rows["pred_rul"].values
    y_flt_p  = rows["is_fault_pred"].values
    y_flt_pr = rows["fault_prob"].values

    reg  = eval_rul(split["y_rul_test"],   y_rul_p,  config, log)
    clf  = eval_fault(split["y_fault_test"], y_flt_p, y_flt_pr, config, log)
    ano  = eval_anomaly(df_pred, log)

    all_metrics = {"regression": reg, "classification": clf, "anomaly": ano}
    path = save_metrics(all_metrics, "eval_results", config)
    log.info(f"  Metrics saved → {path}")

    return all_metrics
