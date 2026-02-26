"""
src/models/evaluate.py
----------------------
Compute and report all evaluation metrics.
Evaluated on the engine-split hold-out rows (not NASA test file).
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


# -----------------------------------------------------------------
# REGRESSION - RUL
# -----------------------------------------------------------------

def eval_rul(y_true, y_pred, config, log) -> dict:
    """
    RMSE, MAE, R2 and NASA asymmetric score.
    NASA score penalises under-predicting RUL (late warnings) more.
    """
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    err   = y_pred - y_true
    nasa  = float(np.sum(
        np.where(err < 0,
                 np.exp(-err / 13) - 1,
                 np.exp( err / 10) - 1)
    ))

    ev = config["evaluation"]
    log.info("-" * 52)
    log.info("  RUL REGRESSION METRICS")
    log.info("-" * 52)
    log.info(f"  RMSE        : {rmse:.3f}  (target < {ev['rmse_target']})")
    log.info(f"  MAE         : {mae:.3f}  (target < {ev['mae_target']})")
    log.info(f"  R2          : {r2:.4f}  (target > {ev['r2_target']})")
    log.info(f"  NASA Score  : {nasa:.1f}  (lower = better)")

    grade = ("EXCELLENT"        if r2 >= ev["r2_target"] else
             "GOOD"             if r2 >= 0.75 else "NEEDS IMPROVEMENT")
    log.info(f"  Grade       : {grade}")

    return {"rmse": rmse, "mae": mae, "r2": r2,
            "nasa_score": nasa, "grade": grade}


# -----------------------------------------------------------------
# CLASSIFICATION - FAULT
# -----------------------------------------------------------------

def eval_fault(y_true, y_pred, y_proba, config, log) -> dict:
    """Accuracy, Precision, Recall, F1, AUC-ROC."""
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    auc  = float(roc_auc_score(y_true, y_proba))
    cm   = confusion_matrix(y_true, y_pred).tolist()

    ev = config["evaluation"]
    log.info("-" * 52)
    log.info("  FAULT CLASSIFICATION METRICS")
    log.info("-" * 52)
    log.info(f"  Accuracy    : {acc*100:.2f}%")
    log.info(f"  Precision   : {prec*100:.2f}%")
    log.info(f"  Recall      : {rec*100:.2f}%   <- most critical")
    log.info(f"  F1 Score    : {f1*100:.2f}%")
    log.info(f"  AUC-ROC     : {auc:.4f}  (target > {ev['auc_target']})")
    log.info(f"  Confusion   : TN={cm[0][0]}  FP={cm[0][1]}  "
             f"FN={cm[1][0]}  TP={cm[1][1]}")

    grade = ("EXCELLENT"   if auc >= ev["auc_target"] else
             "GOOD"        if auc >= 0.90 else "NEEDS IMPROVEMENT")
    log.info(f"  Grade       : {grade}")
    log.info("\n" + classification_report(
        y_true, y_pred, target_names=["Healthy", "Fault"]))

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc, "confusion_matrix": cm, "grade": grade}


# -----------------------------------------------------------------
# ANOMALY SUMMARY
# -----------------------------------------------------------------

def eval_anomaly(df, log) -> dict:
    n     = len(df)
    n_iso = int(df["anomaly_iso"].sum())
    n_lof = int(df["anomaly_lof"].sum())
    n_ens = int(df["anomaly_ens"].sum())

    log.info("-" * 52)
    log.info("  ANOMALY DETECTION SUMMARY")
    log.info("-" * 52)
    log.info(f"  Total rows        : {n:,}")
    log.info(f"  Isolation Forest  : {n_iso:,}  ({n_iso/n*100:.2f}%)")
    log.info(f"  LOF               : {n_lof:,}  ({n_lof/n*100:.2f}%)")
    log.info(f"  Ensemble (agreed) : {n_ens:,}  ({n_ens/n*100:.2f}%)")

    return {"total": n, "iso": n_iso, "lof": n_lof,
            "ensemble": n_ens, "pct_ens": round(n_ens/n*100, 2)}


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------

def evaluate_all(df_pred: pd.DataFrame, split: dict,
                 config: dict = None) -> dict:
    """
    Evaluate on df_pred which MUST be the held-out engine rows
    (split['df_test']) -- these have ground-truth RUL and fault labels.

    df_pred contains both predictions and true labels because it was
    created from split['df_test'] in main.py::stage_predict().
    """
    config = config or load_config()
    log    = get_logger(__name__, config)
    banner("STAGE 5 - EVALUATE")

    # Sanity check
    required = ["RUL", "fault", "pred_rul", "is_fault_pred", "fault_prob"]
    missing  = [c for c in required if c not in df_pred.columns]
    if missing:
        log.error(f"  Missing columns in df_pred: {missing}")
        return {}

    valid = df_pred.dropna(subset=required)
    if len(valid) == 0:
        log.error("  No valid rows after dropping NaN -- check pipeline.")
        return {}

    n_fault = int(valid["fault"].sum())
    n_total = len(valid)
    log.info(f"  Hold-out rows   : {n_total:,}")
    log.info(f"  Fault rows      : {n_fault:,}  ({n_fault/n_total*100:.1f}%)")
    log.info(f"  Healthy rows    : {n_total-n_fault:,}")

    y_rul_true  = valid["RUL"].values
    y_rul_pred  = valid["pred_rul"].values
    y_flt_true  = valid["fault"].values.astype(int)
    y_flt_pred  = valid["is_fault_pred"].values.astype(int)
    y_flt_proba = valid["fault_prob"].values

    reg  = eval_rul(y_rul_true, y_rul_pred, config, log)
    clf  = eval_fault(y_flt_true, y_flt_pred, y_flt_proba, config, log)
    ano  = eval_anomaly(df_pred, log)

    all_metrics = {"regression": reg, "classification": clf, "anomaly": ano}
    path = save_metrics(all_metrics, "eval_results", config)
    log.info(f"  Metrics saved: {path}")

    return all_metrics
