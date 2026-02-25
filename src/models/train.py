"""
src/models/train.py
───────────────────
Train all models:
  1. XGBoost RUL Regressor      (predicts cycles remaining)
  2. XGBoost Fault Classifier   (binary: near-failure or not)
  3. Isolation Forest           (unsupervised anomaly detection)
  4. Local Outlier Factor       (unsupervised anomaly detection)

Milestone M4 — Identifying Faulty Sensors
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import xgboost as xgb
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.helpers import (load_config, get_logger, save_artifact,
                                timer, banner)


# ─────────────────────────────────────────────────────────────
# ENGINE-LEVEL SPLIT (prevents data leakage)
# ─────────────────────────────────────────────────────────────

def engine_split(df: pd.DataFrame, feature_cols: list,
                 test_size: float = 0.20,
                 random_state: int = 42) -> dict:
    """
    Split by engine ID — NOT by row.

    Why this matters
    ────────────────
    A random row split lets the model see cycle 50 of engine #5
    during training and predict cycle 51 during testing.
    That is unrealistic (and cheating).
    Engine-level split ensures test engines are completely unseen.
    """
    units    = df["unit_id"].unique()
    tr_units, te_units = train_test_split(
        units, test_size=test_size, random_state=random_state
    )
    tr = df["unit_id"].isin(tr_units)
    te = df["unit_id"].isin(te_units)

    X_tr = df.loc[tr, feature_cols].fillna(0).values
    X_te = df.loc[te, feature_cols].fillna(0).values
    y_rul_tr    = df.loc[tr, "RUL"].values
    y_rul_te    = df.loc[te, "RUL"].values
    y_fault_tr  = df.loc[tr, "fault"].values
    y_fault_te  = df.loc[te, "fault"].values

    # StandardScaler on features (separate from MinMax on raw sensors)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)
    save_artifact(scaler, "feat_scaler",
                  {"paths": {"saved_models": "models/saved_models/"}})

    return {
        "X_train"      : X_tr_sc,
        "X_test"       : X_te_sc,
        "y_rul_train"  : y_rul_tr,
        "y_rul_test"   : y_rul_te,
        "y_fault_train": y_fault_tr,
        "y_fault_test" : y_fault_te,
        "feat_scaler"  : scaler,
        "tr_units"     : tr_units,
        "te_units"     : te_units,
    }


# ─────────────────────────────────────────────────────────────
# RUL REGRESSOR
# ─────────────────────────────────────────────────────────────

@timer
def train_rul(X_tr, y_tr, X_te, y_te, config: dict, log) -> xgb.XGBRegressor:
    """
    XGBoost Gradient Boosted Trees for RUL regression.

    Key settings
    ────────────
    early_stopping_rounds : halts if val RMSE stops improving
    subsample / colsample : random subsampling prevents overfitting
    """
    p = config["models"]["xgb_regressor"]
    log.info("  Training XGBoost RUL Regressor …")

    model = xgb.XGBRegressor(
        n_estimators          = p["n_estimators"],
        max_depth             = p["max_depth"],
        learning_rate         = p["learning_rate"],
        subsample             = p["subsample"],
        colsample_bytree      = p["colsample_bytree"],
        objective             = p["objective"],
        early_stopping_rounds = p["early_stopping_rounds"],
        eval_metric           = p["eval_metric"],
        random_state          = p["random_state"],
        n_jobs                = -1,
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              verbose=False)

    log.info(f"  Best iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────────────────────
# FAULT CLASSIFIER
# ─────────────────────────────────────────────────────────────

@timer
def train_fault(X_tr, y_tr, X_te, y_te, config: dict,
                log) -> xgb.XGBClassifier:
    """
    XGBoost Classifier for binary fault detection.

    scale_pos_weight
    ────────────────
    Dataset is imbalanced (far more healthy than fault samples).
    scale_pos_weight = n_negative / n_positive tells XGBoost
    to weight fault samples more heavily during training.
    """
    p     = config["models"]["xgb_classifier"]
    n_neg = (y_tr == 0).sum()
    n_pos = (y_tr == 1).sum()
    spw   = n_neg / max(n_pos, 1)

    log.info(f"  Training XGBoost Fault Classifier … "
             f"(healthy:{n_neg:,}  fault:{n_pos:,}  spw:{spw:.1f})")

    model = xgb.XGBClassifier(
        n_estimators      = p["n_estimators"],
        max_depth         = p["max_depth"],
        learning_rate     = p["learning_rate"],
        scale_pos_weight  = spw,
        eval_metric       = p["eval_metric"],
        random_state      = p["random_state"],
        use_label_encoder = False,
        n_jobs            = -1,
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              verbose=False)
    return model


# ─────────────────────────────────────────────────────────────
# ANOMALY DETECTORS
# ─────────────────────────────────────────────────────────────

@timer
def train_anomaly(X_all: np.ndarray, config: dict, log) -> dict:
    """
    Fit two unsupervised anomaly detectors.

    Isolation Forest  — isolates anomalies by random splits;
                        anomalies are isolated in fewer splits.
    Local Outlier Factor — compares each point to its neighbours;
                           low-density points are outliers.
    novelty=True allows LOF to predict on new (unseen) data.
    """
    cont = config["anomaly"]["contamination"]
    log.info(f"  Training Isolation Forest (contamination={cont}) …")
    iso = IsolationForest(
        n_estimators  = config["anomaly"]["iso_n_estimators"],
        contamination = cont,
        random_state  = 42,
        n_jobs        = -1,
    )
    iso.fit(X_all)

    log.info(f"  Training LOF (n_neighbors={config['anomaly']['lof_n_neighbors']}) …")
    lof = LocalOutlierFactor(
        n_neighbors   = config["anomaly"]["lof_n_neighbors"],
        contamination = cont,
        novelty       = True,
        n_jobs        = -1,
    )
    lof.fit(X_all)

    return {"iso_forest": iso, "lof": lof}


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

@timer
def train_all(df_train: pd.DataFrame, feature_cols: list,
              config: dict = None) -> dict:
    """
    Run the complete training pipeline and save all artifacts.

    Returns
    -------
    dict: rul_model, fault_model, anomaly_models, split
    """
    config = config or load_config()
    log    = get_logger(__name__, config)
    banner("STAGE 4 · TRAIN MODELS")

    pc = config["preprocessing"]

    # ── Split ────────────────────────────────────────────
    log.info("  Splitting by engine (no leakage) …")
    split = engine_split(df_train, feature_cols,
                         pc["test_size"], pc["random_state"])
    log.info(f"  Train: {len(split['X_train']):,} samples  "
             f"Test: {len(split['X_test']):,} samples")
    save_artifact(feature_cols, "feature_cols", config)

    # ── RUL Regressor ────────────────────────────────────
    rul_model = train_rul(
        split["X_train"], split["y_rul_train"],
        split["X_test"],  split["y_rul_test"],
        config, log
    )
    save_artifact(rul_model, "xgb_rul_model", config)
    log.info("  XGBoost RUL saved ✓")

    # ── Fault Classifier ─────────────────────────────────
    fault_model = train_fault(
        split["X_train"], split["y_fault_train"],
        split["X_test"],  split["y_fault_test"],
        config, log
    )
    save_artifact(fault_model, "xgb_fault_clf", config)
    log.info("  XGBoost Fault Classifier saved ✓")

    # ── Anomaly Models ───────────────────────────────────
    anomaly = train_anomaly(split["X_train"], config, log)
    save_artifact(anomaly["iso_forest"], "iso_forest", config)
    save_artifact(anomaly["lof"],        "lof_model",  config)
    log.info("  Anomaly detectors saved ✓")

    banner("ALL MODELS TRAINED & SAVED ✓")
    return {
        "rul_model"   : rul_model,
        "fault_model" : fault_model,
        "anomaly"     : anomaly,
        "split"       : split,
    }


if __name__ == "__main__":
    from src.data.make_dataset       import make_dataset
    from src.data.preprocess         import preprocess
    from src.features.build_features import build_features

    cfg   = load_config()
    raw   = make_dataset(cfg)
    proc  = preprocess(raw["train"], raw["test"], cfg)
    feats = build_features(proc["train"], proc["test"],
                           proc["active_sensors"], cfg)
    train_all(feats["train"], feats["feature_cols"], cfg)
