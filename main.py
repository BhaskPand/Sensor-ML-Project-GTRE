"""
main.py
-------
Single entry point for the full pipeline.

Commands
--------
  python main.py train      ->  load -> preprocess -> features -> train
  python main.py evaluate   ->  evaluate saved models on hold-out set
  python main.py visualize  ->  generate all plots
  python main.py all        ->  everything end-to-end (default)

Examples
--------
  python main.py all
  python main.py train
  python main.py evaluate
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.helpers           import (load_config, get_logger,
                                          ensure_dirs, banner, load_artifact,
                                          save_artifact)
from src.data.make_dataset       import make_dataset
from src.data.preprocess         import preprocess
from src.features.build_features import build_features
from src.models.train            import train_all, engine_split
from src.models.predict          import (load_all_models, predict_batch,
                                          SensorValidator, sensor_fault_scores)
from src.models.evaluate         import evaluate_all
from src.visualization.visualize import generate_all_plots


# -----------------------------------------------------------------
# PIPELINE STAGES
# -----------------------------------------------------------------

def stage_data(config):
    raw  = make_dataset(config)
    proc = preprocess(raw["train"], raw["test"], config)
    return raw, proc


def stage_features(proc, config):
    return build_features(proc["train"], proc["test"],
                          proc["active_sensors"], config)


def stage_train(feat, config):
    """
    Train models on feat['train'].
    Returns trained dict which includes split['df_test'] --
    the held-out engine rows WITH true RUL and fault labels.
    This is what we evaluate on.
    """
    return train_all(feat["train"], feat["feature_cols"], config)


def stage_predict(trained, models, config, log):
    """
    Run predictions on the held-out engine rows from training split.

    KEY FIX: We evaluate on split['df_test'] (hold-out training engines)
    NOT on the NASA test file. The NASA test file engines don't run to
    failure so they have no true RUL labels to evaluate against.

    split['df_test'] already has:
      - All feature columns (produced by build_features)
      - True RUL labels
      - True fault labels
    """
    split  = trained["split"]
    df_eval = split["df_test"].copy()   # held-out rows WITH labels

    feature_cols = load_artifact("feature_cols", config)

    # Apply the saved feat_scaler (same one used at training time)
    feat_scaler  = load_artifact("feat_scaler", config)
    X_eval = df_eval[feature_cols].fillna(0).values
    X_sc   = feat_scaler.transform(X_eval)

    # Run all models
    df_eval["pred_rul"]      = np.clip(
        models["rul_model"].predict(X_sc), 0, None)
    df_eval["fault_prob"]    = models["fault_model"].predict_proba(X_sc)[:, 1]
    df_eval["is_fault_pred"] = (df_eval["fault_prob"] >= 0.5).astype(int)

    min_votes = config["anomaly"]["ensemble_min_votes"]
    iso_flags = (models["iso_forest"].predict(X_sc) == -1).astype(int)
    lof_flags = (models["lof"].predict(X_sc)        == -1).astype(int)
    df_eval["anomaly_iso"] = iso_flags
    df_eval["anomaly_lof"] = lof_flags
    df_eval["anomaly_ens"] = ((iso_flags + lof_flags) >= min_votes).astype(int)

    log.info(f"  Evaluated on {len(df_eval):,} hold-out rows  "
             f"({split['te_units'].shape[0]} unseen engines)")
    log.info(f"  Ensemble anomalies: {df_eval['anomaly_ens'].sum():,}")

    return df_eval


def stage_visualize(feat, df_pred, trained, config):
    sensors = load_artifact("active_sensors", config)
    pca     = load_artifact("pca_model",      config)
    generate_all_plots(
        df_train = feat["train"],
        df_pred  = df_pred,
        sensors  = sensors,
        pca      = pca,
        split    = trained["split"],
        config   = config,
    )


# -----------------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------------

def print_summary(metrics: dict) -> None:
    if not metrics:
        return
    reg = metrics.get("regression", {})
    clf = metrics.get("classification", {})
    banner("FINAL RESULTS SUMMARY")
    print(f"  RUL Regression")
    print(f"    RMSE       : {reg.get('rmse', 0):.3f} cycles")
    print(f"    R2         : {reg.get('r2', 0):.4f}")
    print(f"    NASA Score : {reg.get('nasa_score', 0):.1f}")
    print(f"    Grade      : {reg.get('grade', 'N/A')}")
    print()
    print(f"  Fault Classification")
    print(f"    Accuracy   : {clf.get('accuracy', 0)*100:.2f}%")
    print(f"    Recall     : {clf.get('recall', 0)*100:.2f}%  (most critical)")
    print(f"    F1 Score   : {clf.get('f1', 0)*100:.2f}%")
    print(f"    AUC-ROC    : {clf.get('auc', 0):.4f}")
    print(f"    Grade      : {clf.get('grade', 'N/A')}")
    print()
    print("  Plots   -> reports/figures/")
    print("  Models  -> models/saved_models/")
    print()


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="NASA CMAPSS Sensor Fault Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument(
        "command",
        nargs   = "?",
        default = "all",
        choices = ["all", "train", "evaluate", "visualize"],
        help    = "Pipeline stage to run (default: all)"
    )
    p.add_argument(
        "--config",
        default = "config/config.yaml",
        help    = "Path to config file"
    )
    return p.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)
    log    = get_logger("main", config)
    ensure_dirs(config)

    log.info(f"Command : {args.command.upper()}")
    log.info(f"Dataset : {config['data']['dataset_id']}")

    cmd     = args.command
    feat    = None
    trained = None
    df_pred = None
    metrics = None

    # -- TRAIN ------------------------------------------------
    if cmd in ("all", "train"):
        raw, proc = stage_data(config)
        feat      = stage_features(proc, config)
        trained   = stage_train(feat, config)

    # -- EVALUATE ---------------------------------------------
    if cmd in ("all", "evaluate"):

        # If we only ran evaluate (not train), reload from disk
        if feat is None or trained is None:
            log.info("Loading saved artifacts from disk ...")
            did      = config["data"]["dataset_id"]
            pdir     = Path(config["paths"]["processed_data"])

            df_train_f = pd.read_csv(pdir / f"train_{did}_features.csv")
            feature_cols = load_artifact("feature_cols", config)
            feat = {
                "train"       : df_train_f,
                "feature_cols": feature_cols,
            }

            # Reconstruct the same split (same random_state = same engines)
            split = engine_split(
                df_train_f, feature_cols,
                config["preprocessing"]["test_size"],
                config["preprocessing"]["random_state"]
            )
            # Override with saved scaler (don't refit)
            split["feat_scaler"] = load_artifact("feat_scaler", config)
            X_te = df_train_f[
                df_train_f["unit_id"].isin(split["te_units"])
            ][feature_cols].fillna(0).values
            split["X_test"] = split["feat_scaler"].transform(X_te)
            trained = {"split": split}

        models  = load_all_models(config)
        df_pred = stage_predict(trained, models, config, log)
        metrics = evaluate_all(df_pred, trained["split"], config)

        # Save predictions for later visualisation
        out = Path(config["paths"]["processed_data"]) / "predictions.csv"
        df_pred.to_csv(out, index=False)
        log.info(f"Predictions saved -> {out}")

        if cmd == "all":
            print_summary(metrics)

    # -- VISUALIZE --------------------------------------------
    if cmd in ("all", "visualize"):

        if feat is None:
            did        = config["data"]["dataset_id"]
            pdir       = Path(config["paths"]["processed_data"])
            df_train_f = pd.read_csv(pdir / f"train_{did}_features.csv")
            feat       = {"train": df_train_f,
                          "feature_cols": load_artifact("feature_cols", config)}

        if df_pred is None:
            pdir    = Path(config["paths"]["processed_data"])
            df_pred = pd.read_csv(pdir / "predictions.csv")

        if trained is None:
            feature_cols = load_artifact("feature_cols", config)
            split = engine_split(
                feat["train"], feature_cols,
                config["preprocessing"]["test_size"],
                config["preprocessing"]["random_state"]
            )
            split["feat_scaler"] = load_artifact("feat_scaler", config)
            trained = {"split": split}

        stage_visualize(feat, df_pred, trained, config)

    banner("PIPELINE COMPLETE [OK]")


if __name__ == "__main__":
    main()
