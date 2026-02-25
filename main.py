"""
main.py
───────
Single entry point for the full pipeline.

Commands
────────
  python main.py train      →  Full pipeline: load → preprocess → features → train
  python main.py evaluate   →  Load saved models, evaluate on test set
  python main.py predict    →  Run predictions, save to CSV
  python main.py visualize  →  Generate all plots
  python main.py all        →  Everything end-to-end  (default)

Examples
────────
  python main.py all
  python main.py train
  python main.py evaluate
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.helpers             import load_config, get_logger, ensure_dirs, banner, load_artifact
from src.data.make_dataset         import make_dataset
from src.data.preprocess           import preprocess
from src.features.build_features   import build_features
from src.models.train              import train_all
from src.models.predict            import (load_all_models, predict_batch,
                                            SensorValidator, sensor_fault_scores)
from src.models.evaluate           import evaluate_all
from src.visualization.visualize   import generate_all_plots


# ─────────────────────────────────────────────────────────────
# PIPELINE STAGES
# ─────────────────────────────────────────────────────────────

def stage_data(config):
    raw  = make_dataset(config)
    proc = preprocess(raw["train"], raw["test"], config)
    return raw, proc


def stage_features(proc, config):
    return build_features(proc["train"], proc["test"],
                          proc["active_sensors"], config)


def stage_train(feat, config):
    return train_all(feat["train"], feat["feature_cols"], config)


def stage_evaluate(feat, trained, config, log):
    models  = load_all_models(config)
    df_pred = predict_batch(feat["test"], models, config, log)

    # Build SensorValidator from healthy portion of training data
    sensors     = load_artifact("active_sensors", config)
    df_healthy  = feat["train"][feat["train"]["life_pct"] <= 0.30]
    validator   = SensorValidator(df_healthy, sensors, config)

    # Sensor fault scores
    df_norm = df_pred[df_pred["anomaly_ens"] == 0]
    df_anom = df_pred[df_pred["anomaly_ens"] == 1]
    if len(df_anom) > 0:
        scores = sensor_fault_scores(df_norm, df_anom, sensors)
        log.info("\n  Top faulty sensors:")
        for s, v in scores.head(5).items():
            log.info(f"    {s}: {v:.4f}")

    metrics = evaluate_all(df_pred, trained["split"], config)
    return df_pred, metrics, validator


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


# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

def print_summary(metrics: dict) -> None:
    reg = metrics["regression"]
    clf = metrics["classification"]
    banner("FINAL RESULTS SUMMARY")
    print(f"  RUL Regression")
    print(f"    RMSE       : {reg['rmse']:.3f} cycles")
    print(f"    R²         : {reg['r2']:.4f}")
    print(f"    NASA Score : {reg['nasa_score']:.1f}")
    print(f"    Grade      : {reg['grade']}")
    print()
    print(f"  Fault Classification")
    print(f"    Accuracy   : {clf['accuracy']*100:.2f}%")
    print(f"    Recall     : {clf['recall']*100:.2f}%  ← most critical")
    print(f"    F1 Score   : {clf['f1']*100:.2f}%")
    print(f"    AUC-ROC    : {clf['auc']:.4f}")
    print(f"    Grade      : {clf['grade']}")
    print()
    print("  All plots saved to  reports/figures/")
    print("  All models saved to models/saved_models/")
    print()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="NASA CMAPSS Sensor Fault Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument(
        "command",
        nargs    = "?",
        default  = "all",
        choices  = ["all", "train", "evaluate", "predict", "visualize"],
        help     = "Pipeline stage to run (default: all)"
    )
    p.add_argument(
        "--config",
        default = "config/config.yaml",
        help    = "Path to config file (default: config/config.yaml)"
    )
    return p.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)
    log    = get_logger("main", config)
    ensure_dirs(config)

    log.info(f"Command  : {args.command.upper()}")
    log.info(f"Dataset  : {config['data']['dataset_id']}")

    cmd = args.command

    # ── TRAIN (includes data + features + train) ──────────
    if cmd in ("all", "train"):
        raw, proc = stage_data(config)
        feat      = stage_features(proc, config)
        trained   = stage_train(feat, config)

    # ── EVALUATE ──────────────────────────────────────────
    if cmd in ("all", "evaluate", "predict"):
        if cmd not in ("all", "train"):
            # Load from disk
            did      = config["data"]["dataset_id"]
            pdir     = Path(config["paths"]["processed_data"])
            df_train = pd.read_csv(pdir / f"train_{did}_features.csv")
            df_test  = pd.read_csv(pdir / f"test_{did}_features.csv")
            feat_cols= load_artifact("feature_cols", config)
            feat     = {
                "train"       : df_train,
                "test"        : df_test,
                "feature_cols": feat_cols,
            }
            # Reconstruct a minimal split dict
            from src.models.train import engine_split
            from src.utils.helpers import load_config as _lc
            split   = engine_split(df_train, feat_cols,
                                   config["preprocessing"]["test_size"],
                                   config["preprocessing"]["random_state"])
            trained = {"split": split}

        models  = load_all_models(config)
        df_pred = predict_batch(feat["test"], models, config, log)
        metrics = evaluate_all(df_pred, trained["split"], config)

        # Save predictions
        out = Path(config["paths"]["processed_data"]) / "predictions.csv"
        df_pred.to_csv(out, index=False)
        log.info(f"Predictions saved → {out}")

        if cmd == "all":
            print_summary(metrics)

    # ── VISUALIZE ─────────────────────────────────────────
    if cmd in ("all", "visualize"):
        if "df_pred" not in dir():
            did      = config["data"]["dataset_id"]
            pdir     = Path(config["paths"]["processed_data"])
            df_pred  = pd.read_csv(pdir / "predictions.csv")
            df_train_f = pd.read_csv(pdir / f"train_{did}_features.csv")
            feat_cols= load_artifact("feature_cols", config)
            feat     = {"train": df_train_f, "feature_cols": feat_cols}
            from src.models.train import engine_split
            split    = engine_split(df_train_f, feat_cols,
                                    config["preprocessing"]["test_size"],
                                    config["preprocessing"]["random_state"])
            trained  = {"split": split}

        stage_visualize(feat, df_pred, trained, config)

    banner("PIPELINE COMPLETE ✓")


if __name__ == "__main__":
    main()
