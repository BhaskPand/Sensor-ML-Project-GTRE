"""
src/data/make_dataset.py
────────────────────────
Load raw NASA CMAPSS .txt files → compute RUL labels
→ save train/test/rul CSVs to data/processed/.

Milestone M1 — Data Analysis Stage
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.helpers import load_config, get_logger, timer, banner

COLUMNS = [
    "unit_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    *[f"s{i}" for i in range(1, 22)]
]

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]


# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────

def load_cmapss(path: str) -> pd.DataFrame:
    """Load a whitespace-separated CMAPSS file (no header)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return df.dropna(axis=1, how="all").reset_index(drop=True)


def load_rul(path: str) -> pd.DataFrame:
    """Load RUL ground-truth file. One value per test engine."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["true_rul"])
    df["unit_id"] = df.index + 1
    return df[["unit_id", "true_rul"]]


# ─────────────────────────────────────────────────────────────
# RUL COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_train_rul(df: pd.DataFrame, clip: int = 125) -> pd.DataFrame:
    """
    Add RUL column to training data.
    RUL = max_cycle_for_engine − current_cycle, clipped at `clip`.
    Clipping at 125: engines look healthy early in life, so we
    don't penalise the model for over-predicting high RUL values.
    """
    max_c = df.groupby("unit_id")["cycle"].transform("max")
    df    = df.copy()
    df["RUL"] = (max_c - df["cycle"]).clip(upper=clip)
    return df


def attach_test_rul(df_test: pd.DataFrame, df_rul: pd.DataFrame,
                    clip: int = 125) -> pd.DataFrame:
    """
    Attach RUL label to the LAST cycle of each test engine.
    NASA provides the true remaining life after the last observation.
    """
    last = (df_test.groupby("unit_id")["cycle"]
            .max().reset_index()
            .merge(df_rul, on="unit_id"))
    last["RUL"] = last["true_rul"].clip(upper=clip)
    df_test = df_test.merge(
        last[["unit_id", "cycle", "RUL"]],
        on=["unit_id", "cycle"], how="left"
    )
    return df_test


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

@timer
def make_dataset(config: dict = None) -> dict:
    """
    Full data loading pipeline.
    Returns dict: {train, test, rul}
    """
    config = config or load_config()
    log    = get_logger(__name__, config)
    banner("STAGE 1 · LOAD RAW DATA")

    raw_dir  = Path(config["paths"]["raw_data"])
    proc_dir = Path(config["paths"]["processed_data"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    did   = config["data"]["dataset_id"]
    clip  = config["preprocessing"]["rul_clip"]

    train_path = raw_dir / config["data"]["train_file"]
    test_path  = raw_dir / config["data"]["test_file"]
    rul_path   = raw_dir / config["data"]["rul_file"]

    # ── Validate files exist ────────────────────────────────
    for p in [train_path, test_path, rul_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"\n  ✗ File not found: {p}"
                f"\n  → Place NASA CMAPSS files in:  {raw_dir.resolve()}"
            )

    log.info(f"Loading {did} dataset …")
    df_train = load_cmapss(train_path)
    df_test  = load_cmapss(test_path)
    df_rul   = load_rul(rul_path)

    log.info("Computing RUL labels …")
    df_train = compute_train_rul(df_train, clip=clip)
    df_test  = attach_test_rul(df_test, df_rul, clip=clip)

    # ── Summary ─────────────────────────────────────────────
    log.info(f"  Train  →  {len(df_train):,} rows  |  {df_train['unit_id'].nunique()} engines")
    log.info(f"  Test   →  {len(df_test):,}  rows  |  {df_test['unit_id'].nunique()}  engines")
    log.info(f"  RUL range (train): 0 – {df_train['RUL'].max()}")
    log.info(f"  Missing values: {df_train.isnull().sum().sum()}")

    # ── Save ────────────────────────────────────────────────
    df_train.to_csv(proc_dir / f"train_{did}_raw.csv", index=False)
    df_test .to_csv(proc_dir / f"test_{did}_raw.csv",  index=False)
    df_rul  .to_csv(proc_dir / f"rul_{did}.csv",       index=False)
    log.info("Raw CSVs saved to data/processed/ ✓")

    return {"train": df_train, "test": df_test, "rul": df_rul}


if __name__ == "__main__":
    make_dataset()
