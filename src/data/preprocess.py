"""
src/data/preprocess.py
----------------------
Clean, smooth, scale sensor data.
Identifies active sensors, removes noise, clips outliers,
applies MinMax scaling, and saves fitted scalers.

Milestone M1 -- Data Pre-Processing and Cleaning
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.helpers import load_config, get_logger, save_artifact, timer, banner

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OP_COLS     = ["op_setting_1", "op_setting_2", "op_setting_3"]


# -------------------------------------------------------------
# SENSOR SELECTION
# -------------------------------------------------------------

def identify_sensors(df: pd.DataFrame, std_thresh: float = 0.01):
    """
    Split sensors into active (informative) and constant (useless).
    A sensor is constant if its std across ALL training data < threshold.
    """
    std     = df[SENSOR_COLS].std()
    const   = std[std < std_thresh].index.tolist()
    active  = [s for s in SENSOR_COLS if s not in const]
    return active, const


# -------------------------------------------------------------
# CLEANING
# -------------------------------------------------------------

def drop_useless_cols(df: pd.DataFrame, const_sensors: list) -> pd.DataFrame:
    """Remove constant sensors and operating condition columns."""
    drop = [c for c in const_sensors + OP_COLS if c in df.columns]
    return df.drop(columns=drop)


def impute_missing(df: pd.DataFrame, sensors: list, log) -> pd.DataFrame:
    """Fill missing values with per-column median."""
    n = df[sensors].isnull().sum().sum()
    if n == 0:
        log.info("  No missing values [OK]")
        return df
    log.warning(f"  {n} missing values found -- imputing with median")
    df = df.copy()
    for s in sensors:
        if df[s].isnull().any():
            df[s] = df[s].fillna(df[s].median())
    return df


def clip_outliers(df: pd.DataFrame, sensors: list,
                  factor: float = 3.0) -> pd.DataFrame:
    """
    Clip values beyond Q1 − factorxIQR and Q3 + factorxIQR.
    factor=3 is conservative -- only extreme outliers are clipped.
    """
    df = df.copy()
    for s in sensors:
        q1, q3 = df[s].quantile(0.25), df[s].quantile(0.75)
        iqr    = q3 - q1
        df[s]  = df[s].clip(lower=q1 - factor*iqr,
                             upper=q3 + factor*iqr)
    return df


# -------------------------------------------------------------
# SMOOTHING
# -------------------------------------------------------------

def rolling_smooth(df: pd.DataFrame, sensors: list,
                   window: int = 5) -> pd.DataFrame:
    """
    Per-engine rolling mean over `window` cycles.
    Removes cycle-to-cycle random noise while preserving trend.
    min_periods=1 ensures no NaNs at the start of each engine's life.
    """
    df = df.copy()
    for uid in df["unit_id"].unique():
        m = df["unit_id"] == uid
        for s in sensors:
            df.loc[m, s] = (df.loc[m, s]
                             .rolling(window, min_periods=1)
                             .mean()
                             .astype(df[s].dtype))
    return df


# -------------------------------------------------------------
# SCALING
# -------------------------------------------------------------

def fit_and_scale(df_train: pd.DataFrame, df_test: pd.DataFrame,
                  sensors: list, config: dict):
    """
    Fit MinMaxScaler on training data -> transform both sets.
    Scaler is saved for use at inference time.
    Returns (df_train_scaled, df_test_scaled, scaler).
    """
    scaler = MinMaxScaler()
    df_tr  = df_train.copy()
    df_te  = df_test.copy()

    df_tr[sensors] = scaler.fit_transform(df_train[sensors])
    df_te[sensors] = scaler.transform(df_test[sensors])

    save_artifact(scaler,  "minmax_scaler", config)
    return df_tr, df_te, scaler


# -------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------

@timer
def preprocess(df_train: pd.DataFrame, df_test: pd.DataFrame,
               config: dict = None) -> dict:
    """
    Full preprocessing pipeline.

    Steps
    -----
    1. Identify active vs constant sensors
    2. Drop constant sensors + operating settings
    3. Impute missing values
    4. Clip outliers (IQR method)
    5. Rolling mean smoothing per engine
    6. Compute life_pct (position in engine lifecycle)
    7. Fit MinMaxScaler on train -> transform both
    8. Save scaler + active sensor list
    9. Write processed CSVs

    Returns dict: {train, test, active_sensors, scaler}
    """
    config = config or load_config()
    log    = get_logger(__name__, config)
    banner("STAGE 2 · PREPROCESS")

    pc   = config["preprocessing"]
    did  = config["data"]["dataset_id"]
    pdir = Path(config["paths"]["processed_data"])
    pdir.mkdir(parents=True, exist_ok=True)

    # 1 - Sensor selection
    active, const = identify_sensors(df_train, pc["std_threshold"])
    log.info(f"  Active sensors  ({len(active)}): {active}")
    log.info(f"  Constant sensors ({len(const)}): {const}")

    # 2 - Drop useless columns
    df_train = drop_useless_cols(df_train, const)
    df_test  = drop_useless_cols(df_test,  const)

    # 3 - Missing values
    df_train = impute_missing(df_train, active, log)
    df_test  = impute_missing(df_test,  active, log)

    # 4 - Outlier clipping
    df_train = clip_outliers(df_train, active, pc["outlier_iqr_factor"])
    df_test  = clip_outliers(df_test,  active, pc["outlier_iqr_factor"])
    log.info(f"  Outliers clipped (IQRx{pc['outlier_iqr_factor']}) [OK]")

    # 5 - Smoothing
    w        = pc["rolling_window"]
    df_train = rolling_smooth(df_train, active, w)
    df_test  = rolling_smooth(df_test,  active, w)
    log.info(f"  Rolling smooth (window={w}) [OK]")

    # 6 - Life fraction
    for df in [df_train, df_test]:
        mx = df.groupby("unit_id")["cycle"].transform("max")
        df["life_pct"] = df["cycle"] / mx

    # 7 - Scale
    df_train, df_test, scaler = fit_and_scale(
        df_train, df_test, active, config
    )
    log.info("  MinMax scaling [OK]")

    # 8 - Save sensor list
    save_artifact(active, "active_sensors", config)

    # 9 - Write CSVs
    df_train.to_csv(pdir / f"train_{did}_processed.csv", index=False)
    df_test .to_csv(pdir / f"test_{did}_processed.csv",  index=False)
    log.info("Preprocessing complete [OK]")

    return {
        "train"         : df_train,
        "test"          : df_test,
        "active_sensors": active,
        "scaler"        : scaler,
    }


if __name__ == "__main__":
    from src.data.make_dataset import make_dataset
    cfg  = load_config()
    data = make_dataset(cfg)
    preprocess(data["train"], data["test"], cfg)
