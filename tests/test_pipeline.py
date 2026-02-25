"""
tests/test_pipeline.py
──────────────────────
Unit tests for every pipeline module.
Run with:   pytest tests/ -v
"""

import pytest
import numpy  as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

def _dummy(n_engines: int = 5, cycles: int = 40) -> pd.DataFrame:
    """Create a minimal fake CMAPSS DataFrame for fast unit testing."""
    rng  = np.random.default_rng(42)
    rows = []
    for uid in range(1, n_engines + 1):
        for c in range(1, cycles + 1):
            row = {"unit_id": uid, "cycle": c}
            for s in [f"s{i}" for i in range(1, 22)]:
                row[s] = rng.normal(10, 1)
            rows.append(row)
    df = pd.DataFrame(rows)
    mx = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"]      = (mx - df["cycle"]).clip(upper=125)
    df["life_pct"] = df["cycle"] / mx
    df["fault"]    = (df["RUL"] < 10).astype(int)
    return df


@pytest.fixture
def dummy():
    return _dummy()


@pytest.fixture
def cfg():
    from src.utils.helpers import load_config
    return load_config("config/config.yaml")


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

class TestHelpers:
    def test_load_config_keys(self, cfg):
        for k in ["data", "models", "paths", "preprocessing"]:
            assert k in cfg

    def test_get_logger_returns_logger(self, cfg):
        import logging
        from src.utils.helpers import get_logger
        log = get_logger("test", cfg)
        assert isinstance(log, logging.Logger)

    def test_timer_preserves_return(self):
        from src.utils.helpers import timer
        @timer
        def add(a, b):
            return a + b
        assert add(2, 3) == 5

    def test_ensure_dirs_creates(self, cfg, tmp_path):
        from src.utils.helpers import ensure_dirs
        cfg2 = {"paths": {"test_dir": str(tmp_path / "new_dir")}}
        ensure_dirs(cfg2)
        assert (tmp_path / "new_dir").exists()


# ─────────────────────────────────────────────────────────────
# MAKE_DATASET
# ─────────────────────────────────────────────────────────────

class TestMakeDataset:
    def test_compute_train_rul_range(self, dummy):
        from src.data.make_dataset import compute_train_rul
        df = dummy.drop(columns=["RUL", "life_pct", "fault"])
        out = compute_train_rul(df, clip=125)
        assert "RUL" in out.columns
        assert out["RUL"].min() == 0
        assert out["RUL"].max() <= 125

    def test_rul_clip_works(self, dummy):
        from src.data.make_dataset import compute_train_rul
        df  = dummy.drop(columns=["RUL", "life_pct", "fault"])
        out = compute_train_rul(df, clip=50)
        assert out["RUL"].max() <= 50

    def test_columns_present(self, dummy):
        assert "unit_id" in dummy.columns
        assert "cycle"   in dummy.columns
        assert "s1"      in dummy.columns


# ─────────────────────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_identify_sensors_drops_constant(self, dummy):
        from src.data.preprocess import identify_sensors
        dummy2 = dummy.copy()
        dummy2["s1"] = 5.0          # constant sensor
        active, const = identify_sensors(dummy2, 0.01)
        assert "s1" in const
        assert "s2" in active

    def test_rolling_smooth_no_nans(self, dummy):
        from src.data.preprocess import rolling_smooth
        sensors = [f"s{i}" for i in range(1, 8)]
        out = rolling_smooth(dummy, sensors, window=5)
        assert out[sensors].isnull().sum().sum() == 0

    def test_rolling_smooth_shape(self, dummy):
        from src.data.preprocess import rolling_smooth
        sensors = [f"s{i}" for i in range(1, 8)]
        out = rolling_smooth(dummy, sensors, window=5)
        assert out.shape == dummy.shape

    def test_clip_outliers_removes_extreme(self, dummy):
        from src.data.preprocess import clip_outliers
        df      = dummy.copy()
        sensors = [f"s{i}" for i in range(1, 22)]
        df.loc[0, "s3"] = 1e9
        out = clip_outliers(df, sensors, factor=3.0)
        assert out["s3"].max() < 1e9

    def test_impute_fills_nans(self, dummy):
        import logging
        from src.data.preprocess import impute_missing
        df = dummy.copy()
        df.loc[0, "s2"] = np.nan
        sensors = [f"s{i}" for i in range(1, 22)]
        log = logging.getLogger("test")
        out = impute_missing(df, sensors, log)
        assert out["s2"].isnull().sum() == 0


# ─────────────────────────────────────────────────────────────
# BUILD_FEATURES
# ─────────────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_rolling_features_added(self, dummy):
        from src.features.build_features import add_rolling_features
        sensors = ["s2", "s3", "s7"]
        out = add_rolling_features(dummy, sensors, window=5)
        assert "s2_rmean" in out.columns
        assert "s3_rstd"  in out.columns

    def test_rolling_no_nans(self, dummy):
        from src.features.build_features import add_rolling_features
        sensors = ["s2", "s3"]
        out = add_rolling_features(dummy, sensors, window=5)
        assert out["s2_rmean"].isnull().sum() == 0

    def test_health_index_created(self, dummy):
        from src.features.build_features import compute_health_index
        sensors = ["s2", "s3", "s7"]
        out = compute_health_index(dummy, sensors)
        assert "health_index" in out.columns
        assert not out["health_index"].isnull().all()

    def test_health_index_empty_sensors(self, dummy):
        from src.features.build_features import compute_health_index
        out = compute_health_index(dummy, [])
        assert "health_index" in out.columns
        assert (out["health_index"] == 0.0).all()

    def test_find_best_k_in_range(self, dummy):
        from src.features.build_features import find_best_k
        sensors = ["s2", "s3", "s7", "s11"]
        X = dummy[sensors].values
        k = find_best_k(X, k_min=2, k_max=5, seed=42)
        assert 2 <= k <= 5

    def test_fault_label_binary(self, dummy):
        from src.features.build_features import add_fault_label
        out = add_fault_label(dummy, threshold=10)
        assert set(out["fault"].unique()).issubset({0, 1})

    def test_fault_label_threshold(self, dummy):
        from src.features.build_features import add_fault_label
        out = add_fault_label(dummy, threshold=10)
        assert (out.loc[out["RUL"] < 10, "fault"] == 1).all()
        assert (out.loc[out["RUL"] >= 10, "fault"] == 0).all()


# ─────────────────────────────────────────────────────────────
# MODELS — PREDICT
# ─────────────────────────────────────────────────────────────

class TestSensorValidator:
    @pytest.fixture
    def validator(self, dummy, cfg):
        from src.models.predict import SensorValidator
        sensors   = [f"s{i}" for i in range(1, 8)]
        df_healthy = dummy[dummy["life_pct"] <= 0.3]
        return SensorValidator(df_healthy, sensors, cfg), sensors

    def test_healthy_reading_ok(self, validator, dummy):
        v, sensors = validator
        reading = {s: float(dummy[s].mean()) for s in sensors}
        report  = v.validate(reading)
        assert report["OVERALL"] == "HEALTHY"

    def test_extreme_value_fault(self, validator, dummy):
        v, sensors = validator
        reading = {s: 1e9 for s in sensors}
        report  = v.validate(reading)
        assert report["OVERALL"] == "FAULT DETECTED"

    def test_per_sensor_status_keys(self, validator, dummy):
        v, sensors = validator
        reading = {s: float(dummy[s].mean()) for s in sensors}
        report  = v.validate(reading)
        for s in sensors:
            assert s in report
            assert "status"  in report[s]
            assert "z_score" in report[s]
            assert "value"   in report[s]

    def test_fault_sensor_identified(self, validator, dummy):
        v, sensors = validator
        reading = {s: float(dummy[s].mean()) for s in sensors}
        reading[sensors[0]] = 1e9   # inject one bad sensor
        report = v.validate(reading)
        assert report[sensors[0]]["status"] == "FAULT"


class TestSensorFaultScores:
    def test_returns_series(self, dummy):
        from src.models.predict import sensor_fault_scores
        sensors = ["s2", "s3", "s7"]
        normal  = dummy[dummy["RUL"] >= 10]
        anom    = dummy[dummy["RUL"] <  10]
        scores  = sensor_fault_scores(normal, anom, sensors)
        assert hasattr(scores, "values")
        assert len(scores) == 3

    def test_sorted_descending(self, dummy):
        from src.models.predict import sensor_fault_scores
        sensors = ["s2", "s3", "s7", "s11"]
        normal  = dummy[dummy["RUL"] >= 10]
        anom    = dummy[dummy["RUL"] <  10]
        scores  = sensor_fault_scores(normal, anom, sensors)
        vals    = list(scores.values)
        assert vals == sorted(vals, reverse=True)


# ─────────────────────────────────────────────────────────────
# MODELS — EVALUATE
# ─────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_rul_metrics_present(self, cfg):
        import logging
        from src.models.evaluate import eval_rul
        log    = logging.getLogger("test")
        y_true = np.array([100, 80, 60, 40, 20, 10])
        y_pred = np.array([ 95, 82, 58, 38, 22, 12])
        result = eval_rul(y_true, y_pred, cfg, log)
        for k in ["rmse", "mae", "r2", "nasa_score", "grade"]:
            assert k in result

    def test_rul_rmse_positive(self, cfg):
        import logging
        from src.models.evaluate import eval_rul
        log    = logging.getLogger("test")
        y_true = np.array([100, 80, 60])
        y_pred = np.array([ 90, 75, 65])
        result = eval_rul(y_true, y_pred, cfg, log)
        assert result["rmse"] > 0

    def test_fault_metrics_present(self, cfg):
        import logging
        from src.models.evaluate import eval_fault
        log    = logging.getLogger("test")
        y_true  = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred  = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        y_proba = np.array([0.1, 0.6, 0.9, 0.8, 0.2, 0.4, 0.1, 0.7])
        result  = eval_fault(y_true, y_pred, y_proba, cfg, log)
        for k in ["accuracy", "precision", "recall", "f1", "auc", "grade"]:
            assert k in result

    def test_perfect_classifier(self, cfg):
        import logging
        from src.models.evaluate import eval_fault
        log     = logging.getLogger("test")
        y       = np.array([0, 0, 0, 1, 1, 1])
        result  = eval_fault(y, y, y.astype(float), cfg, log)
        assert result["f1"]  == pytest.approx(1.0)
        assert result["auc"] == pytest.approx(1.0)

    def test_anomaly_summary_keys(self):
        import logging
        from src.models.evaluate import eval_anomaly
        log = logging.getLogger("test")
        df  = pd.DataFrame({
            "anomaly_iso": [0, 1, 0, 1, 0],
            "anomaly_lof": [0, 1, 0, 0, 0],
            "anomaly_ens": [0, 1, 0, 0, 0],
        })
        result = eval_anomaly(df, log)
        for k in ["total", "iso", "lof", "ensemble", "pct_ens"]:
            assert k in result
        assert result["total"]    == 5
        assert result["ensemble"] == 1
