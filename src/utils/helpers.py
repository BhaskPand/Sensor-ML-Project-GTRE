"""
src/utils/helpers.py
--------------------
Shared utilities: config loader, logger, path helpers,
model persistence, timing decorator, and data validation.
"""

import os, sys, time, logging, yaml, json, joblib, io
from pathlib import Path
from functools import wraps
from datetime import datetime

# == Windows Unicode fix ==========================================
# Windows terminals (cp1252) crash on Unicode chars like checkmarks
# and arrows. Force UTF-8 encoding on stdout/stderr so they display
# as ? at worst instead of raising UnicodeEncodeError.
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
    except AttributeError:
        pass  # already wrapped (e.g. inside pytest / Jupyter)


# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------

def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML config file and return as dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with open(p) as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------
# LOGGER
# -----------------------------------------------------------------

def get_logger(name: str, config: dict = None) -> logging.Logger:
    """Return a configured logger writing to console (and file)."""
    level_str = "INFO"
    log_file  = None
    if config:
        level_str = config.get("logging", {}).get("level", "INFO")
        log_dir   = config.get("paths", {}).get("logs")
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_file = Path(log_dir) / "pipeline.log"

    level  = getattr(logging, level_str.upper(), logging.INFO)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s  %(name)s - %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler - UTF-8 safe on Windows
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler - always UTF-8
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# -----------------------------------------------------------------
# PATH HELPERS
# -----------------------------------------------------------------

def ensure_dirs(config: dict) -> None:
    """Create all directories defined in config['paths']."""
    for path in config.get("paths", {}).values():
        Path(path).mkdir(parents=True, exist_ok=True)


def get_path(config: dict, key: str, filename: str = "") -> Path:
    """Return Path(config.paths[key] / filename)."""
    base = Path(config["paths"][key])
    return base / filename if filename else base


# -----------------------------------------------------------------
# MODEL PERSISTENCE
# -----------------------------------------------------------------

def save_artifact(obj, name: str, config: dict, logger=None) -> str:
    """Save any Python object as a .pkl artifact."""
    d = Path(config["paths"]["saved_models"])
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.pkl"
    joblib.dump(obj, p)
    if logger:
        logger.info(f"  Saved: {p}")
    return str(p)


def load_artifact(name: str, config: dict):
    """Load a .pkl artifact by name."""
    p = Path(config["paths"]["saved_models"]) / f"{name}.pkl"
    if not p.exists():
        raise FileNotFoundError(
            f"Artifact not found: {p}\n  Run 'python main.py train' first."
        )
    return joblib.load(p)


def save_metrics(metrics: dict, name: str, config: dict) -> str:
    """Save a metrics dict as a JSON file in reports/."""
    d = Path(config["paths"]["reports"])
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p  = d / f"{name}_{ts}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return str(p)


# -----------------------------------------------------------------
# DECORATORS
# -----------------------------------------------------------------

def timer(func):
    """Decorator: prints execution time of wrapped function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0     = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"  [time] {func.__name__}  {elapsed:.2f}s")
        return result
    return wrapper


# -----------------------------------------------------------------
# CONSOLE HELPERS
# -----------------------------------------------------------------

def banner(text: str, char: str = "=", width: int = 60) -> None:
    line = char * width
    print(f"\n{line}\n  {text}\n{line}")
