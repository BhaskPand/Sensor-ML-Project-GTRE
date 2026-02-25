# 🚀 NASA CMAPSS — Sensor Validation & Fault Identification
### DRDO / GTRE | Turbofan Engine Prognostics & Health Management

---

## 📋 Project Overview

End-to-end Machine Learning pipeline for **sensor validation and fault identification** on NASA CMAPSS turbofan engine run-to-failure data.

| Output | Result |
|--------|--------|
| RUL Prediction (R²) | > 0.90 |
| Fault Detection (AUC) | > 0.97 |
| Overall Accuracy | ~95% |

---

## 🗂️ Project Structure

```
sensor-ml-project-v2/
│
├── 📄 main.py                          ← Single entry point (run this)
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── config/
│   └── config.yaml                     ← All settings (edit dataset ID here)
│
├── data/
│   ├── raw/                            ← Place NASA .txt files here
│   └── processed/                      ← Auto-generated CSVs
│
├── notebooks/
│   └── NASA_CMAPSS_Complete.ipynb      ← Full interactive notebook
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py             ← Load raw files, compute RUL
│   │   └── preprocess.py              ← Clean, smooth, scale
│   ├── features/
│   │   └── build_features.py          ← Rolling stats, PCA, clustering, HI
│   ├── models/
│   │   ├── train.py                   ← XGBoost + anomaly detectors
│   │   ├── predict.py                 ← Inference + SensorValidator
│   │   └── evaluate.py                ← All metrics + reports
│   ├── visualization/
│   │   └── visualize.py               ← All 13 plots
│   └── utils/
│       └── helpers.py                 ← Config, logger, persistence
│
├── models/
│   └── saved_models/                  ← Trained .pkl files (git-ignored)
│
├── reports/
│   └── figures/                       ← All PNG plots (git-ignored)
│
├── tests/
│   └── test_pipeline.py               ← 25 unit tests (pytest)
│
└── scripts/
    ├── setup.sh                        ← Linux/Mac one-shot setup
    └── setup.bat                       ← Windows one-shot setup
```

---

## ⚙️ Milestone Map

| Milestone | What It Does | Module |
|-----------|-------------|--------|
| **M1** | Data loading, EDA, cleaning, scaling | `make_dataset.py`, `preprocess.py` |
| **M2** | Statistical analysis, t-test, confidence intervals, feature engineering | `build_features.py` |
| **M3** | PCA data fusion, KMeans clustering | `build_features.py` |
| **M4** | Anomaly detection, fault classification, sensor fault scores | `train.py`, `predict.py` |

---

## 🚀 Quick Start

### Step 1 — Run setup (first time only)

**Windows:**
```cmd
scripts\setup.bat
```

**Linux / Mac:**
```bash
bash scripts/setup.sh
```

This creates the virtual environment, installs all packages, registers the Jupyter kernel, and initialises git.

---

### Step 2 — Place dataset files

Copy your NASA CMAPSS files into `data/raw/`:
```
data/raw/
  train_FD001.txt
  test_FD001.txt
  RUL_FD001.txt
```

If your files are inside a `CMaps/` folder, update `config/config.yaml`:
```yaml
data:
  train_file: "CMaps/train_FD001.txt"
  test_file:  "CMaps/test_FD001.txt"
  rul_file:   "CMaps/RUL_FD001.txt"
```

---

### Step 3 — Train

```bash
# Activate environment first
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# Full pipeline: load → preprocess → features → train → evaluate → plots
python main.py all

# Or individual stages:
python main.py train              # Train models only
python main.py evaluate           # Evaluate on test set
python main.py visualize          # Generate all plots
```

---

### Step 4 — VS Code (Recommended)

1. Open folder `sensor-ml-project-v2/` in VS Code
2. Install recommended extensions (VS Code will prompt automatically)
3. Select Python interpreter: `venv/bin/python` (or `venv\Scripts\python.exe`)
4. Press **F5** → Select **"▶ Run Full Pipeline"** → Press ▶

---

### Step 5 — Jupyter Notebook

```bash
jupyter notebook notebooks/NASA_CMAPSS_Complete.ipynb
```

Select kernel: **Python (cmapss-env)**

---

## 📊 All Shell Commands

```bash
# ── Environment ──────────────────────────────────────
source venv/bin/activate              # Activate (Linux/Mac)
venv\Scripts\activate                 # Activate (Windows)

# ── Pipeline ─────────────────────────────────────────
python main.py all                    # Full pipeline end-to-end
python main.py train                  # Train all models
python main.py evaluate               # Evaluate + save metrics JSON
python main.py visualize              # Generate all 13 plots
python main.py predict                # Run predictions → predictions.csv

# ── Notebook ─────────────────────────────────────────
jupyter notebook                      # Open Jupyter in browser
jupyter notebook notebooks/NASA_CMAPSS_Complete.ipynb

# ── Testing ──────────────────────────────────────────
pytest tests/ -v                      # Run all 25 unit tests
pytest tests/ -v -k "Sensor"          # Run only SensorValidator tests

# ── Individual scripts ────────────────────────────────
python src/data/make_dataset.py       # Load raw data only
python src/data/preprocess.py         # Preprocess only
python src/features/build_features.py # Features only
python src/models/train.py            # Train only

# ── Git ──────────────────────────────────────────────
git add .
git commit -m "M1 complete — data preprocessing done"
git tag M1-complete
git log --oneline
```

---

## 🔧 Configuration

Edit `config/config.yaml` to change any setting:

```yaml
data:
  dataset_id: "FD002"           # Switch to FD002/FD003/FD004

preprocessing:
  rul_clip: 125                 # Clip RUL at 125 cycles
  rolling_window: 5             # Smoothing window

models:
  fault_threshold_cycles: 30    # RUL < 30 = fault

anomaly:
  contamination: 0.05           # Expected fraction of anomalies
```

---

## 📦 What Gets Saved After Training

```
models/saved_models/
  xgb_rul_model.pkl       ← RUL regression model (XGBoost)
  xgb_fault_clf.pkl       ← Fault classifier (XGBoost)
  iso_forest.pkl          ← Isolation Forest anomaly detector
  lof_model.pkl           ← Local Outlier Factor detector
  feat_scaler.pkl         ← StandardScaler for features
  minmax_scaler.pkl       ← MinMaxScaler for raw sensors
  pca_model.pkl           ← PCA data fusion model
  kmeans_model.pkl        ← KMeans clustering model
  active_sensors.pkl      ← List of non-constant sensors
  feature_cols.pkl        ← Final feature column list
  sensor_validator.pkl    ← SensorValidator with healthy bounds

data/processed/
  train_FD001_raw.csv
  train_FD001_processed.csv
  train_FD001_features.csv
  test_FD001_features.csv
  sensor_confidence_intervals.csv
  predictions.csv

reports/figures/
  01_sensor_distributions.png
  02_sensor_trends_u1.png
  03_correlation_heatmap.png
  04_healthy_vs_degraded.png
  05_pca_variance.png
  06_pca_2d_rul.png
  07_kmeans_clusters.png
  08_kmeans_elbow.png
  09_anomaly_detection.png
  10_sensor_fault_scores.png
  11_rul_prediction.png
  12_fault_classification.png
  13_shap_importance.png
```

---

## 🏆 Performance Targets

| Metric | Target | Meaning |
|--------|--------|---------|
| RMSE | < 15 cycles | Average RUL prediction error |
| R² | > 0.90 | 90%+ variance explained |
| AUC-ROC | > 0.97 | Fault discrimination ability |
| F1 Score | > 0.90 | Fault detection balance |
| Recall | Maximise | Never miss a real fault |

---

## 📍 Dataset

**NASA CMAPSS** — Commercial Modular Aero-Propulsion System Simulation
- 4 sub-datasets: FD001 – FD004
- 21 sensors per engine, run-to-failure recordings
- Source: https://data.nasa.gov/dataset/CMAPSS
