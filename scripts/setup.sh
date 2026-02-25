#!/bin/bash
# ============================================================
#  NASA CMAPSS — Complete VS Code Setup Script
#  Run this ONCE after extracting the zip:
#
#    Windows (Git Bash / PowerShell):  bash scripts/setup.sh
#    Linux / Mac:                      bash scripts/setup.sh
#
#  What it does:
#    1. Creates Python virtual environment (venv)
#    2. Installs all dependencies from requirements.txt
#    3. Registers venv as Jupyter kernel
#    4. Initialises git repository
#    5. Prints the exact commands to train / run the project
# ============================================================

set -e   # exit immediately on any error

# ── colours for pretty output ────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
BLU='\033[0;34m'; CYN='\033[0;36m'; NC='\033[0m'

bar()  { echo -e "${BLU}══════════════════════════════════════════════════${NC}"; }
ok()   { echo -e "${GRN}  ✓  $1${NC}"; }
info() { echo -e "${CYN}  →  $1${NC}"; }
warn() { echo -e "${YLW}  ⚠  $1${NC}"; }
err()  { echo -e "${RED}  ✗  $1${NC}"; exit 1; }

# ── check we are in the project root ─────────────────────────
if [ ! -f "main.py" ]; then
    err "Run this script from the project root (where main.py lives)"
fi

bar
echo -e "${YLW}  NASA CMAPSS — VS Code Setup${NC}"
bar

# ─────────────────────────────────────────────────────────────
# 1. Python check
# ─────────────────────────────────────────────────────────────
info "Checking Python version …"
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    err "Python not found. Install Python 3.9+ from https://python.org"
fi

PYTHON=$(command -v python3 || command -v python)
VERSION=$($PYTHON --version 2>&1)
ok "Found: $VERSION  ($PYTHON)"

# ─────────────────────────────────────────────────────────────
# 2. Create virtual environment
# ─────────────────────────────────────────────────────────────
bar
info "Creating virtual environment → venv/ …"
if [ -d "venv" ]; then
    warn "venv/ already exists — skipping creation"
else
    $PYTHON -m venv venv
    ok "Virtual environment created"
fi

# ─────────────────────────────────────────────────────────────
# 3. Activate venv
# ─────────────────────────────────────────────────────────────
info "Activating virtual environment …"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    ACTIVATE="venv/Scripts/activate"
else
    ACTIVATE="venv/bin/activate"
fi

if [ ! -f "$ACTIVATE" ]; then
    err "Could not find activate script at $ACTIVATE"
fi

source "$ACTIVATE"
ok "venv activated"

# ─────────────────────────────────────────────────────────────
# 4. Upgrade pip silently
# ─────────────────────────────────────────────────────────────
info "Upgrading pip …"
pip install --upgrade pip --quiet
ok "pip upgraded"

# ─────────────────────────────────────────────────────────────
# 5. Install requirements
# ─────────────────────────────────────────────────────────────
bar
info "Installing project dependencies (this takes 2–5 min) …"
pip install -r requirements.txt
ok "All dependencies installed"

# ─────────────────────────────────────────────────────────────
# 6. Register Jupyter kernel
# ─────────────────────────────────────────────────────────────
bar
info "Registering Jupyter kernel 'cmapss-env' …"
python -m ipykernel install --user --name cmapss-env \
       --display-name "Python (cmapss-env)"
ok "Jupyter kernel registered"

# ─────────────────────────────────────────────────────────────
# 7. Create reports/logs directory
# ─────────────────────────────────────────────────────────────
mkdir -p reports/logs reports/figures
ok "Output directories ready"

# ─────────────────────────────────────────────────────────────
# 8. Git init
# ─────────────────────────────────────────────────────────────
bar
info "Initialising git repository …"
if [ -d ".git" ]; then
    warn ".git already exists — skipping git init"
else
    git init -q
    git add .
    git commit -q -m "Initial commit — NASA CMAPSS sensor fault detection"
    ok "Git repository initialised and first commit made"
fi

# ─────────────────────────────────────────────────────────────
# 9. Verify dataset files
# ─────────────────────────────────────────────────────────────
bar
info "Checking for NASA CMAPSS dataset files …"
MISSING=0
for f in "train_FD001.txt" "test_FD001.txt" "RUL_FD001.txt"; do
    if [ ! -f "data/raw/$f" ] && [ ! -f "data/raw/CMaps/$f" ]; then
        warn "Not found: data/raw/$f"
        MISSING=1
    else
        ok "Found: $f"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    warn "Copy your NASA CMAPSS files to data/raw/ before running training."
    warn "Expected files: train_FD001.txt  test_FD001.txt  RUL_FD001.txt"
    warn "If they are inside a CMaps/ subfolder, update config/config.yaml:"
    warn "  data.train_file: CMaps/train_FD001.txt"
fi

# ─────────────────────────────────────────────────────────────
# DONE — Print command reference
# ─────────────────────────────────────────────────────────────
bar
echo -e "${GRN}"
echo "  ✅  SETUP COMPLETE!"
echo -e "${NC}"
bar
echo ""
echo -e "${YLW}  ACTIVATE ENVIRONMENT (run this every new terminal session):${NC}"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
echo "    venv\\Scripts\\activate"
else
echo "    source venv/bin/activate"
fi
echo ""
echo -e "${YLW}  TRAIN THE MODEL:${NC}"
echo "    python main.py train"
echo ""
echo -e "${YLW}  FULL PIPELINE (train + evaluate + plots):${NC}"
echo "    python main.py all"
echo ""
echo -e "${YLW}  EVALUATE ONLY (after training):${NC}"
echo "    python main.py evaluate"
echo ""
echo -e "${YLW}  GENERATE PLOTS ONLY:${NC}"
echo "    python main.py visualize"
echo ""
echo -e "${YLW}  RUN JUPYTER NOTEBOOK:${NC}"
echo "    jupyter notebook notebooks/NASA_CMAPSS_Complete.ipynb"
echo ""
echo -e "${YLW}  RUN TESTS:${NC}"
echo "    pytest tests/ -v"
echo ""
echo -e "${YLW}  IN VS CODE:${NC}"
echo "    1. Open folder: sensor-ml-project-v2/"
echo "    2. Press F5 to open Run panel"
echo "    3. Select '▶ Run Full Pipeline' and press the green ▶ button"
echo ""
bar
