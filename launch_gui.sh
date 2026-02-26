#!/bin/bash
echo "============================================================"
echo " NASA CMAPSS  --  Engine Health Monitor"
echo " DRDO / GTRE  Turbofan PHM System"
echo "============================================================"
echo ""
echo " Starting web dashboard on http://localhost:8501"
echo " Press Ctrl+C to stop"
echo ""

cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
pip show streamlit >/dev/null 2>&1 || pip install streamlit --quiet

streamlit run app.py \
  --server.port 8501 \
  --browser.gatherUsageStats false \
  --theme.base dark \
  --theme.primaryColor "#7eb8f7" \
  --theme.backgroundColor "#0a0e1a" \
  --theme.secondaryBackgroundColor "#0d1a2e" \
  --theme.textColor "#c8d6e5"
