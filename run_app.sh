#!/bin/bash
set -e

echo "=========================================================="
echo "Cat & Dog Classification Method Comparison - Streamlit App"
echo "=========================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.10+ from https://www.python.org/"
    exit 1
fi

echo "[1/3] Checking dependencies..."
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Installing dependencies from requirements.txt..."
    pip3 install -r requirements.txt
else
    echo "Dependencies OK"
fi

echo ""
echo "[2/3] Checking model files..."
if [ ! -f "models/custom_cnn_best.keras" ]; then
    echo "NOTICE: Trained model weights not found at models/custom_cnn_best.keras."
    echo "You can train a model using: python3 scripts/train.py --model custom_cnn"
    echo ""
else
    echo "Model weights found: models/custom_cnn_best.keras"
fi

echo ""
echo "[3/3] Starting Streamlit app..."
echo "App will open at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""
streamlit run app/app.py
