@echo off
echo ==========================================================
echo Cat ^& Dog Classification Method Comparison - Streamlit App
echo ==========================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo Dependencies OK
)

echo.
echo [2/3] Checking model files...
if not exist "models\custom_cnn_best.keras" (
    echo NOTICE: Trained model weights not found at models\custom_cnn_best.keras.
    echo You can train a model using: python scripts\train.py --model custom_cnn
    echo.
) else (
    echo Model weights found: models\custom_cnn_best.keras
)

echo.
echo [3/3] Starting Streamlit app...
echo App will open at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run app\app.py
pause
