# Windows Compatible Makefile for Telco Customer Churn Analysis Project
.PHONY: all clean install setup train-pipeline data-pipeline streaming-inference run-all help test lint format notebook mlflow-ui mlflow-start mlflow-clean mlflow-install stop-all check-env info

# Configuration
PYTHON = python
VENV = .venv
VENV_ACTIVATE = $(VENV)\Scripts\activate.bat
VENV_PYTHON = $(VENV)\Scripts\python.exe
VENV_PIP = $(VENV)\Scripts\pip.exe
MLFLOW_PORT = 5000
PROJECT_NAME = telco-churn-analysis

# Default target
all: help


help:
	@echo "=================================================================="
	@echo "  TELCO CUSTOMER CHURN ANALYSIS - MAKEFILE COMMANDS"
	@echo "=================================================================="
	@echo ""
	@echo "Setup and Installation:"
	@echo "  make install             - Install project dependencies and setup environment"
	@echo "  make setup               - Complete project setup with verification"
	@echo "  make check-env           - Check environment and dependencies"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make data-pipeline       - Run the data processing pipeline"
	@echo "  make clean-data          - Clean processed data artifacts"
	@echo ""
	@echo "Model Pipeline:"
	@echo "  make train-pipeline      - Run the model training pipeline"
	@echo "  make clean-models        - Clean trained models and evaluations"
	@echo ""
	@echo "Inference Pipeline:"
	@echo "  make streaming-inference - Run streaming inference with sample data"
	@echo "  make inference-demo      - Run enhanced inference demonstrations"
	@echo "  make clean-predictions   - Clean prediction artifacts"
	@echo ""
	@echo "Complete Workflow:"
	@echo "  make run-all             - Run complete ML pipeline (data + train + inference)"
	@echo "  make run-pipeline        - Run data and training pipelines only"
	@echo ""
	@echo "MLflow and Monitoring:"
	@echo "  make mlflow-ui           - Start MLflow UI server"
	@echo "  make mlflow-clean        - Clean MLflow artifacts"
	@echo "  make mlflow-install      - Install/reinstall MLflow"
	@echo ""
	@echo "Development and Testing:"
	@echo "  make test                - Run unit tests"
	@echo "  make lint                - Run code linting"
	@echo "  make format              - Format code with black"
	@echo "  make notebook            - Start Jupyter notebook server"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean               - Clean all artifacts"
	@echo "  make clean-all           - Complete cleanup including virtual environment"
	@echo "  make stop-all            - Stop all running processes"
	@echo ""
	@echo "=================================================================="

# Install project dependencies and set up environment
install:
	@echo "=================================================================="
	@echo "  INSTALLING TELCO CHURN ANALYSIS DEPENDENCIES"
	@echo "=================================================================="
	@echo ""
	@echo "Creating virtual environment..."
	@if not exist $(VENV) $(PYTHON) -m venv $(VENV)
	@echo "✓ Virtual environment created"
	@echo ""
	@echo "Upgrading pip..."
	@$(VENV_PYTHON) -m pip install --upgrade pip
	@echo "✓ Pip upgraded"
	@echo ""
	@echo "Installing project dependencies..."
	@$(VENV_PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"
	@echo ""
	@echo "Creating necessary directories..."
	@if not exist artifacts mkdir artifacts
	@if not exist artifacts\data mkdir artifacts\data
	@if not exist artifacts\models mkdir artifacts\models
	@if not exist artifacts\evaluation mkdir artifacts\evaluation
	@if not exist artifacts\predictions mkdir artifacts\predictions
	@if not exist artifacts\encode mkdir artifacts\encode
	@if not exist data mkdir data
	@if not exist data\processed mkdir data\processed
	@echo "✓ Directory structure created"
	@echo ""
	@echo "=================================================================="
	@echo "  INSTALLATION COMPLETED SUCCESSFULLY!"
	@echo "=================================================================="
	@echo ""
	@echo "To activate the virtual environment, run:"
	@echo "  $(VENV_ACTIVATE)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make setup' for complete project setup"
	@echo "  2. Run 'make run-all' to execute the complete ML pipeline"
	@echo "=================================================================="

# Complete project setup with verification
setup: install
	@echo "=================================================================="
	@echo "  COMPLETE PROJECT SETUP AND VERIFICATION"
	@echo "=================================================================="
	@echo ""
	@echo "Verifying Python environment..."
	@$(VENV_PYTHON) --version
	@echo ""
	@echo "Verifying key dependencies..."
	@$(VENV_PYTHON) -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('✓ Core ML libraries verified')"
	@$(VENV_PYTHON) -c "import xgboost, lightgbm; print('✓ Boosting libraries verified')"
	@$(VENV_PYTHON) -c "import yaml, joblib; print('✓ Utility libraries verified')"
	@echo ""
	@echo "Checking project structure..."
	@if exist config.yaml echo ✓ Configuration file found
	@if exist requirements.txt echo ✓ Requirements file found
	@if exist src echo ✓ Source directory found
	@if exist utils echo ✓ Utils directory found
	@if exist pipelines echo ✓ Pipelines directory found
	@echo ""
	@echo "=================================================================="
	@echo "  PROJECT SETUP COMPLETED SUCCESSFULLY!"
	@echo "=================================================================="
	@echo "Your Telco Churn Analysis environment is ready for use."
	@echo "=================================================================="

# Check environment and dependencies
check-env:
	@echo "=================================================================="
	@echo "  ENVIRONMENT AND DEPENDENCY CHECK"
	@echo "=================================================================="
	@echo ""
	@echo "Python Environment:"
	@if exist $(VENV) (echo ✓ Virtual environment exists) else (echo ✗ Virtual environment missing - run 'make install')
	@if exist $(VENV_PYTHON) $(VENV_PYTHON) --version
	@echo ""
	@echo "Project Structure:"
	@if exist config.yaml (echo ✓ config.yaml) else (echo ✗ config.yaml missing)
	@if exist requirements.txt (echo ✓ requirements.txt) else (echo ✗ requirements.txt missing)
	@if exist src (echo ✓ src/ directory) else (echo ✗ src/ directory missing)
	@if exist utils (echo ✓ utils/ directory) else (echo ✗ utils/ directory missing)
	@if exist pipelines (echo ✓ pipelines/ directory) else (echo ✗ pipelines/ directory missing)
	@if exist artifacts (echo ✓ artifacts/ directory) else (echo ✗ artifacts/ directory missing)
	@echo ""
	@echo "Data Files:"
	@if exist Telco-Customer-Churn.csv (echo ✓ Raw data file) else (echo ⚠ Raw data file missing)
	@if exist artifacts\data\X_train.csv (echo ✓ Training data processed) else (echo ⚠ Training data not processed)
	@echo ""
	@echo "=================================================================="

# Run data pipeline
data-pipeline: check-env
	@echo "=================================================================="
	@echo "  RUNNING DATA PROCESSING PIPELINE"
	@echo "=================================================================="
	@echo ""
	@echo "Starting data ingestion and preprocessing..."
	@$(VENV_ACTIVATE) && $(PYTHON) pipelines/data_pipeline.py
	@echo ""
	@echo "✓ Data pipeline completed successfully!"
	@echo "=================================================================="

# Clean processed data artifacts
clean-data:
	@echo "Cleaning processed data artifacts..."
	@if exist artifacts\data rmdir /s /q artifacts\data
	@if exist data\processed rmdir /s /q data\processed
	@if exist temp_imputed.csv del temp_imputed.csv
	@echo "✓ Data artifacts cleaned"

# Run training pipeline with MLflow tracking
train-pipeline: check-env
	@echo "=================================================================="
	@echo "  RUNNING MODEL TRAINING PIPELINE WITH MLFLOW"
	@echo "=================================================================="
	@echo ""
	@echo "Starting model training and evaluation with MLflow tracking..."
	@$(VENV_ACTIVATE) && $(PYTHON) pipelines/training_pipeline.py
	@echo ""
	@echo "✓ Training pipeline with MLflow completed successfully!"
	@echo "  • View experiments at: http://localhost:5000"
	@echo "  • Run 'make mlflow-start' to launch MLflow UI"
	@echo "=================================================================="



# Clean trained models and evaluations
clean-models:
	@echo "Cleaning model artifacts..."
	@if exist artifacts\models rmdir /s /q artifacts\models
	@if exist artifacts\evaluation rmdir /s /q artifacts\evaluation
	@if exist mlruns rmdir /s /q mlruns
	@echo "✓ Model artifacts cleaned"

# Run streaming inference pipeline
streaming-inference: check-env
	@echo "=================================================================="
	@echo "  RUNNING STREAMING INFERENCE PIPELINE"
	@echo "=================================================================="
	@echo ""
	@echo "Starting streaming inference with sample customer data..."
	@$(VENV_ACTIVATE) && $(PYTHON) pipelines/streaming_inference_pipeline.py
	@echo ""
	@echo "✓ Streaming inference completed successfully!"
	@echo "=================================================================="

# Run enhanced inference demonstrations
inference-demo: check-env
	@echo "=================================================================="
	@echo "  RUNNING ENHANCED INFERENCE DEMONSTRATIONS"
	@echo "=================================================================="
	@echo ""
	@echo "Starting comprehensive inference demonstrations..."
	@$(VENV_ACTIVATE) && $(PYTHON) example_model_inference_usage.py
	@echo ""
	@echo "✓ Inference demonstrations completed successfully!"
	@echo "=================================================================="

# Clean prediction artifacts
clean-predictions:
	@echo "Cleaning prediction artifacts..."
	@if exist artifacts\predictions rmdir /s /q artifacts\predictions
	@echo "✓ Prediction artifacts cleaned"

# Run data and training pipelines only
run-pipeline: data-pipeline train-pipeline
	@echo "=================================================================="
	@echo "  DATA AND TRAINING PIPELINES COMPLETED"
	@echo "=================================================================="
	@echo ""
	@echo "✓ Data processing completed"
	@echo "✓ Model training completed"
	@echo ""
	@echo "Next step: Run 'make streaming-inference' for inference testing"
	@echo "=================================================================="

# Run all pipelines in sequence
run-all: data-pipeline train-pipeline streaming-inference
	@echo "=================================================================="
	@echo "  COMPLETE ML PIPELINE EXECUTION FINISHED"
	@echo "=================================================================="
	@echo ""
	@echo "Pipeline Summary:"
	@echo "  ✓ Data Processing Pipeline - Completed"
	@echo "  ✓ Model Training with MLflow tracking - Completed"  
	@echo "  ✓ Streaming Inference - Completed"
	@echo ""
	@echo "Results available in:"
	@echo "  • Trained Models:    artifacts/models/"
	@echo "  • Evaluations:       artifacts/evaluation/"
	@echo "  • Predictions:       artifacts/predictions/"
	@echo "  • MLflow Experiments: mlruns/"
	@echo ""
	@echo "Run 'make mlflow-start' to view experiment tracking"
	@echo "=================================================================="

# Install/reinstall MLflow
mlflow-install:
	@echo "=================================================================="
	@echo "  INSTALLING/REINSTALLING MLFLOW"
	@echo "=================================================================="
	@echo ""
	@echo "Installing MLflow and dependencies..."
	@$(VENV_PIP) install mlflow>=2.0.0
	@echo "Installing additional MLflow dependencies..."
	@$(VENV_PIP) install boto3 azure-storage-blob google-cloud-storage
	@echo "Verifying MLflow installation..."
	@$(VENV_PYTHON) -c "import mlflow; print(f'✓ MLflow version: {mlflow.__version__} installed successfully')"
	@echo "✓ MLflow installation completed"
	@echo "=================================================================="

# Alternative MLflow UI command (more reliable)
mlflow-start:
	@echo "=================================================================="
	@echo "  STARTING MLFLOW UI (ALTERNATIVE METHOD)"
	@echo "=================================================================="
	@echo ""
	@if not exist mlruns mkdir mlruns
	@echo "Starting MLflow UI on http://localhost:$(MLFLOW_PORT)"
	@echo "Press Ctrl+C to stop the MLflow UI"
	@echo ""
	@$(VENV_PYTHON) -m mlflow ui --host 127.0.0.1 --port $(MLFLOW_PORT) --backend-store-uri file:./mlruns

# Start MLflow UI
mlflow-ui: check-env
	@echo "=================================================================="
	@echo "  STARTING MLFLOW UI SERVER"
	@echo "=================================================================="
	@echo ""
	@echo "Checking MLflow installation..."
	@$(VENV_PYTHON) -c "import mlflow; print(f'✓ MLflow version: {mlflow.__version__}')" || (echo "✗ MLflow not found - installing..." && $(VENV_PIP) install mlflow>=2.0.0)
	@echo ""
	@echo "Initializing MLflow tracking..."
	@if not exist mlruns mkdir mlruns
	@echo "MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)"
	@echo "Press Ctrl+C to stop the MLflow UI"
	@echo ""
	@echo "=================================================================="
	@$(VENV_PYTHON) -m mlflow ui --host 127.0.0.1 --port $(MLFLOW_PORT) --backend-store-uri file:./mlruns

# Clean MLflow artifacts
mlflow-clean:
	@echo "Cleaning MLflow artifacts..."
	@if exist mlruns rmdir /s /q mlruns
	@if exist mlartifacts rmdir /s /q mlartifacts
	@echo "✓ MLflow artifacts cleaned"







# Run unit tests
test: check-env
	@echo "=================================================================="
	@echo "  RUNNING UNIT TESTS"
	@echo "=================================================================="
	@echo ""
	@if exist tests ($(VENV_ACTIVATE) && $(PYTHON) -m pytest tests/ -v --cov=src) else (echo "⚠ Tests directory not found - skipping tests")
	@echo ""
	@echo "=================================================================="

# Run code linting
lint: check-env
	@echo "=================================================================="
	@echo "  RUNNING CODE LINTING"
	@echo "=================================================================="
	@echo ""
	@echo "Linting source code..."
	@$(VENV_ACTIVATE) && $(PYTHON) -m flake8 src/ --max-line-length=88 --exclude=__pycache__
	@echo "Linting pipelines..."
	@$(VENV_ACTIVATE) && $(PYTHON) -m flake8 pipelines/ --max-line-length=88 --exclude=__pycache__
	@echo "Linting utils..."
	@$(VENV_ACTIVATE) && $(PYTHON) -m flake8 utils/ --max-line-length=88 --exclude=__pycache__
	@echo "✓ Linting completed"
	@echo "=================================================================="

# Format code with black
format: check-env
	@echo "=================================================================="
	@echo "  FORMATTING CODE WITH BLACK"
	@echo "=================================================================="
	@echo ""
	@echo "Formatting source code..."
	@$(VENV_ACTIVATE) && $(PYTHON) -m black src/ --line-length=88
	@echo "Formatting pipelines..."
	@$(VENV_ACTIVATE) && $(PYTHON) -m black pipelines/ --line-length=88
	@echo "Formatting utils..."
	@$(VENV_ACTIVATE) && $(PYTHON) -m black utils/ --line-length=88
	@echo "✓ Code formatting completed"
	@echo "=================================================================="

# Start Jupyter notebook server
notebook: check-env
	@echo "=================================================================="
	@echo "  STARTING JUPYTER NOTEBOOK SERVER"
	@echo "=================================================================="
	@echo ""
	@echo "Jupyter will open in your default browser"
	@echo "Press Ctrl+C to stop the notebook server"
	@echo ""
	@echo "=================================================================="
	@$(VENV_ACTIVATE) && jupyter notebook

# Stop all running processes
stop-all:
	@echo "=================================================================="
	@echo "  STOPPING ALL RUNNING PROCESSES"
	@echo "=================================================================="
	@echo ""
	@echo "Finding and stopping MLflow processes..."
	@-powershell -Command "Get-Process | Where-Object {$$_.ProcessName -like '*mlflow*' -or $$_.ProcessName -like '*gunicorn*'} | Stop-Process -Force" 2>nul || echo "No MLflow processes found."
	@echo ""
	@echo "Finding and stopping processes on port $(MLFLOW_PORT)..."
	@-for /f "tokens=5" %%a in ('netstat -ano ^| findstr :$(MLFLOW_PORT)') do taskkill /pid %%a /f 2>nul || echo "No processes found on port $(MLFLOW_PORT)."
	@echo ""
	@echo "Finding and stopping Jupyter processes..."
	@-powershell -Command "Get-Process | Where-Object {$$_.ProcessName -like '*jupyter*'} | Stop-Process -Force" 2>nul || echo "No Jupyter processes found."
	@echo ""
	@echo "✓ All processes stopped"
	@echo "=================================================================="

# Clean up artifacts
clean:
	@echo "=================================================================="
	@echo "  CLEANING PROJECT ARTIFACTS"
	@echo "=================================================================="
	@echo ""
	@echo "Cleaning model artifacts..."
	@if exist artifacts\models rmdir /s /q artifacts\models
	@echo "Cleaning evaluation artifacts..."
	@if exist artifacts\evaluation rmdir /s /q artifacts\evaluation  
	@echo "Cleaning prediction artifacts..."
	@if exist artifacts\predictions rmdir /s /q artifacts\predictions
	@echo "Cleaning processed data..."
	@if exist data\processed rmdir /s /q data\processed
	@echo "Cleaning temporary files..."
	@if exist temp_imputed.csv del temp_imputed.csv
	@echo "Cleaning Python cache..."
	@for /r %%i in (__pycache__) do @if exist "%%i" rmdir /s /q "%%i"
	@for /r %%i in (*.pyc) do @if exist "%%i" del /q "%%i"
	@echo "Cleaning MLflow artifacts..."
	@if exist mlruns rmdir /s /q mlruns
	@if exist mlartifacts rmdir /s /q mlartifacts
	@echo ""
	@echo "✓ Cleanup completed successfully!"
	@echo "=================================================================="

# Complete cleanup including virtual environment  
clean-all: clean stop-all
	@echo "=================================================================="
	@echo "  COMPLETE PROJECT CLEANUP"
	@echo "=================================================================="
	@echo ""
	@echo "Removing virtual environment..."
	@if exist $(VENV) rmdir /s /q $(VENV)
	@echo "Removing additional artifacts..."
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist .coverage del .coverage
	@echo ""
	@echo "✓ Complete cleanup finished!"
	@echo "=================================================================="
	@echo ""
	@echo "To restart the project:"
	@echo "  1. Run 'make install' to recreate environment"
	@echo "  2. Run 'make run-all' to execute pipelines"
	@echo "=================================================================="





# Project info
info:
	@echo "=================================================================="
	@echo "  TELCO CUSTOMER CHURN ANALYSIS PROJECT"
	@echo "=================================================================="
	@echo ""
	@echo "Project: $(PROJECT_NAME)"
	@echo "Python Version: "
	@if exist $(VENV_PYTHON) $(VENV_PYTHON) --version
	@echo "MLflow Port: $(MLFLOW_PORT)"
	@echo ""
	@echo "Key Directories:"
	@echo "  • Source Code:    src/"
	@echo "  • Pipelines:      pipelines/"
	@echo "  • Configuration:  utils/"
	@echo "  • Artifacts:      artifacts/"
	@echo "  • Data:           data/"
	@echo "  • Notebooks:      Notebook/"
	@echo ""
	@echo "Key Files:"
	@echo "  • Configuration:  config.yaml"
	@echo "  • Requirements:   requirements.txt"
	@echo "  • Raw Data:       Telco-Customer-Churn.csv"
	@echo ""
	@echo "Service URLs (when running):"
	@echo "  • MLflow UI:      http://localhost:$(MLFLOW_PORT)"
	@echo ""
	@echo "================================================================="