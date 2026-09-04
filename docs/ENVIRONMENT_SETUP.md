# TEJAS Environment Setup Guide

## 1. Python Version
- **Recommended Python**: Python 3.10 - 3.14 (AMD64 / x86_64)

## 2. Setting Up a Virtual Environment
From the root directory of the TEJAS repository:
```bash
python -m venv .venv
```
Activate the environment:
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```

## 3. Installing Dependencies
Install all requirements with pip:
```bash
pip install -r requirements.txt
```

*(Optional PyTorch CPU-only installation for fast setup)*:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 4. Running the Pipeline & Models
Always execute scripts from the **project root directory**:

- **Phase 2 (Preprocessing)**:
  ```bash
  python data_pipeline/preprocess.py
  ```
- **Phase 4 (Feature Extraction & QA)**:
  ```bash
  python data_pipeline/feature_extraction.py
  python data_pipeline/feature_qa.py
  ```
- **Phase 5A (Feature-Based Classifiers: RF & HistGradientBoosting)**:
  ```bash
  python train_phase5a.py
  ```
- **Phase 5B (1D-CNN Dynamometer Card Classifier)**:
  ```bash
  python train_phase5b_cnn.py
  ```
