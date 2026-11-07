# 🚀 Complete Run & Installation Commands Guide

This document contains every single command required to set up, validate, train, evaluate, and launch the **Real-Time CCTV Fight Detection System** from start to finish.

---

## 📋 Table of Contents
1. [Virtual Environment Setup](#1-virtual-environment-setup)
2. [Dependency Installation (GPU vs CPU)](#2-dependency-installation-gpu-vs-cpu)
3. [Environment Configuration (.env)](#3-environment-configuration-env)
4. [Hardware & Laptop Detection](#4-hardware--laptop-detection)
5. [Pretrained Model Download](#5-pretrained-model-download)
6. [Dataset Path & Integrity Validation](#6-dataset-path--integrity-validation)
7. [Source-Aware Dataset Splitting](#7-source-aware-dataset-splitting)
8. [Model Training (Stage 1, 2, and 3)](#8-model-training)
9. [Automatic Error Logging & AI Diagnostics](#9-automatic-error-logging--ai-diagnostics)
10. [Model Evaluation on Held-Out Test Set](#10-model-evaluation-on-held-out-test-set)
11. [Launch Streamlit Surveillance Web UI](#11-launch-streamlit-surveillance-web-ui)
12. [Run Automated Test Suite](#12-run-automated-test-suite)

---

## 1. Virtual Environment Setup

Open **Windows PowerShell** in the project directory:

```powershell
cd c:\Users\Soundar\Documents\fight_detection
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> **Note:** If PowerShell displays an execution policy error, enable script execution for the current session:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```

---

## 2. Dependency Installation (GPU vs CPU)

### Option A: If your laptop has a supported NVIDIA GPU with CUDA
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Option B: If your laptop runs on CPU (No NVIDIA GPU)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 3. Environment Configuration (.env)

The `.env` file is located at the root of the project. Open it in any text editor and specify your dataset paths:

```env
RWF2000_DATA_DIR=C:/Datasets/RWF-2000
NTU_CCTV_FIGHTS_DATA_DIR=C:/Datasets/NTU-CCTV-Fights
MODEL_ID=OpenGVLab/VideoMAEv2-Base
DEVICE=auto
```

---

## 4. Hardware & Laptop Detection

Inspects CPU, RAM, and GPU. Never assumes NVIDIA presence simply from an Intel i7 CPU. Automatically selects safe batch sizes and conservative CPU settings if no CUDA GPU is detected:

```powershell
python -m utils.hardware_detector
```

- **Output Report:** `reports/hardware_report.txt` and `reports/hardware_report.json`

---

## 5. Pretrained Model Download

Downloads and verifies the official VideoMAE V2 Base architecture and weights into your local Hugging Face cache:

```powershell
python -m models.download_weights
```

---

## 6. Dataset Path & Integrity Validation

Rigorously checks your configured paths, verifies that video files exist, tests video decoding, checks for Fight/Non-Fight classes, and parses NTU temporal annotations:

```powershell
python -m data.dataset_validator
```

---

## 7. Source-Aware Dataset Splitting

Partitions your clips into 70% Train, 15% Validation, and 15% Test without video source leakage:

```powershell
python -m data.dataset_splitter
```

- **Generated CSVs:** `outputs/splits/train.csv`, `outputs/splits/validation.csv`, `outputs/splits/test.csv`

---

## 8. Model Training

The training script automatically checks hardware, validates datasets, prints a pre-training summary, and asks for confirmation (`yes/no`).

### Stage 1: Train Classification Head (Backbone Frozen)
```powershell
python -m training.train --stage 1 --epochs 10
```

### Stage 2: Unfreeze Final Transformer Blocks
```powershell
python -m training.train --stage 2 --epochs 15 --resume checkpoints/best_model.pth
```

### Stage 3: Full Fine-Tuning (Optional)
```powershell
python -m training.train --stage 3 --epochs 5 --resume checkpoints/best_model.pth
```

> **Tip:** You can pass `-y` to skip the interactive confirmation prompt:
> ```powershell
> python -m training.train --stage 1 --epochs 10 -y
> ```

---

## 9. Automatic Error Logging & AI Diagnostics

If **any** error occurs during training (e.g. CUDA OOM, corrupt file, tensor mismatch):
1. The full traceback is logged to: `logs/training_errors.log`
2. An AI diagnostic report is generated: `reports/training_error_diagnostic.md`
   - Identifies the root cause.
   - Provides the exact suggested fix and command.

---

## 10. Model Evaluation on Held-Out Test Set

Evaluates the best checkpoint strictly on unseen test clips:

```powershell
python -m training.evaluator
```

- **Metrics Generated:**
  - `reports/final_metrics.json`
  - `reports/classification_report.txt`
  - `reports/confusion_matrix.png`
  - `reports/roc_curve.png`
  - `reports/error_analysis.csv`

---

## 11. Launch Streamlit Surveillance Web UI

Starts the interactive web application:

```powershell
streamlit run app.py
```

Open **`http://localhost:8501`** in your web browser.
- **Dashboard:** System health and model info.
- **Upload & Analyze Video:** Upload `.mp4`, `.avi`, `.mov`, `.mkv` files and detect fights.
- **Detection Results:** `< FIGHT DETECTED >` or `< NO FIGHT DETECTED >` alert banners.
- **Temporal Timeline:** Interactive timeline displaying exact fight start and end timestamps.
- **Model Evaluation:** Held-out test performance graphs and confusion matrices.
- **Logs & Diagnostics:** Live log viewer and error reports.

---

## 12. Run Automated Test Suite

Runs all unit tests for hardware, video preprocessing, datasets, and alerts:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```
Or with pytest:
```powershell
pytest -v
```

