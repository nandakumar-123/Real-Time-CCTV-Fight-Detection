# Real-Time-CCTV-Fight-Detection

> **Quick Start:** For a complete step-by-step list of terminal commands from virtual environment creation to running the Streamlit dashboard, see **[RUN_COMMANDS.md](file:///c:/Users/Soundar/Documents/fight_detection/RUN_COMMANDS.md)**.

A modular, hardware-aware deep learning system for automated physical fight and violence detection in surveillance footage using **VideoMAE V2 Base** transfer learning.

---

## 📌 Architecture Overview

```
                        ┌───────────────────────────────┐
                        │      Surveillance Video       │
                        │   (MP4 / AVI / MOV / MKV)     │
                        └───────────────┬───────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │    Temporal Window Slicing    │
                        │    (2.0s Duration, 1.0s Step) │
                        └───────────────┬───────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │   Uniform 16-Frame Sampling   │
                        │      (224 × 224 RGB Normal)   │
                        └───────────────┬───────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │      VideoMAE V2 Base         │
                        │    Pretrained Spatiotemporal  │
                        │        Transformer Model      │
                        └───────────────┬───────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │ Custom Classification Head    │
                        │    [Dropout -> Linear(768,2)] │
                        └───────────────┬───────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │ Temporal Smoothing & Events   │
                        │  - Glitch Filtering           │
                        │  - Start & End Timestamps     │
                        └───────────────┬───────────────┘
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
        ┌───────────────────────────────┐  ┌───────────────────────────────┐
        │       Streamlit Web UI        │  │     Alert & Incident Logs     │
        │ - < FIGHT DETECTED > Alert    │  │ - outputs/events/*.json       │
        │ - Interactive Timeline Graph  │  │ - outputs/predictions/*.json  │
        │ - Held-out Model Metrics      │  │ - logs/application.log        │
        └───────────────────────────────┘  └───────────────────────────────┘
```

---

## ⚙️ Why VideoMAE V2 Base?

- **Spatiotemporal Attention**: Unlike 2D CNNs or frame-averaging networks, VideoMAE V2 employs dual masked autoencoding pre-trained on extensive video corpuses, capturing subtle temporal motions, sudden aggressive lunges, and physical struggles.
- **Surveillance Adaptability**: Pretrained weights provide general spatiotemporal representations that fine-tune effectively on low-framerate and varying CCTV camera angles with high fight recall.

---

## 📂 Datasets

1. **RWF-2000**:
   - 2,000 real-world surveillance video clips (1,000 Fight, 1,000 Non-Fight).
   - Real-world indoor/outdoor CCTV recordings (~5 seconds each).
2. **NTU CCTV-Fights**:
   - 1,000 long-form videos (280 actual CCTV recordings, 720 other camera views).
   - 2,414 annotated physical fight instances with start and end timestamps.
   - Handled via `data/annotation_parser.py` to extract positive and negative temporal intervals without full-video mislabeling.

---

## 🛠️ Step-by-Step Installation Instructions

> [!IMPORTANT]
> In accordance with user requirements, packages are not installed automatically. Follow the manual commands below in Windows PowerShell.

### 1. Create and Activate Virtual Environment

Open Windows PowerShell in the project directory:

```powershell
cd c:\Users\Soundar\Documents\fight_detection
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*(If script execution is restricted on your machine, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*.

### 2. Install PyTorch

Choose either **GPU (CUDA)** or **CPU**:

#### For NVIDIA GPU with CUDA 12.1+:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### For CPU Only:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Required Dependencies

```powershell
pip install -r requirements.txt
```

---

## ⚙️ Configuration (.env)

Copy the template environment file:

```powershell
cp .env.example .env
```

Edit `.env` to point to your physical dataset directories:

```ini
RWF2000_DATA_DIR=C:/Datasets/RWF-2000
NTU_CCTV_FIGHTS_DATA_DIR=C:/Datasets/NTU-CCTV-Fights
DEVICE=auto
BATCH_SIZE=auto
EPOCHS=30
```

---

## 🚀 Execution & Workflow Commands

### Step 1: Automatic Hardware Inspection

Inspect your CPU, RAM, and GPU capabilities. The system automatically categorizes your system into Scenario A, B, C, or D (CPU) and generates safety reports:

```powershell
python -m utils.hardware_detector
```

- Report generated: `reports/hardware_report.txt` and `reports/hardware_report.json`

### Step 2: Download & Verify Pretrained Model Weights (User-Controlled)

Download and verify the pretrained VideoMAE V2 Base weights into your local cache:

```powershell
python -m models.download_weights
```

### Step 3: Dataset Integrity Audit

Audit your dataset videos, frame rates, file formats, corrupted clips, and duplicates:

```powershell
python -m data.dataset_inspector
```

- Reports generated: `reports/dataset_summary.json`, `reports/dataset_summary.csv`, `reports/dataset_inspection.txt`

### Step 4: Source-Aware Data Splitting

Partition clips into 70% Train, 15% Validation, and 15% Test without video source leakage:

```powershell
python -m data.dataset_splitter
```

- Splits generated: `outputs/splits/train.csv`, `outputs/splits/validation.csv`, `outputs/splits/test.csv`

### Step 5: Staged Model Fine-Tuning

Execute hardware-aware transfer learning:

```powershell
# Stage 1: Frozen backbone (trains classification head)
python -m training.train --stage 1 --epochs 10

# Stage 2: Unfreeze final transformer blocks with differential learning rates
python -m training.train --stage 2 --epochs 15 --resume checkpoints/best_model.pth

# Stage 3: Full fine-tuning (if supported by your hardware)
python -m training.train --stage 3 --epochs 5 --resume checkpoints/best_model.pth
```

### Step 6: Independent Evaluation

Run evaluation strictly against the held-out test split:

```powershell
python -m training.evaluator
```

- Outputs generated:
  - `reports/final_metrics.json`
  - `reports/classification_report.txt`
  - `reports/confusion_matrix.png`
  - `reports/roc_curve.png`
  - `reports/error_analysis.csv`

### Step 7: Launch the Streamlit Web Application

Launch the interactive web UI:

```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Step 8: Run Automated Unit Tests

```powershell
pytest -v
```

# Stage 3: Full fine-tuning (if supported by your hardware)

python -m training.train --stage 3 --epochs 5 --resume checkpoints/best_model.pth

```

### Step 5: Independent Evaluation

Run evaluation strictly against the held-out test split:

```powershell
python -m training.evaluator
```

- Outputs generated:
  - `reports/final_metrics.json`
  - `reports/classification_report.txt`
  - `reports/confusion_matrix.png`
  - `reports/roc_curve.png`
  - `reports/error_analysis.csv`

### Step 6: Launch the Streamlit Web Application

Launch the interactive web UI:

```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Step 7: Run Automated Unit Tests

```powershell
pytest -v
```

---

## 📊 Streamlit Application Pages

1. **Dashboard**: Live system health, CPU/VRAM usage, model status, and operational count.
2. **Upload & Analyze Video**: Drag-and-drop video footage, preview video, adjust thresholds, and click `ANALYSE VIDEO`.
3. **Detection Results**: Displays `< FIGHT DETECTED >` or `< NO FIGHT DETECTED >` banner, event duration, timestamps, and download buttons (JSON / CSV).
4. **Temporal Timeline**: Interactive chart plotting probability over time, highlighting physical fight spans.
5. **Model Evaluation**: Displays test metrics, confusion matrix, and ROC curves.
6. **Logs & Diagnostics**: Multi-stream log viewer and automated error root-cause diagnostics.

---

## 🔒 Resource Safety & Hardware Scenarios

- **Scenario A (>= 24GB VRAM)**: Batch size 4, mixed precision (AMP), 4 workers.
- **Scenario B (16GB VRAM)**: Batch size 2, mixed precision (AMP), gradient accumulation.
- **Scenario C (8-12GB VRAM)**: Batch size 1, mixed precision (AMP), gradient accumulation (8 steps).
- **Scenario D (CPU Only)**: Batch size 1, 0 dataloader workers (Windows stability), frozen backbone mode to avoid thermal throttling.

---

## 📜 Future 24/7 Live CCTV Deployment

The initial scope covers uploaded video testing and offline fine-tuning. For full 24/7 live RTSP camera streams, connect an RTSP stream loop into `inference/video_inference.py` using OpenCV or PyAV with an alert webhook dispatcher.
#� �R�e�a�l�-�T�i�m�e�-�C�C�T�V�-�F�i�g�h�t�-�D�e�t�e�c�t�i�o�n�
�
�#� �R�e�a�l�-�T�i�m�e�-�C�C�T�V�-�F�i�g�h�t�-�D�e�t�e�c�t�i�o�n�
�
�
