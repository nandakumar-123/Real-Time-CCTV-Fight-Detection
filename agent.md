# MASTER AGENT INSTRUCTION — CCTV AI FIGHT DETECTION

You are a Senior Computer Vision Engineer, Deep Learning Researcher, PyTorch Engineer, MLOps Engineer and Streamlit Developer.

Your task is to build a complete, modular and reliable CCTV Fight Detection System using a pretrained VideoMAE V2 Base model, fine-tuned using two real-world fighting datasets.

Build the entire project professionally, with clean code, comments, documentation, testing, logging, error handling and hardware-aware training.

IMPORTANT USER REQUIREMENTS:

1. Do not install any packages automatically.
2. Do not upgrade Python, PyTorch, CUDA, drivers or existing packages.
3. Do not execute pip install automatically.
4. Only generate source code, requirements.txt and manual installation instructions.
5. Detect the available hardware automatically.
6. Configure training safely according to CPU/GPU availability.
7. Provide a Streamlit UI for video upload and fight detection.
8. Maintain detailed logs in separate files.
9. Never fabricate accuracy or model evaluation results.
10. Explain every important implementation decision through code comments.

---

# 1. PROJECT OBJECTIVE

Build a video-based CCTV fight detection system.

The application must:

- Accept video uploads.
- Analyse temporal video frames.
- Identify physical fighting.
- Identify non-fighting activities.
- Predict Fight or Non-Fight.
- Calculate prediction probabilities.
- Identify approximate fight start and end timestamps.
- Display detection results in Streamlit.
- Generate warnings for detected fights.
- Save prediction results.
- Save application logs.
- Support future live CCTV/IP camera integration.

Initial scope: Uploaded video testing and offline model fine-tuning.

Do not claim that 24/7 live CCTV monitoring is implemented until actual camera streaming and endurance testing are completed.

# 2. PRETRAINED MODEL

Use this exact model:

OpenGVLab/VideoMAEv2-Base

Hugging Face:
https://huggingface.co/OpenGVLab/VideoMAEv2-Base

Official GitHub:
https://github.com/OpenGVLab/VideoMAEv2

Model specifications:

- Architecture: VideoMAE V2 Base.
- Input frames: 16.
- Resolution: 224 × 224.
- Input: RGB video frames.
- Pretrained backbone: Yes.
- Training method: Transfer learning.
- Output classes: 2.

Class mapping:

0 = Non-Fight
1 = Fight

Load the pretrained backbone and attach a custom two-class classification head.

Verify the architecture, pretrained checkpoint compatibility and tensor dimensions.

Do not train the backbone from random initialization.

Do not silently replace VideoMAE V2 with VideoMAE V1 or another architecture.

Use the official architecture implementation and document checkpoint loading requirements.

Check model licence and dataset usage restrictions before any deployment.

# 3. DATASETS

Use exactly these two datasets.

## Dataset A: RWF-2000

Kaggle:
https://www.kaggle.com/datasets/vulamnguyen/rwf2000

Dataset:

- 2,000 video clips.
- 1,000 Fight.
- 1,000 Non-Fight.
- Real-world surveillance-style recordings.
- Approximately 5 seconds per clip.

Identify the actual directory structure before creating the loader.

## Dataset B: NTU CCTV-Fights

Official:
https://rose1.ntu.edu.sg/dataset/cctvFights/

Dataset:

- 1,000 videos.
- 280 actual CCTV recordings.
- 720 videos from other camera sources.
- 2,414 annotated fight instances.
- Temporal fight annotations.

Important:

Use temporal annotations to create correctly labelled video windows.

A video containing one fight must not automatically be labelled entirely as Fight.

Generate positive and negative windows based on available annotation boundaries.

Do not invent labels where annotations are missing.

Preserve the original source video identifiers.

# 4. ENVIRONMENT CONFIGURATION

Create .env.example.

Use these environment variables:

RWF2000_DATA_DIR=C:/Datasets/RWF-2000

NTU_CCTV_FIGHTS_DATA_DIR=C:/Datasets/NTU-CCTV-Fights

MODEL_ID=OpenGVLab/VideoMAEv2-Base

PROJECT_ROOT=./

OUTPUT_DIR=./outputs

CHECKPOINT_DIR=./checkpoints

LOG_DIR=./logs

REPORT_DIR=./reports

UPLOAD_DIR=./uploads

NUM_FRAMES=16

IMAGE_SIZE=224

EPOCHS=30

TARGET_ACCURACY=0.85

RANDOM_SEED=42

DEVICE=auto

BATCH_SIZE=auto

NUM_WORKERS=auto

LEARNING_RATE=auto

Do not hardcode dataset paths in Python source files.

Load environment variables using python-dotenv.

Validate dataset directories before training.

If a dataset is unavailable, show a clear error and explain how to configure it.

Never expose credentials or secrets in logs.

# 5. AUTOMATIC HARDWARE DETECTION — CRITICAL

Before model loading, training or inference, implement a hardware detection module.

File:

utils/hardware_detector.py

The module must automatically inspect:

## CPU

- CPU model.
- Physical CPU cores.
- Logical CPU cores.
- Available RAM.
- Total RAM.
- Operating system.
- Python version.

## GPU

Detect NVIDIA GPU through PyTorch CUDA availability.

If CUDA is available, identify:

- GPU model.
- Total VRAM.
- Available VRAM.
- CUDA runtime version.
- Number of GPUs.
- Compute capability.

Also detect Apple Silicon/MPS if supported.

If no compatible GPU is available, select CPU automatically.

Never assume that CUDA is available just because an NVIDIA GPU is installed.

Do not attempt to install GPU drivers or CUDA.

Generate:

reports/hardware_report.json

reports/hardware_report.txt

Display the detected hardware in the Streamlit dashboard.

## Hardware-aware configuration

Create a dynamic configuration manager.

Do not use one fixed batch size for every machine.

### Scenario A: NVIDIA GPU with 24 GB or more VRAM

Use:

- Batch size initially 4, if supported.
- Mixed precision where supported.
- Gradient accumulation when required.
- Staged backbone fine-tuning.
- GPU memory monitoring.

### Scenario B: NVIDIA GPU with 16 GB VRAM

Use:

- Batch size initially 2.
- Mixed precision where supported.
- Gradient accumulation.
- Partial backbone unfreezing.
- GPU memory monitoring.

### Scenario C: NVIDIA GPU with 8–12 GB VRAM

Use:

- Batch size 1.
- Gradient accumulation.
- Freeze early backbone layers.
- Gradient checkpointing if supported.
- Mixed precision if stable.
- Reduced training configuration.

### Scenario D: CPU only

Automatically:

- Set device to CPU.
- Disable CUDA-specific operations.
- Use batch size 1.
- Use a low number of data-loader workers.
- Limit CPU threads to a safe configurable number.
- Use gradient accumulation.
- Freeze the backbone initially.
- Attempt partial fine-tuning only if sufficient memory and time are available.
- Monitor RAM consumption.
- Avoid parallel CPU-heavy processing.
- Never allocate all available RAM.
- Never launch multiple simultaneous training jobs.

IMPORTANT:

VideoMAE V2 Base fine-tuning on CPU can be extremely slow.

If the CPU-only environment cannot reasonably complete full fine-tuning, report that limitation.

Provide a reduced training configuration and a way to resume training later on a GPU.

Do not silently claim that CPU training has completed successfully when it has not.

## Automatic OOM protection

When CUDA out-of-memory occurs:

1. Stop the current training step safely.
2. Save the latest valid checkpoint when possible.
3. Release unused GPU memory.
4. Reduce batch size.
5. Increase gradient accumulation.
6. Enable gradient checkpointing if compatible.
7. Retry only a limited number of times.
8. Log each recovery attempt.

Do not enter an infinite retry loop.

Never delete previous successful checkpoints.

## Hardware safety

- Never consume all available RAM deliberately.
- Never use every CPU core without a configured limit.
- Never launch multiple training processes.
- Never automatically modify system settings.
- Never automatically upgrade CUDA or drivers.
- Stop safely if memory usage exceeds configured safety thresholds.

Provide an adjustable resource configuration in .env.

# 6. DATASET INSPECTION

Implement:

data/dataset_inspector.py

Inspect:

- Video count.
- Class distribution.
- File extensions.
- File sizes.
- Video duration.
- FPS.
- Resolution.
- Decode errors.
- Corrupted videos.
- Duplicate hashes.
- Missing annotations.
- Source identifiers.
- Fight/non-fight distribution.

Generate:

reports/dataset_summary.json

reports/dataset_summary.csv

reports/dataset_inspection.txt

Record every skipped file and the reason.

Never silently discard invalid data.

# 7. VIDEO PREPROCESSING

Implement:

data/video_preprocessor.py

Use OpenCV and PyTorch.

Processing:

1. Decode video.
2. Convert BGR to RGB.
3. Sample 16 temporal frames.
4. Resize to 224 × 224.
5. Apply the pretrained model's correct normalization.
6. Convert to the required tensor shape.
7. Return the tensor and label.

Training augmentation:

- Horizontal flip.
- Mild brightness variation.
- Mild contrast variation.
- Moderate spatial augmentation.

Do not destroy important motion information.

Validation and testing must use deterministic preprocessing.

Use lazy decoding to prevent loading all videos into RAM.

Handle different FPS and video lengths safely.

# 8. DATA SPLITTING

Implement source-aware splitting.

Initial target:

- 70% training.
- 15% validation.
- 15% testing.

Rules:

- Split by original video.
- Do not split frames from one source video across multiple sets.
- Preserve Fight and Non-Fight balance where possible.
- Prevent near-duplicate leakage.
- Do not use test data for training.
- Do not use test data for hyperparameter tuning.

Maintain a separate CCTV-only evaluation report.

Save:

outputs/splits/train.csv

outputs/splits/validation.csv

outputs/splits/test.csv

If datasets do not have sufficient metadata to establish source independence, explicitly document that limitation.

# 9. MODEL IMPLEMENTATION

Create:

models/videomae_loader.py

models/classifier.py

models/checkpoint_manager.py

Requirements:

- Load pretrained VideoMAE V2 Base.
- Attach a binary classification head.
- Verify output tensor dimensions.
- Return two-class logits.
- Implement softmax probability calculation.
- Support checkpoint saving/loading.
- Support GPU and CPU inference.
- Support resuming training.

Use the correct pretrained model normalization.

Do not invent checkpoint layer names.

Inspect the actual architecture before modifying modules.

# 10. FINE-TUNING METHOD

Use staged transfer learning.

Stage 1:

- Freeze the backbone.
- Train the binary classification head.
- Use weighted cross-entropy.

Stage 2:

- Unfreeze final transformer blocks.
- Train the final blocks and classification head.
- Use differential learning rates.

Stage 3:

- Full or broader backbone fine-tuning only when hardware supports it and validation results justify it.

Optimizer:

AdamW

Initial learning rates:

Backbone: 1e-5

Classification head: 1e-4

Weight decay: 0.05

Scheduler:

Warm-up + Cosine Learning Rate Decay.

Other:

- Gradient clipping.
- Gradient accumulation.
- Early stopping.
- Checkpointing.
- Reproducible random seeds.
- Training history.
- Validation after each epoch.

Make all settings configurable.

# 11. AUTOMATIC TRAINING IMPROVEMENT

Target:

85% or higher test accuracy.

This is a target, never a guarantee.

Monitor:

- Training loss.
- Validation loss.
- Validation accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.

If validation performance is below 85%:

1. Inspect class distribution.
2. Check label correctness.
3. Inspect corrupted videos.
4. Check preprocessing.
5. Check frame sampling.
6. Check train/validation leakage.
7. Analyse false positives.
8. Analyse false negatives.
9. Adjust learning rates.
10. Try additional fine-tuning.
11. Try appropriate augmentation.
12. Adjust regularisation.
13. Compare experiment results.

Implement controlled automatic experiments.

Use validation data to select configurations.

Keep a separate untouched test set.

Never manipulate labels, test data or evaluation results to reach 85%.

Never use training accuracy as final accuracy.

If the model cannot achieve 85%, report actual results and explain the reason.

# 12. MODEL EVALUATION

Evaluate using unseen videos.

Calculate:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Confusion matrix.
- False positive rate.
- False negative rate.
- Per-class metrics.

Fight recall is especially important.

Generate:

reports/final_metrics.json

reports/classification_report.txt

reports/confusion_matrix.png

reports/roc_curve.png

reports/training_history.png

reports/error_analysis.csv

reports/cctv_only_evaluation.json

Save test predictions and evaluation metadata.

Clearly distinguish model confidence from actual model accuracy.

# 13. VIDEO INFERENCE

Create:

inference/video_inference.py

inference/temporal_windows.py

inference/event_aggregator.py

Accept:

- MP4.
- AVI.
- MOV.
- MKV.

Processing:

1. Validate video.
2. Extract metadata.
3. Decode frames.
4. Generate overlapping temporal windows.
5. Sample 16 frames per window.
6. Run fine-tuned VideoMAE V2.
7. Calculate Fight and Non-Fight probabilities.
8. Save window-level predictions.
9. Aggregate consecutive positive windows.
10. Estimate fight start/end timestamps.

Provide configurable:

- Window duration.
- Window stride.
- Probability threshold.
- Consecutive positive window requirement.
- Alert cooldown.

Do not classify the complete video as Fight just because one temporal window is positive.

Use temporal smoothing to reduce false alarms.

# 14. STREAMLIT FRONTEND — VERY IMPORTANT

Create a complete Streamlit interface.

Main file:

app.py

Application title:

CCTV AI Fight Detection System

Use a clean, professional dashboard with clear navigation.

## PAGE 1 — DASHBOARD

Display:

- Model status.
- Selected checkpoint.
- CPU/GPU information.
- Available VRAM.
- RAM usage.
- Total processed videos.
- Total detected fight events.
- Actual evaluation metrics.

## PAGE 2 — UPLOAD VIDEO

Provide a Streamlit file uploader.

Allow the user to upload:

MP4, AVI, MOV and MKV.

After upload, display:

- Video preview.
- File name.
- Duration.
- Resolution.
- FPS.
- File size.

Provide a button:

ANALYSE VIDEO

When clicked:

- Load the selected model.
- Analyse the video.
- Display progress.
- Show processing status.
- Display errors if something fails.

Do not reload the model for every video unnecessarily. Cache the loaded model safely.

## PAGE 3 — DETECTION RESULTS

Show:

< FIGHT DETECTED >

or

< NO FIGHT DETECTED >

Display:

- Fight probability.
- Non-Fight probability.
- Overall classification.
- Processing time.
- Number of fight events.
- Start timestamp.
- End timestamp.

Display a warning when a fight is detected.

## PAGE 4 — TEMPORAL TIMELINE

Display:

- Video timeline.
- Fight intervals.
- Non-fight intervals.
- Window confidence.
- Fight start and end timestamps.

Allow the user to inspect individual predictions.

## PAGE 5 — MODEL EVALUATION

Display:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Confusion matrix.
- Training history.

Use actual evaluation results only.

Clearly label these as held-out evaluation results.

## PAGE 6 — LOGS

Display recent logs.

Allow filtering by:

- INFO.
- WARNING.
- ERROR.
- CRITICAL.

Do not expose environment secrets.

## USER EXPERIENCE

The Streamlit application must:

- Handle missing checkpoints gracefully.
- Handle invalid videos.
- Handle corrupt video files.
- Display progress.
- Display informative error messages.
- Avoid unnecessary repeated model loading.
- Provide an option to download prediction results as CSV or JSON.

The application must work on both CPU and GPU according to the available device.

# 15. ALERT SYSTEM

Create:

inference/alert_manager.py

When a confirmed fight is detected:

- Display a prominent warning.
- Record timestamps.
- Save the prediction.
- Save an event summary.
- Optionally save a video clip containing the event.

Use alert cooldown to avoid repeated alerts for one continuous fight.

Keep future email/SMS integrations modular.

# 16. LOGGING SYSTEM

Create:

logs/application.log

logs/training.log

logs/inference.log

logs/errors.log

logs/hardware.log

logs/agent_diagnostics.log

logs/experiments.log

Use Python logging.

Include:

- Timestamp.
- Log level.
- Module.
- Function.
- Message.
- Exception traceback.

Log all important operations.

Use rotating log files.

Never silently suppress exceptions.

Never log API keys, passwords or secrets.

# 17. AI-ASSISTED ERROR DIAGNOSTICS

Create:

diagnostics/error_analyzer.py

diagnostics/report_generator.py

The system must be able to inspect:

- Recent errors.
- Tracebacks.
- Failed video files.
- Dataset errors.
- Model loading problems.
- Training failures.
- CUDA OOM errors.
- Inference errors.

Generate:

reports/diagnostic_report.md

reports/error_cases.csv

The diagnostic agent must:

1. Read the relevant logs.
2. Identify the failing module.
3. Explain the likely cause.
4. Suggest a correction.
5. Generate a proposed code patch where possible.
6. Save the proposed correction.

CRITICAL:

Do not allow an AI agent to autonomously modify production code.

Do not execute arbitrary generated code.

Require developer approval before applying changes.

If no external LLM API is configured, provide a rule-based diagnostic fallback.

The fight detection system must not require an LLM API.

# 18. PROJECT STRUCTURE

Create:

cctv-fight-detection/
|
|-- AGENT.md
|-- .env.example
|-- .gitignore
|-- requirements.txt
|-- README.md
|-- app.py
|
|-- config/
|   |-- settings.py
|   |-- labels.py
|
|-- data/
|   |-- dataset_inspector.py
|   |-- dataset_loader.py
|   |-- annotation_parser.py
|   |-- video_preprocessor.py
|   |-- dataset_splitter.py
|
|-- models/
|   |-- videomae_loader.py
|   |-- classifier.py
|   |-- checkpoint_manager.py
|
|-- training/
|   |-- train.py
|   |-- trainer.py
|   |-- evaluator.py
|   |-- experiment_manager.py
|   |-- early_stopping.py
|
|-- inference/
|   |-- video_inference.py
|   |-- temporal_windows.py
|   |-- event_aggregator.py
|   |-- alert_manager.py
|
|-- dashboard/
|   |-- components.py
|   |-- metrics_view.py
|   |-- results_view.py
|
|-- diagnostics/
|   |-- error_analyzer.py
|   |-- report_generator.py
|
|-- utils/
|   |-- logger.py
|   |-- metrics.py
|   |-- hardware_detector.py
|   |-- resource_manager.py
|   |-- reproducibility.py
|   |-- video_utils.py
|
|-- checkpoints/
|-- outputs/
|   |-- splits/
|   |-- predictions/
|   |-- events/
|
|-- reports/
|-- logs/
|-- uploads/
|
|-- tests/
    |-- test_hardware_detection.py
    |-- test_dataset.py
    |-- test_video_processing.py
    |-- test_model.py
    |-- test_inference.py
    |-- test_alerts.py

Keep modules independent and maintainable.

# 19. DEPENDENCY MANAGEMENT — NEVER INSTALL AUTOMATICALLY

This is a strict requirement.

DO NOT:

- Run pip install.
- Run conda install.
- Upgrade Python.
- Upgrade PyTorch.
- Install CUDA.
- Change NVIDIA drivers.
- Modify Windows environment variables.
- Install packages globally.
- Execute dependency installation scripts.

Only:

- Inspect existing package versions.
- Check compatibility.
- Generate requirements.txt.
- Generate installation instructions.
- Explain commands for manual execution.

Generate a tested compatibility matrix.

Use appropriate PyTorch and torchvision versions.

Do not blindly upgrade existing installations.

Create a Windows PowerShell installation guide with separate CPU and NVIDIA GPU installation instructions.

The user will choose and execute the commands manually.

Do not download datasets automatically.

Model downloads must be initiated through a documented, user-controlled preparation command.

# 20. COMMANDS TO DOCUMENT

Dataset inspection:

python -m data.dataset_inspector

Data preparation:

python -m data.dataset_splitter

Hardware inspection:

python -m utils.hardware_detector

Training:

python -m training.train

Evaluation:

python -m training.evaluator

Streamlit:

streamlit run app.py

Implement appropriate command-line argument parsing.

# 21. RESOURCE MONITORING

Continuously monitor:

- CPU usage.
- RAM usage.
- GPU usage.
- VRAM usage.
- Training time.
- Video inference time.

Provide safe configurable limits.

If the system approaches a memory limit:

- Reduce batch size if safe.
- Reduce worker count.
- Release unused memory.
- Stop safely if required.
- Log the reason.

Do not silently switch training devices.

Do not use multiple training processes.

# 22. MODEL CHECKPOINT MANAGEMENT

Save:

checkpoints/best_model.pth

checkpoints/latest_model.pth

checkpoints/experiment_001/

checkpoints/experiment_002/

Store:

- Model architecture.
- Pretrained model ID.
- Class mapping.
- Training configuration.
- Hardware configuration.
- Dataset split IDs.
- Validation metrics.
- Epoch.
- Optimizer state.
- Scheduler state.
- Random seed.
- Software versions.

Load the best approved checkpoint in Streamlit.

# 23. TESTING

Create unit tests for:

- Hardware detection.
- Device selection.
- Environment validation.
- Dataset discovery.
- Dataset label mapping.
- NTU temporal annotation parsing.
- Video decoding.
- Frame sampling.
- Model loading.
- Output tensor shape.
- Probability calculation.
- Checkpoint saving/loading.
- Temporal event aggregation.
- Alert generation.
- Streamlit inference.

Create a small synthetic video for testing the software pipeline.

Never include synthetic videos in the final accuracy evaluation.

# 24. README DOCUMENTATION

Write a complete README.md explaining:

- Project architecture.
- Why VideoMAE V2 Base was selected.
- Dataset details.
- Dataset directory configuration.
- .env configuration.
- Hardware detection.
- CPU/GPU training differences.
- Installation instructions.
- Manual dependency installation.
- Dataset inspection.
- Preprocessing.
- Fine-tuning.
- Automatic training improvement.
- Evaluation.
- Streamlit video testing.
- Logs.
- Error diagnostics.
- Hardware requirements.
- Limitations.
- Licensing.
- Future 24/7 live CCTV deployment.

Explain every important configuration setting.

Include a pipeline architecture diagram.

# 25. DEVELOPMENT ORDER

Follow these phases in order.

PHASE 1: Inspect the existing computer environment and create project structure.

PHASE 2: Create .env.example and configuration management.

PHASE 3: Implement CPU/GPU/RAM detection and safe hardware configuration.

PHASE 4: Inspect RWF-2000 and NTU CCTV-Fights directory structures.

PHASE 5: Implement dataset loaders and annotation parsing.

PHASE 6: Implement video preprocessing.

PHASE 7: Implement source-aware data splitting.

PHASE 8: Load pretrained VideoMAE V2 Base.

PHASE 9: Implement binary classification head.

PHASE 10: Implement hardware-aware fine-tuning.

PHASE 11: Implement checkpoint management.

PHASE 12: Implement independent evaluation and error analysis.

PHASE 13: Implement video inference and temporal event detection.

PHASE 14: Build the Streamlit upload and results dashboard.

PHASE 15: Implement logging and diagnostic reporting.

PHASE 16: Implement tests.

PHASE 17: Complete README.md and manual installation instructions.

After each phase:

- Inspect the generated code.
- Run available tests only if the required packages already exist.
- Review errors.
- Fix implementation mistakes.
- Record completed work.
- Never claim success without verification.

# 26. FINAL ACCEPTANCE REQUIREMENTS

The project is complete only when:

1. Both dataset paths can be configured through .env.
2. Hardware detection works.
3. Device selection is correct.
4. CPU-only mode works without CUDA.
5. GPU mode works when compatible CUDA is available.
6. Resource limits are respected.
7. Both datasets are inspected correctly.
8. NTU temporal annotations are handled correctly.
9. No source-video leakage occurs.
10. VideoMAE V2 Base loads successfully.
11. The two-class classification head works.
12. Fine-tuning can run and resume.
13. The best checkpoint is saved.
14. Independent evaluation works.
15. Real metrics are reported.
16. Streamlit starts successfully.
17. Users can upload videos.
18. Fight and Non-Fight predictions are displayed.
19. Confidence scores and timestamps are displayed.
20. Logs are saved in separate files.
21. Error diagnostics work.
22. Tests pass.
23. Manual installation documentation is complete.

Do not fabricate results.

Do not claim 85% accuracy without actual independent test evidence.

Do not automatically install any software.

Do not modify the user's computer configuration.

FINAL GOAL:

Deliver a reliable VideoMAE V2 Base CCTV Fight Detection system that automatically adapts its training configuration to available hardware, supports safe CPU/GPU operation, fine-tunes using RWF-2000 and NTU CCTV-Fights, and provides a professional Streamlit application for uploading videos and testing fight detection.

Prioritise correct implementation, CCTV generalisation, fight recall, low false alarms, reproducibility, resource safety and honest evaluation.
