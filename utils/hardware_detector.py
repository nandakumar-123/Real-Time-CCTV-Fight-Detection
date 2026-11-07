"""
Hardware Detection and Dynamic Configuration Manager.
Inspects CPU, RAM, GPU model, VRAM, CUDA availability, and PyTorch compatibility.
Never assumes NVIDIA GPU presence from CPU model (e.g. Intel i7).
Automatically selects safe GPU or conservative CPU execution parameters.
"""

import sys
import os
import json
import platform
import psutil
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

from config.settings import settings
from utils.logger import hardware_logger

@dataclass
class HardwareProfile:
    os_name: str
    os_version: str
    python_version: str
    cpu_model: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    device_type: str # 'cuda', 'mps', or 'cpu'
    is_nvidia_gpu_verified: bool
    gpu_count: int
    gpu_name: Optional[str]
    gpu_total_vram_gb: float
    gpu_available_vram_gb: float
    cuda_version: Optional[str]
    compute_capability: Optional[str]
    pytorch_version: str
    recommended_batch_size: int
    recommended_workers: int
    recommended_lr_backbone: float
    recommended_lr_head: float
    gradient_accumulation_steps: int
    use_mixed_precision: bool
    freeze_backbone_initially: bool
    scenario: str # 'Scenario A', 'Scenario B', 'Scenario C', or 'Scenario D (CPU)'

def detect_hardware() -> HardwareProfile:
    """
    Rigorously inspects laptop CPU, RAM, and GPU/CUDA environments.
    Strictly verifies NVIDIA hardware before enabling CUDA.
    """
    # 1. CPU & System metrics
    os_name = platform.system()
    os_version = platform.platform()
    python_version = platform.python_version()
    cpu_model = platform.processor() or platform.machine() or "Generic CPU"
    cpu_physical = psutil.cpu_count(logical=False) or 1
    cpu_logical = psutil.cpu_count(logical=True) or 1

    vm = psutil.virtual_memory()
    total_ram_gb = round(vm.total / (1024 ** 3), 2)
    available_ram_gb = round(vm.available / (1024 ** 3), 2)

    # 2. PyTorch & CUDA / GPU Inspection
    pytorch_version = "Not Installed"
    device_type = "cpu"
    is_nvidia_gpu_verified = False
    gpu_count = 0
    gpu_name = None
    gpu_total_vram_gb = 0.0
    gpu_available_vram_gb = 0.0
    cuda_version = None
    compute_capability = None

    try:
        import torch
        pytorch_version = torch.__version__

        # Critical Check: PyTorch CUDA availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            raw_name = torch.cuda.get_device_name(0)
            gpu_name = raw_name
            cuda_version = torch.version.cuda or "Unknown"

            # Check if actual NVIDIA device
            if "nvidia" in raw_name.lower() or "geforce" in raw_name.lower() or "quadro" in raw_name.lower() or "rtx" in raw_name.lower() or "gtx" in raw_name.lower() or "tesla" in raw_name.lower():
                is_nvidia_gpu_verified = True
                device_type = "cuda"

                cap = torch.cuda.get_device_capability(0)
                compute_capability = f"{cap[0]}.{cap[1]}"

                props = torch.cuda.get_device_properties(0)
                gpu_total_vram_gb = round(props.total_memory / (1024 ** 3), 2)
                mem_allocated = torch.cuda.memory_allocated(0)
                gpu_available_vram_gb = round((props.total_memory - mem_allocated) / (1024 ** 3), 2)
            else:
                # Detected non-NVIDIA accelerator; fallback to CPU for safety
                device_type = "cpu"
                hardware_logger.warning(f"Device '{raw_name}' is not a verified NVIDIA GPU. Defaulting to CPU mode.")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            gpu_count = 1
            gpu_name = "Apple Silicon MPS"
        else:
            device_type = "cpu"

    except ImportError:
        device_type = "cpu"
        hardware_logger.warning("PyTorch not installed. Hardware detection will report CPU default.")
    except Exception as e:
        device_type = "cpu"
        hardware_logger.warning(f"CUDA probe encountered error: {e}. Defaulting to CPU.")

    # 3. Dynamic Hardware-Aware Configuration Strategy
    if device_type == "cuda" and is_nvidia_gpu_verified:
        if gpu_total_vram_gb >= 22.0:
            scenario = "Scenario A (>= 24GB VRAM)"
            batch_size = 4
            workers = min(4, cpu_physical)
            grad_accum = 2
            mixed_prec = True
            freeze_backbone = False
            lr_backbone = 1e-5
            lr_head = 1e-4
        elif gpu_total_vram_gb >= 14.0:
            scenario = "Scenario B (16GB VRAM)"
            batch_size = 2
            workers = min(2, cpu_physical)
            grad_accum = 4
            mixed_prec = True
            freeze_backbone = True
            lr_backbone = 1e-5
            lr_head = 1e-4
        else:
            scenario = "Scenario C (8-12GB VRAM)"
            batch_size = 1
            workers = 1
            grad_accum = 8
            mixed_prec = True
            freeze_backbone = True
            lr_backbone = 5e-6
            lr_head = 1e-4
    else:
        # Scenario D: CPU Mode (Strictly conservative settings for laptop safety)
        scenario = "Scenario D (CPU Mode - Conservative)"
        batch_size = 1
        workers = 0 # 0 workers prevents multiprocessing locks/freezes on Windows CPU
        grad_accum = 16
        mixed_prec = False
        freeze_backbone = True
        lr_backbone = 1e-6
        lr_head = 1e-4

    # Allow manual override from settings if user explicitly requested
    if settings.batch_size != "auto":
        try:
            batch_size = int(settings.batch_size)
        except ValueError:
            pass

    if settings.num_workers != "auto":
        try:
            workers = int(settings.num_workers)
        except ValueError:
            pass

    return HardwareProfile(
        os_name=os_name,
        os_version=os_version,
        python_version=python_version,
        cpu_model=cpu_model,
        cpu_physical_cores=cpu_physical,
        cpu_logical_cores=cpu_logical,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        device_type=device_type,
        is_nvidia_gpu_verified=is_nvidia_gpu_verified,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_total_vram_gb=gpu_total_vram_gb,
        gpu_available_vram_gb=gpu_available_vram_gb,
        cuda_version=cuda_version,
        compute_capability=compute_capability,
        pytorch_version=pytorch_version,
        recommended_batch_size=batch_size,
        recommended_workers=workers,
        recommended_lr_backbone=lr_backbone,
        recommended_lr_head=lr_head,
        gradient_accumulation_steps=grad_accum,
        use_mixed_precision=mixed_prec,
        freeze_backbone_initially=freeze_backbone,
        scenario=scenario
    )

def print_hardware_summary(profile: HardwareProfile):
    """Prints a clear, formatted terminal report of system hardware."""
    print("=" * 75)
    print("💻 LAPTOP HARDWARE & ACCELERATION PROFILE")
    print("=" * 75)
    print(f"  Operating System:    {profile.os_name} ({profile.os_version})")
    print(f"  Python Version:      {profile.python_version}")
    print(f"  PyTorch Version:     {profile.pytorch_version}")
    print(f"  CPU Model:           {profile.cpu_model}")
    print(f"  CPU Cores:           {profile.cpu_physical_cores} Physical / {profile.cpu_logical_cores} Logical")
    print(f"  System RAM:          {profile.available_ram_gb} GB free / {profile.total_ram_gb} GB total")
    print("-" * 75)
    if profile.device_type == "cuda" and profile.is_nvidia_gpu_verified:
        print(f"  NVIDIA GPU Detected: YES")
        print(f"  GPU Model Name:      {profile.gpu_name}")
        print(f"  Total VRAM:          {profile.gpu_total_vram_gb} GB")
        print(f"  Available VRAM:      {profile.gpu_available_vram_gb} GB")
        print(f"  CUDA Version:        {profile.cuda_version}")
        print(f"  Compute Capability:  {profile.compute_capability}")
    else:
        print(f"  NVIDIA GPU Detected: NO (or CUDA not enabled in PyTorch)")
        print(f"  Note:                Intel i7/CPU does NOT provide CUDA acceleration.")
        print(f"  Selected Device:     CPU MODE (Conservative Memory Settings Active)")
    print("-" * 75)
    print(f"  Hardware Strategy:   {profile.scenario}")
    print(f"  Target Device:       {profile.device_type.upper()}")
    print(f"  Batch Size:          {profile.recommended_batch_size}")
    print(f"  DataLoader Workers:  {profile.recommended_workers}")
    print(f"  Gradient Accum Steps:{profile.gradient_accumulation_steps}")
    print(f"  Backbone Mode:       {'Frozen (Head Training)' if profile.freeze_backbone_initially else 'Trainable'}")
    print("=" * 75)

def generate_hardware_reports(profile: HardwareProfile) -> None:
    """Saves reports to reports/hardware_report.json and .txt."""
    rep_dir = settings.report_dir
    rep_dir.mkdir(parents=True, exist_ok=True)

    json_path = rep_dir / "hardware_report.json"
    txt_path = rep_dir / "hardware_report.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(profile), f, indent=4)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("CCTV AI FIGHT DETECTION — HARDWARE INSPECTION REPORT\n")
        f.write("================================================================================\n")
        f.write(f"OS: {profile.os_name} | Python: {profile.python_version} | PyTorch: {profile.pytorch_version}\n")
        f.write(f"CPU: {profile.cpu_model} ({profile.cpu_physical_cores} cores)\n")
        f.write(f"RAM: {profile.available_ram_gb} GB available / {profile.total_ram_gb} GB total\n")
        f.write(f"Device: {profile.device_type.upper()}\n")
        f.write(f"GPU: {profile.gpu_name or 'None (CPU Mode)'}\n")
        f.write(f"Total VRAM: {profile.gpu_total_vram_gb} GB\n")
        f.write(f"CUDA: {profile.cuda_version or 'N/A'}\n")
        f.write(f"Scenario: {profile.scenario}\n")
        f.write("================================================================================\n")

if __name__ == "__main__":
    prof = detect_hardware()
    print_hardware_summary(prof)
    generate_hardware_reports(prof)
