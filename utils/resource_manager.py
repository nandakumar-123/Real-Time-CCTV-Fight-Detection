"""
System Resource Manager and OOM Protection.
Monitors memory and compute usage, manages thread limits, and implements
graceful recovery procedures during Out-Of-Memory (OOM) events.
"""

import gc
import psutil
from typing import Dict, Any, Tuple

from config.settings import settings
from utils.logger import hardware_logger, app_logger

class ResourceManager:
    """Monitors CPU/GPU resource utilization and applies safety boundaries."""
    
    def __init__(self):
        self.max_ram_percent = settings.max_ram_percent
        self.max_vram_percent = settings.max_vram_percent
        self.max_cpu_threads = settings.max_cpu_threads
        self._configure_thread_limits()

    def _configure_thread_limits(self):
        """Restricts PyTorch CPU thread count to prevent freezing system multitasking."""
        try:
            import torch
            torch.set_num_threads(min(self.max_cpu_threads, psutil.cpu_count(logical=True) or 4))
        except ImportError:
            pass

    @staticmethod
    def get_resource_snapshot() -> Dict[str, Any]:
        """Returns instantaneous snapshot of CPU, RAM, and GPU memory."""
        vm = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)

        snapshot = {
            "ram_used_gb": round((vm.total - vm.available) / (1024 ** 3), 2),
            "ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "ram_percent": vm.percent,
            "cpu_percent": cpu_pct,
            "gpu_allocated_gb": 0.0,
            "gpu_reserved_gb": 0.0,
            "gpu_percent": 0.0
        }

        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                props = torch.cuda.get_device_properties(0)
                total = props.total_memory / (1024 ** 3)
                snapshot["gpu_allocated_gb"] = round(allocated, 2)
                snapshot["gpu_reserved_gb"] = round(reserved, 2)
                snapshot["gpu_percent"] = round((reserved / total) * 100, 1) if total > 0 else 0.0
        except Exception:
            pass

        return snapshot

    def check_safety_thresholds(self) -> Tuple[bool, str]:
        """
        Validates if resource usage is within safe operational limits.
        Returns (is_safe, message).
        """
        snapshot = self.get_resource_snapshot()

        if snapshot["ram_percent"] > self.max_ram_percent:
            msg = f"RAM usage ({snapshot['ram_percent']}%) exceeds safety limit ({self.max_ram_percent}%)."
            hardware_logger.warning(msg)
            return False, msg

        if snapshot["gpu_percent"] > self.max_vram_percent:
            msg = f"VRAM usage ({snapshot['gpu_percent']}%) exceeds safety limit ({self.max_vram_percent}%)."
            hardware_logger.warning(msg)
            return False, msg

        return True, "Resources within safe boundaries."

    @staticmethod
    def release_memory():
        """Aggressively flushes Python garbage collector and PyTorch CUDA caching."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

resource_manager = ResourceManager()

