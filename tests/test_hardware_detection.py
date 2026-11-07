"""
Unit tests for hardware detection and dynamic configuration logic.
"""

import pytest
from utils.hardware_detector import detect_hardware, HardwareProfile
from utils.resource_manager import resource_manager

def test_hardware_detection_profile():
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.cpu_physical_cores >= 1
    assert profile.total_ram_gb > 0.0
    assert profile.device_type in ["cpu", "cuda", "mps"]
    assert profile.recommended_batch_size >= 1
    assert profile.scenario.startswith("Scenario")

def test_resource_manager_thresholds():
    snapshot = resource_manager.get_resource_snapshot()
    assert "ram_percent" in snapshot
    assert "cpu_percent" in snapshot
    is_safe, msg = resource_manager.check_safety_thresholds()
    assert isinstance(is_safe, bool)

