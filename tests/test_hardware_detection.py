"""
Unit tests for hardware detection and dynamic configuration logic.
Supports both pytest and python -m unittest.
"""

import unittest
from utils.hardware_detector import detect_hardware, HardwareProfile
from utils.resource_manager import resource_manager

class TestHardwareDetection(unittest.TestCase):
    def test_hardware_detection_profile(self):
        profile = detect_hardware()
        self.assertIsInstance(profile, HardwareProfile)
        self.assertGreaterEqual(profile.cpu_physical_cores, 1)
        self.assertGreater(profile.total_ram_gb, 0.0)
        self.assertIn(profile.device_type, ["cpu", "cuda", "mps"])
        self.assertGreaterEqual(profile.recommended_batch_size, 1)
        self.assertTrue(profile.scenario.startswith("Scenario"))

    def test_resource_manager_thresholds(self):
        snapshot = resource_manager.get_resource_snapshot()
        self.assertIn("ram_percent", snapshot)
        self.assertIn("cpu_percent", snapshot)
        is_safe, msg = resource_manager.check_safety_thresholds()
        self.assertIsInstance(is_safe, bool)

if __name__ == "__main__":
    unittest.main()
