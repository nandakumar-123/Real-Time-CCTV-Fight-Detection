"""
Unit tests for dataset parsing, label mappings, and splitting.
Supports both pytest and python -m unittest.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from config.labels import LABEL_FIGHT, LABEL_NON_FIGHT, ID_TO_LABEL, LABEL_TO_ID
from data.annotation_parser import NTUAnnotationParser, VideoInterval
from data.dataset_validator import DatasetValidator

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_label_mappings(self):
        self.assertEqual(ID_TO_LABEL[LABEL_FIGHT], "Fight")
        self.assertEqual(ID_TO_LABEL[LABEL_NON_FIGHT], "Non-Fight")
        self.assertEqual(LABEL_TO_ID["Fight"], LABEL_FIGHT)
        self.assertEqual(LABEL_TO_ID["Non-Fight"], LABEL_NON_FIGHT)

    def test_ntu_annotation_parsing(self):
        annot_dir = self.test_dir / "annotations"
        annot_dir.mkdir(parents=True)
        sample_annot = annot_dir / "sample_annotations.txt"
        sample_annot.write_text("video_001 2.0 6.0\nvideo_002 0.0 4.0\n")

        parser = NTUAnnotationParser(self.test_dir)
        video_durations = {"video_001.mp4": 10.0, "video_002.mp4": 5.0}
        intervals = parser.parse_annotations(video_durations)

        self.assertGreaterEqual(len(intervals), 2)
        labels = [inv.label for inv in intervals]
        self.assertIn(LABEL_FIGHT, labels)
        self.assertIn(LABEL_NON_FIGHT, labels)

    def test_dataset_validator_empty_env(self):
        validator = DatasetValidator()
        validator.rwf_dir_raw = ""
        validator.ntu_dir_raw = ""
        res = validator.validate_all()
        self.assertFalse(res["overall_valid"])
        self.assertGreaterEqual(len(res["errors"]), 2)

if __name__ == "__main__":
    unittest.main()
