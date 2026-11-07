"""
Unit tests for dataset parsing, label mappings, and splitting.
"""

from pathlib import Path
import pytest
from config.labels import LABEL_FIGHT, LABEL_NON_FIGHT, ID_TO_LABEL, LABEL_TO_ID
from data.annotation_parser import NTUAnnotationParser, VideoInterval

def test_label_mappings():
    assert ID_TO_LABEL[LABEL_FIGHT] == "Fight"
    assert ID_TO_LABEL[LABEL_NON_FIGHT] == "Non-Fight"
    assert LABEL_TO_ID["Fight"] == LABEL_FIGHT
    assert LABEL_TO_ID["Non-Fight"] == LABEL_NON_FIGHT

def test_ntu_annotation_parsing(tmp_path):
    # Create mock annotation file
    annot_dir = tmp_path / "annotations"
    annot_dir.mkdir(parents=True)
    sample_annot = annot_dir / "sample_annotations.txt"
    sample_annot.write_text("video_001 2.0 6.0\nvideo_002 0.0 4.0\n")

    parser = NTUAnnotationParser(tmp_path)
    video_durations = {"video_001.mp4": 10.0, "video_002.mp4": 5.0}
    intervals = parser.parse_annotations(video_durations)

    assert len(intervals) >= 2
    # Verify fight and non-fight segments exist
    labels = [inv.label for inv in intervals]
    assert LABEL_FIGHT in labels
    assert LABEL_NON_FIGHT in labels

def test_dataset_validator_empty_env():
    from data.dataset_validator import DatasetValidator
    validator = DatasetValidator()
    # If environment variables are empty, validate_all should detect errors
    validator.rwf_dir_raw = ""
    validator.ntu_dir_raw = ""
    res = validator.validate_all()
    assert res["overall_valid"] is False
    assert len(res["errors"]) >= 2

