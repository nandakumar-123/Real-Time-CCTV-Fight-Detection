"""
Dataset Path and Integrity Validator.
Rigorously checks RWF-2000 and NTU CCTV-Fights dataset directories,
verifying file existence, folder structures, codecs, and temporal annotations.
Halts execution with detailed problem reports and corrections if invalid.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2

from config.settings import settings
from utils.video_utils import SUPPORTED_EXTENSIONS
from data.annotation_parser import NTUAnnotationParser

class DatasetValidationError(Exception):
    """Custom exception raised when dataset directory or structure validation fails."""
    pass

class DatasetValidator:
    """Performs deep validation of RWF-2000 and NTU CCTV-Fights datasets."""

    def __init__(self):
        self.rwf_dir_raw = os.getenv("RWF2000_DATA_DIR", "").strip()
        self.ntu_dir_raw = os.getenv("NTU_CCTV_FIGHTS_DATA_DIR", "").strip()
        self.validation_log = []

    def validate_all(self) -> Dict[str, Any]:
        """
        Executes all validation checks.
        Returns a dict of validation results, or raises DatasetValidationError on fatal error.
        """
        results = {
            "rwf2000": {"valid": False, "details": {}},
            "ntu_cctv_fights": {"valid": False, "details": {}},
            "overall_valid": False,
            "errors": []
        }

        # 1. Check if paths are provided in .env
        if not self.rwf_dir_raw:
            msg = (
                "[VALIDATION ERROR: RWF2000_DATA_DIR is empty in .env]\n"
                "  Problem: The environment variable RWF2000_DATA_DIR has no value.\n"
                "  Correction: Open .env and specify your local RWF-2000 path.\n"
                "  Example: RWF2000_DATA_DIR=C:/Datasets/RWF-2000"
            )
            results["errors"].append(msg)

        if not self.ntu_dir_raw:
            msg = (
                "[VALIDATION ERROR: NTU_CCTV_FIGHTS_DATA_DIR is empty in .env]\n"
                "  Problem: The environment variable NTU_CCTV_FIGHTS_DATA_DIR has no value.\n"
                "  Correction: Open .env and specify your local NTU CCTV-Fights path.\n"
                "  Example: NTU_CCTV_FIGHTS_DATA_DIR=C:/Datasets/NTU-CCTV-Fights"
            )
            results["errors"].append(msg)

        if results["errors"]:
            return results

        # 2. Validate RWF-2000
        rwf_path = Path(self.rwf_dir_raw)
        rwf_valid, rwf_info, rwf_errs = self._validate_rwf2000(rwf_path)
        results["rwf2000"]["valid"] = rwf_valid
        results["rwf2000"]["details"] = rwf_info
        results["errors"].extend(rwf_errs)

        # 3. Validate NTU CCTV-Fights
        ntu_path = Path(self.ntu_dir_raw)
        ntu_valid, ntu_info, ntu_errs = self._validate_ntu(ntu_path)
        results["ntu_cctv_fights"]["valid"] = ntu_valid
        results["ntu_cctv_fights"]["details"] = ntu_info
        results["errors"].extend(ntu_errs)

        results["overall_valid"] = (rwf_valid and ntu_valid and len(results["errors"]) == 0)
        return results

    def _validate_rwf2000(self, path: Path) -> Tuple[bool, Dict[str, Any], List[str]]:
        info = {
            "path": str(path.resolve()) if path.exists() else str(path),
            "exists": False,
            "total_videos": 0,
            "fight_videos": 0,
            "non_fight_videos": 0,
            "sample_decode_ok": False
        }
        errors = []

        if not path.exists():
            errors.append(
                f"[RWF-2000 Path Not Found]\n"
                f"  Tested Path: '{path}'\n"
                f"  Problem: Directory does not exist on disk.\n"
                f"  Correction: Verify the folder location in Windows Explorer and update RWF2000_DATA_DIR in .env."
            )
            return False, info, errors

        info["exists"] = True

        # Scan for video files and class structure
        video_files = []
        fight_count = 0
        non_fight_count = 0

        for root, _, files in os.walk(path):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    video_files.append(p)
                    p_str = str(p).lower()
                    if "fight" in p_str and "nonfight" not in p_str and "non_fight" not in p_str:
                        fight_count += 1
                    else:
                        non_fight_count += 1

        info["total_videos"] = len(video_files)
        info["fight_videos"] = fight_count
        info["non_fight_videos"] = non_fight_count

        if len(video_files) == 0:
            errors.append(
                f"[RWF-2000 No Videos Found]\n"
                f"  Directory: '{path}'\n"
                f"  Problem: No supported video files (.mp4, .avi, .mkv, .mov) were found.\n"
                f"  Correction: Ensure RWF-2000 clips are extracted into this directory."
            )
            return False, info, errors

        if fight_count == 0 or non_fight_count == 0:
            errors.append(
                f"[RWF-2000 Label Distribution Missing]\n"
                f"  Found: {fight_count} Fight, {non_fight_count} Non-Fight clips.\n"
                f"  Problem: Both Fight and Non-Fight categories must be identifiable in the folder structure (e.g., Fight/ and NonFight/ folders).\n"
                f"  Correction: Ensure folder structure matches standard RWF-2000 (train/Fight, train/NonFight, val/Fight, val/NonFight)."
            )
            return False, info, errors

        # Test video decode on first sample
        sample_v = video_files[0]
        cap = cv2.VideoCapture(str(sample_v))
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                info["sample_decode_ok"] = True
            else:
                errors.append(f"Failed to decode frames from sample RWF-2000 video: {sample_v.name}")
        else:
            errors.append(f"OpenCV could not open sample RWF-2000 video: {sample_v.name}")

        is_valid = len(errors) == 0
        return is_valid, info, errors

    def _validate_ntu(self, path: Path) -> Tuple[bool, Dict[str, Any], List[str]]:
        info = {
            "path": str(path.resolve()) if path.exists() else str(path),
            "exists": False,
            "total_videos": 0,
            "annotation_files_found": 0,
            "parsed_intervals_count": 0,
            "sample_decode_ok": False
        }
        errors = []

        if not path.exists():
            errors.append(
                f"[NTU CCTV-Fights Path Not Found]\n"
                f"  Tested Path: '{path}'\n"
                f"  Problem: Directory does not exist on disk.\n"
                f"  Correction: Verify the folder location in Windows Explorer and update NTU_CCTV_FIGHTS_DATA_DIR in .env."
            )
            return False, info, errors

        info["exists"] = True

        # Scan for videos
        video_files = []
        for root, _, files in os.walk(path):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    video_files.append(p)

        info["total_videos"] = len(video_files)
        if len(video_files) == 0:
            errors.append(
                f"[NTU CCTV-Fights No Videos Found]\n"
                f"  Directory: '{path}'\n"
                f"  Problem: No supported video files (.mp4, .avi, .mkv, .mov) found.\n"
                f"  Correction: Ensure NTU CCTV-Fights footage is located in this directory."
            )
            return False, info, errors

        # Scan for annotations
        annot_files = list(path.glob("**/*.txt")) + list(path.glob("**/*.csv"))
        info["annotation_files_found"] = len(annot_files)

        if len(annot_files) == 0:
            errors.append(
                f"[NTU CCTV-Fights Missing Annotations]\n"
                f"  Directory: '{path}'\n"
                f"  Problem: No temporal annotation files (.txt or .csv) found in the dataset folder.\n"
                f"  Correction: Provide the NTU CCTV-Fights timestamp annotation files in the directory or an 'annotations/' subfolder."
            )
            return False, info, errors

        # Test annotation parsing
        parser = NTUAnnotationParser(path)
        dummy_durations = {v.name: 10.0 for v in video_files[:5]}
        intervals = parser.parse_annotations(dummy_durations)
        info["parsed_intervals_count"] = len(intervals)

        if len(intervals) == 0:
            errors.append(
                f"[NTU CCTV-Fights Annotation Parse Failure]\n"
                f"  Problem: Could not extract valid temporal fight intervals from annotation files.\n"
                f"  Correction: Verify annotation format (must contain video identifier, start time, and end time)."
            )
            return False, info, errors

        # Test video decode on first sample
        sample_v = video_files[0]
        cap = cv2.VideoCapture(str(sample_v))
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                info["sample_decode_ok"] = True
            else:
                errors.append(f"Failed to decode frames from sample NTU video: {sample_v.name}")
        else:
            errors.append(f"OpenCV could not open sample NTU video: {sample_v.name}")

        is_valid = len(errors) == 0
        return is_valid, info, errors

def print_validation_report(results: Dict[str, Any]):
    """Prints a clear, formatted terminal validation report."""
    print("=" * 75)
    print("📁 DATASET DIRECTORY & INTEGRITY VALIDATION REPORT")
    print("=" * 75)

    # RWF-2000
    rwf = results["rwf2000"]
    print(f"Dataset 1: RWF-2000")
    print(f"  Configured Path:    {rwf['details'].get('path', 'Not Set')}")
    print(f"  Directory Exists:   {'YES' if rwf['details'].get('exists') else 'NO'}")
    print(f"  Videos Discovered:  {rwf['details'].get('total_videos', 0)} ({rwf['details'].get('fight_videos', 0)} Fight, {rwf['details'].get('non_fight_videos', 0)} Non-Fight)")
    print(f"  Video Decode Test:  {'PASSED' if rwf['details'].get('sample_decode_ok') else 'FAILED'}")
    print(f"  Validation Status:  {'✅ VALID' if rwf['valid'] else '❌ INVALID'}")
    print("-" * 75)

    # NTU CCTV-Fights
    ntu = results["ntu_cctv_fights"]
    print(f"Dataset 2: NTU CCTV-Fights")
    print(f"  Configured Path:    {ntu['details'].get('path', 'Not Set')}")
    print(f"  Directory Exists:   {'YES' if ntu['details'].get('exists') else 'NO'}")
    print(f"  Videos Discovered:  {ntu['details'].get('total_videos', 0)}")
    print(f"  Annotation Files:   {ntu['details'].get('annotation_files_found', 0)}")
    print(f"  Temporal Intervals: {ntu['details'].get('parsed_intervals_count', 0)} intervals parsed")
    print(f"  Video Decode Test:  {'PASSED' if ntu['details'].get('sample_decode_ok') else 'FAILED'}")
    print(f"  Validation Status:  {'✅ VALID' if ntu['valid'] else '❌ INVALID'}")
    print("=" * 75)

    if results["errors"]:
        print("\n🚨 VALIDATION DEFICIENCIES DETECTED:")
        for i, err in enumerate(results["errors"], 1):
            print(f"\n{i}. {err}")
        print("\n" + "=" * 75)
        print("ACTION REQUIRED: Correct the issues above in your .env file or dataset directories.")
        print("Training cannot proceed until all datasets are validated.")
        print("=" * 75 + "\n")
    else:
        print("\n✅ ALL DATASETS SUCCESSFULLY VALIDATED! Ready for training.\n")

if __name__ == "__main__":
    validator = DatasetValidator()
    res = validator.validate_all()
    print_validation_report(res)
    if not res["overall_valid"]:
        sys.exit(1)

