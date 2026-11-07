"""
Main Training Script for CCTV AI Fight Detection.
Integrates strict laptop hardware detection, .env dataset validation,
pre-training safety reports, explicit user confirmation, and automatic
error logging into separate files for AI diagnostic analysis.
Entrypoint: python -m training.train [OPTIONS]
"""

import argparse
import sys
import traceback
from pathlib import Path

from config.settings import settings
from utils.reproducibility import seed_everything
from utils.logger import train_logger, training_error_logger
from utils.hardware_detector import detect_hardware, print_hardware_summary, generate_hardware_reports
from data.dataset_validator import DatasetValidator, print_validation_report
from diagnostics.report_generator import DiagnosticReportGenerator
from models.classifier import CCTVFightClassifier
from models.checkpoint_manager import CheckpointManager
from data.dataset_loader import FightDetectionDataset, create_dataloader
from training.trainer import CCTVTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train CCTV Fight Detection Model")
    parser.add_argument("--epochs", type=int, default=settings.epochs, help="Number of training epochs")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="Transfer learning stage (1=Head only, 2=Final blocks, 3=Full)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--yes", "-y", action="store_true", help="Automatically confirm prompt and proceed with training")
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(settings.random_seed)

    print("\n" + "#" * 75)
    print(" CCTV AI FIGHT DETECTION — PRE-TRAINING INITIALIZATION")
    print("#" * 75 + "\n")

    # =========================================================================
    # STEP 1: Rigorous Laptop Hardware Inspection & Report
    # =========================================================================
    profile = detect_hardware()
    generate_hardware_reports(profile)
    print_hardware_summary(profile)

    # =========================================================================
    # STEP 2: Strict .env and Dataset Integrity Validation
    # =========================================================================
    validator = DatasetValidator()
    validation_results = validator.validate_all()
    print_validation_report(validation_results)

    if not validation_results["overall_valid"]:
        print("[HALT] Training aborted due to dataset validation failure.")
        print("Please resolve the deficiencies listed above and re-run.\n")
        sys.exit(1)

    # =========================================================================
    # STEP 3: Check/Generate Source-Aware Dataset Splits
    # =========================================================================
    train_csv = settings.splits_dir / "train.csv"
    val_csv = settings.splits_dir / "validation.csv"

    if not train_csv.exists() or not val_csv.exists():
        print("Generating source-aware train/val/test splits (70/15/15)...")
        from data.dataset_splitter import create_splits, save_splits
        t_df, v_df, te_df = create_splits(settings.random_seed)
        save_splits(t_df, v_df, te_df)

    # =========================================================================
    # STEP 4: Explicit User Confirmation
    # =========================================================================
    if not args.yes:
        print("\n" + "-" * 75)
        try:
            user_input = input("❓ Do you want to proceed with training with the above hardware configuration and validated datasets? (yes/no): ").strip().lower()
        except EOFError:
            user_input = "no"

        if user_input not in ["yes", "y"]:
            print("\n[STOPPED] Training cancelled by user. No training was initiated.\n")
            sys.exit(0)
        print("-" * 75 + "\n")

    # =========================================================================
    # STEP 5: Datasets, Model & Training Execution with Automated Error Logging
    # =========================================================================
    execution_context = {
        "stage": args.stage,
        "epochs": args.epochs,
        "batch_size": args.batch_size or profile.recommended_batch_size,
        "device": profile.device_type,
        "hardware_scenario": profile.scenario
    }

    try:
        train_dataset = FightDetectionDataset(split_csv_path=train_csv, is_training=True)
        val_dataset = FightDetectionDataset(split_csv_path=val_csv, is_training=False)

        batch_size = args.batch_size or profile.recommended_batch_size
        train_loader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=profile.recommended_workers
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        model = CCTVFightClassifier()
        ckpt_mgr = CheckpointManager()

        if args.resume:
            model, _ = ckpt_mgr.load_checkpoint(model, Path(args.resume), device=profile.device_type)
            print(f"Resumed model state from {args.resume}")

        trainer = CCTVTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            stage=args.stage
        )
        trainer.fit()

    except Exception as e:
        tb_str = traceback.format_exc()
        diag_gen = DiagnosticReportGenerator()
        report_p = diag_gen.log_and_diagnose_training_error(
            exc=e,
            tb_str=tb_str,
            context=execution_context
        )

        print("\n" + "!" * 75)
        print("🚨 CRITICAL TRAINING ERROR DETECTED")
        print("!" * 75)
        print(f"Error Type:     {type(e).__name__}")
        print(f"Error Message:  {e}")
        print("-" * 75)
        print(f"📁 Full stack trace logged to: logs/training_errors.log")
        print(f"📄 AI Diagnostic Report saved to: {report_p}")
        print("!" * 75)
        print("Tip: An AI assistant or developer can inspect reports/training_error_diagnostic.md")
        print("to immediately read the root cause, suggested fix, and patch hint.")
        print("!" * 75 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
