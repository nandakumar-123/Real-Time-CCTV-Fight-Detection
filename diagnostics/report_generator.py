"""
Diagnostic Report Generator.
Aggregates categorized system diagnostics and outputs reports/diagnostic_report.md
and reports/training_error_diagnostic.md so an AI coding agent or developer can
immediately read, understand the root cause, and correct errors.
"""

from pathlib import Path
import csv
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

from config.settings import settings
from diagnostics.error_analyzer import ErrorAnalyzer, DIAGNOSTIC_RULES
from utils.resource_manager import resource_manager

class DiagnosticReportGenerator:
    """Produces structured markdown and CSV reports summarizing system diagnostics."""

    def __init__(self):
        self.analyzer = ErrorAnalyzer()
        self.report_dir = settings.report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def log_and_diagnose_training_error(
        self,
        exc: Exception,
        tb_str: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Automatically invoked when a training error occurs.
        Saves full stack trace to logs/training_errors.log and produces
        reports/training_error_diagnostic.md with root-cause analysis and proposed fixes.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        res_snap = resource_manager.get_resource_snapshot()
        error_type = type(exc).__name__
        error_msg = str(exc)

        # 1. Match error against diagnostic rules
        diagnosis = self.analyzer._diagnose_line(f"{error_type}: {error_msg}\n{tb_str}")

        # 2. Write to logs/training_errors.log
        err_log_path = settings.log_dir / "training_errors.log"
        with open(err_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*75}\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write(f"EXCEPTION: {error_type}: {error_msg}\n")
            f.write(f"CATEGORY:  {diagnosis['category']}\n")
            f.write(f"HARDWARE:  RAM: {res_snap['ram_used_gb']}/{res_snap['ram_total_gb']}GB | VRAM Allocated: {res_snap['gpu_allocated_gb']}GB\n")
            if context:
                f.write(f"CONTEXT:   {context}\n")
            f.write(f"{'-'*75}\nTRACEBACK:\n{tb_str}\n{'='*75}\n")

        # 3. Generate reports/training_error_diagnostic.md
        md_path = self.report_dir / "training_error_diagnostic.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# 🚨 AI Training Error Diagnostic Report\n\n")
            f.write(f"**Occurred At:** `{timestamp}`  \n")
            f.write(f"**Exception Class:** `{error_type}`  \n")
            f.write(f"**Diagnostic Category:** `{diagnosis['category']}`\n\n")

            f.write("## 🔍 Likely Root Cause\n")
            f.write(f"{diagnosis['cause']}\n\n")

            f.write("## 💡 Recommended Correction\n")
            f.write(f"{diagnosis['suggestion']}\n\n")

            f.write("## 🛠️ Actionable Patch / Command\n")
            f.write(f"```text\n{diagnosis['patch_hint']}\n```\n\n")

            if context:
                f.write("## ⚙️ Training Execution Context\n")
                f.write(f"- **Epoch:** {context.get('epoch', 'N/A')}\n")
                f.write(f"- **Stage:** {context.get('stage', 'N/A')}\n")
                f.write(f"- **Batch Size:** {context.get('batch_size', 'N/A')}\n")
                f.write(f"- **Device:** {context.get('device', 'N/A')}\n\n")

            f.write("## 💻 System Resources at Failure\n")
            f.write(f"- **RAM Used:** {res_snap['ram_used_gb']} GB / {res_snap['ram_total_gb']} GB ({res_snap['ram_percent']}%)\n")
            f.write(f"- **GPU Memory Allocated:** {res_snap['gpu_allocated_gb']} GB\n\n")

            f.write("## 📜 Full Stack Trace\n")
            f.write("```python\n")
            f.write(tb_str)
            f.write("```\n")

        return md_path

    def generate_reports(self) -> Dict[str, Path]:
        errors = self.analyzer.scan_recent_errors()

        md_path = self.report_dir / "diagnostic_report.md"
        csv_path = self.report_dir / "error_cases.csv"

        # 1. Write CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Log File", "Category", "Likely Cause", "Suggested Fix", "Raw Excerpt"])
            for err in errors:
                writer.writerow([
                    err["log_file"],
                    err["category"],
                    err["cause"],
                    err["suggestion"],
                    err["raw_message"][:200]
                ])

        # 2. Write Markdown
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# CCTV AI Fight Detection — System Diagnostic Report\n\n")
            f.write(f"**Generated:** {timestamp}\n")
            f.write(f"**Total Error Incidents Detected:** {len(errors)}\n\n")
            f.write("> [!IMPORTANT]\n")
            f.write("> All recommended patches and environment adjustments require developer review.\n\n")

            if not errors:
                f.write("### Status: All Systems Operational\n")
                f.write("No critical errors or tracebacks detected in active system logs.\n")
            else:
                f.write("## Identified Diagnostic Findings\n\n")
                for i, err in enumerate(errors, 1):
                    f.write(f"### {i}. {err['category']} (Source: `{err['log_file']}`)\n\n")
                    f.write(f"- **Likely Cause:** {err['cause']}\n")
                    f.write(f"- **Recommended Fix:** {err['suggestion']}\n")
                    f.write(f"- **Proposed Action:** `{err['patch_hint']}`\n\n")
                    f.write("```text\n")
                    f.write(f"{err['raw_message'][:300]}\n")
                    f.write("```\n\n")

        return {"markdown_report": md_path, "csv_report": csv_path}
