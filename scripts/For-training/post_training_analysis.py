"""
Post-Training Cross-Validation Analysis
========================================
Supplementary material for:
"Implementing Optimized Computer Vision Algorithm To Underwater Imagery
 For Identification and Spatial Analysis Of Epibenthic Fauna In
 Shallow Lagoon Waters"

Purpose
-------
Run this script AFTER all K-fold training runs have completed
(e.g. after dataset_Kfold_splitter-trainer.py has finished).

It reconstructs training curves from the per-epoch results.csv files
written by Ultralytics, and runs a clean model.val() pass on each
fold's best.pt weights to produce confusion matrices, PR curves, and
full per-class metrics — all without triggering the matplotlib
recursion error that affects Python 3.13+ when plots are generated
inside the Ultralytics trainer callbacks.

Outputs (written to the user-specified output directory)
--------------------------------------------------------
  fold_N_training_curves.png  — loss + mAP curves for every fold
  fold_N_val/                 — Ultralytics val output folder per fold
                                (confusion_matrix.png, PR_curve.png, ...)
  per_fold_metrics.csv        — one row per fold, all metrics
  cv_summary.csv              — mean ± CI across folds (publication table)
  cv_summary.xlsx             — same data formatted for supplementary material
  analysis_log.txt            — full console transcript

Dependencies
------------
    pip install ultralytics pandas matplotlib openpyxl numpy

Usage
-----
    python post_training_analysis.py
    Follow the interactive prompts — no command-line arguments needed.

Notes
-----
  - matplotlib is forced to the 'Agg' non-interactive backend before any
    import of ultralytics to prevent the deepcopy/observer recursion that
    crashes on Python 3.13+.
  - GPU memory is explicitly released between folds via del + gc.collect()
    + torch.cuda.empty_cache() to prevent VRAM accumulation across folds.
"""

# ── Force non-interactive matplotlib backend BEFORE any ultralytics import ──
import matplotlib
matplotlib.use("Agg")

import gc
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import yaml
from ultralytics import YOLO


# ============================================================================
# Console UI helpers  (consistent with dataset_Kfold_splitter-trainer.py)
# ============================================================================

def _hr(char: str = "─", width: int = 64) -> str:
    return char * width

def header(title: str) -> None:
    print(f"\n{_hr('═')}")
    print(f"  {title}")
    print(_hr('═'))

def section(title: str) -> None:
    print(f"\n{_hr()}")
    print(f"  {title}")
    print(_hr())

def banner(title: str) -> None:
    print(f"\n{'#' * 64}")
    print(f"#  {title}")
    print(f"{'#' * 64}")

def prompt_str(message: str, default: str = "") -> str:
    indicator = f" [default: {default}]" if default else ""
    while True:
        raw = input(f"  {message}{indicator}: ").strip().strip('"').strip("'")
        if raw:
            return raw
        if default:
            return default
        print("    ✗ Cannot be empty.")

def prompt_int(message: str, default: int, lo: int | None = None,
               hi: int | None = None) -> int:
    while True:
        raw = input(f"  {message} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            v = int(raw)
            if lo is not None and v < lo:
                print(f"    ✗ Must be >= {lo}."); continue
            if hi is not None and v > hi:
                print(f"    ✗ Must be <= {hi}."); continue
            return v
        except ValueError:
            print("    ✗ Enter a whole number.")

def prompt_float(message: str, default: float, lo: float | None = None,
                 hi: float | None = None) -> float:
    while True:
        raw = input(f"  {message} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            v = float(raw)
            if lo is not None and v < lo:
                print(f"    ✗ Must be >= {lo}."); continue
            if hi is not None and v > hi:
                print(f"    ✗ Must be <= {hi}."); continue
            return v
        except ValueError:
            print("    ✗ Enter a number.")

def prompt_yes_no(message: str, default: str = "y") -> bool:
    ind = "Y/n" if default.lower() == "y" else "y/N"
    while True:
        raw = input(f"  {message} ({ind}): ").strip().lower()
        if raw == "":
            return default.lower() == "y"
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("    ✗ Enter y or n.")

def prompt_existing_dir(message: str, default: str = "") -> Path:
    indicator = f" [default: {default}]" if default else ""
    while True:
        raw = input(f"  {message}{indicator}: ").strip().strip('"').strip("'")
        if not raw and default:
            raw = default
        p = Path(raw)
        if p.exists() and p.is_dir():
            return p
        print(f"    ✗ Directory not found: {p}")

def prompt_output_dir(message: str, default: str = "") -> Path:
    indicator = f" [default: {default}]" if default else ""
    raw = input(f"  {message}{indicator}: ").strip().strip('"').strip("'")
    if not raw and default:
        raw = default
    p = Path(raw) if raw else Path(default)
    p.mkdir(parents=True, exist_ok=True)
    return p

def prompt_choice(message: str, choices: dict) -> str:
    keys = list(choices.keys())
    print(f"\n  {message}")
    for k, desc in choices.items():
        print(f"    [{k}]  {desc}")
    while True:
        raw = input(f"  Your choice ({'/'.join(keys)}): ").strip().upper()
        if raw in keys:
            return raw
        print(f"    ✗ Enter one of: {', '.join(keys)}")


# ============================================================================
# Logging — mirrors console output to a transcript file
# ============================================================================

class TeeLogger:
    """Duplicates stdout to a log file."""
    def __init__(self, log_path: Path):
        self._terminal = sys.stdout
        self._log      = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)
        self._log.flush()

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()


# ============================================================================
# GPU helpers
# ============================================================================

def select_device() -> str:
    if torch.cuda.is_available():
        idx  = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem  = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        print(f"  [GPU] CUDA device {idx}: {name}  ({mem:.1f} GB VRAM)")
        torch.backends.cudnn.benchmark = False   # safe for variable-size val
        return str(idx)
    print("  [CPU] CUDA not available — running on CPU.")
    return "cpu"

def free_gpu(label: str = "") -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated() / 1024 ** 3
        tag  = f"  [{label}] " if label else "  "
        print(f"{tag}VRAM after cleanup: {used:.3f} GB allocated")


# ============================================================================
# Fold discovery
# ============================================================================

def discover_fold_dirs(runs_dir: Path) -> list[Path]:
    """
    Return sorted list of Ultralytics output directories that look like
    fold training runs (fold_N_training, fold_N_training2, etc.).
    """
    candidates = sorted(runs_dir.glob("fold_*training*"))
    if not candidates:
        # Fallback: any directory that contains weights/best.pt
        candidates = sorted(
            [d for d in runs_dir.iterdir()
             if d.is_dir() and (d / "weights" / "best.pt").exists()]
        )
    return candidates


def extract_fold_number(fold_dir: Path) -> int:
    """Parse fold index from directory name (e.g. fold_3_training -> 3)."""
    for part in fold_dir.name.split("_"):
        if part.isdigit():
            return int(part)
    return 0


def find_yaml_for_fold(fold_idx: int, splits_root: Path,
                        yaml_mode: str, global_yaml: Path | None) -> Path | None:
    """
    Locate the YAML config to use for validation of a given fold.

    yaml_mode:
      'per_fold'  — look for fold_N/config.yaml inside splits_root
      'global'    — use global_yaml for every fold
    """
    if yaml_mode == "per_fold":
        candidate = splits_root / f"fold_{fold_idx}" / "config.yaml"
        if candidate.exists():
            return candidate
        # Try without subfolder structure
        for p in splits_root.glob(f"*fold_{fold_idx}*/config.yaml"):
            return p
        print(f"    ✗ Per-fold YAML not found for fold {fold_idx} "
              f"(looked in {splits_root / f'fold_{fold_idx}'}). "
              f"Falling back to global YAML.")
        return global_yaml
    return global_yaml


# ============================================================================
# Training curves
# ============================================================================

# Column names Ultralytics writes to results.csv (with leading spaces stripped)
_CURVE_PANELS = [
    # (csv_column,             y-axis label,           panel title)
    ("train/box_loss",        "Loss",                  "Box Loss — Train"),
    ("val/box_loss",          "Loss",                  "Box Loss — Val"),
    ("train/cls_loss",        "Loss",                  "Cls Loss — Train"),
    ("val/cls_loss",          "Loss",                  "Cls Loss — Val"),
    ("train/dfl_loss",        "Loss",                  "DFL Loss — Train"),
    ("metrics/mAP50(B)",      "mAP",                   "mAP@50"),
    ("metrics/mAP50-95(B)",   "mAP",                   "mAP@50-95"),
    ("metrics/precision(B)",  "Score",                 "Precision"),
    ("metrics/recall(B)",     "Score",                 "Recall"),
]


def plot_training_curves(fold_dir: Path, fold_idx: int,
                          output_dir: Path) -> bool:
    """
    Read results.csv and write a multi-panel training-curve PNG.
    Returns True on success.
    """
    csv_path = fold_dir / "results.csv"
    if not csv_path.exists():
        print(f"    ✗ results.csv not found in {fold_dir} — curves skipped.")
        return False

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()   # Ultralytics adds leading spaces

    if "epoch" not in df.columns:
        print(f"    ✗ 'epoch' column missing in results.csv — curves skipped.")
        return False

    available = [p for p in _CURVE_PANELS if p[0] in df.columns]
    if not available:
        print(f"    ✗ No recognised metric columns found — curves skipped.")
        return False

    n_panels = len(available)
    n_cols   = 3
    n_rows   = (n_panels + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f"Fold {fold_idx} — Training Curves", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.45, wspace=0.35)

    for i, (col, ylabel, title) in enumerate(available):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        ax.plot(df["epoch"], df[col], linewidth=1.5, color="#1f77b4")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Annotate final value
        final_val = df[col].iloc[-1]
        ax.annotate(f"{final_val:.4f}",
                    xy=(df["epoch"].iloc[-1], final_val),
                    xytext=(-5, 6), textcoords="offset points",
                    fontsize=7, color="#d62728")

    out_path = output_dir / f"fold_{fold_idx}_training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Training curves  → {out_path.name}")
    return True


# ============================================================================
# Validation
# ============================================================================

def run_fold_validation(fold_dir: Path, fold_idx: int,
                         yaml_path: Path, val_params: dict,
                         device: str, output_dir: Path) -> dict | None:
    """
    Load best.pt and run model.val() with plots=True (safe because the
    Agg backend is already active).  Returns a metrics dict or None.
    """
    weights = fold_dir / "weights" / "best.pt"
    if not weights.exists():
        print(f"    ✗ best.pt not found in {fold_dir / 'weights'} — skipped.")
        return None
    if not yaml_path or not yaml_path.exists():
        print(f"    ✗ YAML config not found ({yaml_path}) — skipped.")
        return None

    val_out_dir = output_dir / f"fold_{fold_idx}_val"
    val_out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    try:
        results = model.val(
            data      = str(yaml_path),
            split     = val_params["split"],
            conf      = val_params["conf"],
            iou       = val_params["iou"],
            imgsz     = val_params["imgsz"],
            batch     = val_params["batch"],
            plots     = True,            # safe: Agg backend is active
            save_json = False,
            project   = str(output_dir),
            name      = f"fold_{fold_idx}_val",
            exist_ok  = True,
            device    = device,
            verbose   = True,
        )

        rd  = results.results_dict

        # Overall metrics
        metrics = {
            "fold"          : fold_idx,
            "precision_all" : float(rd.get("metrics/precision(B)", 0)),
            "recall_all"    : float(rd.get("metrics/recall(B)",    0)),
            "mAP50_all"     : float(rd.get("metrics/mAP50(B)",     0)),
            "mAP50-95_all"  : float(rd.get("metrics/mAP50-95(B)", 0)),
        }

        # Per-class metrics (results.ap_class_index aligns with results.box)
        try:
            names = results.names   # {0: 'Paranemonia sp.', ...}
            box   = results.box     # has .p, .r, .ap50, .ap per class

            if hasattr(box, "p") and box.p is not None:
                for cls_idx, cls_name in names.items():
                    safe = cls_name.replace(" ", "_").replace(".", "")
                    if cls_idx < len(box.p):
                        metrics[f"precision_{safe}"] = float(box.p[cls_idx])
                        metrics[f"recall_{safe}"]    = float(box.r[cls_idx])
                        metrics[f"mAP50_{safe}"]     = float(box.ap50[cls_idx])
                        metrics[f"mAP50-95_{safe}"]  = float(box.ap[cls_idx])
        except Exception as e:
            print(f"    ⚠  Per-class extraction failed: {e}")

        # Print summary to console
        print(f"\n    {'─'*50}")
        print(f"    Fold {fold_idx} — Validation results  [{val_params['split']} split]")
        print(f"    {'─'*50}")
        print(f"    {'Metric':<22}  {'Value':>8}")
        for k, v in metrics.items():
            if k != "fold":
                print(f"    {k:<22}  {v:>8.4f}")
        print(f"    {'─'*50}\n")

        return metrics

    except Exception as e:
        print(f"    ✗ Validation failed for fold {fold_idx}: {e}")
        return None

    finally:
        del model
        free_gpu(f"fold {fold_idx} cleanup")


# ============================================================================
# Summary & export
# ============================================================================

def compute_summary(all_metrics: list[dict]) -> pd.DataFrame:
    """
    Compute mean, std, and 95 % CI across folds for every numeric metric.
    Returns a DataFrame with one row per metric.
    """
    if not all_metrics:
        return pd.DataFrame()

    # Collect all numeric metric keys (excluding 'fold')
    keys = [k for k in all_metrics[0].keys() if k != "fold"]

    rows = []
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        if not vals:
            continue
        arr  = np.array(vals, dtype=float)
        mean = float(arr.mean())
        std  = float(arr.std())
        ci   = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        row  = {"metric": k, "mean": mean, "std": std, "ci_95": ci,
                "n_folds": len(arr)}
        for m in all_metrics:
            if k in m:
                row[f"fold_{m['fold']}"] = m[k]
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary_table(summary_df: pd.DataFrame, n_folds: int) -> None:
    banner(f"CROSS-VALIDATION SUMMARY  ({n_folds} folds)")
    print(f"  {'Metric':<28}  {'Mean':>8}  {'± CI 95%':>10}  {'Std':>8}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*10}  {'─'*8}")
    for _, row in summary_df.iterrows():
        print(f"  {row['metric']:<28}  {row['mean']:>8.4f}  "
              f"{row['ci_95']:>10.4f}  {row['std']:>8.4f}")
    print()


def export_excel(summary_df: pd.DataFrame, per_fold_df: pd.DataFrame,
                 output_dir: Path) -> Path:
    """
    Write a two-sheet Excel workbook:
      Sheet 1 — Per-fold metrics (one row per fold, one column per metric)
      Sheet 2 — CV summary table (mean ± CI, formatted for supplementary)
    """
    out_path = output_dir / "cv_metrics.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

        # ── Sheet 1: Per-fold raw metrics ───────────────────────────────────
        per_fold_df.to_excel(writer, sheet_name="Per-Fold Metrics",
                             index=False)
        ws1 = writer.sheets["Per-Fold Metrics"]

        # Auto-width columns
        for col_cells in ws1.columns:
            max_len = max(len(str(c.value)) if c.value else 0 for c in col_cells)
            ws1.column_dimensions[col_cells[0].column_letter].width = max_len + 3

        # ── Sheet 2: CV summary ──────────────────────────────────────────────
        # Pivot to a publication-friendly layout: Metric | Mean ± CI | Std | Folds
        pub_rows = []
        for _, row in summary_df.iterrows():
            fold_vals = [v for k, v in row.items()
                         if str(k).startswith("fold_")]
            pub_rows.append({
                "Metric"          : row["metric"],
                "Mean"            : round(row["mean"], 4),
                "± CI 95%"        : round(row["ci_95"], 4),
                "Std"             : round(row["std"],   4),
                "N folds"         : int(row["n_folds"]),
                "Per-fold values" : ", ".join(f"{v:.4f}" for v in fold_vals),
            })
        pub_df = pd.DataFrame(pub_rows)
        pub_df.to_excel(writer, sheet_name="CV Summary", index=False)

        ws2 = writer.sheets["CV Summary"]
        for col_cells in ws2.columns:
            max_len = max(len(str(c.value)) if c.value else 0 for c in col_cells)
            ws2.column_dimensions[col_cells[0].column_letter].width = max_len + 3

    print(f"  ✓ Excel workbook      → {out_path.name}")
    return out_path


def export_csvs(summary_df: pd.DataFrame, per_fold_df: pd.DataFrame,
                output_dir: Path) -> None:
    p1 = output_dir / "per_fold_metrics.csv"
    p2 = output_dir / "cv_summary.csv"
    per_fold_df.to_csv(p1, index=False)
    summary_df.to_csv(p2,  index=False)
    print(f"  ✓ Per-fold CSV        → {p1.name}")
    print(f"  ✓ CV summary CSV      → {p2.name}")


# ============================================================================
# Interactive configuration
# ============================================================================

def ask_paths() -> dict:
    section("Directory configuration")

    runs_dir = prompt_existing_dir(
        "Ultralytics runs directory (contains fold_N_training folders)",
        default="runs/detect"
    )

    output_dir = prompt_output_dir(
        "Output directory for analysis results",
        default=str(Path("fold_analysis") /
                    datetime.now().strftime("%Y%m%d_%H%M%S"))
    )

    section("YAML configuration")
    yaml_mode_choice = prompt_choice(
        "How should config YAMLs be located for validation?",
        {
            "A": "Per-fold  — use fold_N/config.yaml from the splits directory "
                 "(created by dataset_Kfold_splitter-trainer.py)",
            "B": "Global    — use a single YAML for all folds",
        }
    )
    yaml_mode   = "per_fold" if yaml_mode_choice == "A" else "global"
    splits_root = None
    global_yaml = None

    if yaml_mode == "per_fold":
        splits_root = prompt_existing_dir(
            "Splits root directory (parent of fold_1/, fold_2/, ...)"
        )
    else:
        while True:
            raw = prompt_str("Path to the global config.yaml")
            p   = Path(raw)
            if p.exists() and p.suffix == ".yaml":
                global_yaml = p; break
            print(f"    ✗ File not found or not a .yaml: {p}")

    return {
        "runs_dir"    : runs_dir,
        "output_dir"  : output_dir,
        "yaml_mode"   : yaml_mode,
        "splits_root" : splits_root,
        "global_yaml" : global_yaml,
    }


def ask_val_params() -> dict:
    section("Validation parameters  (press Enter to accept defaults)")
    split_choice = prompt_choice(
        "Which split to run validation on?",
        {
            "1": "test  — held-out test set (recommended for final reporting)",
            "2": "val   — validation split used during training",
        }
    )
    return {
        "split" : "test" if split_choice == "1" else "val",
        "conf"  : prompt_float("Confidence threshold", default=0.25, lo=0.0, hi=1.0),
        "iou"   : prompt_float("IoU threshold",        default=0.70, lo=0.0, hi=1.0),
        "imgsz" : prompt_int("Image size (px)",        default=640,  lo=32),
        "batch" : prompt_int("Batch size",             default=8,    lo=1),
    }


def ask_analysis_options() -> dict:
    section("Analysis options")
    return {
        "do_curves"    : prompt_yes_no("Generate training curve plots?",           "y"),
        "do_val"       : prompt_yes_no("Run model.val() for confusion matrix / PR curve / metrics?", "y"),
        "do_excel"     : prompt_yes_no("Export Excel workbook for supplementary material?", "y"),
        "fold_filter"  : _ask_fold_filter(),
    }

def _ask_fold_filter() -> list[int] | None:
    if prompt_yes_no("Analyse all detected folds?", "y"):
        return None   # None means all
    raw = input("  Enter fold numbers to include (comma-separated, e.g. 1,3,5): ")
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
    except Exception:
        print("    ✗ Could not parse — analysing all folds.")
        return None


# ============================================================================
# Main
# ============================================================================

def main():
    header("Post-Training Cross-Validation Analysis")
    print("  Supplementary tool — Lagoon Epibenthos YOLO study")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}\n")

    print("  Detecting compute device ...")
    device = select_device()
    print()

    # ── Collect configuration ────────────────────────────────────────────────
    paths   = ask_paths()
    options = ask_analysis_options()
    val_params = ask_val_params() if options["do_val"] else {}

    runs_dir    = paths["runs_dir"]
    output_dir  = paths["output_dir"]
    yaml_mode   = paths["yaml_mode"]
    splits_root = paths["splits_root"]
    global_yaml = paths["global_yaml"]

    # ── Start logging to file ────────────────────────────────────────────────
    log_path = output_dir / "analysis_log.txt"
    tee      = TeeLogger(log_path)
    sys.stdout = tee

    print(f"\n  Runs dir   : {runs_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Log file   : {log_path}")

    # ── Discover fold directories ────────────────────────────────────────────
    section("Discovering fold training directories")
    fold_dirs = discover_fold_dirs(runs_dir)

    if not fold_dirs:
        print(f"  ✗ No fold training directories found in {runs_dir}.")
        print("    Expected folders named fold_N_training* containing weights/best.pt")
        sys.stdout = tee._terminal; tee.close(); return

    fold_filter = options["fold_filter"]
    if fold_filter:
        fold_dirs = [d for d in fold_dirs
                     if extract_fold_number(d) in fold_filter]

    print(f"  Found {len(fold_dirs)} fold(s):")
    for d in fold_dirs:
        has_weights = (d / "weights" / "best.pt").exists()
        has_csv     = (d / "results.csv").exists()
        status      = []
        if has_weights: status.append("weights ✓")
        if has_csv:     status.append("results.csv ✓")
        print(f"    {d.name:<40}  {', '.join(status) or 'WARNING: missing files'}")

    if not prompt_yes_no(f"\n  Proceed with {len(fold_dirs)} fold(s)?", "y"):
        print("  Aborted.")
        sys.stdout = tee._terminal; tee.close(); return

    # ── Main analysis loop ───────────────────────────────────────────────────
    all_metrics = []

    for fold_dir in fold_dirs:
        fold_idx = extract_fold_number(fold_dir)

        banner(f"FOLD {fold_idx}  |  {fold_dir.name}  |  {time.strftime('%H:%M:%S')}")

        # Training curves
        if options["do_curves"]:
            section(f"Fold {fold_idx} — Training curves")
            plot_training_curves(fold_dir, fold_idx, output_dir)

        # Validation
        if options["do_val"]:
            section(f"Fold {fold_idx} — Validation  [{val_params['split']} split]")
            yaml_path = find_yaml_for_fold(fold_idx, splits_root,
                                           yaml_mode, global_yaml)
            if yaml_path:
                metrics = run_fold_validation(fold_dir, fold_idx, yaml_path,
                                              val_params, device, output_dir)
                if metrics:
                    all_metrics.append(metrics)

        free_gpu(f"after fold {fold_idx}")

    # ── Summary & export ─────────────────────────────────────────────────────
    if all_metrics:
        section("Computing cross-validation summary")

        per_fold_df = pd.DataFrame(all_metrics)
        summary_df  = compute_summary(all_metrics)

        print_summary_table(summary_df, len(all_metrics))

        section("Saving outputs")
        export_csvs(summary_df, per_fold_df, output_dir)

        if options["do_excel"]:
            export_excel(summary_df, per_fold_df, output_dir)

    else:
        print("\n  No validation metrics collected — CSV/Excel export skipped.")

    # ── Final report ─────────────────────────────────────────────────────────
    banner("ANALYSIS COMPLETE")
    print(f"  All outputs written to: {output_dir}")
    print(f"  Log saved to          : {log_path}")
    print(f"  {_hr('═')}\n")

    sys.stdout = tee._terminal
    tee.close()


if __name__ == "__main__":
    main()