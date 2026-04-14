"""
YOLO K-Fold Dataset Splitter & Cross-Validation Trainer
========================================================
Supplementary material for:
"Implementing Optimized Computer Vision Algorithm To Underwater Imagery
 For Identification and Spatial Analysis Of Epibenthic Fauna In
 Shallow Lagoon Waters"

Three operating modes (selected interactively at runtime):
  A) Split only      — partition dataset into k stratified folds
                       (train / val / test), write per-fold YOLO YAML configs.
  B) Split + Train   — as above, then fine-tune YOLOv11s on every fold and
                       report aggregated cross-validation metrics.
  C) Train + Validate — train on an already-split dataset and run model.val()
                        on each fold's test split, producing per-fold and
                        overall test-set performance reports.

Split logic
-----------
A FIXED test set (sized to your requested test ratio) is held out first and
is SHARED across all folds — it is never seen during training or validation.
The remaining images are then partitioned by KFold into k train/val pairs.
This is the statistically correct design: the test set is genuinely held out
from the entire cross-validation process.

  Example — 91 images, 5 folds, 70/20/10:
    test   = 9  images (fixed, same for every fold)
    val    ≈ 16 images (rotates — 1/k of the remaining 82)
    train  ≈ 66 images (the rest of the remaining 82)
    Actual ratios ≈ 72.5% / 17.6% / 9.9%  ✓

Dependencies
------------
    pip install ultralytics scikit-learn numpy pyyaml tqdm

Usage
-----
    python dataset_Kfold_splitter-trainer.py
    Follow the interactive console prompts — no command-line arguments needed.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# GPU / device selection
# ---------------------------------------------------------------------------

def select_device() -> str:
    """
    Return the best available compute device string for Ultralytics.
    Printed once at startup; the result is passed to model.train() / model.val().

    Returns
    -------
    '0'   — first CUDA GPU  (fastest)
    'cpu' — fallback
    """
    if torch.cuda.is_available():
        idx  = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem  = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        print(f"  [GPU] CUDA device {idx}: {name}  ({mem:.1f} GB VRAM)")
        # cuDNN autotuner: selects fastest conv algorithm for the fixed image size.
        torch.backends.cudnn.benchmark = True
        return str(idx)       # e.g. '0'
    else:
        print("  [CPU] CUDA not available — running on CPU (slow).")
        return "cpu" 


# ---------------------------------------------------------------------------
# Console UI helpers
# ---------------------------------------------------------------------------

def _hr(char: str = "─", width: int = 60) -> str:
    return char * width


def header(title: str) -> None:
    print(f"\n{_hr('═')}")
    print(f"  {title}")
    print(_hr('═'))


def section(title: str) -> None:
    print(f"\n{_hr()}")
    print(f"  {title}")
    print(_hr())


def prompt_str(message: str, default: str = "") -> str:
    indicator = f" [default: {default}]" if default else ""
    while True:
        raw = input(f"  {message}{indicator}: ").strip().strip('"').strip("'")
        if raw:
            return raw
        if default:
            return default
        print("    ✗ Value cannot be empty. Please try again.")


def prompt_int(message: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    while True:
        raw = input(f"  {message} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if lo is not None and val < lo:
                print(f"    ✗ Must be >= {lo}.")
                continue
            if hi is not None and val > hi:
                print(f"    ✗ Must be <= {hi}.")
                continue
            return val
        except ValueError:
            print("    ✗ Please enter a whole number.")


def prompt_float(message: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    while True:
        raw = input(f"  {message} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if lo is not None and val < lo:
                print(f"    ✗ Must be >= {lo}.")
                continue
            if hi is not None and val > hi:
                print(f"    ✗ Must be <= {hi}.")
                continue
            return val
        except ValueError:
            print("    ✗ Please enter a number.")


def prompt_yes_no(message: str, default: str = "y") -> bool:
    indicator = "Y/n" if default.lower() == "y" else "y/N"
    while True:
        raw = input(f"  {message} ({indicator}): ").strip().lower()
        if raw == "":
            return default.lower() == "y"
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("    ✗ Please enter y or n.")


def prompt_existing_dir(message: str) -> Path:
    while True:
        raw = prompt_str(message)
        p = Path(raw)
        if p.exists() and p.is_dir():
            return p
        print(f"    ✗ Directory not found: {p}")


def prompt_output_dir(message: str, default: str = "") -> Path:
    raw = prompt_str(message, default=default)
    p = Path(raw)
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
        print(f"    ✗ Please enter one of: {', '.join(keys)}")


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"    ✗ Could not load YAML {path}: {e}")
        raise


def write_fold_yaml(fold_path: Path, fold_number: int,
                    class_names: dict, class_weights: dict) -> Path:
    config = {
        "path":       str(fold_path),
        "train":      "images/train",
        "val":        "images/val",
        "test":       "images/test",
        "nc":         len(class_names),
        "names":      class_names,
        "weights":    class_weights,
        "single_cls": False,
        "rect":       False,
        "fold":       fold_number,
    }
    yaml_path = fold_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


# ---------------------------------------------------------------------------
# Dataset splitting  (FIXED ratio logic)
# ---------------------------------------------------------------------------

def collect_images(dataset_path: Path) -> list:
    img_dir = dataset_path / "images"
    if not img_dir.exists():
        raise FileNotFoundError(
            f"Expected an 'images/' subdirectory inside {dataset_path}."
        )
    files = sorted(
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.jpeg")) +
        list(img_dir.glob("*.png"))
    )
    if not files:
        raise ValueError(f"No images (jpg/jpeg/png) found in {img_dir}")
    return files


def copy_split(file_list: list, images_dst: Path, labels_dst: Path) -> tuple:
    """Copy images and matching YOLO .txt labels. Returns (copied, missing)."""
    copied = missing = 0
    for img in tqdm(file_list,
                    desc=f"  -> {images_dst.parent.name}/{images_dst.name}",
                    leave=False, unit="file"):
        shutil.copy2(img, images_dst / img.name)
        label = img.parent.parent / "labels" / img.with_suffix(".txt").name
        if label.exists():
            shutil.copy2(label, labels_dst / label.name)
            copied += 1
        else:
            missing += 1
    return copied, missing


def create_k_fold_splits(
        dataset_path:  Path,
        output_path:   Path,
        n_splits:      int,
        train_ratio:   float,
        val_ratio:     float,
        seed:          int,
        class_names:   dict,
        class_weights: dict,
) -> list:
    """
    Partition *dataset_path* into k folds with correct train/val/test ratios.

    Strategy
    --------
    1. A FIXED test set (test_ratio % of all images) is held out once using
       train_test_split.  It is IDENTICAL across all folds.
    2. The remaining images are partitioned by KFold into k train/val pairs.

    This ensures:
      - The test set is never touched during training or validation.
      - The val set rotates so each image is validated exactly once.
      - Reported ratios honour the user's requested split.

    Each fold directory receives:
      images/{train, val, test}/
      labels/{train, val, test}/
      config.yaml
      README.txt

    Returns a list of fold root Paths.
    """
    image_files = collect_images(dataset_path)
    n = len(image_files)
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)

    print(f"\n  Found {n} images.")
    print(f"  Requested split  : {train_ratio:.0%} train / "
          f"{val_ratio:.0%} val / {test_ratio:.0%} test")

    # ── Step 1: carve out the fixed test set ─────────────────────────────
    all_idx      = np.arange(n)
    trainval_idx, test_idx = train_test_split(
        all_idx, test_size=test_ratio, random_state=seed, shuffle=True
    )
    test_files = [image_files[i] for i in test_idx]

    n_tv = len(trainval_idx)
    n_t  = len(test_idx)
    # val fraction WITHIN the trainval pool that matches the requested ratio
    val_fraction_of_tv = val_ratio / (train_ratio + val_ratio)

    print(f"  Fixed test set   : {n_t} images  "
          f"(actual {n_t/n:.1%})")
    print(f"  Train+val pool   : {n_tv} images  "
          f"(KFold will rotate val at ~{val_fraction_of_tv:.0%} of this pool "
          f"= ~{round(n_tv * val_fraction_of_tv)} images per fold)")

    # ── Step 2: KFold on the trainval pool ───────────────────────────────
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_paths = []

    for fold_idx, (inner_train_pos, inner_val_pos) in enumerate(
            kf.split(np.arange(n_tv)), start=1):

        print(f"\n  Processing fold {fold_idx}/{n_splits} ...")

        train_files = [image_files[trainval_idx[i]] for i in inner_train_pos]
        val_files   = [image_files[trainval_idx[i]] for i in inner_val_pos]

        # ── Directory layout ─────────────────────────────────────────────
        fold_path = output_path / f"fold_{fold_idx}"
        for split in ("train", "val", "test"):
            (fold_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (fold_path / "labels" / split).mkdir(parents=True, exist_ok=True)

        copy_split(train_files, fold_path / "images/train", fold_path / "labels/train")
        copy_split(val_files,   fold_path / "images/val",   fold_path / "labels/val")
        copy_split(test_files,  fold_path / "images/test",  fold_path / "labels/test")

        write_fold_yaml(fold_path, fold_idx, class_names, class_weights)

        with open(fold_path / "README.txt", "w") as f:
            f.write(f"K-Fold split {fold_idx} of {n_splits}\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Train  images : {len(train_files)}  "
                    f"({len(train_files)/n:.1%} of total)\n")
            f.write(f"Val    images : {len(val_files)}  "
                    f"({len(val_files)/n:.1%} of total)\n")
            f.write(f"Test   images : {len(test_files)}  "
                    f"({len(test_files)/n:.1%} of total)  [fixed across all folds]\n\n")
            f.write(f"Requested split: {train_ratio:.0%} / "
                    f"{val_ratio:.0%} / {test_ratio:.0%}\n\n")
            f.write("Class weights:\n")
            for idx, name in class_names.items():
                f.write(f"  {name} (class {idx}): {class_weights[idx]}\n")

        print(f"  Fold {fold_idx}: "
              f"{len(train_files)} train ({len(train_files)/n:.1%}) | "
              f"{len(val_files)} val ({len(val_files)/n:.1%}) | "
              f"{len(test_files)} test ({len(test_files)/n:.1%})")
        fold_paths.append(fold_path)

    print(f"\n  All {n_splits} folds written to: {output_path}")
    return fold_paths


# ---------------------------------------------------------------------------
# Training & Validation
# ---------------------------------------------------------------------------

def train_single_fold(fold_idx: int, yaml_path: Path,
                      output_path: Path, train_params: dict,
                      device: str = "cpu") -> dict | None:
    print(f"\n{'─' * 60}")
    print(f"  Training fold {fold_idx} ...")
    print(f"{'─' * 60}")

    model = YOLO("yolo11s.pt")
    params = dict(train_params)
    params["data"]   = str(yaml_path)
    params["name"]   = f"fold_{fold_idx}_training"
    params["seed"]   = train_params.get("seed", 42) + fold_idx
    params["device"] = device          # GPU/CPU

    try:
        results = model.train(**params)
    except Exception as e:
        print(f"  ✗ Training fold {fold_idx} failed: {e}")
        return None

    if results is None or not hasattr(results, "results_dict"):
        print(f"  ✗ Could not extract metrics from fold {fold_idx}.")
        return None

    rd = results.results_dict
    metrics = {
        "fold":      fold_idx,
        "mAP50":     float(rd.get("metrics/mAP50(B)", 0)),
        "mAP50-95":  float(rd.get("metrics/mAP50-95(B)", 0)),
        "precision": float(rd.get("metrics/precision(B)", 0)),
        "recall":    float(rd.get("metrics/recall(B)", 0)),
    }

    with open(output_path / f"fold_{fold_idx}_train_results.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"  Fold {fold_idx} — mAP50: {metrics['mAP50']:.4f} | "
          f"mAP50-95: {metrics['mAP50-95']:.4f} | "
          f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")
    return metrics


def validate_on_test(fold_idx: int, yaml_path: Path, weights_path: Path,
                     output_path: Path, val_params: dict,
                     device: str = "cpu") -> dict | None:
    print(f"\n{'─' * 60}")
    print(f"  Validating fold {fold_idx} on test split ...")
    print(f"{'─' * 60}")

    if not weights_path.exists():
        print(f"  ✗ Weights not found: {weights_path}")
        return None

    model = YOLO(str(weights_path))
    try:
        results = model.val(data=str(yaml_path), split="test",
                              device=device, **val_params)
    except Exception as e:
        print(f"  ✗ Validation fold {fold_idx} failed: {e}")
        return None

    rd = results.results_dict
    metrics = {
        "fold":      fold_idx,
        "mAP50":     float(rd.get("metrics/mAP50(B)", 0)),
        "mAP50-95":  float(rd.get("metrics/mAP50-95(B)", 0)),
        "precision": float(rd.get("metrics/precision(B)", 0)),
        "recall":    float(rd.get("metrics/recall(B)", 0)),
    }

    with open(output_path / f"fold_{fold_idx}_test_val_results.txt", "w") as f:
        f.write(f"TEST SET VALIDATION — Fold {fold_idx}\n")
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"  Fold {fold_idx} TEST — mAP50: {metrics['mAP50']:.4f} | "
          f"mAP50-95: {metrics['mAP50-95']:.4f} | "
          f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")
    return metrics


def summarise_cv(all_metrics: list, output_path: Path, phase: str = "cv") -> None:
    if not all_metrics:
        print("  No results to summarise.")
        return

    header("Cross-validation summary")
    keys = ["mAP50", "mAP50-95", "precision", "recall"]
    summary = {}

    for k in keys:
        vals = [m[k] for m in all_metrics]
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        ci   = 1.96 * std / np.sqrt(len(vals))
        summary[k] = {"mean": mean, "std": std, "ci": ci, "values": vals}
        print(f"  {k:<12}  {mean:.4f} +/- {ci:.4f}  "
              f"(std {std:.4f}  |  folds: {', '.join(f'{v:.4f}' for v in vals)})")

    report_path = output_path / f"{phase}_summary.txt"
    with open(report_path, "w") as f:
        f.write(f"YOLO {phase.upper()} SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Folds completed : {len(all_metrics)}\n")
        f.write(f"Model           : yolo11s.pt\n\n")
        for k, d in summary.items():
            f.write(f"{k}:\n")
            f.write(f"  Mean : {d['mean']:.4f} +/- {d['ci']:.4f}\n")
            f.write(f"  Std  : {d['std']:.4f}\n")
            f.write(f"  Per fold: {', '.join(f'{v:.4f}' for v in d['values'])}\n\n")
    print(f"\n  Summary saved -> {report_path}")


# ---------------------------------------------------------------------------
# Interactive configuration
# ---------------------------------------------------------------------------

def ask_class_setup() -> tuple:
    section("Class configuration")
    use_defaults = prompt_yes_no(
        "Use default classes (Paranemonia / Anemonia / Brachyura)?", default="y"
    )
    if use_defaults:
        class_names = {0: "Paranemonia sp.", 1: "Anemonia sp.", 2: "Brachyura sp."}
    else:
        n_cls = prompt_int("Number of classes", default=3, lo=1, hi=80)
        class_names = {i: prompt_str(f"Name for class {i}", default=f"class_{i}")
                       for i in range(n_cls)}

    section("Class weights  (higher = model penalised more for missing that class)")
    use_default_w = prompt_yes_no(
        "Use default weights  1 : 15 : 20  (Paranemonia : Anemonia : Brachyura)?",
        default="y"
    )
    if use_default_w and len(class_names) == 3:
        class_weights = {0: 1.0, 1: 15.0, 2: 20.0}
    else:
        class_weights = {
            idx: prompt_float(f"Weight for '{name}' (class {idx})", default=1.0, lo=0.1)
            for idx, name in class_names.items()
        }

    print("\n  Class configuration:")
    for idx, name in class_names.items():
        print(f"    [{idx}]  {name}  — weight {class_weights[idx]}")
    return class_names, class_weights


def ask_split_ratios() -> tuple:
    section("Train / Val / Test split ratios")
    presets = {"1": "70 / 20 / 10  (default)", "2": "80 / 10 / 10", "3": "Custom"}
    choice = prompt_choice("Select a split preset:", presets)
    if choice == "1":
        return 0.70, 0.20
    elif choice == "2":
        return 0.80, 0.10
    else:
        while True:
            train_r = prompt_float("Train ratio (e.g. 0.70)", default=0.70, lo=0.1, hi=0.9)
            val_r   = prompt_float("Val   ratio (e.g. 0.20)", default=0.20, lo=0.05, hi=0.9)
            test_r  = round(1.0 - train_r - val_r, 6)
            if test_r < 0.01:
                print(f"    ✗ Test ratio would be {test_r:.2%} — too small. Try again.")
                continue
            print(f"    -> Test ratio will be: {test_r:.0%}")
            return train_r, val_r


def ask_train_params() -> dict:
    section("Training hyperparameters  (press Enter to accept defaults)")
    return {
        "epochs":        prompt_int("Epochs",                        default=200,    lo=1),
        "batch":         prompt_int("Batch size",                     default=16,     lo=1),
        "imgsz":         prompt_int("Image size (px)",                default=640,    lo=32),
        "patience":      prompt_int("Early-stop patience (0 = off)",  default=50,     lo=0),
        "optimizer":     "AdamW",
        "lr0":           prompt_float("Initial LR",                   default=0.00084, lo=1e-6),
        "lrf":           prompt_float("Final LR fraction",            default=0.00964, lo=1e-6),
        "weight_decay":  prompt_float("Weight decay",                 default=0.0002, lo=0.0),
        "dropout":       prompt_float("Dropout",                      default=0.25,   lo=0.0, hi=0.9),
        "cos_lr":        True,
        "multi_scale":   True,
        "cache":         "disk",
        "deterministic": True,
        "amp":           True,
        "pretrained":    True,
        "seed":          42,
    }


def ask_val_params() -> dict:
    section("Validation parameters")
    return {
        "imgsz": prompt_int("Image size (px)",         default=640,  lo=32),
        "batch": prompt_int("Batch size",               default=16,   lo=1),
        "conf":  prompt_float("Confidence threshold",   default=0.25, lo=0.0, hi=1.0),
        "iou":   prompt_float("IoU threshold",          default=0.70, lo=0.0, hi=1.0),
    }


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def mode_split_only():
    header("MODE A — Create K-fold splits only")
    dataset_path   = prompt_existing_dir("Dataset path (must contain images/ and labels/)")
    output_path    = prompt_output_dir("Output directory for k-fold splits")
    n_splits       = prompt_int("Number of folds", default=5, lo=2, hi=20)
    seed           = prompt_int("Random seed",     default=42)
    train_r, val_r = ask_split_ratios()
    class_names, class_weights = ask_class_setup()

    print(f"\n  Ready to create {n_splits} folds from {dataset_path}")
    if not prompt_yes_no("Proceed?"):
        print("  Aborted.")
        return

    create_k_fold_splits(
        dataset_path=dataset_path, output_path=output_path,
        n_splits=n_splits, train_ratio=train_r, val_ratio=val_r,
        seed=seed, class_names=class_names, class_weights=class_weights,
    )


def mode_split_and_train():
    header("MODE B — Create K-fold splits and train")
    dataset_path   = prompt_existing_dir("Dataset path (must contain images/ and labels/)")
    output_path    = prompt_output_dir("Output directory for folds and results")
    n_splits       = prompt_int("Number of folds", default=5, lo=2, hi=20)
    seed           = prompt_int("Random seed",     default=42)
    train_r, val_r = ask_split_ratios()
    class_names, class_weights = ask_class_setup()
    train_params   = ask_train_params()

    print(f"\n  Ready to create {n_splits} folds and train on each.")
    if not prompt_yes_no("Proceed?"):
        print("  Aborted.")
        return

    device = select_device()

    fold_paths = create_k_fold_splits(
        dataset_path=dataset_path, output_path=output_path,
        n_splits=n_splits, train_ratio=train_r, val_ratio=val_r,
        seed=seed, class_names=class_names, class_weights=class_weights,
    )

    all_metrics = []
    for fold_idx, fold_path in enumerate(fold_paths, start=1):
        m = train_single_fold(fold_idx, fold_path / "config.yaml",
                              output_path, train_params, device=device)
        if m:
            all_metrics.append(m)

    summarise_cv(all_metrics, output_path, phase="cross_validation_train")


def mode_train_and_validate():
    header("MODE C — Train on existing splits and validate on test sets")
    splits_root = prompt_existing_dir(
        "Directory containing fold_1/, fold_2/, ... subdirectories"
    )
    detected = sorted(splits_root.glob("fold_*"))
    if not detected:
        print("  ✗ No fold_* directories found. Aborting.")
        return
    print(f"  Detected {len(detected)} fold(s).")

    output_path  = prompt_output_dir("Output directory for results", default=str(splits_root))
    train_params = ask_train_params()
    val_params   = ask_val_params()

    print(f"\n  Ready to train {len(detected)} folds and validate each on its test split.")
    if not prompt_yes_no("Proceed?"):
        print("  Aborted.")
        return

    device = select_device()

    train_metrics = []
    val_metrics   = []

    for fold_idx, fold_path in enumerate(detected, start=1):
        yaml_path = fold_path / "config.yaml"
        if not yaml_path.exists():
            print(f"  ✗ config.yaml missing in {fold_path} — skipping.")
            continue

        tm = train_single_fold(fold_idx, yaml_path, output_path, train_params,
                               device=device)
        if tm:
            train_metrics.append(tm)

        # Locate best weights (standard Ultralytics output path)
        weights_path = (
            Path("runs") / "detect" / f"fold_{fold_idx}_training" / "weights" / "best.pt"
        )
        if not weights_path.exists():
            candidates = list(output_path.glob(f"*fold_{fold_idx}*/weights/best.pt"))
            if candidates:
                weights_path = candidates[0]

        vm = validate_on_test(fold_idx, yaml_path, weights_path, output_path,
                              val_params, device=device)
        if vm:
            val_metrics.append(vm)

    if train_metrics:
        summarise_cv(train_metrics, output_path, phase="cross_validation_train")
    if val_metrics:
        summarise_cv(val_metrics,   output_path, phase="test_set_validation")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    header("YOLO K-Fold Dataset Splitter & Cross-Validation Trainer")
    print("  Supplementary tool — Lagoon Epibenthos YOLO study\n")
    print("  Detecting compute device ...")
    select_device()   # informational — actual device is passed per-mode
    print()
    print("  NOTE: a single fixed test set is held out first (your requested")
    print("  test %), then KFold rotates the remaining images as train/val.")
    print("  This ensures the test set is never seen during CV training.\n")

    mode = prompt_choice(
        "Select operating mode:",
        {
            "A": "Split only       — partition dataset into k folds, write YAML configs",
            "B": "Split + Train    — split dataset, then train YOLOv11s on every fold",
            "C": "Train + Validate — train on an existing split, validate on test sets",
        }
    )

    if mode == "A":
        mode_split_only()
    elif mode == "B":
        mode_split_and_train()
    elif mode == "C":
        mode_train_and_validate()

    print(f"\n{_hr('=')}")
    print("  Done.")
    print(_hr('=') + "\n")


if __name__ == "__main__":
    main()