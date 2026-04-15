"""
YOLO Dataset Splitter & Cross-Validation Trainer
=================================================
Supplementary material for:
"Implementing Optimized Computer Vision Algorithm To Underwater Imagery
 For Identification and Spatial Analysis Of Epibenthic Fauna In
 Shallow Lagoon Waters"

Operating modes
---------------
  A  Split only       — create splits and YAML configs, no training
  B  Split + Train    — create splits then train YOLOv11s on every fold
  C  Train + Validate — train on an already-split dataset folder

Split strategies (asked interactively in modes A and B)
-------------------------------------------------------
  1  K-fold, static test set
       A fixed test set (your requested %) is held out once and is SHARED
       across all folds.  The remainder is partitioned by KFold so that every
       image appears in the val split exactly once.
       Best for rigorous cross-validation reporting.

  2  K-fold, no test set
       Pure train/val K-fold — no test split is created.
       Use when your test set lives in a separate dataset.

  3  Fully random, static test set
       A fixed test set is drawn once.  Each of the N splits then
       independently samples a random train/val split from the remainder.
       Splits may overlap — intended for training-stability experiments.

  4  Fully random, per-fold test sets
       Each of the N splits independently draws its own random
       train/val/test partition.  Every fold is completely independent.

All strategies write the same fold_N/ directory structure so every output
is compatible with the trainer in modes B and C.

Dependencies
------------
    pip install ultralytics scikit-learn numpy pyyaml tqdm

Usage
-----
    python dataset_Kfold_splitter-trainer.py
    Follow the interactive prompts — no command-line arguments needed.
"""

import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
from ultralytics import YOLO
import sys
sys.setrecursionlimit(5000)   # default is 1000; Ultralytics+matplotlib needs more


# ============================================================================
# GPU detection
# ============================================================================

def select_device() -> str:
    """
    Detect and return the best available Ultralytics device string.
    Enables cuDNN benchmark mode when CUDA is present.
    """
    if torch.cuda.is_available():
        idx  = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem  = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        print(f"  [GPU] CUDA device {idx}: {name}  ({mem:.1f} GB VRAM)")
          
        return str(idx)
    print("  [CPU] CUDA not available — running on CPU (slow).")
    return "cpu"


# ============================================================================
# Console UI helpers
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
    """Prominent banner — easy to spot when scrolling / copy-pasting."""
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

def prompt_int(message: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    while True:
        raw = input(f"  {message} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            v = int(raw)
            if lo is not None and v < lo:
                print(f"    ✗ Must be >= {lo}.")
                continue
            if hi is not None and v > hi:
                print(f"    ✗ Must be <= {hi}.")
                continue
            return v
        except ValueError:
            print("    ✗ Enter a whole number.")

def prompt_float(message: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    while True:
        raw = input(f"  {message} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            v = float(raw)
            if lo is not None and v < lo:
                print(f"    ✗ Must be >= {lo}.")
                continue
            if hi is not None and v > hi:
                print(f"    ✗ Must be <= {hi}.")
                continue
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

def prompt_existing_dir(message: str) -> Path:
    while True:
        raw = prompt_str(message)
        p = Path(raw)
        if p.exists() and p.is_dir():
            return p
        print(f"    ✗ Not found: {p}")

def prompt_output_dir(message: str, default: str = "") -> Path:
    p = Path(prompt_str(message, default=default))
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
# YAML helpers
# ============================================================================

def write_fold_yaml(fold_path: Path, fold_number: int,
                    class_names: dict, class_weights: dict,
                    has_test: bool = True) -> Path:
    """
    Write a YOLO-compatible data config for one fold.
    If has_test is False the 'test' key is omitted so YOLO does not
    try to evaluate a non-existent split.
    """
    config = {
        "path":       str(fold_path),
        "train":      "images/train",
        "val":        "images/val",
        "nc":         len(class_names),
        "names":      class_names,
        "weights":    class_weights,
        "single_cls": False,
        "rect":       False,
        "fold":       fold_number,
    }
    if has_test:
        config["test"] = "images/test"
    yaml_path = fold_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


# ============================================================================
# File I/O helpers
# ============================================================================

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


def copy_split(files: list, img_dst: Path, lbl_dst: Path) -> tuple:
    """Copy images + YOLO label files.  Returns (copied, missing_labels)."""
    copied = missing = 0
    for img in files:
        shutil.copy2(img, img_dst / img.name)
        label = img.parent.parent / "labels" / img.with_suffix(".txt").name
        if label.exists():
            shutil.copy2(label, lbl_dst / label.name)
            copied += 1
        else:
            missing += 1
    return copied, missing


def make_split_dirs(fold_path: Path, splits: list) -> None:
    for s in splits:
        (fold_path / "images" / s).mkdir(parents=True, exist_ok=True)
        (fold_path / "labels" / s).mkdir(parents=True, exist_ok=True)


def write_readme(fold_path: Path, fold_idx: int, n_splits: int,
                 counts: dict, ratios: dict, strategy: str,
                 class_names: dict, class_weights: dict) -> None:
    with open(fold_path / "README.txt", "w") as f:
        f.write(f"Fold {fold_idx} of {n_splits}  —  strategy: {strategy}\n")
        f.write("=" * 48 + "\n\n")
        total = sum(counts.values())
        for split, n in counts.items():
            note = "  [fixed across all folds]" if split == "test" and "static" in strategy else ""
            f.write(f"  {split:<6}: {n:>4} images  ({n/total:.1%}){note}\n")
        f.write(f"\n  Requested ratios: "
                f"train {ratios['train']:.0%} / val {ratios['val']:.0%}"
                + (f" / test {ratios.get('test', 0):.0%}" if "test" in ratios else "") + "\n\n")
        f.write("Class weights:\n")
        for idx, name in class_names.items():
            f.write(f"  {name} (class {idx}): {class_weights[idx]}\n")


# ============================================================================
# Split strategy implementations
# ============================================================================

def _progress_label(fold_idx: int, n_splits: int, strategy: str) -> str:
    return f"Fold {fold_idx}/{n_splits}  [{strategy}]"


def create_kfold_static_test(dataset_path, output_path, n_splits,
                              train_ratio, val_ratio, seed,
                              class_names, class_weights) -> list:
    """
    Strategy 1 — K-fold with a single fixed test set.
    Test set is drawn first; KFold rotates train/val over the remainder.
    Every image appears in val exactly once.
    """
    images    = collect_images(dataset_path)
    n         = len(images)
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)

    print(f"\n  Images: {n}  |  Strategy: K-fold + static test set")
    print(f"  Requested: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")

    all_idx       = np.arange(n)
    trainval_idx, test_idx = train_test_split(
        all_idx, test_size=test_ratio, random_state=seed, shuffle=True
    )
    test_files = [images[i] for i in test_idx]

    n_tv              = len(trainval_idx)
    val_frac_of_tv    = val_ratio / (train_ratio + val_ratio)
    print(f"  Fixed test : {len(test_idx)} images  ({len(test_idx)/n:.1%})")
    print(f"  Train+val pool : {n_tv} images  (val rotates at ~{val_frac_of_tv:.0%})")

    kf         = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_paths = []

    for fold_idx, (tr_pos, va_pos) in enumerate(kf.split(np.arange(n_tv)), start=1):
        print(f"\n  {_progress_label(fold_idx, n_splits, 'kfold-static-test')} ...")
        train_files = [images[trainval_idx[i]] for i in tr_pos]
        val_files   = [images[trainval_idx[i]] for i in va_pos]

        fold_path = output_path / f"fold_{fold_idx}"
        make_split_dirs(fold_path, ["train", "val", "test"])
        copy_split(train_files, fold_path / "images/train", fold_path / "labels/train")
        copy_split(val_files,   fold_path / "images/val",   fold_path / "labels/val")
        copy_split(test_files,  fold_path / "images/test",  fold_path / "labels/test")
        write_fold_yaml(fold_path, fold_idx, class_names, class_weights, has_test=True)
        write_readme(fold_path, fold_idx, n_splits,
                     {"train": len(train_files), "val": len(val_files), "test": len(test_files)},
                     {"train": train_ratio, "val": val_ratio, "test": test_ratio},
                     "kfold-static-test", class_names, class_weights)
        print(f"  -> {len(train_files)} train ({len(train_files)/n:.1%}) | "
              f"{len(val_files)} val ({len(val_files)/n:.1%}) | "
              f"{len(test_files)} test ({len(test_files)/n:.1%})")
        fold_paths.append(fold_path)

    return fold_paths


def create_kfold_no_test(dataset_path, output_path, n_splits,
                         train_ratio, val_ratio, seed,
                         class_names, class_weights) -> list:
    """
    Strategy 2 — Pure K-fold, no test split.
    All images participate in train/val rotation.
    """
    images = collect_images(dataset_path)
    n      = len(images)
    print(f"\n  Images: {n}  |  Strategy: K-fold, no test set")
    print(f"  Each fold: ~{train_ratio:.0%} train / ~{val_ratio:.0%} val  "
          f"(KFold uses 1/{n_splits} = {1/n_splits:.0%} as val)")

    kf         = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_idx    = np.arange(n)
    fold_paths = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(all_idx), start=1):
        print(f"\n  {_progress_label(fold_idx, n_splits, 'kfold-no-test')} ...")
        train_files = [images[i] for i in tr_idx]
        val_files   = [images[i] for i in va_idx]

        fold_path = output_path / f"fold_{fold_idx}"
        make_split_dirs(fold_path, ["train", "val"])
        copy_split(train_files, fold_path / "images/train", fold_path / "labels/train")
        copy_split(val_files,   fold_path / "images/val",   fold_path / "labels/val")
        write_fold_yaml(fold_path, fold_idx, class_names, class_weights, has_test=False)
        write_readme(fold_path, fold_idx, n_splits,
                     {"train": len(train_files), "val": len(val_files)},
                     {"train": train_ratio, "val": val_ratio},
                     "kfold-no-test", class_names, class_weights)
        print(f"  -> {len(train_files)} train ({len(train_files)/n:.1%}) | "
              f"{len(val_files)} val ({len(val_files)/n:.1%})")
        fold_paths.append(fold_path)

    return fold_paths


def create_random_static_test(dataset_path, output_path, n_splits,
                               train_ratio, val_ratio, seed,
                               class_names, class_weights) -> list:
    """
    Strategy 3 — Fully random splits with a single static test set.
    Test is drawn once. Each of the N splits independently samples a
    random train/val partition from the remainder (splits may overlap).
    """
    images     = collect_images(dataset_path)
    n          = len(images)
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)

    print(f"\n  Images: {n}  |  Strategy: random splits + static test set")
    print(f"  Requested: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")

    all_idx               = np.arange(n)
    trainval_idx, test_idx = train_test_split(
        all_idx, test_size=test_ratio, random_state=seed, shuffle=True
    )
    test_files = [images[i] for i in test_idx]
    print(f"  Fixed test : {len(test_idx)} images  ({len(test_idx)/n:.1%})")

    fold_paths  = []
    val_frac_tv = val_ratio / (train_ratio + val_ratio)

    for fold_idx in range(1, n_splits + 1):
        print(f"\n  {_progress_label(fold_idx, n_splits, 'random-static-test')} ...")
        tr_pos, va_pos = train_test_split(
            trainval_idx,
            test_size=val_frac_tv,
            random_state=seed + fold_idx,
            shuffle=True,
        )
        train_files = [images[i] for i in tr_pos]
        val_files   = [images[i] for i in va_pos]

        fold_path = output_path / f"fold_{fold_idx}"
        make_split_dirs(fold_path, ["train", "val", "test"])
        copy_split(train_files, fold_path / "images/train", fold_path / "labels/train")
        copy_split(val_files,   fold_path / "images/val",   fold_path / "labels/val")
        copy_split(test_files,  fold_path / "images/test",  fold_path / "labels/test")
        write_fold_yaml(fold_path, fold_idx, class_names, class_weights, has_test=True)
        write_readme(fold_path, fold_idx, n_splits,
                     {"train": len(train_files), "val": len(val_files), "test": len(test_files)},
                     {"train": train_ratio, "val": val_ratio, "test": test_ratio},
                     "random-static-test", class_names, class_weights)
        print(f"  -> {len(train_files)} train ({len(train_files)/n:.1%}) | "
              f"{len(val_files)} val ({len(val_files)/n:.1%}) | "
              f"{len(test_files)} test ({len(test_files)/n:.1%})")
        fold_paths.append(fold_path)

    return fold_paths


def create_random_per_fold_test(dataset_path, output_path, n_splits,
                                train_ratio, val_ratio, seed,
                                class_names, class_weights) -> list:
    """
    Strategy 4 — Fully random, independent test set per fold.
    Each fold independently draws its own train/val/test partition.
    Splits may overlap.  Maximises independence between folds.
    """
    images     = collect_images(dataset_path)
    n          = len(images)
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)

    print(f"\n  Images: {n}  |  Strategy: fully random, per-fold test sets")
    print(f"  Each fold independently: {train_ratio:.0%} train / "
          f"{val_ratio:.0%} val / {test_ratio:.0%} test")

    all_idx     = np.arange(n)
    val_frac_tv = val_ratio / (train_ratio + val_ratio)
    fold_paths  = []

    for fold_idx in range(1, n_splits + 1):
        print(f"\n  {_progress_label(fold_idx, n_splits, 'random-per-fold-test')} ...")
        tv_idx, te_idx = train_test_split(
            all_idx,
            test_size=test_ratio,
            random_state=seed + fold_idx * 100,
            shuffle=True,
        )
        tr_idx, va_idx = train_test_split(
            tv_idx,
            test_size=val_frac_tv,
            random_state=seed + fold_idx * 100 + 1,
            shuffle=True,
        )
        train_files = [images[i] for i in tr_idx]
        val_files   = [images[i] for i in va_idx]
        test_files  = [images[i] for i in te_idx]

        fold_path = output_path / f"fold_{fold_idx}"
        make_split_dirs(fold_path, ["train", "val", "test"])
        copy_split(train_files, fold_path / "images/train", fold_path / "labels/train")
        copy_split(val_files,   fold_path / "images/val",   fold_path / "labels/val")
        copy_split(test_files,  fold_path / "images/test",  fold_path / "labels/test")
        write_fold_yaml(fold_path, fold_idx, class_names, class_weights, has_test=True)
        write_readme(fold_path, fold_idx, n_splits,
                     {"train": len(train_files), "val": len(val_files), "test": len(test_files)},
                     {"train": train_ratio, "val": val_ratio, "test": test_ratio},
                     "random-per-fold-test", class_names, class_weights)
        print(f"  -> {len(train_files)} train ({len(train_files)/n:.1%}) | "
              f"{len(val_files)} val ({len(val_files)/n:.1%}) | "
              f"{len(test_files)} test ({len(test_files)/n:.1%})")
        fold_paths.append(fold_path)

    return fold_paths


def create_splits(split_cfg: dict) -> list:
    """Dispatch to the correct split strategy based on split_cfg."""
    strategy = split_cfg["strategy"]
    common   = {k: split_cfg[k] for k in
                ("dataset_path", "output_path", "n_splits",
                 "train_ratio", "val_ratio", "seed",
                 "class_names", "class_weights")}
    if strategy == 1:
        return create_kfold_static_test(**common)
    elif strategy == 2:
        return create_kfold_no_test(**common)
    elif strategy == 3:
        return create_random_static_test(**common)
    elif strategy == 4:
        return create_random_per_fold_test(**common)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# Training & Validation
# ============================================================================

def _print_fold_banner(fold_idx: int, n_folds: int, phase: str) -> None:
    """Prominent per-fold marker — easy to locate when scrolling the console."""
    ts = time.strftime("%H:%M:%S")
    banner(f"FOLD {fold_idx}/{n_folds}  |  {phase.upper()}  |  {ts}")


def _print_metrics_block(label: str, metrics: dict) -> None:
    """
    Print a clearly delimited metrics block that is easy to copy-paste.
    """
    print(f"\n{'─' * 64}")
    print(f"  METRICS  —  {label}")
    print(f"{'─' * 64}")
    for k, v in metrics.items():
        if k != "fold":
            print(f"    {k:<16} {v:.4f}")
    print(f"{'─' * 64}\n")


def train_single_fold(fold_idx: int, n_folds: int, yaml_path: Path,
                      train_params: dict, device: str) -> dict | None:
    _print_fold_banner(fold_idx, n_folds, "TRAINING")

    model  = YOLO("yolo11s.pt")
    params = dict(train_params)
    params["data"]   = str(yaml_path)
    params["name"]   = f"fold_{fold_idx}_training"
    params["seed"]   = train_params.get("seed", 42) + fold_idx
    params["device"] = device

    try:
        results = model.train(**params)
    except Exception as e:
        print(f"  ✗ Training fold {fold_idx} failed: {e}")
        return None

    if results is None or not hasattr(results, "results_dict"):
        print(f"  ✗ Metrics unavailable for fold {fold_idx}.")
        return None

    rd = results.results_dict
    metrics = {
        "fold":      fold_idx,
        "mAP50":     float(rd.get("metrics/mAP50(B)",    0)),
        "mAP50-95":  float(rd.get("metrics/mAP50-95(B)", 0)),
        "precision": float(rd.get("metrics/precision(B)", 0)),
        "recall":    float(rd.get("metrics/recall(B)",    0)),
    }
    _print_metrics_block(f"Fold {fold_idx}/{n_folds} — VAL split (end of training)", metrics)
    return metrics


def validate_on_test(fold_idx: int, n_folds: int,
                     yaml_path: Path, weights_path: Path,
                     val_params: dict, device: str,
                     output_path: Path) -> dict | None:
    _print_fold_banner(fold_idx, n_folds, "TEST SET VALIDATION")

    if not weights_path.exists():
        print(f"  ✗ Weights not found: {weights_path}")
        return None

    # Confirm the config actually has a test split before running
    cfg = {}
    try:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    if "test" not in cfg:
        print("  ✗ config.yaml has no 'test' key — this fold has no test split. Skipping.")
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
        "mAP50":     float(rd.get("metrics/mAP50(B)",    0)),
        "mAP50-95":  float(rd.get("metrics/mAP50-95(B)", 0)),
        "precision": float(rd.get("metrics/precision(B)", 0)),
        "recall":    float(rd.get("metrics/recall(B)",    0)),
    }
    _print_metrics_block(f"Fold {fold_idx}/{n_folds} — TEST split", metrics)

    log = output_path / f"fold_{fold_idx}_test_validation.txt"
    with open(log, "w") as f:
        f.write(f"TEST SET VALIDATION — Fold {fold_idx}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Weights : {weights_path}\n")
        f.write(f"Config  : {yaml_path}\n\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")
    return metrics


def find_weights(fold_idx: int, output_path: Path) -> Path:
    """Locate best.pt written by Ultralytics for a given fold."""
    standard = (Path("runs") / "detect" /
                f"fold_{fold_idx}_training" / "weights" / "best.pt")
    if standard.exists():
        return standard
    candidates = list(output_path.glob(f"*fold_{fold_idx}*/weights/best.pt"))
    if candidates:
        return candidates[0]
    return standard   # return non-existent path; caller handles missing file


def summarise(all_metrics: list, output_path: Path, phase: str) -> None:
    if not all_metrics:
        print("  No results to summarise.")
        return

    banner(f"SUMMARY — {phase.upper()}")
    keys = ["mAP50", "mAP50-95", "precision", "recall"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        ci   = 1.96 * std / np.sqrt(len(vals))
        summary[k] = {"mean": mean, "std": std, "ci": ci, "values": vals}
        print(f"  {k:<16} {mean:.4f} +/- {ci:.4f}  "
              f"(std {std:.4f}  |  folds: {', '.join(f'{v:.4f}' for v in vals)})")

    out = output_path / f"{phase}_summary.txt"
    with open(out, "w") as f:
        f.write(f"YOLO — {phase.upper()}\n")
        f.write("=" * 48 + "\n\n")
        f.write(f"Folds : {len(all_metrics)}\n")
        f.write(f"Model : yolo11s.pt\n\n")
        for k, d in summary.items():
            f.write(f"{k}:\n")
            f.write(f"  Mean  : {d['mean']:.4f} +/- {d['ci']:.4f}\n")
            f.write(f"  Std   : {d['std']:.4f}\n")
            f.write(f"  Folds : {', '.join(f'{v:.4f}' for v in d['values'])}\n\n")
    print(f"\n  Summary saved -> {out}")


# ============================================================================
# Interactive configuration helpers
# ============================================================================

def ask_class_setup() -> tuple:
    section("Class configuration")
    if prompt_yes_no("Use default classes (Paranemonia / Anemonia / Brachyura)?", "y"):
        class_names = {0: "Paranemonia sp.", 1: "Anemonia sp.", 2: "Brachyura sp."}
    else:
        n_cls = prompt_int("Number of classes", default=3, lo=1, hi=80)
        class_names = {i: prompt_str(f"Name for class {i}", default=f"class_{i}")
                       for i in range(n_cls)}

    section("Class weights  (inverse-frequency — higher = penalise more for missing)")
    if prompt_yes_no("Use default weights  1 : 15 : 20?", "y") and len(class_names) == 3:
        class_weights = {0: 1.0, 1: 15.0, 2: 20.0}
    else:
        class_weights = {
            idx: prompt_float(f"Weight for '{name}' (class {idx})", default=1.0, lo=0.1)
            for idx, name in class_names.items()
        }
    print("\n  Classes:")
    for idx, name in class_names.items():
        print(f"    [{idx}]  {name}  — weight {class_weights[idx]}")
    return class_names, class_weights


def ask_split_ratios(has_test: bool) -> tuple:
    """
    Ask for split ratios.  If has_test is False only train/val are needed.
    Returns (train_r, val_r).  test_r = 1 - train_r - val_r (or 0).
    """
    section("Split ratios")
    if has_test:
        presets = {"1": "70 / 20 / 10  (default)", "2": "80 / 10 / 10", "3": "Custom"}
        c = prompt_choice("Select preset:", presets)
        if c == "1": return 0.70, 0.20
        if c == "2": return 0.80, 0.10
        while True:
            tr = prompt_float("Train ratio (e.g. 0.70)", 0.70, lo=0.1, hi=0.9)
            va = prompt_float("Val   ratio (e.g. 0.20)", 0.20, lo=0.05, hi=0.9)
            te = round(1.0 - tr - va, 6)
            if te < 0.01:
                print(f"    ✗ Test ratio {te:.2%} too small."); continue
            print(f"    -> Test ratio: {te:.0%}")
            return tr, va
    else:
        presets = {"1": "70 / 30  (default)", "2": "80 / 20", "3": "Custom"}
        c = prompt_choice("Select preset (train / val, no test):", presets)
        if c == "1": return 0.70, 0.30
        if c == "2": return 0.80, 0.20
        while True:
            tr = prompt_float("Train ratio (e.g. 0.70)", 0.70, lo=0.1, hi=0.95)
            va = round(1.0 - tr, 6)
            print(f"    -> Val ratio: {va:.0%}")
            return tr, va


def ask_split_strategy() -> int:
    """
    Return an integer 1-4 for the desired split strategy.
    Also returns whether a test set exists (for ratio prompting).
    """
    section("Split strategy")
    print("  Choose how to partition images into train / val / test splits.\n")
    strats = {
        "1": "K-fold + static test set   (rigorous CV; test never seen during training)",
        "2": "K-fold, no test set        (pure train/val rotation; test set lives elsewhere)",
        "3": "Fully random + static test  (independent random splits; shared test set)",
        "4": "Fully random, per-fold test (independent random splits; each fold owns its test)",
    }
    c = prompt_choice("Strategy:", strats)
    return int(c)


def ask_train_params() -> dict:
    section("Training hyperparameters  (press Enter to accept defaults)")
    return {
        "epochs":        prompt_int("Epochs",                        default=100,      lo=1),
        "batch":         prompt_int("Batch size",                    default=16,      lo=1),
        "imgsz":         prompt_int("Image size (px)",               default=640,     lo=32),
        "patience":      prompt_int("Early-stop patience (0=off)",   default=0,       lo=0),
        "optimizer":     "AdamW",
        "lr0":           prompt_float("Initial LR",                  default=0.00084, lo=1e-6),
        "lrf":           prompt_float("Final LR fraction",           default=0.00964, lo=1e-6),
        "weight_decay":  prompt_float("Weight decay",                default=0.0002,  lo=0.0),
        "dropout":       prompt_float("Dropout",                     default=0.25,    lo=0.0, hi=0.9),
        "cos_lr":        True,
        "multi_scale":   False,
        "cache":         "disk",
        "deterministic": True,
        "amp":           True,
        "pretrained":    True,
        "seed":          42,
    }


def ask_val_params() -> dict:
    section("Test-validation parameters")
    return {
        "imgsz": prompt_int("Image size (px)",       default=640,  lo=32),
        "batch": prompt_int("Batch size",             default=16,   lo=1),
        "conf":  prompt_float("Confidence threshold", default=0.25, lo=0.0, hi=1.0),
        "iou":   prompt_float("IoU threshold",        default=0.70, lo=0.0, hi=1.0),
    }


# ============================================================================
# Core training loop (shared by modes B and C)
# ============================================================================

def run_training_loop(fold_paths: list, train_params: dict, device: str,
                      output_path: Path,
                      run_test_val: bool, val_params: dict) -> None:
    """
    Train on every fold.  Optionally follow each training run with a
    model.val(split='test') on the held-out test split.
    """
    n_folds       = len(fold_paths)
    train_metrics = []
    test_metrics  = []

    for fold_idx, fold_path in enumerate(fold_paths, start=1):
        yaml_path = fold_path / "config.yaml"
        if not yaml_path.exists():
            print(f"  ✗ config.yaml missing in {fold_path} — skipping.")
            continue

        tm = train_single_fold(fold_idx, n_folds, yaml_path, train_params, device)
        if tm:
            train_metrics.append(tm)

        if run_test_val:
            weights = find_weights(fold_idx, output_path)
            vm = validate_on_test(fold_idx, n_folds, yaml_path, weights,
                                  val_params, device, output_path)
            if vm:
                test_metrics.append(vm)

    if train_metrics:
        summarise(train_metrics, output_path, phase="cv_train_val_metrics")
    if test_metrics:
        summarise(test_metrics, output_path, phase="cv_test_metrics")


# ============================================================================
# Operating mode handlers
# ============================================================================

def mode_split_only():
    header("MODE A — Create splits only")

    dataset_path = prompt_existing_dir("Dataset path (must contain images/ and labels/)")
    output_path  = prompt_output_dir("Output directory for fold directories")
    n_splits     = prompt_int("Number of splits / folds", default=5, lo=2, hi=50)
    seed         = prompt_int("Random seed", default=42)
    strategy     = ask_split_strategy()
    has_test     = strategy != 2      # strategy 2 has no test set
    train_r, val_r = ask_split_ratios(has_test)
    class_names, class_weights = ask_class_setup()

    print(f"\n  Ready: {n_splits} folds from {dataset_path}")
    if not prompt_yes_no("Proceed?"):
        print("  Aborted."); return

    create_splits({
        "strategy": strategy, "dataset_path": dataset_path,
        "output_path": output_path, "n_splits": n_splits,
        "train_ratio": train_r, "val_ratio": val_r,
        "seed": seed, "class_names": class_names, "class_weights": class_weights,
    })
    print(f"\n  Done — splits written to: {output_path}")


def mode_split_and_train():
    header("MODE B — Create splits and train")

    dataset_path = prompt_existing_dir("Dataset path (must contain images/ and labels/)")
    output_path  = prompt_output_dir("Output directory for folds and training results")
    n_splits     = prompt_int("Number of splits / folds", default=5, lo=2, hi=50)
    seed         = prompt_int("Random seed", default=42)
    strategy     = ask_split_strategy()
    has_test     = strategy != 2
    train_r, val_r = ask_split_ratios(has_test)
    class_names, class_weights = ask_class_setup()
    train_params = ask_train_params()

    run_test_val = False
    val_params   = {}
    if has_test:
        run_test_val = prompt_yes_no(
            "After each fold trains, also validate against its test split?", "y"
        )
        if run_test_val:
            val_params = ask_val_params()

    print(f"\n  Ready: {n_splits} folds, train on each"
          + (", then validate on test split" if run_test_val else "") + ".")
    if not prompt_yes_no("Proceed?"):
        print("  Aborted."); return

    device = select_device()

    fold_paths = create_splits({
        "strategy": strategy, "dataset_path": dataset_path,
        "output_path": output_path, "n_splits": n_splits,
        "train_ratio": train_r, "val_ratio": val_r,
        "seed": seed, "class_names": class_names, "class_weights": class_weights,
    })

    run_training_loop(fold_paths, train_params, device,
                      output_path, run_test_val, val_params)


def mode_train_and_validate():
    header("MODE C — Train on existing splits")

    splits_root = prompt_existing_dir(
        "Directory containing fold_1/, fold_2/, ... subdirectories"
    )
    detected = sorted(splits_root.glob("fold_*"))
    if not detected:
        print("  ✗ No fold_* directories found. Aborting."); return
    print(f"  Detected {len(detected)} fold(s).")

    # Check whether any fold has a test split
    sample_cfg = {}
    try:
        with open(detected[0] / "config.yaml") as f:
            sample_cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    has_test = "test" in sample_cfg

    output_path  = prompt_output_dir("Output directory for results",
                                     default=str(splits_root))
    train_params = ask_train_params()

    run_test_val = False
    val_params   = {}
    if has_test:
        run_test_val = prompt_yes_no(
            "After each fold trains, also validate against its test split?", "y"
        )
        if run_test_val:
            val_params = ask_val_params()
    else:
        print("  (No test split detected in configs — skipping test validation option.)")

    print(f"\n  Ready to train {len(detected)} fold(s)"
          + (", then validate on test splits" if run_test_val else "") + ".")
    if not prompt_yes_no("Proceed?"):
        print("  Aborted."); return

    device = select_device()
    run_training_loop(list(detected), train_params, device,
                      output_path, run_test_val, val_params)


# ============================================================================
# Entry point
# ============================================================================

def main():
    header("YOLO Dataset Splitter & Cross-Validation Trainer")
    print("  Supplementary tool — Lagoon Epibenthos YOLO study\n")
    print("  Detecting compute device ...")
    select_device()
    print()

    mode = prompt_choice(
        "Select operating mode:",
        {
            "A": "Split only       — create fold directories and YAML configs",
            "B": "Split + Train    — create splits then train YOLOv11s on each fold",
            "C": "Train + Validate — train on an already-split dataset folder",
        }
    )

    if mode == "A":
        mode_split_only()
    elif mode == "B":
        mode_split_and_train()
    elif mode == "C":
        mode_train_and_validate()

    print(f"\n{_hr('=')}")
    print("  All done.")
    print(_hr('=') + "\n")


if __name__ == "__main__":
    main()