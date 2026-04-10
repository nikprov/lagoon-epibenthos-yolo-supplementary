import os
import shutil
import random
from pathlib import Path
import yaml
from sklearn.model_selection import KFold
import numpy as np
from ultralytics import YOLO


def create_fold_config(fold_path: Path, fold_number: int) -> None:
    """
    Creates a YOLO configuration file for a specific fold, incorporating
    the three-class structure with appropriate weights for class imbalance.

    Args:
        fold_path: Path to the fold directory
        fold_number: The number of the current fold
    """
    # Define class names dictionary for the three classes
    class_names = {
        0: 'Paranemonia sp.',
        1: 'Anemonia sp.',
        2: 'Brachyura sp.'
    }

    # Define class weights to handle the significant imbalance
    class_weights = {
        0: 1.0,   # Majority class (Paranemonia)
        1: 15.0,  # Minority class (Anemonia) - weighted higher
        2: 20.0   # Rare class (Brachyura) - weighted highest
    }

    # Create configuration with additional training parameters
    config = {
        'path': str(fold_path),  # Dataset root directory
        'train': 'images/train',  # Train images relative to path
        'val': 'images/val',  # Val images relative to path
        'test': 'images/test',  # Test set path (if available)
        'names': class_names,  # Class names
        'weights': class_weights,  # Class weights for imbalance
        'nc': 3,  # Number of classes
        'single_cls': False,  # Multi-class detection
        'rect': False,  # Rectangular training
        'fold': fold_number  # Record which fold this is
    }

    # Save the configuration
    with open(fold_path / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def copy_files_with_labels(file_list, images_dest_dir, labels_dest_dir):
    """
    Copies image files and their corresponding label files to destination directories.

    Args:
        file_list: List of image file paths
        images_dest_dir: Destination directory for images
        labels_dest_dir: Destination directory for labels
    """
    copied_count = 0
    missing_labels = 0

    for img_path in file_list:
        # Copy image
        shutil.copy2(img_path, images_dest_dir / img_path.name)

        # Copy corresponding label file
        label_path = Path(str(img_path).replace('images', 'labels')).with_suffix('.txt')
        if label_path.exists():
            shutil.copy2(label_path, labels_dest_dir / label_path.name)
            copied_count += 1
        else:
            print(f"Warning: Label file not found for {img_path.name}")
            missing_labels += 1

    return copied_count, missing_labels


def load_yaml(file_path):
    """
    Safely load a YAML configuration file.

    Args:
        file_path: Path to the YAML file to load

    Returns:
        dict: Loaded YAML content

    Raises:
        yaml.YAMLError: If the YAML file is malformed
        FileNotFoundError: If the file doesn't exist
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    except FileNotFoundError:
        print(f"Configuration file not found: {file_path}")
        raise


def create_k_fold_splits(
        original_dataset_path: str,
        splits_base_path: Path,
        n_splits: int = 5,
        seed: int = 42
) -> None:
    """
    Creates k-fold splits of the dataset ensuring each sample appears once in validation.
    Handles a three-class dataset with appropriate class weighting.

    Args:
        original_dataset_path: Path to original dataset
        splits_base_path: Where to save the splits
        n_splits: Number of folds
        seed: Random seed for reproducibility
    """
    # Get list of all image files
    original_images_dir = Path(original_dataset_path) / 'images'
    image_files = list(original_images_dir.glob('*.jpg')) + list(original_images_dir.glob('*.png'))

    if not image_files:
        raise ValueError(f"No image files found in {original_images_dir}. Check the path and file extensions.")

    print(f"Found {len(image_files)} images for k-fold splitting")

    # Initialize K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Create folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(image_files))), 1):
        print(f"\nProcessing fold {fold}/{n_splits}...")

        # Get train and validation files for this fold
        train_files = [image_files[i] for i in train_idx]
        val_files = [image_files[i] for i in val_idx]

        # Create fold directory structure
        fold_path = splits_base_path / f"fold_{fold}"
        train_images_dir = fold_path / 'images' / 'train'
        val_images_dir = fold_path / 'images' / 'val'
        train_labels_dir = fold_path / 'labels' / 'train'
        val_labels_dir = fold_path / 'labels' / 'val'

        # Also create a test directory structure (empty for now)
        test_images_dir = fold_path / 'images' / 'test'
        test_labels_dir = fold_path / 'labels' / 'test'

        # Create directories
        for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir,
                         test_images_dir, test_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Copy files for this fold
        train_results = copy_files_with_labels(train_files, train_images_dir, train_labels_dir)
        val_results = copy_files_with_labels(val_files, val_images_dir, val_labels_dir)

        # Create fold-specific config with three-class structure
        create_fold_config(fold_path, fold)

        print(f"Created fold {fold}:")
        print(
            f"  Training: {len(train_files)} images ({train_results[0]} with labels, {train_results[1]} missing labels)")
        print(f"  Validation: {len(val_files)} images ({val_results[0]} with labels, {val_results[1]} missing labels)")

        # Create a readme file to explain the fold structure
        with open(fold_path / "README.txt", "w") as readme:
            readme.write(f"K-Fold Split {fold} of {n_splits}\n")
            readme.write("=========================\n\n")
            readme.write(f"Training images: {len(train_files)}\n")
            readme.write(f"Validation images: {len(val_files)}\n")
            readme.write("\nClass weights applied:\n")
            readme.write("- Paranemonia sp. (class 0): 1.0\n")
            readme.write("- Anemonia sp. (class 1): 15.0\n")
            readme.write("- Brachyura sp. (class 2): 20.0\n\n")
            readme.write("These weights help balance training for the imbalanced dataset.\n")


def train_k_folds(base_config_path: str, splits_base_path: Path, n_splits: int = 5):
    """
    Trains YOLO model on each fold and tracks performance.
    Handles the three-class structure with appropriate metrics.

    Args:
        base_config_path: Path to base YOLO config
        splits_base_path: Path containing the k-fold splits
        n_splits: Number of folds
    """
    # Initialize results tracking for each class
    fold_results = []
    class_results = {
        'Paranemonia sp.': [],
        'Anemonia sp.': [],
        'Brachyura sp.': []
    }

    for fold in range(1, n_splits + 1):
        print(f"\n{'=' * 60}")
        print(f"Training Fold {fold}/{n_splits}")
        print(f"{'=' * 60}")

        # Update config for this fold
        fold_config = load_yaml(base_config_path)
        fold_config.update({
            'data': str(splits_base_path / f"fold_{fold}" / 'config.yaml'),
            'name': f"fold_{fold}_training",
            'seed': 42 + fold,  # Unique seed per fold
            # Add any other specific training parameters
            'epochs': 200,
            'batch': 16,
            'patience': 50,
            'imgsz': 640
        })

        # Train model on this fold
        try:
            model = YOLO('yolo11s.pt')  # You can change the model size as needed
            results = model.train(**fold_config)

            # Extract metrics
            if results is None or not hasattr(results, 'results_dict'):
                print(f"Warning: Unable to extract metrics from fold {fold} results")
                continue

            fold_metrics = {
                'fold': fold,
                'mAP50': float(results.results_dict['metrics/mAP50(B)']),
                'mAP50-95': float(results.results_dict['metrics/mAP50-95(B)']),
                'precision': float(results.results_dict['metrics/precision(B)']),
                'recall': float(results.results_dict['metrics/recall(B)']),
            }

            # Add fold results to tracking
            fold_results.append(fold_metrics)

            # Log results to file
            with open(splits_base_path / f"fold_{fold}_results.txt", 'w') as f:
                for metric, value in fold_metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")

            print(f"\nFold {fold} training completed:")
            for metric, value in fold_metrics.items():
                if metric != 'fold':
                    print(f"  {metric}: {value:.4f}")

        except Exception as e:
            print(f"Error training fold {fold}: {str(e)}")
            continue

    # Calculate cross-validation metrics
    if fold_results:
        calculate_cv_metrics(fold_results)
    else:
        print("\nNo successful training runs to analyze.")


def calculate_cv_metrics(fold_results):
    """
    Calculates and displays cross-validation metrics with confidence intervals.
    Creates a comprehensive summary report.
    """
    print("\n" + "=" * 40)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 40)

    metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
    summary_data = {}

    for metric in metrics:
        values = [result[metric] for result in fold_results]
        mean = np.mean(values)
        std = np.std(values)
        ci = 1.96 * std / np.sqrt(len(values))  # 95% confidence interval

        print(f"\n{metric}:")
        print(f"  Mean: {mean:.4f} ± {ci:.4f}")
        print(f"  Standard deviation: {std:.4f}")
        print(f"  Individual folds: {', '.join([f'{v:.4f}' for v in values])}")

        summary_data[metric] = {
            'mean': mean,
            'std': std,
            'ci': ci,
            'values': values
        }

    # Create a summary report file
    with open("cross_validation_summary.txt", 'w') as f:
        f.write("YOLO CROSS-VALIDATION SUMMARY\n")
        f.write("============================\n\n")
        f.write(f"Number of folds: {len(fold_results)}\n")
        f.write(f"Model: yolo11s.pt\n")
        f.write(f"Classes: Paranemonia sp., Anemonia sp., Brachyura sp.\n\n")

        f.write("METRICS SUMMARY\n")
        f.write("--------------\n")
        for metric, data in summary_data.items():
            f.write(f"\n{metric}:\n")
            f.write(f"  Mean: {data['mean']:.4f} ± {data['ci']:.4f}\n")
            f.write(f"  Standard deviation: {data['std']:.4f}\n")
            f.write(f"  Values per fold: {', '.join([f'{v:.4f}' for v in data['values']])}\n")

    print(f"\nDetailed summary saved to cross_validation_summary.txt")


def main():
    """
    Main function to execute K-fold cross-validation process.
    Handles the three-class dataset structure with appropriate weighting.
    """
    # Define paths
    original_dataset = "./path/to/original/dataset"  # Path to original dataset with 'images' and 'labels' subdirectories
    output_base = "./path/to/output/directory"  # Base path where splits and results will be saved
    splits_base_path = Path(output_base) / 'kfold_splits_3class_augmented'
    base_config_path = "./For-inference/args.yaml"  # Path to base YOLO config

    # Number of folds
    n_splits = 5

    # Create base directory for all splits
    splits_base_path.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating {n_splits}-fold cross-validation splits for 3-class dataset in: {splits_base_path}")

    # Create the K-fold splits
    create_k_fold_splits(
        original_dataset_path=original_dataset,
        splits_base_path=splits_base_path,
        n_splits=n_splits,
        seed=45  # Seed for reproducibility
    )

    # Ask user if they want to train models on the splits
    train_models = input("\nDo you want to train models on the created folds? (y/n): ").lower().strip()

    if train_models == 'y':
        print("\nStarting cross-validation training with three-class structure...")
        train_k_folds(
            base_config_path=base_config_path,
            splits_base_path=splits_base_path,
            n_splits=n_splits
        )
    else:
        print("\nK-fold splits created successfully. Training skipped.")
        print(f"Splits are available at: {splits_base_path}")
        print("Each split includes configuration for the three-class structure with appropriate class weights.")


if __name__ == "__main__":
    main()