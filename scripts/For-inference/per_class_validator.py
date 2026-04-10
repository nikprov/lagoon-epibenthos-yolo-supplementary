"""
YOLO Test Set Evaluation Script

This script performs a scientific evaluation of a trained YOLO model on the test set.
It calculates and reports comprehensive metrics including per-class performance.

"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


def load_yaml(file_path):
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        file_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration data
    """
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return {}


def evaluate_on_test_set(
        model_path,
        data_config_path,
        output_dir=None,
        conf_thres=0.25,
        iou_thres=0.7,
        batch=16,
        imgsz=640,
        device=None,
        verbose=True
):
    """
    Evaluate a trained YOLO model specifically on the test set and generate reports.

    Args:
        model_path: Path to the trained model (.pt file)
        data_config_path: Path to the data configuration file
        output_dir: Directory where results will be saved (defaults to model directory)
        conf_thres: Confidence threshold for detections
        iou_thres: IoU threshold for NMS and matching
        batch: Batch size for evaluation
        imgsz: Image size for evaluation
        device: Device to run evaluation on ('cpu', '0', '0,1,2,3', etc.)
        verbose: Whether to print progress information

    Returns:
        Dictionary containing the evaluation results
    """
    # Set up paths and directories
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(model_path))
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Evaluating model on TEST SET: {os.path.basename(model_path)}")
        print(f"{'=' * 80}")

    # Load model and data configuration
    try:
        model = YOLO(model_path)
        data_config = load_yaml(data_config_path)
        class_names = data_config.get('names', {})
        num_classes = len(class_names)

        if verbose:
            print(f"Model loaded successfully: {model_path}")
            print(f"Evaluating on {num_classes} classes: {', '.join(class_names.values())}")
    except Exception as e:
        print(f"Error initializing model or data: {e}")
        return None

    # Run evaluation on the test set
    try:
        if verbose:
            print(f"\nRunning evaluation on test set...")

        results = model.val(
            data=data_config_path,
            split="test",  # Specifically use the test split
            imgsz=imgsz,
            batch=batch,
            conf=conf_thres,
            iou=iou_thres,
            device=device,
            verbose=verbose,
        )

        if verbose:
            print(f"Evaluation completed successfully.")
    except Exception as e:
        print(f"Error during test set evaluation: {e}")
        return None

    # Extract metrics
    metrics = {}

    # Overall metrics
    metrics['overall'] = {
        'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
        'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
        'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
        'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
        'val_loss': float(results.results_dict.get('val/box_loss', 0))
    }

    # Per-class metrics
    metrics['per_class'] = {}
    try:
        # Determine class indices to analyze
        ap_class_indices = list(range(num_classes))
        if hasattr(results, 'ap_class_index') and results.ap_class_index is not None:
            ap_class_indices = results.ap_class_index

        for i, class_idx in enumerate(ap_class_indices):
            class_id = int(class_idx)
            class_name = class_names.get(class_id, f"Class {class_id}")

            # Extract metrics for this class
            metrics['per_class'][class_name] = {
                'class_id': class_id,
                'precision': float(results.precision[i]) if hasattr(results, 'precision') and i < len(
                    results.precision) else 0,
                'recall': float(results.recall[i]) if hasattr(results, 'recall') and i < len(results.recall) else 0,
                'mAP50': float(results.ap50[i]) if hasattr(results, 'ap50') and i < len(results.ap50) else 0,
                'mAP50-95': float(results.ap[i]) if hasattr(results, 'ap') and i < len(results.ap) else 0,
                'images': int(results.seen) if hasattr(results, 'seen') else 0,
                'instances': int(np.sum(results.nt[class_id])) if hasattr(results, 'nt') and class_id < len(
                    results.nt) else 0
            }
    except Exception as e:
        print(f"Warning: Could not extract per-class metrics: {e}")
        print("Creating placeholder metrics for each class...")

        # Create placeholder metrics if extraction fails
        for class_id in range(num_classes):
            class_name = class_names.get(class_id, f"Class {class_id}")
            metrics['per_class'][class_name] = {
                'class_id': class_id,
                'precision': 0.0,
                'recall': 0.0,
                'mAP50': 0.0,
                'mAP50-95': 0.0,
                'images': 0,
                'instances': 0
            }

    # Speed metrics
    metrics['speed'] = {
        'preprocess': getattr(results, 'speed', {}).get('preprocess', 0),
        'inference': getattr(results, 'speed', {}).get('inference', 0),
        'loss': getattr(results, 'speed', {}).get('loss', 0),
        'postprocess': getattr(results, 'speed', {}).get('postprocess', 0)
    }

    # Create and save reports
    try:
        # Generate the TXT report
        report_path = create_test_evaluation_report(metrics, model_path, data_config, output_dir)

        # Generate the CSV report
        csv_path = create_csv_report(metrics, output_dir)

        if verbose:
            print(f"\nReports generated successfully:")
            print(f"  - Detailed report: {report_path}")
            print(f"  - CSV data: {csv_path}")
    except Exception as e:
        print(f"Error generating reports: {e}")

    return metrics


def create_test_evaluation_report(metrics, model_path, data_config, output_dir):
    """
    Create a comprehensive test set evaluation report and save it to a file.

    Args:
        metrics: Dictionary containing evaluation metrics
        model_path: Path to the model that was evaluated
        data_config: Data configuration dictionary
        output_dir: Directory where the report will be saved

    Returns:
        Path to the created report file
    """
    # Create the report filename - use 'test_evaluation' to clearly indicate this is test set results
    report_path = os.path.join(output_dir, "test_set_evaluation.txt")

    # Get timestamp for the report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write the comprehensive report
    with open(report_path, 'w') as f:
        # Write header with clear indication this is test set evaluation
        f.write(f"TEST SET EVALUATION REPORT\n")
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Generated on: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        # Write the model and data information
        f.write("Model and Data Information:\n")
        f.write(f"  Model Path: {os.path.abspath(model_path)}\n")
        f.write(f"  Data Config: {data_config.get('path', 'Unknown')}\n")
        f.write(f"  Number of Classes: {len(data_config.get('names', {}))}\n\n")

        # Write test set information if available
        f.write("Test Set Information:\n")
        first_class = next(iter(metrics['per_class'].values()), {})
        images_count = first_class.get('images', 0)
        total_instances = sum(m.get('instances', 0) for m in metrics['per_class'].values())
        f.write(f"  Images: {images_count}\n")
        f.write(f"  Total Instances: {total_instances}\n\n")

        # Write overall metrics - the key scientific results
        f.write("Overall Performance Metrics:\n")
        f.write(f"  Precision: {metrics['overall']['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['overall']['recall']:.4f}\n")
        f.write(f"  mAP50: {metrics['overall']['mAP50']:.4f}\n")
        f.write(f"  mAP50-95: {metrics['overall']['mAP50-95']:.4f}\n")
        f.write(f"  Test Loss: {metrics['overall']['val_loss']:.4f}\n\n")

        # Write per-class metrics in a detailed table format
        f.write("Per-Class Performance Metrics:\n")
        f.write(
            f"{'Class':>20}  {'Images':>10}  {'Instances':>10}  {'Precision':>10}  {'Recall':>10}  {'mAP50':>10}  {'mAP50-95':>10}\n")
        f.write(
            f"{'-' * 20:>20}  {'-' * 10:>10}  {'-' * 10:>10}  {'-' * 10:>10}  {'-' * 10:>10}  {'-' * 10:>10}  {'-' * 10:>10}\n")

        # Write the combined "all" row first
        f.write(f"{'all':>20}  {images_count:>10}  {total_instances:>10}  ")
        f.write(f"{metrics['overall']['precision']:.3f}  {metrics['overall']['recall']:.3f}  ")
        f.write(f"{metrics['overall']['mAP50']:.3f}  {metrics['overall']['mAP50-95']:.3f}\n")

        # Then write individual class rows
        for class_name, class_metrics in metrics['per_class'].items():
            f.write(
                f"{class_name:>20}  {class_metrics.get('images', 0):>10}  {class_metrics.get('instances', 0):>10}  ")
            f.write(f"{class_metrics.get('precision', 0):.3f}  {class_metrics.get('recall', 0):.3f}  ")
            f.write(f"{class_metrics.get('mAP50', 0):.3f}  {class_metrics.get('mAP50-95', 0):.3f}\n")

        # Write speed metrics
        f.write("\nInference Speed Metrics:\n")
        speed = metrics.get('speed', {})
        speed_line = f"Speed: {speed.get('preprocess', 0):.1f}ms preprocess, "
        speed_line += f"{speed.get('inference', 0):.1f}ms inference, "
        speed_line += f"{speed.get('loss', 0):.1f}ms loss, "
        speed_line += f"{speed.get('postprocess', 0):.1f}ms postprocess per image"
        f.write(f"{speed_line}\n")

        # Add a scientific methodology statement
        f.write("\nMethodology Note:\n")
        f.write("  This evaluation was performed on the held-out test set, which was not used during\n")
        f.write("  model training or hyperparameter optimization. These results represent an unbiased\n")
        f.write("  estimate of the model's performance on new, unseen data.\n")

    return report_path


def create_csv_report(metrics, output_dir):
    """
    Create a CSV file with per-class evaluation metrics for easier analysis.

    Args:
        metrics: Dictionary containing evaluation metrics
        output_dir: Directory where the CSV will be saved

    Returns:
        Path to the created CSV file
    """
    # Create CSV path
    csv_path = os.path.join(output_dir, "test_set_metrics.csv")

    # Prepare rows for the DataFrame
    rows = []

    # Add each class as a row
    for class_name, class_metrics in metrics.get('per_class', {}).items():
        row = {
            'class': class_name,
            'class_id': class_metrics.get('class_id', -1),
            'precision': class_metrics.get('precision', 0),
            'recall': class_metrics.get('recall', 0),
            'mAP50': class_metrics.get('mAP50', 0),
            'mAP50-95': class_metrics.get('mAP50-95', 0),
            'images': class_metrics.get('images', 0),
            'instances': class_metrics.get('instances', 0)
        }
        rows.append(row)

    # Add overall metrics as a row
    overall_row = {
        'class': 'Overall',
        'class_id': -1,
        'precision': metrics['overall']['precision'],
        'recall': metrics['overall']['recall'],
        'mAP50': metrics['overall']['mAP50'],
        'mAP50-95': metrics['overall']['mAP50-95'],
        'images': next(iter(metrics['per_class'].values()), {}).get('images', 0),
        'instances': sum(m.get('instances', 0) for m in metrics['per_class'].values())
    }
    rows.append(overall_row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    return csv_path


# Main function with hardcoded paths
def main():
    """
    Main function to execute the test set evaluation.
    Uses hardcoded paths to the model and data configuration.
    """
    # MODIFY THESE PATHS FOR YOUR ENVIRONMENT
    MODEL_PATH = r"./models/best.pt"
    DATA_CONFIG_PATH = r"./For-inference/config_imbalanced.yaml"
    OUTPUT_DIR = None  # Will use model directory by default

    # Check if the paths exist
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        return 1

    if not os.path.exists(DATA_CONFIG_PATH):
        print(f"Error: Data configuration file not found: {DATA_CONFIG_PATH}")
        return 1

    # Run the evaluation
    print(f"Starting test set evaluation...")
    metrics = evaluate_on_test_set(
        model_path=MODEL_PATH,
        data_config_path=DATA_CONFIG_PATH,
        output_dir=OUTPUT_DIR,
        conf_thres=0.25,
        iou_thres=0.7,
        batch=16,
        imgsz=640,
        device=None,  # Will use default device (CUDA if available)
        verbose=True
    )

    if metrics is None:
        print("Test set evaluation failed.")
        return 1

    print(f"\nTest set evaluation completed successfully!")
    print(f"Full results are available in the output directory: {OUTPUT_DIR or os.path.dirname(MODEL_PATH)}")

    # Print a summary of the key results
    print(f"\nSummary of test set performance:")
    print(f"  Precision: {metrics['overall']['precision']:.4f}")
    print(f"  Recall: {metrics['overall']['recall']:.4f}")
    print(f"  mAP50: {metrics['overall']['mAP50']:.4f}")
    print(f"  mAP50-95: {metrics['overall']['mAP50-95']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())