import os
import multiprocessing
import ultralytics
from ultralytics import YOLO
import time


def main():
    # Define the training folder name at the beginning
    train_folder_name = 'name-your-folder-here'

    multiprocessing.freeze_support()

    # Load pretrained model - using small variant for faster training/testing
    model = ultralytics.YOLO('yolo11s.pt')

    # Train the model with enhanced augmentation and class imbalance handling
    # Adjust per your dataset and requirements
    print(f"Starting training: {train_folder_name}")
    results = model.train(
        data="config_imbalanced.yaml",
        name=train_folder_name,

        # Training duration and batch settings
        epochs=190,
        batch=16,  # Smaller batch size can help with imbalanced datasets
        patience=0,
        # for overfit regulation
        weight_decay=0.0002,  # Add L2 regularization
        dropout=0.25,  # Add dropout layers - default: 0.0
        optimizer='AdamW',

        # Warmup and learning rate settings
        warmup_epochs=2,  # Increase warmup to 10 for better stability
        cos_lr=True,  # Cosine learning rate schedule
        lr0=0.00084,  # Initial learning rate
        lrf=0.00964,  # Final learning rate fraction

        # Aggressive augmentation strategy
        mosaic=0.699,  # Maximum mosaic augmentation
        mixup=0.11542,  # Increase mix-up for better generalization, def:0.1
        copy_paste=0.1,  # Copy-paste augmentation, def:0.1

        # Color augmentation - important for underwater imagery
        hsv_h=0.127,  # Moderate hue variation for underwater color shifts
        hsv_s=0.54,  # Balanced saturation adjustment
        hsv_v=0.5,  # Conservative brightness variation

        # Geometric augmentations
        degrees=5.0,  # Reduced rotation for underwater perspective
        translate=0.029,  # Moderate translation, def:0.1
        scale=0.2,  # Balanced scaling -> 0.5
        shear=0.0,  # Minimal shear to preserve organism shape, def:0.0
        fliplr=0.67575,  # Horizontal flip
        flipud=0.0,  # No vertical flip for aquatic species

        # Multi-scale training to handle size variations
        multi_scale=True,
        imgsz=640,

        # Early stopping mosaic for final fine-tuning
        close_mosaic=0,

        # Other training settings
        cache='disk',
        seed=50,
        deterministic=True,
        save=True,
        workers=8,
        pretrained=True,
        amp=True,  # Automatic mixed precision

        cls= 0.793,
        dfl= 1.5,

        # Validation settings
        val=True,  # Enable validation
        rect=False,  # Keep aspect ratio during validation

        iou=0.7,  # high IOU threshold for many overlapped species

    )

    print("\nTraining completed. Saving results...")

    # Create model output directory and results file paths
    model_dir = os.path.join('runs', 'detect', train_folder_name)
    os.makedirs(model_dir, exist_ok=True)
    results_file = os.path.join(model_dir, "training_results.txt")

    # Save basic metrics to a text file
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(f"Training Summary for {train_folder_name}\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Write training completed message with metrics
            f.write("Training completed. Overall metrics:\n")
            if results and hasattr(results, 'results_dict'):
                for metric, value in results.results_dict.items():
                    if 'metrics/' in metric:
                        f.write(f"{metric}: {value:.4f}\n")

            # Write training parameters without unicode characters
            # f.write("\nTraining Parameters:\n")
            # f.write(f"Epochs: 200, Batch size: 16\n")
            # f.write(f"Learning rate: 0.001 to 0.00001 (cosine schedule)\n")
            # f.write(f"Augmentation: mosaic=1.0, mixup=0.2, copy_paste=0.2\n")
    except Exception as e:
        print(f"Warning: Could not save training results text file: {e}")

    # Export the best model to ONNX format - this is the critical part
    try:
        model_path = os.path.join(model_dir, 'weights', 'best.pt')
        if os.path.exists(model_path):
            print(f"Exporting model to ONNX format from: {model_path}")
            export_model = YOLO(model_path)

            # Export with error handling
            onnx_path = os.path.splitext(model_path)[0] + '.onnx'
            export_model.export(format='onnx')

            if os.path.exists(onnx_path):
                print(f"ONNX export successful: {onnx_path}")
            else:
                print(f"Warning: ONNX file not found after export")
        else:
            print(f"Warning: Best model not found at {model_path}. ONNX export skipped.")
    except Exception as e:
        print(f"Error exporting model to ONNX: {str(e)}")
        print("Try manual export after training completion")

    print("Process completed!")


if __name__ == '__main__':
    main()