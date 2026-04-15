import matplotlib
matplotlib.use('Agg')  # Must precede any pyplot/YOLO import to prevent
                       # RecursionError in matplotlib Path.__deepcopy__ on Python >=3.14, Possibly 3.13 too.
import torch
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# GPU / device selection  (shared logic with yolo_trainer_for_imbalanced.py)
# ---------------------------------------------------------------------------

def select_device() -> str:
    if torch.cuda.is_available():
        idx  = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem  = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        print(f"  [GPU] CUDA device {idx}: {name}  ({mem:.1f} GB VRAM)")
        
        return str(idx)
    else:
        print("  [CPU] CUDA not available — tuning on CPU (slow).")
        return "cpu"


device = select_device()

# Initialize the model
model = YOLO("yolo11s.pt")

# ---------------------------------------------------------------------------
# Search space
# Active parameters are those that were still being refined in the final
# tuning round; commented-out entries show what was fixed earlier.
# ---------------------------------------------------------------------------
search_space = {
    # Learning parameters
    #"lr0": (0.0005, 0.005),
    #"lrf": (0.005, 0.05),
    "dropout": (0.01, 0.3),

    # Augmentation parameters
    #"hsv_h": (0.1, 0.4),
    #"hsv_s": (0.3, 0.9),
    #"hsv_v": (0.2, 0.5),

    # Training transformations
    #"scale": (0.2, 0.8),
    #"mosaic": (0.4, 1.0),
    #"mixup": (0.1, 0.7),
    #"copy_paste": (0.1, 0.5),
    "flipud": (0.01, 0.8),
    #"fliplr": (0.0, 0.8),
    "degrees": (0.01, 45.0),
    "shear": (0.01, 5.0),
    "translate": (0.01, 0.25),
    "perspective": (0.01, 0.4),

    # Class / loss balance
    #"cls": (0.3, 2.0),
    #"weight_decay": (0.0001, 0.001),

    # Integer parameters
    #"warmup_epochs": (0, 10),
    #"close_mosaic": [0, 2, 5, 10]
}

# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
results = model.tune(
    data="config_imbalanced.yaml",
    device=device,            # <-- GPU/CPU device
    epochs=50,
    iterations=60,
    space=search_space,
    optimizer="AdamW",
    plots=True,
    save=True,
    seed=48,
    val=True,
    deterministic=True,
    batch=16,
    # Fixed hyperparameters established in the prior manual search:
    lr0=0.00099,
    lrf=0.00715,
    hsv_h=0.11622,
    hsv_s=0.79876,
    hsv_v=0.222,
    scale=0.30327,
    mosaic=0.8865,
    mixup=0.13397,
    copy_paste=0.22137,
    fliplr=0.52646,
    translate=0.13784,
    cls=0.40984,
    weight_decay=0.0007,
    warmup_epochs=1.76162,
)

# After tuning completes, load the best model with:
# best_model = YOLO("runs/detect/tune/weights/best.pt")
