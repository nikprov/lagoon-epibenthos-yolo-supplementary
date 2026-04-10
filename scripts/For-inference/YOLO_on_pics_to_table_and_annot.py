"""
YOLO Inference, Annotation & Detection Table Builder
=====================================================
Runs a trained YOLOv11s model over a folder of images, saves
JPEG-compressed annotated copies, and writes per-image detection
counts and mean confidence scores to a CSV file.

Usage
-----
Edit the three constants in the __main__ block and run:
    python YOLO_on_pics_to_table_and_annot.py

Dependencies
------------
    pip install ultralytics opencv-python-headless pillow pandas tqdm
"""

from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


def process_images_with_yolo(
        model_path: str,
        image_folder: str,
        output_folder: str,
        confidence_threshold: float = 0.4,
        compression_quality: int = 75
) -> pd.DataFrame:
    """
    Run YOLO inference on every image in *image_folder*.

    For each image the function:
      - saves a JPEG-compressed annotated copy to <output_folder>/annotated_images/
      - records per-class detection counts and mean confidence scores

    After all images are processed a CSV summary is written to
    <output_folder>/detection_results.csv.

    Parameters
    ----------
    model_path           : path to trained YOLO weights (.pt)
    image_folder         : directory containing input images (jpg / jpeg / png)
    output_folder        : root directory for all outputs
    confidence_threshold : detections below this confidence are discarded
    compression_quality  : JPEG quality for annotated images (0-100)

    Returns
    -------
    pd.DataFrame with one row per input image
    """
    annotated_dir = Path(output_folder) / "annotated_images"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    class_names = model.names

    image_paths = sorted(
        list(Path(image_folder).glob('*.jpg')) +
        list(Path(image_folder).glob('*.jpeg')) +
        list(Path(image_folder).glob('*.png'))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_folder}")

    results_data = []

    for image_path in tqdm(image_paths, desc="Inferencing", unit="img"):
        results = model(str(image_path), conf=confidence_threshold, verbose=False)

        counts = {i: 0 for i in range(len(class_names))}
        confidences = {i: [] for i in range(len(class_names))}

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                counts[cls] += 1
                confidences[cls].append(float(box.conf[0]))

        mean_confs = {
            cls: round(float(np.mean(confs)), 6) if confs else 0.0
            for cls, confs in confidences.items()
        }

        # Save annotated image (YOLO returns BGR, PIL needs RGB)
        annotated_bgr = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(annotated_rgb).save(
            annotated_dir / f"annotated_{image_path.name}",
            "JPEG", quality=compression_quality
        )

        row: dict = {'image_name': image_path.name}
        for i in range(len(class_names)):
            row[f'count_{class_names[i]}'] = counts[i]
            row[f'mean_conf_{class_names[i]}'] = float(mean_confs[i])
        results_data.append(row)

        del results   # free GPU/RAM immediately

    df = pd.DataFrame(results_data)
    csv_path = Path(output_folder) / "detection_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved -> {csv_path}")

    return df


if __name__ == "__main__":
    MODEL_PATH        = 'models/best.pt'
    IMAGE_FOLDER      = 'imagery-files/enhanced'
    OUTPUT_FOLDER     = 'imagery-files/enhanced/output'

    df = process_images_with_yolo(
        model_path           = MODEL_PATH,
        image_folder         = IMAGE_FOLDER,
        output_folder        = OUTPUT_FOLDER,
        confidence_threshold = 0.4,
        compression_quality  = 75,
    )
    print(f"Done - {len(df)} images processed.")
