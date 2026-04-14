import albumentations as A
import cv2
import numpy as np
from pathlib import Path
import yaml
import shutil
from PIL import Image


class UnderwaterAugmenter:
    def __init__(self, common_multiplier=2, rare_multiplier=5):
        
        self.common_multiplier = common_multiplier  
        self.rare_multiplier = rare_multiplier  
        # Configure augmentation pipeline with conservative transformations
        self.transform = A.Compose([
            # Color adjustments remain the same
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.8
            ),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),  
                rotate=(-10, 10),  
                translate_percent={
                    'x': (-0.08, 0.08),
                    'y': (-0.08, 0.08)
                },
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7
            ),
            # Noise and blur
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.GaussianBlur(p=0.5)
            ], p=0.3),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3,  # Only keep boxes that remain at least 30% visible
            check_each_transform=True  # Validate boxes after each transform
        ))

    def clip_bbox(self, bbox):
        """Clip bbox coordinates to valid range [0, 1]"""
        x, y, w, h = bbox
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        w = np.clip(w, 0, 1 - x)  # Ensure width doesn't extend beyond 1
        h = np.clip(h, 0, 1 - y)  # Ensure height doesn't extend beyond 1
        return [x, y, w, h]

    def augment_dataset(self, image_dir: Path, labels_dir: Path, output_dir: Path):
        """Augment dataset with improved error handling and bbox validation"""
        # Create output directories
        output_img_dir = output_dir / 'images'
        output_label_dir = output_dir / 'labels'
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        # Copy original dataset
        print("Copying original dataset...")
        for img_path in image_dir.glob('*.jpg'):
            shutil.copy2(img_path, output_img_dir / img_path.name)
            label_path = labels_dir / img_path.with_suffix('.txt').name
            if label_path.exists():
                shutil.copy2(label_path, output_label_dir / label_path.name)

        print("Starting augmentation...")
        for img_path in image_dir.glob('*.jpg'):
            label_path = labels_dir / img_path.with_suffix('.txt').name

            if not label_path.exists():
                print(f"Warning: No label file found for {img_path.name}")
                continue

            # Read image and labels
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                with open(label_path) as f:
                    lines = f.readlines()

                bboxes = []
                class_labels = []
                for line in lines:
                    cls, x, y, w, h = map(float, line.strip().split())
                    # Pre-validate and clip bbox coordinates
                    bbox = self.clip_bbox([x, y, w, h])
                    bboxes.append(bbox)
                    class_labels.append(cls)

                # Skip if no valid bounding boxes
                if not bboxes:
                    print(f"Warning: No valid bounding boxes in {img_path.name}")
                    continue

                # Determine augmentation count
                has_rare = any(cls == 2 for cls in class_labels)
                num_augmentations = self.rare_multiplier if has_rare else self.common_multiplier

                # Generate augmentations
                for i in range(num_augmentations):
                    try:
                        transformed = self.transform(
                            image=image,
                            bboxes=bboxes,
                            class_labels=class_labels
                        )

                        # Skip if no bounding boxes survived transformation
                        if not transformed['bboxes']:
                            continue

                        aug_img_path = output_img_dir / f"{img_path.stem}_aug_{i}{img_path.suffix}"
                        aug_label_path = output_label_dir / f"{img_path.stem}_aug_{i}.txt"

                        # Save augmented image with JPEG compression.
                        # transformed['image'] is already RGB (albumentations preserves the
                        # channel order of its input, which we converted BGR→RGB above).
                        # Do NOT pass through cv2.cvtColor again — that would swap R and B
                        # a second time and silently produce wrong colours in the saved files.
                        pil_image = Image.fromarray(transformed['image'])
                        pil_image.save(str(aug_img_path), 'JPEG', quality=75)

                        # Save labels with post-validation
                        with open(aug_label_path, 'w') as f:
                            for bbox, cls in zip(transformed['bboxes'], transformed['class_labels']):
                                bbox = self.clip_bbox(bbox)  # Extra safety check
                                f.write(f"{int(cls)} {' '.join(map(str, bbox))}\n")

                    except Exception as e:
                        print(f"Warning: Failed augmentation {i} for {img_path.name}: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
                continue

        print(f"Augmentation complete. Output saved to {output_dir}")

if __name__ == "__main__":
    # Convert string paths to Path objects
    base_dir = Path(r"D:\OneDrive\HCMR docs\Projects\Yolov8 CV benthic epifauna\dataset_old_and_new")
    output_dir = Path(
        r"D:\OneDrive\HCMR docs\Projects\Yolov8 CV benthic epifauna\dataset_old_and_new\dataset_augmented_3")

    augmenter = UnderwaterAugmenter(
        common_multiplier=3,
        rare_multiplier=5
    )

    # Run augmentation
    augmenter.augment_dataset(
        image_dir=base_dir / "images",
        labels_dir=base_dir / "labels",
        output_dir=output_dir
    )