"""
CLAHE Underwater Image Preprocessor (v2, 10/04/2026)
=============================================================
Supplementary material for:
"Implementing Optimized Computer Vision Algorithms for Underwater Imagery
 for Identification and Spatial Analysis of Epibenthic Fauna in
 Shallow Lagoon Waters"

Description:
    Batch preprocessing pipeline for raw underwater / USV-collected imagery.
    Full pipeline (each step optional and interactively configurable):

        [1] Adaptive Gray-World Color Cast Correction
               – Detects dominant chromatic cast from per-channel means.
               – Rescales weaker channels so all channel means equalise
                 (relative gray-world), without forcing an arbitrary absolute
                 target. Operates in BGR before any colour-space conversion.

        [2] CLAHE  (Contrast Limited Adaptive Histogram Equalisation)
               – Applied to the L* channel in CIE LAB colour space.
               – Runs on the cast-corrected image so luminance enhancement
                 is performed on a spectrally balanced signal.

        [3] Per-channel Histogram Stretching  (optional)
               – Three selectable strategies: cumulative-percentile,
                 min-max, or mean +/- k*sigma.
               – Applied last as a mild global dynamic-range expansion.
                 Recommended tail clipping <= 2 % when used in combination
                 with steps 1 and 2.

    Original EXIF / GPS metadata is preserved in every output file.

Processing-order rationale
--------------------------
    Colour cast correction precedes CLAHE because the BGR -> LAB conversion
    used by CLAHE decomposes luminance from a cast-contaminated signal when
    the image is not chromatically balanced.  Equalising channel means first
    ensures that the L* channel CLAHE enhances is spectrally unbiased.
    Histogram stretching follows as a final global range adjustment; it
    remains useful because gray-world balancing sets channel means but does
    not expand the dynamic range, and CLAHE only addresses local contrast.
    Reference for grey-world theory: Bianco et al. (2015),
    DOI: 10.5194/isprsarchives-XL-5-W5-25-2015.

Dependencies:
    opencv-python, numpy, Pillow, piexif, tqdm

Usage:
    python CLAHE_underwater_preprocessor.py
    The script prompts interactively for all parameters.

Author : Nikolaos Providakis
License: MIT License  (https://opensource.org/licenses/MIT)
"""

import cv2
import numpy as np
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import logging
from PIL import Image
import piexif
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
#  Prompt helpers
# ===========================================================================

def prompt_float(message: str, default: float, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    while True:
        raw = input(f"  {message} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if lo is not None and val < lo:
                print(f"    -> Must be >= {lo}.  Try again.")
                continue
            if hi is not None and val > hi:
                print(f"    -> Must be <= {hi}.  Try again.")
                continue
            return val
        except ValueError:
            print("    -> Not a valid number.  Try again.")


def prompt_int(message: str, default: int, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    while True:
        raw = input(f"  {message} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if lo is not None and val < lo:
                print(f"    -> Must be >= {lo}.  Try again.")
                continue
            if hi is not None and val > hi:
                print(f"    -> Must be <= {hi}.  Try again.")
                continue
            return val
        except ValueError:
            print("    -> Not a valid integer.  Try again.")


def prompt_yes_no(message: str, default: str = "n") -> bool:
    indicator = "Y/n" if default.lower() == "y" else "y/N"
    while True:
        raw = input(f"  {message} ({indicator}): ").strip().lower()
        if raw == "":
            return default.lower() == "y"
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("    -> Please enter 'y' or 'n'.")


def prompt_directory(message: str) -> Path:
    while True:
        raw = input(f"  {message}: ").strip().strip('"').strip("'")
        if not raw:
            print("    -> Path cannot be empty.")
            continue
        return Path(raw)


# ===========================================================================
#  Color cast corrector
# ===========================================================================

class ColorCastCorrector:
    """
    Adaptive gray-world color cast correction for underwater imagery.

    Theory
    ------
    The gray-world assumption (Buchsbaum 1980) states that the average
    reflectance of a natural scene is achromatic, meaning all three colour
    channels should share the same mean intensity.  Deviations from this
    indicate a chromatic cast caused by selective spectral attenuation
    (e.g. red/orange absorption at depth, turbidity-induced green or blue
    shift in lagoon water).

    Implementation
    --------------
    Unlike approaches that rescale each channel to an arbitrary absolute
    target (e.g. 128), this implementation equalises channel means
    *relative to each other*, preserving overall scene luminance while
    removing only the chromatic imbalance.  The reference is the grand mean
    of all three channels, which is luminance-neutral.

    A strength parameter (0-1) blends between no correction (0) and full
    gray-world correction (1), allowing conservative adjustment for scenes
    that are legitimately asymmetric (e.g. dark seagrass beds).

    The correction is applied as a per-channel multiplicative scalar in
    BGR space BEFORE any colour-space conversion, so that subsequent CLAHE
    in LAB space operates on a chromatically balanced luminance signal.

    Reference
    ---------
    Bianco, G., Muzzupappa, M., Bruno, F., Garcia, R., Neumann, L. (2015).
    A new color correction method for underwater imaging.
    ISPRS Archives, XL-5/W5, 25-32.
    DOI: 10.5194/isprsarchives-XL-5-W5-25-2015
    """

    def __init__(self, strength: float = 1.0):
        """
        Parameters
        ----------
        strength : float in [0, 1]
            Blending factor between original (0) and fully corrected (1)
            image.  Values around 0.8-1.0 are recommended for most
            lagoon / shallow-water conditions.
        """
        if not (0.0 <= strength <= 1.0):
            raise ValueError("strength must be in [0, 1]")
        self.strength = strength

    def apply(self, image_bgr: np.ndarray) -> tuple:
        """
        Equalise per-channel means and return corrected image + diagnostics.

        Returns
        -------
        corrected : np.ndarray  (uint8, BGR)
        info      : dict with keys 'dominant', 'means_before', 'means_after'
        """
        b, g, r = cv2.split(image_bgr.astype(np.float32))

        b_mean = float(np.mean(b))
        g_mean = float(np.mean(g))
        r_mean = float(np.mean(r))
        means_before = {"B": b_mean, "G": g_mean, "R": r_mean}

        # Grand mean = luminance-neutral reference
        grand_mean = (b_mean + g_mean + r_mean) / 3.0
        dominant = max(means_before.items(), key=lambda x: x[1])[0]

        def _scale(channel: np.ndarray, ch_mean: float) -> np.ndarray:
            if ch_mean < 1e-6:
                return channel
            # Full gray-world correction factor
            full_factor = grand_mean / ch_mean
            # Blend with identity (factor = 1.0) by 'strength'
            factor = 1.0 + (full_factor - 1.0) * self.strength
            return np.clip(channel * factor, 0, 255)

        b_c = _scale(b, b_mean)
        g_c = _scale(g, g_mean)
        r_c = _scale(r, r_mean)

        corrected = cv2.merge([b_c, g_c, r_c]).astype(np.uint8)

        means_after = {
            "B": float(np.mean(b_c)),
            "G": float(np.mean(g_c)),
            "R": float(np.mean(r_c)),
        }

        info = {
            "dominant": dominant,
            "means_before": means_before,
            "means_after": means_after,
        }
        return corrected, info


# ===========================================================================
#  Histogram stretcher
# ===========================================================================

class HistogramStretcher:
    """
    Per-channel histogram stretching with three selectable strategies.

    Applied as the FINAL step after CLAHE to expand global dynamic range.
    When used in combination with gray-world correction and CLAHE, mild
    settings (tail clipping <= 2 %) are recommended to avoid artefacts.
    """

    STRATEGIES = {
        "1": "cumulative_percentile",
        "2": "min_max",
        "3": "mean_stdev",
    }

    def __init__(self, strategy: str, low_pct: float = 0.5, high_pct: float = 99.5,
                 stdev_factor: float = 2.0):
        if strategy not in self.STRATEGIES.values():
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self.low_pct = low_pct / 100.0
        self.high_pct = high_pct / 100.0
        self.stdev_factor = stdev_factor

    def _stretch_channel(self, channel: np.ndarray) -> np.ndarray:
        if self.strategy == "cumulative_percentile":
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
            cumsum = np.cumsum(hist)
            total = cumsum[-1]
            lo = int(np.searchsorted(cumsum, total * self.low_pct))
            hi = int(np.searchsorted(cumsum, total * self.high_pct))
        elif self.strategy == "min_max":
            lo = int(channel.min())
            hi = int(channel.max())
        elif self.strategy == "mean_stdev":
            mu = channel.mean()
            sigma = channel.std()
            lo = max(0,   int(mu - self.stdev_factor * sigma))
            hi = min(255, int(mu + self.stdev_factor * sigma))
        else:
            return channel

        if hi <= lo:
            return channel

        stretched = np.clip(channel, lo, hi)
        stretched = ((stretched.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        return stretched

    def apply(self, image_bgr: np.ndarray) -> np.ndarray:
        channels = [self._stretch_channel(c) for c in cv2.split(image_bgr)]
        return cv2.merge(channels)


# ===========================================================================
#  Core processor
# ===========================================================================

class UnderwaterImageProcessor:
    """
    Full preprocessing pipeline for raw underwater images collected by USV.

    Pipeline (each stage independently optional)
    ---------------------------------------------
    1. Adaptive gray-world color cast correction  (BGR space)
    2. CLAHE on L* channel                        (CIE LAB space)
    3. Per-channel histogram stretching           (BGR space, final step)

    EXIF / GPS metadata is read from the source file and re-embedded in
    every output JPEG.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_size: tuple = (8, 8),
        jpeg_quality: int = 75,
        color_corrector: Optional[ColorCastCorrector] = None,
        hist_stretcher: Optional[HistogramStretcher] = None,
    ):
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.jpeg_quality = jpeg_quality
        self.color_corrector = color_corrector
        self.hist_stretcher = hist_stretcher

        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_size,
        )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _load_exif(self, image_path: Path):
        try:
            return piexif.load(str(image_path))
        except Exception as exc:
            logger.warning(f"  Could not load EXIF from {image_path.name}: {exc}")
            return None

    def _save_with_exif(self, image_bgr: np.ndarray, original_path: Path,
                        output_path: Path) -> bool:
        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
            exif_dict = self._load_exif(original_path)
            save_kwargs: dict = dict(format="JPEG", quality=self.jpeg_quality, optimize=True)
            if exif_dict:
                save_kwargs["exif"] = piexif.dump(exif_dict)
            pil_img.save(str(output_path), **save_kwargs)
            return True
        except Exception as exc:
            logger.error(f"  Error saving {output_path.name}: {exc}")
            return False

    # ------------------------------------------------------------------
    # Single-image processing
    # ------------------------------------------------------------------

    def process_image(self, image_path: Path, output_path: Path) -> bool:
        """
        Apply the full preprocessing pipeline to one raw underwater image.

        Step 1 -- Color cast correction (if enabled)
            Gray-world equalisation of per-channel means in BGR space.
            Must precede CLAHE so the LAB decomposition is chromatically
            unbiased.

        Step 2 -- CLAHE
            Contrast Limited Adaptive Histogram Equalisation on the L*
            channel of the CIE LAB representation. Operates on the
            cast-corrected image.

        Step 3 -- Histogram stretching (if enabled)
            Global per-channel dynamic-range expansion as a final step.
            Distinct from steps 1-2: corrects neither colour balance nor
            local contrast, only the global intensity range.
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read: {image_path}")

            # -- Step 1: Color cast correction ---------------------------
            if self.color_corrector is not None:
                img, cast_info = self.color_corrector.apply(img)
                logger.debug(
                    f"  {image_path.name} | dominant cast: {cast_info['dominant']} | "
                    f"B {cast_info['means_before']['B']:.1f}->{cast_info['means_after']['B']:.1f}  "
                    f"G {cast_info['means_before']['G']:.1f}->{cast_info['means_after']['G']:.1f}  "
                    f"R {cast_info['means_before']['R']:.1f}->{cast_info['means_after']['R']:.1f}"
                )

            # -- Step 2: CLAHE in LAB colour space -----------------------
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe.apply(l)
            img = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

            # -- Step 3: Histogram stretching ----------------------------
            if self.hist_stretcher is not None:
                img = self.hist_stretcher.apply(img)

            return self._save_with_exif(img, image_path, output_path)

        except Exception as exc:
            logger.error(f"  Error processing {image_path.name}: {exc}")
            return False

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def batch_process(self, input_dir: Path, output_dir: Path,
                      num_workers: int = 4) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(set(
            list(input_dir.glob("*.jpg"))  +
            list(input_dir.glob("*.jpeg")) +
            list(input_dir.glob("*.JPG"))  +
            list(input_dir.glob("*.JPEG"))
        ))

        if not image_files:
            logger.warning(f"No JPEG images found in: {input_dir}")
            return

        output_paths = [output_dir / f"clahe_{img.name}" for img in image_files]
        logger.info(f"Found {len(image_files)} image(s).  Starting batch ...\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.process_image, src, dst): src
                for src, dst in zip(image_files, output_paths)
            }
            successful = 0
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing",
            ):
                if future.result():
                    successful += 1

        logger.info("\nFile-size comparison (up to first 5 images):")
        for i in range(min(5, len(image_files))):
            orig_mb = os.path.getsize(image_files[i]) / 1_048_576
            proc_mb = os.path.getsize(output_paths[i]) / 1_048_576
            logger.info(
                f"  {image_files[i].name}: {orig_mb:.2f} MB -> {proc_mb:.2f} MB"
            )

        logger.info(
            f"\nCompleted: {successful} / {len(image_files)} images processed successfully."
        )


# ===========================================================================
#  Interactive parameter collection
# ===========================================================================

def collect_parameters() -> dict:
    print()
    print("=" * 65)
    print("  CLAHE Underwater Image Preprocessor  (v2)")
    print("  Supplementary tool -- YOLO epibenthic communities publication")
    print("=" * 65)

    # -- Directories ---------------------------------------------------
    print("\n-- Directories --------------------------------------------------")
    while True:
        input_dir = prompt_directory("Input folder (raw images)")
        if input_dir.exists() and input_dir.is_dir():
            break
        print(f"    -> Directory not found: {input_dir}")
    output_dir = prompt_directory("Output folder (created if absent)")

    # -- Step 1: Color cast correction ---------------------------------
    print("\n-- [Step 1]  Adaptive Gray-World Color Cast Correction ----------")
    print("   Equalises per-channel means to remove chromatic bias caused")
    print("   by wavelength-selective attenuation in turbid / shallow water.")
    print("   Applied in BGR space BEFORE CLAHE (see module docstring).")
    apply_cast = prompt_yes_no("Apply color cast correction?", default="y")

    color_corrector = None
    if apply_cast:
        print()
        strength = prompt_float(
            "Correction strength  (0.0 = none  |  1.0 = full gray-world)",
            default=1.0, lo=0.0, hi=1.0,
        )
        color_corrector = ColorCastCorrector(strength=strength)

    # -- Step 2: CLAHE -------------------------------------------------
    print("\n-- [Step 2]  CLAHE Hyperparameters ------------------------------")
    print("   (Press Enter to accept the default shown in brackets)\n")
    clip_limit   = prompt_float("Clip limit",           default=2.0,  lo=0.1, hi=40.0)
    tile_w       = prompt_int(  "Tile width  (px)",     default=8,    lo=1,   hi=64)
    tile_h       = prompt_int(  "Tile height (px)",     default=8,    lo=1,   hi=64)
    tile_size    = (tile_w, tile_h)
    jpeg_quality = prompt_int(  "JPEG quality (1-95)",  default=75,   lo=1,   hi=95)
    num_workers  = prompt_int(  "Parallel workers",     default=4,    lo=1,   hi=32)

    # -- Step 3: Histogram stretching ----------------------------------
    print("\n-- [Step 3]  Per-channel Histogram Stretching (optional) --------")
    print("   Expands global dynamic range after CLAHE.")
    print("   Tip: when combined with steps 1+2, use mild settings")
    print("   (tail clipping <= 2 %) to avoid over-processing.")
    apply_stretch = prompt_yes_no("Apply histogram stretching?", default="n")

    hist_stretcher = None
    if apply_stretch:
        print()
        print("  Stretching strategy:")
        print("    1 -- Cumulative percentile  (clip histogram tails at set %)")
        print("    2 -- Min-Max                (stretch full pixel range)")
        print("    3 -- Mean +/- k*sigma       (clip at mean +/- k std devs)")

        while True:
            choice = input("\n  Select strategy (1/2/3) [default: 1]: ").strip()
            if choice == "":
                choice = "1"
            if choice in HistogramStretcher.STRATEGIES:
                strategy = HistogramStretcher.STRATEGIES[choice]
                break
            print("    -> Invalid choice.  Please enter 1, 2, or 3.")

        if strategy == "cumulative_percentile":
            print()
            low_pct  = prompt_float("Lower tail clipping (%)",
                                    default=0.5, lo=0.0, hi=49.0)
            high_pct = prompt_float("Upper tail clipping (%)",
                                    default=99.5, lo=low_pct + 0.01, hi=100.0)
            hist_stretcher = HistogramStretcher(
                strategy, low_pct=low_pct, high_pct=high_pct
            )

        elif strategy == "min_max":
            hist_stretcher = HistogramStretcher(strategy)

        elif strategy == "mean_stdev":
            print()
            stdev_factor = prompt_float(
                "Number of standard deviations (k)", default=2.0, lo=0.1, hi=10.0
            )
            hist_stretcher = HistogramStretcher(strategy, stdev_factor=stdev_factor)

    return dict(
        input_dir=input_dir,
        output_dir=output_dir,
        clip_limit=clip_limit,
        tile_size=tile_size,
        jpeg_quality=jpeg_quality,
        num_workers=num_workers,
        color_corrector=color_corrector,
        hist_stretcher=hist_stretcher,
    )


def _print_summary(params: dict) -> None:
    cc = params["color_corrector"]
    hs = params["hist_stretcher"]

    print("\n" + "=" * 65)
    print("  Parameter summary")
    print("=" * 65)
    print(f"  Input folder    : {params['input_dir']}")
    print(f"  Output folder   : {params['output_dir']}")
    print()
    if cc:
        print(f"  [1] Color cast  : ENABLED   strength = {cc.strength}")
    else:
        print("  [1] Color cast  : disabled")
    print(f"  [2] CLAHE       : clip_limit={params['clip_limit']}, "
          f"tile={params['tile_size']}, "
          f"quality={params['jpeg_quality']}, "
          f"workers={params['num_workers']}")
    if hs:
        detail = ""
        if hs.strategy == "cumulative_percentile":
            detail = f" [{hs.low_pct*100:.2f}% - {hs.high_pct*100:.2f}%]"
        elif hs.strategy == "mean_stdev":
            detail = f" [k = {hs.stdev_factor}]"
        print(f"  [3] Hist.stretch: ENABLED   {hs.strategy}{detail}")
    else:
        print("  [3] Hist.stretch: disabled")
    print()


# ===========================================================================
#  Entry point
# ===========================================================================

def main() -> None:
    params = collect_parameters()
    _print_summary(params)

    if not prompt_yes_no("Proceed with processing?", default="y"):
        print("Aborted by user.")
        sys.exit(0)

    processor = UnderwaterImageProcessor(
        clip_limit=params["clip_limit"],
        tile_size=params["tile_size"],
        jpeg_quality=params["jpeg_quality"],
        color_corrector=params["color_corrector"],
        hist_stretcher=params["hist_stretcher"],
    )

    processor.batch_process(
        input_dir=params["input_dir"],
        output_dir=params["output_dir"],
        num_workers=params["num_workers"],
    )

    print(f"\nDone!  Processed images saved to: {params['output_dir']}")


if __name__ == "__main__":
    main()
