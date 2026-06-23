"""
Check that the trained YOLO weights exist before running ArtistMatch.

Usage:
    python download_model.py

This script does NOT download anything. It only verifies that best.pt is
present at the expected path. You must supply the weights yourself.
"""

from pathlib import Path
import sys

WEIGHTS_PATH = Path("runs/detect/train/weights/best.pt")


def main():
    if WEIGHTS_PATH.exists():
        print(f"[OK] Weights found: {WEIGHTS_PATH.resolve()}")
        sys.exit(0)

    print(
        f"\n[ERROR] Model weights not found: {WEIGHTS_PATH}\n"
        "\n"
        "  ArtistMatch requires a trained YOLOv8 model (best.pt) to detect\n"
        "  text regions on festival poster images. This file is not included\n"
        "  in the repository because it is too large for standard git.\n"
        "\n"
        "  To fix this, place your trained best.pt at:\n"
        f"    {WEIGHTS_PATH.resolve()}\n"
        "\n"
        "  If you need to train the model, see the dataset/ directory and\n"
        "  the YOLOv8 training docs: https://docs.ultralytics.com/modes/train/\n"
        "\n"
        "  For Docker / Hugging Face Spaces deployment, see README.md.\n",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
