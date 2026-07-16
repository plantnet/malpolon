#!/usr/bin/env python3

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp", ".gif"
}


def main():
    parser = argparse.ArgumentParser(
        description="Recursively detect truncated image files."
    )
    parser.add_argument(
	"--root",
	help="Root directory to search."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="truncated_images.txt",
        help="Output text file (default: truncated_images.txt)",
    )

    args = parser.parse_args()

    # Collect image files so tqdm knows the total
    image_files = [
        p for p in Path(args.root).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    truncated = []

    for path in tqdm(image_files, desc="Checking images", unit="img"):
        try:
            with Image.open(path) as img:
                img.load()  # Force full decoding
        except OSError as e:
            if "image file is truncated" in str(e):
                tqdm.write(f"TRUNCATED: {path}")
                truncated.append(str(path))
            else:
                tqdm.write(f"OTHER ERROR: {path}: {e}")

    with open(args.output, "w") as f:
        f.write("\n".join(truncated))

    print(f"\nFound {len(truncated)} truncated images.")
    print(f"Saved list to {args.output}")


if __name__ == "__main__":
    main()
