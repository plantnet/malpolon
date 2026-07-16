#!/usr/bin/env python3

import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp", ".gif"
}


def check_image(path):
    """
    Returns:
        ("truncated", path)  if image is truncated
        ("other", path, err) for other OSErrors
        None                 if image is OK
    """
    try:
        with Image.open(path) as img:
            img.load()
        return None

    except OSError as e:
        msg = str(e)
        if "image file is truncated" in msg:
            return ("truncated", str(path))
        else:
            return ("other", str(path), msg)


def main():

    parser = argparse.ArgumentParser(
        description="Recursively detect truncated image files."
    )

    parser.add_argument("--root")

    parser.add_argument(
        "-o",
        "--output",
        default="truncated_images.txt",
    )

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=cpu_count(),
        help="Number of worker processes (default: all CPUs).",
    )

    args = parser.parse_args()

    image_files = [
        p for p in Path(args.root).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Found {len(image_files):,} images.")
    print(f"Using {args.jobs} worker processes.")

    truncated = []

    with Pool(args.jobs) as pool:

        for result in tqdm(
            pool.imap_unordered(check_image, image_files, chunksize=64),
            total=len(image_files),
            desc="Checking images",
            unit="img",
        ):

            if result is None:
                continue

            if result[0] == "truncated":
                tqdm.write(f"TRUNCATED: {result[1]}")
                truncated.append(result[1])

            else:
                tqdm.write(f"OTHER ERROR: {result[1]}: {result[2]}")

    with open(args.output, "w") as f:
        f.write("\n".join(truncated))

    print()
    print(f"Found {len(truncated)} truncated images.")
    print(f"Saved list to {args.output}")


if __name__ == "__main__":
    main()
