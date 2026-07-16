#!/usr/bin/env python3

import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count the number of semicolon-separated string elements in a CSV column."
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input CSV file"
    )

    parser.add_argument(
        "-c", "--column",
        default="filepath",
        help="Name of the column to process (default: filepath)"
    )

    parser.add_argument(
        "--sep",
        default=",",
        help="CSV separator character (default: ',')"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Read CSV
    df = pd.read_csv(args.input, sep=args.sep)

    if args.column not in df.columns:
        raise ValueError(
            f"Column '{args.column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    counts = []

    # Iterate over column values
    for value in df[args.column]:
        if pd.isna(value) or value == "":
            count = 0
        else:
            count = len(str(value).split(";"))

        counts.append(count)

    # Print counts
    for idx, count in enumerate(counts):
        print(f"{idx}: {count}")

    # Optional summary
    print("\nTotal rows:", len(counts))
    print("Total elements:", sum(counts))


if __name__ == "__main__":
    main()
