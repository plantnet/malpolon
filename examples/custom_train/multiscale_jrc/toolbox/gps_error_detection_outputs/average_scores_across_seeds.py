import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Average score values across multiple CSV files, grouped by ID."
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        help="Input CSV files (e.g. scores_cosine_seed1.csv scores_cosine_seed2.csv ...)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV file.",
    )
    args = parser.parse_args()
    
     # Default file selection if none provided
    if args.inputs:
        files = args.inputs
    else:
        files = sorted(Path(".").glob("scores_cosine_seed*.csv"))

    if not files:
        raise FileNotFoundError("No input files found.")

    # Read and concatenate
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    # Group by id:
    # - average the score
    # - keep the first value of all other columns
    result = (
        df.groupby("id", as_index=False)
          .agg({
              "label": "first",
              "score": "mean",
              "noise_type": "first"
          })
    )

    # Save
    output_file = "scores_cosine_average.csv"
    result.to_csv(output_file, index=False)

    print(f"Saved averaged scores to {output_file}")

if __name__ == "__main__":
    main()
