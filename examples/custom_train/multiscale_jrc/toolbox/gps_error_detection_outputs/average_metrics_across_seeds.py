import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Average CSVs with mixed types")

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of input CSV files"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file"
    )

    args = parser.parse_args()

    dfs = [pd.read_csv(f) for f in args.inputs]

    # Stack everything
    df_all = pd.concat(dfs, axis=0)

    # Split numeric vs non-numeric columns
    numeric_df = df_all.select_dtypes(include="number")
    string_df = df_all.select_dtypes(exclude="number")

    # Average numeric columns
    numeric_mean = numeric_df.groupby(level=0).mean()
    numeric_mean = numeric_mean.rename(
            columns={"roc_auc": "roc_auc_seeds_mean",
                     "pr_auc": "pr_auc_seeds_mean"}
        )
    numeric_std = numeric_df.groupby(level=0).std()
    numeric_std = numeric_std.rename(
            columns={"roc_auc": "roc_auc_seeds_std",
                     "pr_auc": "pr_auc_seeds_std"}
        )

    # For string columns: keep first occurrence per index
    string_first = string_df.groupby(level=0).first()

    # Combine back
    result = pd.concat([string_first, numeric_mean, numeric_std], axis=1)

    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
