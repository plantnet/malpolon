import os
import argparse
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm


def check_file_exists(file_path):
    return os.path.isfile(file_path)

def check_file_type(file_path, expected_extensions):
    return any(file_path.lower().endswith(ext.lower()) for ext in expected_extensions)

def construct_fpath_with_suffix(base_path, suffix):
    base = Path(base_path)
    return str(base.with_name(base.stem + suffix + base.suffix))

def check_file_existence_in_columns(csv_path, output_path=None):
    """
    Check if files exist at the file paths in a CSV file with multiple columns.
    Save the name of the column under which the file exists in a new column.

    Args:
        csv_path (str): Path to the input CSV file.
        output_path (str): Path to save the output CSV file with results.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create a new column to store the column name where the file exists
    df['exists_in_column'] = None

    # Iterate over each row
    for index, row in df.iterrows():
        for column in df.columns:
            file_path = row[column]
            if isinstance(file_path, str) and os.path.exists(file_path):
                df.at[index, 'exists_in_column'] = column
                break  # Stop checking other columns once a match is found

    # Save the updated DataFrame to a new CSV file
    if output_path is not None:
        df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return df

def check_file_based_on_csv(csv_path, column_names, expected_extensions, suffix):
    df = pd.read_csv(csv_path)
    df['file_path_exists'] = [''] * len(df)
    df['file_type_correct'] = [False] * len(df)

    for rowi, row in tqdm(df.iterrows(), total=len(df)):
        for i, file_path in enumerate(row[column_names]):
            conditions = 0
            # file_path = construct_fpath_with_suffix(file_path, suffix)  # useful for raw metadata from JRC
            file_exists = check_file_exists(file_path)
            file_type_correct = check_file_type(file_path, expected_extensions)

            if file_exists:
                df.loc[rowi, 'file_path_exists'] = file_path
                conditions += 1
            if file_type_correct:
                df.loc[rowi, 'file_type_correct'] = file_type_correct
                conditions += 1
            if conditions >= 2:
                break

    num_existing_files = df['file_path_exists'].apply(lambda x: x != '').sum()
    num_type_correct = df['file_type_correct'].sum()
    print(f"File paths exist for {num_existing_files} / {len(df)} entries ({100 * num_existing_files / len(df):.2f} %).")
    print(f"File types correct for {num_type_correct} / {len(df)} entries ({100 * num_type_correct / len(df):.2f} %).")
    return df

def main():
    parser = argparse.ArgumentParser(description="Check file integrity based on CSV input.")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing file paths.')
    parser.add_argument('--column_names', type=str, nargs='+', required=True, help='Column name in the CSV that contains file paths.')
    parser.add_argument('--expected_extensions', type=str, nargs='+', required=True, help='List of expected file extensions.')
    parser.add_argument('--suffix', type=str, default='_modified', help='Suffix to append to file names for modified files.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output CSV with integrity check results.')

    args = parser.parse_args()

    results_df = check_file_based_on_csv(
        csv_path=args.csv_path,
        column_names=args.column_names,
        expected_extensions=args.expected_extensions,
        suffix=args.suffix
    )

    results_df.to_csv(args.output_path, index=False)
    print(f"Integrity check results saved to {args.output_path}")


if __name__ == '__main__':
    main()
# 1. Find all file paths which don't properly link an existing file (regardless of the reason)
# 2. Separate broken links between those whose fp is wrong VS those whose file is missing
# 3. Go through wrong fp, correct them and retrieve missing files
# 4. Flatten LUCAS data on this CBN-Med subset with a new script
# 5. Re-check integrity on the flattened dataset
# 6. Apply that script to the main dataset
