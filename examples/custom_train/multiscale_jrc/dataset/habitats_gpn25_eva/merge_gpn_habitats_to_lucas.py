#!/usr/bin/env python3

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Merge GPN habitat metadata into an input CSV."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input CSV (data CSV).",
    )

    parser.add_argument(
        "-g",
        "--gpn_habitat_metadata",
        required=True,
        help="CSV containing GPN habitat metadata.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV.",
    )

    parser.add_argument(
        "--merge_key_input",
        default="gpn_habitat_id",
        help="Merge column in the input CSV (default: %(default)s).",
    )

    parser.add_argument(
        "--merge_key_metadata",
        default="raster_value",
        help="Merge column in the metadata CSV (default: %(default)s).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load CSVs
    # ------------------------------------------------------------------

    df_input = pd.read_csv(args.input)
    df_metadata = pd.read_csv(args.gpn_habitat_metadata)

    # ------------------------------------------------------------------
    # Select and rename metadata columns
    # ------------------------------------------------------------------

    df_metadata = df_metadata[
        [
            args.merge_key_metadata,
            "habitat_code",
            "name",
        ]
    ].rename(
        columns={
            "habitat_code": "gpn_habitat_code",
            "name": "gpn_habitat_name",
        }
    )

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    df_output = df_input.merge(
        df_metadata,
        left_on=args.merge_key_input,
        right_on=args.merge_key_metadata,
        how="left",
        validate="many_to_one",
    )

    if args.merge_key_input != args.merge_key_metadata:
        df_output = df_output.drop(columns=args.merge_key_metadata)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    assert len(df_output) == len(df_input), (
        f"Row count changed after merge: "
        f"{len(df_input)} -> {len(df_output)}"
    )

    assert df_output["point_id"].nunique() == df_input["point_id"].nunique(), (
        "Number of unique point_id changed after merge"
    )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def match_lvl1(row):
        gpn_code = row["gpn_habitat_code"]
        lvl1_codes = row["habitats_code_lvl1"]

        if pd.isna(gpn_code) or pd.isna(lvl1_codes):
            return False

        target = str(gpn_code)[0]

        return any(
            str(code).strip()[0] == target
            for code in str(lvl1_codes).split(";")
            if str(code).strip()
        )

    def match_lvl2(row):
        gpn_code = row["gpn_habitat_code"]
        lvl2_codes = row["habitats_code_lvl2"]

        if pd.isna(gpn_code) or pd.isna(lvl2_codes):
            return False

        target = str(gpn_code)[:2]

        return any(
            str(code).strip()[:2] == target
            for code in str(lvl2_codes).split(";")
            if str(code).strip()
        )

    lvl1_matches = df_output.apply(match_lvl1, axis=1).sum()
    lvl2_matches = df_output.apply(match_lvl2, axis=1).sum()

    print(
        f"Rows where gpn_habitat_code level-1 matches habitats_code_lvl1: "
        f"{lvl1_matches}/{len(df_output)} ({100 * lvl1_matches / len(df_output):.2f}%)"
    )

    print(
        f"Rows where gpn_habitat_code level-2 matches habitats_code_lvl2: "
        f"{lvl2_matches}/{len(df_output)} ({100 * lvl2_matches / len(df_output):.2f}%)"
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    df_output.to_csv(args.output, index=False)

    print(f"Wrote {len(df_output)} rows to {args.output}")


if __name__ == "__main__":
    main()
