#!/usr/bin/python3

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

PARSER = argparse.ArgumentParser(description='Preprocess habitat metadata')
PARSER.add_argument('--input_path', '-i',
                    type=str,
                    help='Path to the input CSV')
PARSER.add_argument('--output_path', '-o',
                    type=str,
                    default='habitat_metadata_lvl3.csv',
                    help='Path to the output CSV')
PARSER.add_argument('--habitat_column',
                    type=str,
                    default='Expert.System',
                    help='Habitat column name. Default: Expert.System')
PARSER.add_argument('--keep_columns',
                    type=list,
                    nargs='*',
                    default=['lon', 'lat', 'Expert.System', 'surveyId'],
                    help='Columns to keep in the output CSV. Default: PlotObservationID_eva, lon, lat, Expert.System, surveyId')
PARSER.add_argument("--sep",
                    required=False,
                    default=',',
                    help='Column separator.')

def filter_codes(code: str) -> str:
    """Apply filtering rules to a habitat code.

    Parameters
    ----------
    code : str
        habitat code

    Returns
    -------
    str
        filtered habitat code
    """
    new_code = code
    if '!' in code:
        new_code = code.replace('!', '')
    return new_code


def split_multiple_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Create a new row for each habitat code for rows with multiple codes.

    Parameters
    ----------
    df : pd.DataFrame
        input metadata dataframe

    Returns
    -------
    pd.DataFrame
        output filtered metadata dataframe
    """
    new_df = df.copy()
    for rowi, row in tqdm(df.iterrows(), total=len(df)):
        if ',' in row['habitatId']:
            hids = row['habitatId'].split(',')
            new_df.drop(index=rowi)
            for hid in hids:
                new_row = row.copy()
                new_row['habitatId'] = hid
                new_df.loc[len(new_df)] = new_row
    new_df.reset_index(drop=True, inplace=True)
    return new_df


def habitat_lvl3_to_na(hid: str) -> str:
    """Turn a habitat code that is not level 3 to NA.

    Parameters
    ----------
    hid : str
        habitat code

    Returns
    -------
    str
        transformed habitat code
    """
    res = hid
    if not (len(hid) == 3 or ('MA2' in hid and len(hid) == 5)):
        res = np.nan
    return res


def keep_habitat_lvl3(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out habitats that are not level 3.

    Parameters
    ----------
    df : pd.DataFrame
        input metadata dataframe

    Returns
    -------
    pd.DataFrame
        output filtered metadata dataframe
    """
    df['habitatId'] = df['habitatId'].apply(habitat_lvl3_to_na)
    df = df.dropna()
    return df


def preprocess_habitats_lvl3(df: pd.DataFrame,
                             keep_cols: list = ['PlotObservationID_eva', 'lon', 'lat', 'Expert.System', 'surveyId'],
                             habitat_col: str = 'Expert.System') -> pd.DataFrame:
    """Pre-process habitat metadata to keep only level 3 habitats.

    Parameters
    ----------
    df : pd.DataFrame
        input metadata dataframe
    keep_cols : list, optional
        columns to keep, by default ['PlotObservationID_eva', 'lon', 'lat', 'Expert.System', 'surveyId']
    habitat_col : str, optional
        habitat columns name identifier, by default 'Expert.System'

    Returns
    -------
    pd.DataFrame
        output filtered metadata dataframe
    """
    df = df[keep_cols]
    df.rename(columns={habitat_col: 'habitatId'}, inplace=True)
    df = df[df['habitatId'] != '~']
    df = df.dropna()
    df['habitatId'] = df['habitatId'].apply(filter_codes)
    df = split_multiple_codes(df.copy())
    df = keep_habitat_lvl3(df)
    # df = df.drop_duplicates(subset=['surveyId', 'habitatId']).reset_index(drop=True)
    return df


if __name__ == '__main__':
    args = PARSER.parse_args()
    habitat_metadata = pd.read_csv(args.input_path, sep=args.sep)
    habitat_metadata = preprocess_habitats_lvl3(habitat_metadata, keep_cols=args.keep_columns, habitat_col=args.habitat_column)
    habitat_metadata.to_csv(args.output_path, index=False)
