import os
import re
import csv
import argparse
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Union


def clean_bad_file_paths(
    df,
    file_cols = ['file_path_gisco_north', 'file_path_gisco_south', 'file_path_gisco_east', 'file_path_gisco_west', 'file_path_gisco_point', 'file_path_gisco_cover']
):
    print("[Info] Filtering out badly formatted file paths (turning them to empty strings)..")
    if os.path.exists("pre_processing_bad_file_paths.csv"):
        os.remove("pre_processing_bad_file_paths.csv")
    
    def export_update_rejects(point_id: Union[int, str], file_path: str):
        """Export a rejected file path to the CSV storage file."""

        filename = "pre_processing_bad_file_paths.csv"
        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["point_id", "file_path"])

            writer.writerow([point_id, file_path])

    def regex_filter(row, file_col, verbose=False):
        """Filter file paths based on strict regex rules.
        
        The conditions in match are written so to keep the URL of the server hidden.
        """
        res = ''
        fp = row[file_col]
        point_id = row['point_id']
        re_rule_gisco = "^lucas\/photos\/(\d{4})\/([A-Z]{2})\/(\d{3})\/(\d{3})\/(\d+)([A-Za-z]+)\.(\w+)$"
        re_rule_jeodpp_cover = "^LUCAS\/LUCAS_COVER\/(LUCAS\d{4})\/([A-Z]{2})\/(\d{3})\/(\d{3})\/(\w+)\.(\w+)$"
        
        if all(x in fp for x in ('gisco', 'europa.eu')):
            if re.match(re_rule_gisco, fp.split('europa.eu/')[1]):
                res = fp
            else:
                export_update_rejects(point_id, fp)
                if verbose:
                    print(f'[Warning] File path {fp} for point_id {point_id} in col {file_col} does not match the expected regex rule for gisco paths. It will be turned to an empty string.')
        elif all(x in fp for x in ('jeodpp', 'europa.eu', 'LUCAS_COVER')):
            if re.match(re_rule_jeodpp_cover, fp.split('europa.eu/')[1]):
                res = fp
            else:
                export_update_rejects(point_id, fp)
                if verbose:
                    print(f'[Warning] File path {fp} for point_id {point_id} in col {file_col} does not match the expected regex rule for gisco paths. It will be turned to an empty string.')
        elif 'europa.eu' in fp:
            raise NotImplementedError(f"File path '{fp}' not handled in the cleaning of file paths. Only provide either gisco or jeodpp paths.")
        row[file_col] = res
        return row

    for c in file_cols:
        df[c] = df[c].fillna(value='')
        # df[c] = df[c].apply(lambda x: str(x) if (str(x).lower()).endswith(tuple(['.jpg', '.jpeg', '.png'])) else '')  # Weak filter
        df[c] = df.apply(lambda row: regex_filter(row[['point_id', c]], c) if isinstance(row[c], str) else '', axis=1)[c]  # Hard filter
    if os.path.exists("pre_processing_bad_file_paths.csv") and os.path.getsize("pre_processing_bad_file_paths.csv") > 0:
        n_lines = os.getsize("pre_processing_bad_file_paths.csv")
        print(f"[Info] Found {n_lines} badly formatted file paths were found and exported to pre_processing_bad_file_paths.csv. Please check this file for details.")
    return df


def cut_file_path_to_local_path(
    df,
    file_cols = ['file_path_gisco_north', 'file_path_gisco_south', 'file_path_gisco_east', 'file_path_gisco_west', 'file_path_gisco_point', 'file_path_gisco_cover']
):
    """Adapt the file paths from gisco server format to local."""
    print("[Info] Changing file paths to exclude server path...")
    for c in file_cols:
        if c in df.columns:
            df[c] = df[c].fillna(value='')
            df[c] = df[c].apply(lambda x: '/'.join(x.split('/')[-5:]))
    return df


def gather_file_paths(df):
    """Gather columns of file paths into a single one.
    
    DOES NOT collapses the df to 1 row = 1 point_id.
    """
    print("[Info] Gathering file paths columns into a unique one...")
    df_gathered = df.copy()
    file_cols = ['file_path_gisco_north', 'file_path_gisco_south', 'file_path_gisco_east', 'file_path_gisco_west', 'file_path_gisco_point', 'file_path_gisco_cover']
    df_gathered["file_path"] = (
        df_gathered[file_cols]
        .fillna("")
        .apply(lambda row: ";".join(x for x in row if x), axis=1)
    )
    return df_gathered

def expand(df):
    """Expand the df by melting every file path column in a new row.
    
    Values from other columns are replicated.
    """
    print("[Info] Expanding df by creating a new line for every file path...")
    file_cols = ['file_path_gisco_north', 'file_path_gisco_south', 'file_path_gisco_east', 'file_path_gisco_west', 'file_path_gisco_point', 'file_path_gisco_cover']
    id_vars = [col for col in df.columns if col not in file_cols]

    # Melt to long format, keeping other metadata
    expanded_df = pd.melt(df, id_vars=id_vars, value_vars=file_cols,
                          var_name='image_source', value_name='file_path')
    expanded_df = expanded_df.dropna(subset='file_path')
    expanded_df = cut_file_path_to_local_path(expanded_df, file_cols=['file_path'])
    return expanded_df

def keep_essentials_cols(df, root_path='LUCAS/'):
    """Reduce the size of the dataframe file by keeping only certain columns."""
    print("[Info] Keeping essential columns...")
    if 'full_path' not in df.columns:
        df['full_path'] = df['file_path'].apply(lambda x: ';'.join([f"{root_path}{x}" for x in x.split(';')]))
    cols_to_keep = np.array(['point_id', 'gps_long', 'gps_lat', 'gps_altitude', 'year', 'file_path', 'full_path'])
    cols_exist = np.array([col in df.columns for col in cols_to_keep])
    try:
        assert all(cols_exist)
    except AssertionError as e:
        print(f"\n[Error] The following cols aren't in the input dataframe and thus can't be kept: {cols_to_keep[np.array(~cols_exist)]}\n")
        raise e
    df_essentials = df[cols_to_keep]
    df_essentials = df_essentials.dropna(subset=['point_id', 'gps_long', 'gps_lat', 'file_path'])
    df_essentials['id'] = df_essentials['point_id']
    df_essentials = df_essentials[df_essentials['file_path'] != '']
    return df_essentials
    
def check_existing_file_paths(df_ref, prefix='LUCAS/'):
    """Check if file paths lead to actually existing files on the disk.
    
    [DEPRECATED]
    This was used for the CBN-Med subset. Now, filex are checked using bash script
    `check_file_exist_on_disk_multithread_simple[_progressbar].sh`.
    """
    print("[Info] Checking if file paths exist on disk...")
    df = df_ref.copy()
    df['full_path'] = prefix + df['file_path'].astype(str)
    prefix_missing = prefix+'../../lucas_missing/'
    df['full_path_missing'] = prefix_missing + df['file_path'].astype(str)
    prefix_2022 = prefix+'../../lucas_photos_all_2022/'
    df['full_path_2022'] = prefix_2022 + df['file_path'].astype(str)
    prefix_cover = prefix+'../../lucas_cover/'
    df['full_path_cover'] = prefix_cover + df['file_path'].astype(str)
    df['full_path_cover'] = df['full_path_cover'].apply(lambda x: Path(x).parent / Path(f"LUCAS{x.split('/')[4]}_{str(Path(x).stem)[:-1]}_Cover.jpg") if x.endswith('C.jpg') else x)
    exists = df['full_path'].apply(os.path.exists)
    exists_missing = df['full_path_missing'].apply(os.path.exists)
    exists_2022 = df['full_path_2022'].apply(os.path.exists)
    exists_cover = df['full_path_cover'].apply(os.path.exists)
    df['exists'] = exists | exists_missing | exists_2022 | exists_cover
    diff = len(df) - df['exists'].sum()
    print(f'There are {diff} files in the CSV which were not found on disk. That\'s {100*diff/len(df):3f}% of the data')
    return df

def rename_cols(df):
    print("[Info] Renaming columns...")
    cols_map = {'gps_long': 'lon', 'gps_lat': 'lat'}
    df_renamed = df.rename(columns=cols_map)
    return df_renamed


def remove_row_based_on_fps_list(df, fps_fp=None):
    with open(fps_fp, 'r') as f:
        fps = f.readlines()
    
    values_set = set(fps)

    def clean_path(s):
        return "" if any(p in values_set for p in s.split(';')) else s
        
    # Case 1: the df is expanded (1 row per image)
    if df['point_id'].nunique() >= len(df):
        df = df[~df['file_path'].isin(fps)]
    # Case 2: the df is gathered (1 row per site, multiple images per site)
    else:
        df['file_path'] = df['file_path'].map(clean_path)
        df = df[df['file_path'] != ""]
    

def main(df_fp, fp_out, fps_remove):
    df = pd.read_csv(df_fp)
    df = clean_bad_file_paths(df)
    df = cut_file_path_to_local_path(df)
    # df = gather_file_paths(df)  # ';'-separated string formatting of col "file_path" (gather the values of north, south, west... views to 1 col "file_path" and deletes other cols)
    df = expand(df)  # Legacy formatting: 1 row = 1 image. The dataframe is expanded by how many unique views there are per site.
    
    # /!\ Legacy method used for the CBN-Med subset. See docstring. /!\
    # df = check_existing_file_paths(df)

    df = keep_essentials_cols(df)
    df = rename_cols(df)
    if fps_remove:
        remove_row_based_on_fps_list(df, fps_remove)
    df.to_csv(f'{fp_out}', index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        type=str,
                        help="Input dataset.",)
    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output file path.",)
    parser.add_argument("-r", "--remove",
                        type=str,
                        default=None,
                        help="Filepath to a text file containing file paths to remove.",)
    args = parser.parse_args()                        
    main(args.input, args.output, args.remove)

