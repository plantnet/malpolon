import os
import argparse
import pandas as pd
import numpy as np

from pathlib import Path


def clean_bad_file_paths(
    df,
    file_cols = ['file_path_gisco_north', 'file_path_gisco_south', 'file_path_gisco_east', 'file_path_gisco_west', 'file_path_gisco_point', 'file_path_gisco_cover']
):
    print("[Info] Filtering out badly formatted file paths (turning them to empty strings)..")
    for c in file_cols:
        df[c] = df[c].apply(lambda x: str(x) if (str(x).lower()).endswith(tuple(['.jpg', '.jpeg', '.png'])) else '')
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
    """Gather columns of file paths into a single one."""
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
    """Check if file paths lead to actually existing files on the disk."""
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


def main(df_fp, fp_out):
    df = pd.read_csv(df_fp)
    df = clean_bad_file_paths(df)
    df = cut_file_path_to_local_path(df)
    # df = gather_file_paths(df)
    df = expand(df)
    # df = check_existing_file_paths(df)
    df = keep_essentials_cols(df)
    df = rename_cols(df)
    df.to_csv(f'{fp_out}', index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        type=str,
                        help="Input dataset.",)
    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output file path.",)
    args = parser.parse_args()                        
    main(args.input, args.output)

