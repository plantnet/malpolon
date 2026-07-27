import os
import argparse
import pandas as pd
import numpy as np

from pathlib import Path

def expand(df):
    """Expand the df by melting every file path column in a new row.
    
    Values from other columns are replicated.
    """
    file_cols = ['file_path_gisco_north', 'file_path_gisco_south', 'file_path_gisco_east', 'file_path_gisco_west', 'file_path_gisco_point', 'file_path_gisco_cover']
    id_vars = [col for col in df.columns if col not in file_cols]
     
    # Melt to long format, keeping other metadata
    expanded_df = pd.melt(df, id_vars=id_vars, value_vars=file_cols,
                          var_name='image_source', value_name='file_path')
    expanded_df = expanded_df.dropna(subset='file_path')
    return expanded_df

def remove_non_image(df_ref):
    df = df_ref[df_ref['file_path'].apply(lambda x: isinstance(x, str) and x.lower().endswith('.jpg'))]
    return df

def cut_file_path_to_local_path(df, col='file_path'):
    """Adapt the file paths from gisco server format to local."""
    df2 = df.copy()
    df2['file_path'] = df2['file_path'].apply(lambda x: '/'.join(x.split('/')[-5:]))
    return df2
    

def keep_essentials_cols(df, cols_to_keep):
    """Reduce the size of the dataframe file by keeping only certain columns."""
    cols_to_keep = np.array(cols_to_keep)
    cols_exist = np.array([col in df.columns for col in cols_to_keep])
    try:
        assert all(cols_exist)
    except AssertionError as e:
        print(f"\n[Error] The following cols aren't in the input dataframe and thus can't be kept: {cols_to_keep[np.array(~cols_exist)]}\n")
        raise e
    return df[cols_to_keep]
    
def check_existing_file_paths(df_ref, prefix='LUCAS/'):
     """Check if file paths lead to actually existing files on the disk."""
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
    

def main(df_fp, fp_out):
    df = pd.read_csv(df_fp)
    df = expand(df)
    df = cut_file_path_to_local_path(df)
    df = check_existing_file_paths(df)
    df = keep_essentials_cols(df, ['id', 'gps_long', 'gps_lat', 'gps_altitude', 'file_path', 'full_path', 'image_source'])
    df.to_csv(f'{fp_out}')
    

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
    


