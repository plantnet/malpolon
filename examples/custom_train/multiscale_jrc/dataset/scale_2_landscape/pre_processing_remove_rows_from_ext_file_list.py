import argparse
import pandas as pd


def remove_row_based_on_fps_list(df, fps_fp=None):
    with open(fps_fp, 'r') as f:
        fps = [f.strip() for f in f.readlines()]
    
    values_set = set(fps)

    def clean_path(s):
        return "" if any(p in values_set for p in s.split(';')) else s
        
    # Case 1: the df is expanded (1 row per image)
    if df['point_id'].nunique() <= len(df):
        df = df[~df['file_path'].isin(fps)]
    # Case 2: the df is gathered (1 row per site, multiple images per site)
    else:
        df['file_path'] = df['file_path'].map(clean_path)
        df = df[df['file_path'] != ""]
    return df
        
def main(df_fp, fp_out, fps_remove):
    df = pd.read_csv(df_fp)
    df = remove_row_based_on_fps_list(df, fps_remove)
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