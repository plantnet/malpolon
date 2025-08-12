import argparse
import pandas as pd
from pathlib import Path

INFO = '\033[93m'
RESET = '\033[0m'
LINK = '\033[94m'


def remove_truplicates(df, cols=['lon','lat','speciesKey']):
    truplicates = df[df.duplicated(subset=cols, keep=False)]
    truplicates_counts = df.groupby(cols).size().reset_index(name='count')
    truplicates_cols_only = truplicates_counts[truplicates_counts['count'] > 1]
    df_no_truplicates = df.loc[~df.index.isin(truplicates.index)]
    assert len(df_no_truplicates), (len(df) - truplicates_cols_only['count'].sum())
    tmp = df_no_truplicates.groupby(cols).size().reset_index(name='count')
    assert tmp[tmp['count'] > 1]['count'].sum() == 0
    return df_no_truplicates

def main(fp_df, cols):
    df = pd.read_csv(fp_df)
    df_no_duplicates = remove_truplicates(df, cols=cols)
    out_fp = f'{Path(fp_df).stem}_no_{len(cols)}-duplicates{Path(fp_df).suffix}'
    df_no_duplicates.to_csv(out_fp, index=False)
    print(f'Exported {len(cols)}-duplicates free obs files at: \033[93m{out_fp}\033[0m')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i",
                        nargs=1,
                        required=True,
                        help='Input path of the observation CSV.')
    parser.add_argument("--duplicate_cols", "-d",
                        nargs='+',
                        required=False,
                        type=str,
                        default=['lon','lat','speciesKey'],
                        help='Columns to consider checking for grouped duplicates.')
    
    args = parser.parse_args()
    main(args.input[0], args.duplicate_cols[1])
