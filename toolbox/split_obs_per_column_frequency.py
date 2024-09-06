"""This script splits an obs csv in val/train based on the frequency
of occurrences in the whole dataset.
It does NOT perform a spatial split.
"""

import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--input_path", "-i",
                    nargs=1,
                    required=False,
                    default=['sample_obs.csv'])
PARSER.add_argument("--output_name", "-o",
                    nargs=1,
                    required=False,
                    default=['sample_obs'])
PARSER.add_argument("--filter_id", "-f",
                    nargs=1,
                    required=False,
                    default=['speciesId'],
                    help='Column on which to perform frequency split.')
PARSER.add_argument("--sep",
                    required=False,
                    default=[','],
                    help='Column separator.')
PARSER.add_argument("--ratio", "-r",
                    required=False,
                    type=float,
                    default=0.05,
                    help='Ratio of data to split to validation (by default 5%)')
PARSER.add_argument("--keep_rares",
                    required=False,
                    type=int,
                    default=0,
                    help='This parameter determines how many rare occurrences'
                         ' (with at least 2 elements) should be split in val.')


def main(input_path: str,
         output_name: str,
         filter_id: str = 'speciesId',
         sep: str = ',',
         val_ratio: float = 0.05,
         keep_rares: int = 0):
    """Split an obs csv in val/train.

    Performs a split with equal proportions of classes
    in train and val (if possible depending on the number
    of occurrences per species). If too few species are in
    the obs file, they are not included in the val split.

    The val proportion is defined by the val_ratio argument.

    Parameters
    ----------
    input_path : str
        path to the input CSV
    output_name : str
        output name of the split CSVs
    filter_id : str, optional
        Column of the CSV on which to equally split the data.
        If too few occurrence of a label exist to be split, the occurrences
        will be given to the val set. By default 'speciesId'
    sep : str, optional
        _description_, by default ','
    val_ratio : float, optional
        percentage of data to split in validation, by default 0.05
    keep_rares : int, optional
        This parameter determines how many rare occurrences (with at least 2 elements) should be split in val.
        By defualt 1.
    """
    input_name = input_path[:-4] if input_path.endswith(".csv") else input_path
    pa_train = pd.read_csv(f'{input_name}.csv', sep=sep)
    pa_train['subset'] = ['train'] * len(pa_train)

    # Encode filter column in case to handle any type of data
    label_encoder = LabelEncoder().fit(pa_train[filter_id])
    pa_train[filter_id] = label_encoder.transform(pa_train[filter_id])

    pa_train_uniques = np.unique(pa_train[filter_id], return_counts=True)
    args_sorted = np.argsort(pa_train_uniques[1])
    u_cls_sorted_desc = (pa_train_uniques[0][args_sorted][::-1],
                         pa_train_uniques[1][args_sorted][::-1])

    # Compute the number of classes which do not have enough occurrences to be split by given ratio into an integer value
    indivisible_sid_n_rows = np.sum(u_cls_sorted_desc[1][u_cls_sorted_desc[1] < (1 / val_ratio)])
    pa_val = pd.DataFrame(columns=pa_train.columns)
    for sid, n_sid in zip(tqdm(u_cls_sorted_desc[0]), u_cls_sorted_desc[1]):
        # Rare occurrences are excluded from the val split and kept in train (or kept if keep_rares > 0)
        if n_sid >= (1 / val_ratio):
            df_slice = pa_train[pa_train[filter_id] == sid]
            pa_val = pd.concat([pa_val, df_slice.sample(n=round(n_sid * val_ratio))])
        elif keep_rares and n_sid > keep_rares:
            df_slice = pa_train[pa_train[filter_id] == sid]
            pa_val = pd.concat([pa_val, df_slice.sample(n=keep_rares)])

    pa_val['subset'] = ['val'] * len(pa_val)
    pa_train = pa_train.drop(pa_val.index)

    # Restore original filter_id values
    pa_train[filter_id] = label_encoder.inverse_transform(pa_train[filter_id])
    pa_val[filter_id] = label_encoder.inverse_transform(list(pa_val[filter_id].values))
    pa_train_val = pd.concat([pa_train, pa_val])

    pa_train.to_csv(f'{output_name}_split-{val_ratio*100}%_train.csv', index=False)
    pa_val.to_csv(f'{output_name}_split-{val_ratio*100}%_val.csv', index=False)
    pa_train_val.to_csv(f'{output_name}_split-{val_ratio*100}%_all.csv', index=False)
    print('Exported train_without_val, val, and train_val_split_by_species_frequency csvs.')

    rare_cls = label_encoder.inverse_transform(u_cls_sorted_desc[0])[np.where(u_cls_sorted_desc[1]<(1/val_ratio))]
    rare_cls_counts = u_cls_sorted_desc[1][np.where(u_cls_sorted_desc[1]<(1/val_ratio))]
    print(f'Rare classes were detected in the dataset: {dict(zip(rare_cls, rare_cls_counts))}')
    if keep_rares:
        print(f'{keep_rares} occurrences of rare classes have been included in val (if they contain at least {keep_rares + 1} occurrences).')
    else:
        print(f'Rare classes were not included in val. '
              f'{indivisible_sid_n_rows} rows were not included in val due to indivisibility by {val_ratio} (too few observations to split in at least 1 obs train / 1 obs val).')


if __name__ == '__main__':
    args = PARSER.parse_args()
    main(args.input_path[0], args.output_name[0],
         filter_id=args.filter_id[0], sep=args.sep[0], val_ratio=args.ratio, keep_rares=args.keep_rares)
