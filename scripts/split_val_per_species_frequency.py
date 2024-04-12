from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm


def main(input_name: str, output_name:str, val_ratio: float = 0.05):
    """Split an obs csv in val/train.

    Performs a split with equal proportions of classes
    in train and val (if possible depending on the number
    of occurrences per species). If too few species are in
    the obs file, they are not included in the val split.

    The val proportion is defined by the val_ratio argument.
    
    Input csv is expected to have at least the following columns:
    ['speciesId']
    """
    pa_train = pd.read_csv(f'{input_name}.csv')
    pa_train['subset'] = ['train'] * len(pa_train)
    pa_train_uniques = np.unique(pa_train['speciesId'], return_counts=True)
    args_sorted = np.argsort(pa_train_uniques[1])
    pa_train_uniques_sorted_desc = (pa_train_uniques[0][args_sorted][::-1],
                                    pa_train_uniques[1][args_sorted][::-1])
    n_cls_val = deepcopy(pa_train_uniques_sorted_desc)
    for i, v in enumerate(n_cls_val[1]):
        n_cls_val[1][i] = round(v * val_ratio)

    indivisible_sid_n_rows = np.sum(n_cls_val[1][n_cls_val[1] < (1/val_ratio)])
    pa_val = pd.DataFrame(columns=pa_train.columns)
    for sid, n_sid in zip(tqdm(n_cls_val[0]), n_cls_val[1]):
        if n_sid >= 1:
            df_slice = pa_train[pa_train['speciesId'] == sid]
            pa_val = pd.concat([pa_val, df_slice.sample(n=n_sid)])
    pa_val['subset'] = ['val'] * len(pa_val)
    pa_train = pa_train.drop(pa_val.index)
    pa_train.to_csv(f'{input_name}_without_val-{val_ratio*100}%.csv', index=False)
    pa_val.to_csv(f'{output_name}-{val_ratio*100}%.csv', index=False)
    pa_train_val = pd.concat([pa_train, pa_val])
    pa_train_val.to_csv(f'{input_name}_val-{val_ratio*100}%.csv', index=False)
    print('Exported train_without_val, val, and train_val_split_by_species_frequency csvs.')
    print(f'{indivisible_sid_n_rows} rows were not included in val due to indivisibility by {val_ratio} (too few observations to split in at least 1 obs train / 1 obs val).')

if __name__ == '__main__':
    input_name = 'sample_obs'
    output_name = 'sample_obs'
    main(input_name, output_name)
