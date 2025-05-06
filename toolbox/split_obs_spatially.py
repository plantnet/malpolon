"""This script splits an obs csv in val/train based on the observations'
geogrpahic locations using the Verde package.

Author: Théo Larcher <theo.larcher@inria.fr>
"""

import argparse

import pandas as pd
from verde import train_test_split as spatial_tts

from malpolon.plot.map import plot_observation_dataset as plot_od


def main(input_path: str,
         spacing: float = 10 / 60,
         plot: bool = False,
         val_size: float = 0.15):
    """Perform a spatial train/val split on the input csv file.

    Parameters
    ----------
    obs_path : str
        obs CSV input file's path.
    spacing : float, optional
        size of the spatial split in degrees (or whatever unit the coordinates are in),
        by default 10/60
    plot : bool, optional
        if true, plots the train/val split on a 2D map,
        by default False
    val_size : float, optional
        size of the validaiton split, by default 0.15
    """
    input_name = input_path[:-4] if input_path.endswith(".csv") else input_path
    df = pd.read_csv(f'{input_name}.csv')
    coords, data = {}, {}
    for col in df.columns:
        if col in ['lon', 'lat']:
            coords[col] = df[col].to_numpy()
        else:
            data[col] = df[col].to_numpy()
    train_split, val_split = spatial_tts((coords['lon'], coords['lat']), tuple(data.values()),
                                         spacing=spacing, test_size=val_size)

    df_train = pd.DataFrame({'lon': train_split[0][0], 'lat': train_split[0][1]})
    df_val = pd.DataFrame({'lon': val_split[0][0], 'lat': val_split[0][1]})
    df_train['subset'] = ['train'] * len(df_train)
    df_val['subset'] = ['val'] * len(df_val)
    for train_data, val_data, col in zip(train_split[1], val_split[1], data.keys()):
        df_train[col] = train_data
        df_val[col] = val_data

    df_train_val = pd.concat([df_train, df_val])

    df_train_val.to_csv(f'{input_name}_train_val-{spacing*60}min.csv', index=False)
    df_train.to_csv(f'{input_name}_train-{spacing*60}min.csv', index=False)
    df_val.to_csv(f'{input_name}_val-{spacing*60}min.csv', index=False)

    if plot:
        plot_od(df=df_train_val, show_map=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path",
                        help="Path to the input csv obs file.",
                        default='GLC24_PA_metadata_train',
                        type=str)
    parser.add_argument("-s", "--spacing",
                        help="Size of the spatial split in degrees (or whatever unit the coordinates are in)",
                        default=10 / 60,
                        type=float)
    parser.add_argument("--val_size",
                        help="Size of the validation subset to produce.",
                        default=0.15,
                        type=float)
    parser.add_argument("-p", "--plot",
                        help="If true, plot the train/val split at the end of the script.",
                        action='store_true')
    args = parser.parse_args()
    main(args.input_path, args.spacing, plot=args.plot, val_size=args.val_size)
