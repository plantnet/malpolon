"""GeoLifeCLEF23 time series example module.

This module provides an example of how to load and use GeoLifeCLEF2023 time
series datasets using the framework developped for the challenge and
incorporated in malpolon at `malpolon.data.datasets.geolifeclef2023`.
"""

import argparse
import random

from malpolon.data.datasets.geolifeclef2023 import (
    CSVTimeSeriesProvider, MultipleCSVTimeSeriesProvider, TimeSeriesDataset)


def main(display: bool = True):
    """Run GLC23 time series example script."""
    data_path = 'dataset/sample_data/'  # root path of the data
    # configure providers
    ts_red = CSVTimeSeriesProvider(data_path + 'SatelliteTimeSeries/time_series_red.csv')
    ts_multi = MultipleCSVTimeSeriesProvider(data_path + 'SatelliteTimeSeries/',
                                             select=['red', 'blue'])
    ts_all = MultipleCSVTimeSeriesProvider(data_path + 'SatelliteTimeSeries/')

    # create dataset
    dataset = TimeSeriesDataset(occurrences=data_path + 'Presence_only_occurrences/Presences_only_train_sample.csv',
                                providers=[ts_red, ts_multi, ts_all])

    # print random tensors from dataset
    ids = [random.randint(0, len(dataset) - 1) for i in range(5)]
    for i in ids:
        tensor = dataset[i][0]
        label = dataset[i][1]
        print(f'Tensor type: {type(tensor)}, tensor shape: {tensor.shape}, '
              f'label: {label}')
        if display:
            dataset.plot_ts(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", help="Plot patches.",
                        nargs='*', action='store')
    args = parser.parse_args()
    display = args.plot is not None
    main(display)
