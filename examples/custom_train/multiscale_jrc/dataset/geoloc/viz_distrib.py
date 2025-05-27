import pandas as pd
import numpy as np
import argparse
from math import floor
import matplotlib as mpl
from matplotlib import pyplot as plt

def plot_bars_surveyId_distribution(df_train, df_val, scale='linear'):
     mpl.rcParams['hatch.linewidth'] = 5
     mpl.rcParams['hatch.color'] = 'yellow'
     _, ax = plt.subplots(figsize=(15, 5))
     df_ref_name = 'glc24_pa_train_CBN-med'
     ax.set_title(f"glc24_pa_train_CBN-med: surveyId distribution ({scale})")
     counts_speciesId = pd.concat([df_train, df_val], ignore_index=True)['surveyId'].value_counts()
     counts_speciesId_train = df_train['surveyId'].value_counts()
     counts_speciesId_val = df_val['surveyId'].value_counts()
     counts_speciesId_train = counts_speciesId_train.reindex(counts_speciesId.index, fill_value=0)
     counts_speciesId_val = counts_speciesId_val.reindex(counts_speciesId.index, fill_value=0)
     ax.bar(np.arange(0, len(counts_speciesId), 1), counts_speciesId_train.values, label=f"Dataset {df_ref_name or '???'}_train surveyId counts", facecolor='deepskyblue', edgecolor=None)
     ax.bar(np.arange(0, len(counts_speciesId), 1), counts_speciesId_val.values, label=f"Dataset {df_ref_name or '???'}_val surveyId counts", facecolor='purple', alpha=0.5)
     ax.set_yscale(scale); ax.set_xlabel('surveyId'); ax.set_ylabel(f'Count ({scale} scale)'); ax.legend()
     ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim()); ax2.set_xlabel('Nb of unique surveyId')
     ax.set_xticks(np.arange(len(counts_speciesId))[1::floor(len(counts_speciesId)*0.1)], counts_speciesId.index[1::floor(len(counts_speciesId)*0.1)], rotation=45)
     ax2.set_xticks(np.arange(len(counts_speciesId))[1::floor(len(counts_speciesId)*0.1)], np.arange(len(counts_speciesId))[1::floor(len(counts_speciesId)*0.1)])
     plt.show()
     

def main(fp_df_train, fp_df_val, scale):
    df_train = pd.read_csv(fp_df_train)
    df_val = pd.read_csv(fp_df_val)
    plot_bars_surveyId_distribution(df_train, df_val, scale=scale)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_paths", "-i",
                        nargs='+',
                        required=True)
    parser.add_argument("--scale", "-s",
                        nargs=1,
                        required=False,
                        type=str,
                        default=['linear'])
    args = parser.parse_args()
    main(args.input_paths[0], args.input_paths[1], args.scale[0])
