import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('PN_gbif_France_2005-2025_illustrated_CBN-med_train_val-10.0min.csv')
df_train = df[df['subset']=='train']
df_val = df[df['subset']=='val']

u_train, c_train = np.unique(df_train['speciesKey'], return_counts=True)
u_val, c_val = np.unique(df_val['speciesKey'], return_counts=True)
u = np.union1d(u_train, u_val)

print(f'Nb of unique speciesKey in Train: {len(u_train)}')
print(f'Nb of unique speciesKey in Val: {len(u_val)}')

# Extend both split unique counts to include the species which are missing
c_train_extended = []
for k, v in enumerate(u):
    if not v in u_train:
        c_train_extended.append(0)
    else:
        c_train_extended.append(int(c_train[np.argwhere(u_train==v)][0][0]))
c_train_extended = np.array(c_train_extended)
c_val_extended = []
for k, v in enumerate(u):
    if not v in u_val:
        c_val_extended.append(0)
    else:
        c_val_extended.append(int(c_val[np.argwhere(u_val==v)][0][0]))
c_val_extended = np.array(c_val_extended)

# Prepare descending cumulative counts representation
args_c_train_extended_sort_desc = np.argsort(c_train_extended)[::-1]
args_c_val_extended_sort_desc = np.argsort(c_val_extended)[::-1]

def plot_bars(x1, y1, x2, y2):
    plt.figure()
    plt.bar(x1, y1, label='train', color='blue', alpha=0.5)
    plt.bar(x2, y2, label='val', color='orange')
    n_zeros_train = len(np.where(y1==0)[0])
    n_zeros_val =len(np.where(y2==0)[0])
    plt.title('PN_gbif_France_2005-2025_illustrated train/val unique speciesKey distribution.\n'+\
             f'Nb of unique speciesKey in Train: {len(y1)-n_zeros_train}\n'+\
             f'Nb of unique speciesKey in Val: {len(y2)-n_zeros_val}')
    plt.legend()
    plt.xlabel('Unique speciesKey')
    plt.ylabel('Counts')
    plt.show()
    
plot_bars(np.arange(0, len(u), 1), c_train_extended[args_c_train_extended_sort_desc], np.arange(0, len(u), 1), c_val_extended[args_c_val_extended_sort_desc])
