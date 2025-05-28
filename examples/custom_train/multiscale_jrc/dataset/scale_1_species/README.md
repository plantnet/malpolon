Pl@ntNet & Gbif illustrations of individual plant species go here.

The metadata file to be used is the merge of the `multimedia.tsv` and `PN_gbif_France_2005-2025_illustrated.csv` extract, based on the column `gbifID`, filtered on the desired zone. The 1st file contains the urls to fetch images online; while the 2nd file contains the coordinates.

The URLs to the images are under the column `identifier`.
The file paths are formed by the `gbifID` as the images are dumped directly in the dataset folder.
The GPS coordinates are retrieved from columns `['decimalLongitude', 'decimalLatitude']`.

Pl@ntNet's unique species IDs are registered under `scientificName`. So are other providers', but PN uses the Kew Royal Botanical Garden referencial, which might not be the case of other sources.

In `PN_gbif_France_2005-2025_illustrated_CBN-med.csv` there are 2991 unique species for 117 460 obs. That is ~39 obs per species.
However `speciesKey` has 2900 unique values.

- **Species distrib viz**
```python
df = pd.read_csv('PN_gbif_France_2005-2025_illustrated_CBN-med_train_val-10.0min.csv')
df_train = df[df['subset']=='train']
df_val = df[df['subset']=='val']

u_train, c_train = np.unique(df_train['speciesKey'], return_counts=True)
u_val, c_val = np.unique(df_val['speciesKey'], return_counts=True)
u = np.union1d(u_train, u_val)

# Extend both split unique counts to include the species which are missing
c_train_extended = []
for k, v in enumerate(u):
    if not v in u_train:
        c_train_extended.append(0)
    else:
        c_train_extended.append(c_train[np.argwhere(u_train==v)])
c_train_extended = np.array(c_train_extended)
c_val_extended = []
for k, v in enumerate(u):
    if not v in u_val:
        c_val_extended.append(0)
    else:
        c_val_extended.append(c_val[np.argwhere(u_val==v)])
c_val_extended = np.array(c_val_extended)

# Prepare descending cumulative counts representation
args_c_train_extended_sort_desc = np.argsort(c_train_extended)[::-1]
args_c_val_extended_sort_desc = np.argsort(c_val_extended)[::-1]

def plot_bars(x1, y1, x2, y2):
    plt.figure()
    plt.bar(x1, y1, label='train', color='blue', alpha=0.5)
    plt.bar(x2, y2, label='val', color='orange', alpha=0.5)
    plt.title('PN_gbif_France_2005-2025_illustrated train/val unique speciesKey distribution')
    plt.legend()
    plt.xlabel('Unique speciesKey')
    plt.ylabel('Counts')
    plt.show()
    
plot_bars(np.arange(0, len(u), 1), c_train_extended[args_c_train_extended_sort_desc], np.arange(0, len(u), 1), c_val_extended[args_c_val_extended_sort_desc])
```
