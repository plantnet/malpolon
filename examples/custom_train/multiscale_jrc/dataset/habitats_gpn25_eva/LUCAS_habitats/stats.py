import numpy as np

def count_u_habitats_multilabel(df, col):
    u_habitats = {}
    for rowi, row in df.iterrows():
        ks = str(row[col]).split(';')
        for k in ks:
            u_habitats[k.strip()] = u_habitats.get(k.strip(), 0) + 1
    return u_habitats

def get_basic_stats(df, col_labels, labels_multilabel=True, return_stats=False):
    unique_fids = df['point_id'].unique()
    unique_labels = np.array(list(count_u_habitats_multilabel(df, col_labels).keys()))
    unique_gps = df.value_counts(['lon', 'lat']).reset_index(name='count')

    n_unique_fids = unique_fids.shape[0]
    n_unique_labels = unique_labels.shape[0]
    n_occu_valid_gps = unique_gps['count'].sum()
    n_occu_invalid_gps = df.shape[0] - n_occu_valid_gps
    n_occurrences = df.shape[0]
    n_plots = unique_gps.shape[0]
    n_unique_habitats = unique_labels.shape[0]

    print(f'Nb occurrences (illustrated): {n_occurrences}')
    print(f'Nb unique LUCAS IDs: {n_unique_fids}')
    print(f'Occurrences: {n_occu_valid_gps} ({100*n_occu_valid_gps/n_occurrences:.2f}%) with GPS | {n_occu_invalid_gps} ({100*n_occu_invalid_gps/n_occurrences:.2f}%) without')
    print(f'Nb of unique plots with valid GPS: {n_plots}')
    print()
    print(f'Nb unique habitats: {n_unique_habitats}')
    
    if return_stats:
        return n_unique_fids, n_unique_labels, n_occu_valid_gps, n_occu_invalid_gps, n_occurrences, n_plots, n_unique_habitats

