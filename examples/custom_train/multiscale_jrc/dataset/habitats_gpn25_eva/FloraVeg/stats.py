def get_basic_stats(df_occurrences, df_labels, return_stats=False):
    unique_fids = df_occurrences['id_floraveg'].unique()
    unique_labels = df_labels['habitats'].unique()
    unique_syntaxons_occurrences = df_occurrences['syntaxon_floraveg'].unique()
    unique_syntaxons_labels = df_labels['syntaxons'].unique()
    unique_gps = df_occurrences.value_counts(['lon', 'lat']).reset_index(name='count')

    n_unique_fids = unique_fids.shape[0]
    n_unique_labels = unique_labels.shape[0]
    n_occu_valid_gps = unique_gps['count'].sum()
    n_occu_invalid_gps = df_occurrences.shape[0] - n_occu_valid_gps
    n_occurrences = df_occurrences.shape[0]
    n_plots = unique_gps.shape[0]
    n_unique_habitats = unique_labels.shape[0]

    print(f'Nb occurrences (illustrated): {n_occurrences}')
    print(f'Nb unique FloraVeg IDs: {n_unique_fids}')
    print(f'Occurrences: {n_occu_valid_gps} ({100*n_occu_valid_gps/n_occurrences:.2f}%) with GPS | {n_occu_invalid_gps} ({100*n_occu_invalid_gps/n_occurrences:.2f}%) without')
    print(f'Nb of unique plots with valid GPS: {n_plots}')
    print()
    print(f'Nb unique habitats: {n_unique_habitats}')
    print(f'Nb unique syntaxons (from occurrences file): {len(unique_syntaxons_occurrences)}')
    print(f'Nb unique syntaxons (from labels file): {len(unique_syntaxons_labels)}\n')
    
    if return_stats:
        return n_unique_fids, n_unique_labels, n_occu_valid_gps, n_occu_invalid_gps, n_occurrences, n_plots, n_unique_habitats