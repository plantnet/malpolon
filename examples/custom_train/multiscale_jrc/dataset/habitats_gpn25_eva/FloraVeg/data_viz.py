import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly_upset.plotting import plot_upset
from sklearn.preprocessing import LabelEncoder


def bar_plot_habitats_soft_multilabel(df, counts, title='', plot_width=10, fp_out=False, plot=True):
    counts_per_k_multilabel = counts.groupby(counts).groups
    uhabitats_per_k_multilabel = {}
    colors = ['#FF2A00']
    uhabitats_per_k_multilabel['All'] = df['habitats_code'].nunique()
    bar_labels = [f'{uhabitats_per_k_multilabel['All']}\n({100*uhabitats_per_k_multilabel['All']/uhabitats_per_k_multilabel['All']:.2f}%)']
    for k in range(min(counts_per_k_multilabel.keys()), max(counts_per_k_multilabel.keys())+1, 1):
        uhabitats_per_k_multilabel[str(k)] = df[df['id_floraveg'].isin(counts_per_k_multilabel[k])]['habitats_code'].nunique()
        bar_labels.append(f'{uhabitats_per_k_multilabel[str(k)]}\n({100*uhabitats_per_k_multilabel[str(k)]/uhabitats_per_k_multilabel['All']:.2f}%)')
        colors.append('orange')
    

    # Plot
    fig, ax = plt.subplots(figsize=(plot_width, 5))
    plt.bar(list(uhabitats_per_k_multilabel.keys()), list(uhabitats_per_k_multilabel.values()), color=colors)

    ax.set_xlabel('k')
    ax.set_ylabel("Nb of unique habitats")
    ax.set_ylim([0, 250])
    ax.set_title(
        f"Distribution of unique habitats for each 1-to-k soft multilabel samples\n",
        fontsize=11
    )
    # fig.text(
    #     0.5, 0.94,                     # same horizontal center
    #     f"{title}",
    #     ha='center',
    #     fontsize=15
    # )

    plt.xticks(rotation=0)

    # Add bar labels
    ax.bar_label(ax.containers[0], bar_labels, padding=1, label_type='edge')

    # --- Floating info box ---
    # n_unique_habitats = df['habitats'].nunique()
    # n_unique_syntaxons = df['syntaxons'].nunique()

    # info_text = (
    #     f"Unique habitats: {n_unique_habitats}\n"
    #     f"Unique syntaxons: {n_unique_syntaxons}"
    # )

    # ax.text(
    #     0.16, 0.96,                # position (top-left corner)
    #     info_text,
    #     transform=ax.transAxes,    # relative to axes
    #     fontsize=10,
    #     verticalalignment='top',
    #     horizontalalignment='right',
    #     bbox=dict(
    #         boxstyle="round,pad=0.3",
    #         facecolor="white",
    #         alpha=0.8,
    #         edgecolor="gray"
    #     )
    # )
    # -------------------------
    plt.tight_layout()
    if fp_out:
        plt.savefig(fp_out, bbox_inches='tight')

    if plot:
        plt.show()
    plt.close()
    return fig, counts

def bar_plot_distribution_with_floating_text(df, col, top_k=20, title='', plot_width=10, fp_out=False, plot=True):
    # Count occurrences
    counts = df[col].value_counts()

    # Number of values that appear only once
    n_singletons = (counts == 1).sum()

    counts_plot = counts.head(top_k).copy()

    # Group remaining into "Other"
    if len(counts) > top_k:
        counts_plot["Other"] = counts.iloc[top_k:].sum()

    # Plot
    fig, ax = plt.subplots(figsize=(plot_width, 5))

    counts_plot.plot.bar(ax=ax)

    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(
        f"Habitats with only 1 occurrence: {n_singletons}",
        fontsize=11
    )
    fig.text(
        0.5, 0.94,                     # same horizontal center
        f"{title}",
        ha='center',
        fontsize=15
    )

    plt.xticks(rotation=90)

    # Add bar labels
    ax.bar_label(ax.containers[0], padding=3)

    # --- Floating info box ---
    n_unique_habitats = df['habitats'].nunique()
    n_unique_syntaxons = df['syntaxons'].nunique()

    info_text = (
        f"Unique habitats: {n_unique_habitats}\n"
        f"Unique syntaxons: {n_unique_syntaxons}"
    )

    ax.text(
        0.16, 0.96,                # position (top-left corner)
        info_text,
        transform=ax.transAxes,    # relative to axes
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.8,
            edgecolor="gray"
        )
    )
    # -------------------------

    if fp_out:
        plt.savefig(fp_out, bbox_inches='tight')

    if plot:
        plt.show()
    plt.close()
    return fig, counts

def bar_plot_habitats_distribution_VS_syntaxons(data, top_k=20, title='', plot_width=10, fp_out=False, plot=True):
    # Count occurrences
    counts = (
        data.groupby('habitats')['syntaxons']
          .nunique()
          .reset_index(name='syntaxons')
    )
    unique_habitats = data['habitats'].value_counts()
    display(data.head(3))
    
    # Number of values that appear only once
    n_singletons = (counts[counts['syntaxons'] == 1]).shape[0]

    counts_plot = counts.head(top_k).copy()

    # Group remaining into "Other"
    if len(counts) > top_k:
        counts_plot.loc["Other"] = counts.iloc[top_k:].sum()

    counts_plot = counts_plot.sort_values(by='syntaxons', ascending=False)

    x_vals = list(range(1, len(counts_plot) + 1))
    x_vals2 = np.arange(1, len(unique_habitats)+1, 1)
    y_vals = counts_plot['syntaxons'].values
    y_vals2 = unique_habitats.values

    fig, ax = plt.subplots(figsize=(plot_width, 8))

    # Step plots
    ax.step(x_vals, y_vals, where='mid', color='blue', linestyle='-', label='Syntaxons counts per habitats')
    ax.step(x_vals2, y_vals2, where='mid', color='orange', linestyle='--', label='Habitats value counts')

    # --- Dashed horizontal lines for each unique step value ---
    y_ticks = [x for x in range(0, y_vals2.max(), 10)]
    ax.axhline(y=y_vals2.min(), linestyle='--', linewidth=0.8, alpha=0.5)
    for y in range(1, y_vals2.max(), 1):
        ax.axhline(y=y, linestyle='--', linewidth=0.8, alpha=0.5)
        y_ticks.append(y)
    ax.axhline(y=y_vals2.max(), linestyle='--', linewidth=0.8, alpha=0.5)
    y_ticks.append(y_vals2.min())
    y_ticks.append(y_vals2.max())
    
    # --- Show y-ticks for each dashed line ---
    ax.set_yticks(y_ticks)

    # --- Show only 1 every 10 ticks ---
    tick_step = 10
    ticks = x_vals[::tick_step]

    # Ensure first and last are included
    if x_vals[0] not in ticks:
        ticks = [x_vals[0]] + ticks
    if x_vals[-1] not in ticks:
        ticks = ticks + [x_vals[-1]]

    ax.set_xticks(ticks)

    ax.set_xlabel("Habitat index")
    ax.set_ylabel("Syntaxons")
    ax.set_title(
        f"Habitats with only 1 corresponding syntaxon: {n_singletons}",
        fontsize=11
    )
    ax.legend()

    fig.text(
        0.5, 0.94,
        title,
        ha='center',
        fontsize=15
    )

    if fp_out:
        plt.savefig(fp_out, bbox_inches='tight')

    if plot:
        plt.show()
    plt.close()
    return fig, counts


def step_plot_distribution(df, col='id_floraveg', fp_out=False, plot=True):
    # Count occurrences (already sorted descending)
    counts = df[col].value_counts()

    # Number of values that appear only once
    n_singletons = (counts == 1).sum()

    values = counts.values.astype(int)
    N = len(values)

    fig = plt.figure(figsize=(10,5))
    plt.plot(values)

    # ---- detect step changes ----
    step_positions = np.where(np.diff(values) != 0)[0] + 1

    # include first and last
    step_positions = np.concatenate(([0], step_positions, [N-1]))
    step_positions = np.unique(step_positions)

    # draw vertical dashed lines
    for pos in step_positions:
        plt.axvline(x=pos, linestyle='--', alpha=0.5)

    # set xticks at step positions
    plt.xticks(step_positions, step_positions, rotation=45)

    # ---- floating annotations ----
    plt.text(
        -210,
        values[0],                 # y-position = curve value
        f"{0:.1f}%",
        fontsize=9,
        ha='left',
        va='bottom'
    )
    for pos in step_positions[1:]:
        pct = (pos / N) * 100
        plt.text(
            pos-1,
            values[pos-1],                 # y-position = curve value
            f"{pct:.1f}%",
            fontsize=9,
            ha='left',
            va='bottom'
        )

    plt.xlabel(f'Nb of {col}')
    plt.ylabel("Count")
    plt.title(
        f"Distribution of {col} after merging with labels\n"
        f"Values with only 1 occurrence: {n_singletons}"
    )

    plt.tight_layout()
    if fp_out:
        plt.savefig(fp_out)
    if plot:
        plt.show()
    plt.close()

    counts_percentage = counts.value_counts()
    n = len(counts)
    counts_percentage = counts_percentage.apply(lambda x: 100*x/n)
    df_counts_percentage = pd.DataFrame({f'% of {col}': counts_percentage.values, 'count': counts.value_counts().values}, index=counts_percentage.index)
    return fig, counts, df_counts_percentage

def pie_chart_simple(values, labels, title, fp_out=False, plot=True):
    cmap = plt.get_cmap("rainbow")
    colors = cmap(np.linspace(0.6, 0.8, len(values)))

    def show_values(pct, all_values):
        total = sum(all_values)
        value = f'{(pct):.2f}%\n({int(pct * total / 100.0)})'
        return f"{value}"

    fig = plt.figure()
    plt.pie(
        values,
        labels=labels,
        autopct=lambda pct: show_values(pct, values),  # display values
        wedgeprops={
            "edgecolor": "black",   # border color
            "linewidth": 0.5        # border thickness
        },
        startangle=130,
        colors=colors,
        labeldistance=1.05,
        pctdistance=0.60,
    )

    plt.title(title)
    plt.axis('equal')  # keeps pie circular
    plt.tight_layout()
    if fp_out:
        plt.savefig(fp_out, bbox_inches='tight')
    if plot:
        plt.show()
    plt.close()
    return fig

def pie_chart_multilabels_proportion(df_counts_percentage, fp_out=False, plot=True):
    values = df_counts_percentage['count']# ['% of id_floraveg']
    labels = pd.Series(df_counts_percentage.index.astype(str))
    labels = labels.apply(lambda x: x + ' label' if x=='1' else x + ' labels')

    # Choose a colormap
    cmap = plt.get_cmap("rainbow")
    # Generate colors from the colormap
    colors = cmap(np.linspace(0.15, 0.7, len(values)))

    # Function to display actual values instead of percentages
    def show_values(pct, all_values):
        total = sum(all_values)
        value = f'{pct:.2f}%\n({int(pct/100 * total)})'
        return f"{value}"

    # Create pie chart
    fig = plt.figure()
    plt.pie(
        values,
        labels=labels,
        autopct=lambda pct: show_values(pct, values),  # display values
        wedgeprops={
            "edgecolor": "black",   # border color
            "linewidth": 0.5        # border thickness
        },
        startangle=130,
        colors=colors[::-1],
        labeldistance=1.05,
        pctdistance=0.85,
    )

    plt.title(f"FloraVeg habitats dataset: proportion of multi-labeled images from {df_counts_percentage.index.min()} to {df_counts_percentage.index.max()} max labels")
    plt.axis('equal')  # keeps pie circular
    plt.tight_layout()
    if fp_out:
        plt.savefig(fp_out, bbox_inches='tight')
    if plot:
        plt.show()
    plt.close()
    return fig


def stack_bars_habitats(df_occurrences_merged, counts, df_counts_percentage, title='EUNIS-lvl-1 habitats per nb of matching syntaxons', fp_out='resources/Stack_bar_plot_multi-taxons_habitats.png', plot=True):
    # Each bar has its own list of values (uneven lengths)
    bars_counts = {}
    bars_counts_pct = {}
    bars_labels = {}
    for k in range(df_counts_percentage.index.min(), df_counts_percentage.index.max()+1, 1):
        id_floraveg_multi_k = counts[counts == k].index.tolist()  # Get the FloraVeg IDs which have k-multiple labels
        df_slice_k = df_occurrences_merged[df_occurrences_merged['id_floraveg'].isin(id_floraveg_multi_k)]  # Slice the merged dataframe accordingly
        habitats_multi_k = df_slice_k['habitats_code_lvl1'].value_counts()  # Get the count of unique habitats of this slice
        habitats_multi_k_pct = habitats_multi_k.apply(lambda x: 100*x/habitats_multi_k.sum())
        key = f'Ensemble 1 to {k}'
        bars_counts[key] = habitats_multi_k.values.tolist()
        bars_counts_pct[key] = habitats_multi_k_pct.values.tolist()
        bars_labels[key] = habitats_multi_k.index.tolist()


    x = np.arange(len(bars_counts))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 15))
    small_sized_section = 30
    size_previous_section = np.inf
    x_text_offset_dir = -1
    
    # Plot bar by bar
    # Get all unique habitat labels across bars
    all_labels = sorted(set(
        label
        for labels in bars_labels.values()
        for label in labels
    ))

    # Create stable color mapping
    cmap = plt.get_cmap('tab20')  # or 'tab20b', 'Set3', etc.
    color_map = {
        label: cmap(i % cmap.N)
        for i, label in enumerate(all_labels)
    }
    for i, (bar_name, values) in enumerate(bars_counts.items()):
        bottom = 0

        for value, label, pct in zip(values, bars_labels[bar_name], bars_counts_pct[bar_name]):
            rect = ax.bar(
                x[i],
                value,
                width,
                bottom=bottom,
                color=color_map[label]
            )[0]
            # Add label centered inside section
            x_text = rect.get_x() + rect.get_width()/2
            y_text = bottom + value/2
            if value <= small_sized_section and size_previous_section <= small_sized_section:
                y_text += 30#*x_text_offset_dir
                #x_text_offset_dir *= -1
            ax.text(
                x_text,
                y_text,
                f'{label} ({pct:.2f}%)',
                ha='center',
                va='center',
                color='black'
            )
            bottom += value
            size_previous_section = value

    # X axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(bars_counts.keys())
    ax.set_ylabel("Occurrences")
    ax.set_title(title)
    if fp_out:
        plt.savefig(fp_out, bbox_inches='tight')
    if plot:
        plt.show()
    plt.close()
    return fig

# fig = venn_habitats_per_multilabel_correspondance(counts, fp_out='resources/Venn_diagram_habitats_per_multilabel_correspondance.png', plot=False)
def venn_habitats_per_multilabel_correspondance(counts, fp_out=False, plot=True):
    def format_venn_vals(x, total):
        pct = 100*x/total
        return f'{x}\n({pct:.2f})%'

    samples_per_n_labels = counts.groupby(counts).groups
    habitats_sets = {}
    for k,v in samples_per_n_labels.items():
        u_habitats = df_occurrences_merged[df_occurrences_merged['id_floraveg'].isin(v)]['habitats_code']
        habitats_sets[k] = set(u_habitats)

    n_total_habitats = df_occurrences_merged['habitats_code'].nunique()
    s1 = habitats_sets[1]
    s2_to_5 = set()
    for k in range(2, len(habitats_sets)):
        s2_to_5 = s2_to_5 | habitats_sets[k]

    fig = plt.figure(figsize=(5,5))
    venn2(
        subsets=(len(s1), len(s2_to_5), len(s1 & s2_to_5)),  # (A only, B only, A∩B)
        set_labels=('S1\n', 'S2_to_5\n'),
        subset_label_formatter = (lambda x: format_venn_vals(x, len(s1 | s2_to_5)))
    )
    plt.legend(['S1: 1-to-1 img-label correspondance', 'S2: 1-to-k img-label correspondance w/ k ∈ (2,3,4,5)'], bbox_to_anchor=(1,-0.07), fontsize=10)
    plt.suptitle(f'Images with only 1 label represent {(100*len(s1 & s2_to_5)/len(s1 | s2_to_5)):.2f}% of total habitats ({len(s1 | s2_to_5)})', fontsize=16)
    plt.tight_layout()
    if fp_out:
        plt.savefig(fp_out, bbox_inches='tight')
    if plot:
        plt.show()
    plt.close()
    return fig

def upset_plot_habitats_per_soft_labeling(df_occurrences_merged, counts, plot=True):
    # @article{2014_infovis_upset,
    #     title = {UpSet: Visualization of Intersecting Sets},
    #     author = {Alexander Lex and Nils Gehlenborg and Hendrik Strobelt and Romain Vuillemot and Hanspeter Pfister},
    #     journal = {IEEE Transactions on Visualization and Computer Graphics (InfoVis)},
    #     doi = {10.1109/TVCG.2014.2346248},
    #     volume = {20},
    #     number = {12},
    #     pages = {1983--1992},
    #     year = {2014}
    # }

    nunique_habitats = df_occurrences_merged['habitats_code'].nunique()
    le = LabelEncoder()
    le.fit(df_occurrences_merged['habitats_code'].values)

    set_list = []
    df = pd.DataFrame({'S5': [0]*nunique_habitats,
                       'S4': [0]*nunique_habitats,
                       'S3': [0]*nunique_habitats,
                       'S2': [0]*nunique_habitats,
                       'S1': [0]*nunique_habitats})

    # For each k of 1-to-k soft labelling... 
    for k, v in counts.groupby(counts).groups.items():
        # ...find the unique habitats...
        labels = df_occurrences_merged[df_occurrences_merged['id_floraveg'].isin(v)][['habitats_code']]
        # ...cast them to encoded integer...
        labels['habitats_code'] = le.transform(labels['habitats_code'].values.tolist())
        # ...insert the count of unique habitats h_i in the df in the index h_i
        labels_counts = np.unique_counts(labels['habitats_code'])
        df.loc[labels_counts.values, f'S{k}'] += labels_counts.counts
        # ...convert any count > 0 to 1 to enable using plotly_upset
        df[f'S{k}'] = df[f'S{k}'].apply(lambda x: 1 if x>0 else 0)

    # Plotting
    fig = plot_upset(
        dataframes=[df],
        legendgroups=["Count of unique EUNIS-lvl3 habitats per soft multilabel 1-to-k"],
        marker_size=8,
    )

    fig.update_layout(
        title='Upset plot: up to k=2, soft multilabel 1-to-k contain a significant amount of proper labels',
        font_family="Ubuntu",
        width=1000,
        title_font_size=18,
        legend_font_size=12,
    )
    if plot:
        fig.show()
    return fig

def jaccard_sim_habitat_soft_multilabeling(counts, df_occurrences_merged, fp_out=False):
    dfs = []
    for k,v in counts.groupby(counts).groups.items():
        dfs.append(df_occurrences_merged[df_occurrences_merged['id_floraveg'].isin(v)])
    sets = [set(df["habitats_code"]) for df in dfs]

    n = len(sets)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            intersection = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            matrix[i, j] = intersection / union
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, cmap="Blues",
                xticklabels=[f"k = {i+1}" for i in range(n)],
                yticklabels=[f"k = {i+1}" for i in range(n)],
                ax=ax)

    plt.suptitle("Multi-labeled images contain unique habitats absent from uni-labeled images", fontsize=14)
    plt.title(r'Jaccard Similarity ($\frac{Intersection}{Union}$) of soft multilabel 1-to-$k$', fontsize=11)
    plt.tight_layout()
    if fp_out:
        fig.savefig(fp_out, bbox_inches='tight')
    plt.close()
    return fig


def stacked_bars_habitat_distribution(train, test, title='Title', suptitle='', fp_out=False, plot=True):
    # Count occurrences
    train_counts = train["habitats_code"].value_counts()
    test_counts = test["habitats_code"].value_counts()

    # Align categories
    all_habitats = train_counts.index.union(test_counts.index)
    train_counts = train_counts.reindex(all_habitats, fill_value=0)
    test_counts = test_counts.reindex(all_habitats, fill_value=0)

    # Sort by total frequency
    total_counts = train_counts + test_counts
    sorted_idx = total_counts.sort_values(ascending=False).index

    train_counts = train_counts.loc[sorted_idx]
    test_counts = test_counts.loc[sorted_idx]
    total_counts = total_counts.loc[sorted_idx]

    # Percentage of test vs train
    pct_test_vs_train = test_counts / (train_counts + test_counts)

    x = np.arange(0, len(sorted_idx), 1)

    fig, ax1 = plt.subplots(figsize=(12,6))

    # Stacked bars
    ax1.bar(x, test_counts, color="blue", label="Test")
    ax1.bar(x, train_counts, bottom=test_counts, color="orange", label="Train")

    # Annotate total counts on top
    for i, total in enumerate(total_counts[:1]):
        ax1.text(i, total, str(total), ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel("Habitat code")
    ax1.set_ylabel("Count")
    ax1.set_xticks(x[::5])
    ax1.set_xticklabels(sorted_idx[::5], rotation=45)

    # Secondary axis
    ax2 = ax1.twinx()

    max_count = total_counts.max()

    # Scale percentage so 100% corresponds to max_count
    scaled_pct = pct_test_vs_train * max_count

    ax2.plot(x, scaled_pct, marker="o", color="black", linestyle='--', label="% Test", linewidth=0.5, markersize=2)
    # ax2.plot(scaled_pct, 'lightblue', scaled_pct.rolling(10).mean(), 'blue', scaled_pct, marker="o", label="% Test", linewidth=0.5, markersize=2)

    ax2.set_ylim(0, max_count)
    ax2.set_ylabel("Test proportion (%)")

    # Show percentage scale on the right axis
    ticks = np.linspace(0, max_count, 11)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([10*t for t in np.arange(len(ticks))])
    # ax2.set_yticklabels([f"{int(t/max_count*100)}%" for t in ticks])

    # Titles
    ax1.set_title(title, fontsize=14)
    if suptitle:
        fig.suptitle(suptitle, fontsize=18)

    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center')

    plt.tight_layout()
    if fp_out:
        fig.savefig(fp_out, bbox_inches='tight')
    if plot:
        plt.show()
    return fig

def plot_stacked_bar_multilabel_stratify(
    dict1, dict2, dict3,
    label1="all_data", label2="test_ideal", label3="test_real",
    color1="lightblue", color2="red", color3="limegreen",
    suptitle='Unique habitats distribution',
    title='',
    xlabel="Habitat code",
    ylabel="Count",
    y_min=None,
    y_max=None,
    test_pct=0.10,
    plot=False,
    fp_out=False,
):
    bins = sorted(dict1.keys(), key=lambda b: dict1.get(b, 0), reverse=True)

    # Align values
    values1 = [dict1.get(b, 0) for b in bins]
    values2 = [dict2.get(b, 0) for b in bins]
    values3 = [dict3.get(b, 0) for b in bins]

    # % dict3 over dict1 (avoid division by zero)
    pct_values = [
        (v3 / v1 * 100) if v1 != 0 else 0
        for v1, v3 in zip(values1, values3)
    ]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # --- Bars on primary axis ---
    ax1.bar(bins, values1, label=label1, color=color1)
    ax1.bar(bins, values2, bottom=0, label=label2, color=color2)
    ax1.bar(bins, values3, bottom=0, label=label3, color=color3, alpha=0.8)

    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if y_min is not None or y_max is not None:
        ax1.set_ylim(bottom=y_min, top=y_max)

    ax1.set_xticks(bins[::5])
    ax1.set_xticklabels(bins[::5], rotation=45)

    # --- Secondary axis for percentage line ---
    ax2 = ax1.twinx()
    ax2.plot(
        range(len(bins)),
        pct_values,
        color="black",
        linestyle="--",
        linewidth=1,
        marker="o",
        markersize=3,
        label="% of test data per habitat"
    )
    ax2.set_ylabel("% test data")
    ax2.set_ylim(0, 100)
    ax2.axhline(y=100*test_pct, linestyle='--', linewidth=0.8, alpha=0.8, c='gray')
    ax2.set_yticks(sorted(set(list(range(0, 101, 20)) + [10])))
    
    for tick_value, label in zip(ax2.get_yticks(), ax2.get_yticklabels()):
        if tick_value == 10:
            label.set_color("gray")
        else:
            label.set_color("black")

    # --- Combined legend ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()

    if plot:
        plt.show()
    if fp_out:
        fig.savefig(fp_out, bbox_inches='tight')

    plt.close()
    return fig