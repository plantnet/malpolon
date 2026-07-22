# python get_stats_GPN_x_LUCAS.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


FP_GPN_X_LUCAS = 'data/output/csv/LUCAS_metadata_labels_merged_S3-10%_extended_GPN-labels.csv'
FP_GPN_HABITATS = '../GPN/v1/geoplantnet_v1_habitat_eunis2020_metadata.csv'

def get_lvlk(code, k):
    return code[:k+1] if code.startswith('MA') else code[:k]

def compare_labels(df):
    u_lvl1 = ['R', 'V', 'T', 'S', 'Q', 'N','MA', 'U']
    tp_lvl2, tp_lvl1 = [], []
    gpn_labels = df['gpn_habitat_code_lvl2']
    lucas_labels = df['habitats_code_lvl2']
    for l_label, g_label in zip(gpn_labels, lucas_labels):
        codes = l_label.split(';')
        tp_lvl1.append(1) if any(get_lvlk(g_label, k=1) in code for code in codes) else tp_lvl1.append(0)
        tp_lvl2.append(1) if any(g_label in code for code in codes) else tp_lvl2.append(0)
    df['tp_lvl1'] = tp_lvl1
    df['tp_lvl2'] = tp_lvl2
    
    print("\n--- Real accuracies ---")
    print(f"GPN pseudo-labels EUNIS-lvl2 acc: {100*df['tp_lvl2'].sum()/len(df):.2f}%")
    print(f"GPN pseudo-labels EUNIS-lvl1 acc: {100*df['tp_lvl1'].sum()/len(df):.2f}%")
    print('\n--- Details per EUNIS-lvl1 habitat group ---')
    for lvl1 in u_lvl1:
        df_slice = df[df['habitats_code_lvl2'].str.startswith(lvl1)]
        print(f"Habitat group ['{lvl1}', {len(df_slice)} ({100*len(df_slice)/len(df):.2f}%) samples] - lvl2 acc: {0 if len(df_slice) <= 0 else 100*df_slice['tp_lvl2'].sum()/len(df_slice):.2f}%, lvl1 acc: {0 if len(df_slice) <= 0 else 100*df_slice['tp_lvl1'].sum()/len(df_slice):.2f}%")
    return df

df = pd.read_csv(FP_GPN_X_LUCAS)
df = df.dropna(subset='gpn_habitat_id')
gpn_habitats = pd.read_csv(FP_GPN_HABITATS)
gpn_h_dict = {k: v for k,v in zip(gpn_habitats['raster_value'], gpn_habitats['habitat_code'])}
df['gpn_habitat_code'] = df['gpn_habitat_id'].map(gpn_h_dict)
df.rename(columns={'gpn_habitat_code': 'gpn_habitat_code_lvl3'}, inplace=True)
df['gpn_habitat_code_lvl2'] = df['gpn_habitat_code_lvl3'].apply(get_lvlk, args=(2,))
df['gpn_habitat_code_lvl1'] = df['gpn_habitat_code_lvl3'].apply(get_lvlk, args=(1,))

compare_labels(df)

# Simplify multi-label cases to compute the sklearn metrics
for k in range(1,3,1):
    for rowi, row in df.iterrows():
        gpn_label = row[f'gpn_habitat_code_lvl{k}']
        lucas_label = row[f'habitats_code_lvl{k}']
        if gpn_label in lucas_label:
            df.loc[rowi, f'habitats_code_lvl{k}'] = gpn_label
        else:
            df.loc[rowi, f'habitats_code_lvl{k}'] = lucas_label.split(';')[0]

# Ground truth and predictions
y_true = df["habitats_code_lvl2"]
y_pred = df["gpn_habitat_code_lvl2"]

# Overall metrics
accuracy = accuracy_score(y_true, y_pred)

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

print("\n--- Global EUNIS lvl-2 metrics ---\n")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Macro Precision    : {precision_macro:.4f}")
print(f"Macro Recall       : {recall_macro:.4f}")
print(f"Macro F1-score     : {f1_macro:.4f}")
print(f"Weighted Precision : {precision_weighted:.4f}")
print(f"Weighted Recall    : {recall_weighted:.4f}")
print(f"Weighted F1-score  : {f1_weighted:.4f}")

# ------------------------------------------------------------------
# Random classifier (uniform)
# ------------------------------------------------------------------
rng = np.random.default_rng(42)

classes = np.unique(y_true)
y_pred_random_uniform = rng.choice(classes, size=len(y_true))

acc_random_uniform = accuracy_score(y_true, y_pred_random_uniform)
prf_random_uniform = precision_recall_fscore_support(
    y_true,
    y_pred_random_uniform,
    average="macro",
    zero_division=0,
)

print("\nRandom baseline (uniform)")
print(f"Accuracy        : {acc_random_uniform:.4f}")
print(f"Macro Precision : {prf_random_uniform[0]:.4f}")
print(f"Macro Recall    : {prf_random_uniform[1]:.4f}")
print(f"Macro F1        : {prf_random_uniform[2]:.4f}")

# ------------------------------------------------------------------
# Per-class metrics
# ------------------------------------------------------------------
print("\nPer-class metrics:")
per_class = precision_recall_fscore_support(
    y_true,
    y_pred,
    average=None,
    labels=sorted(y_true.unique()),
    zero_division=0,
)
prf_random_uniform = precision_recall_fscore_support(
    y_true,
    y_pred,
    average=None,
    labels=sorted(y_true.unique()),
    zero_division=0,
)
metrics_random = list(zip(
    sorted(y_true.unique()),
    *prf_random_uniform
))
metrics = list(zip(
    sorted(y_true.unique()),
    *per_class
))
# Sort by F1 score (index 3) in decreasing order
metrics.sort(key=lambda x: x[3], reverse=True)

for (label, p, r, f1, support), (label_rnd, p_rnd, r_rnd, f1_rnd, support_rnd) in zip(metrics, metrics_random):
    print(
        f"{label:15} "
        f"F1={f1:.3f} (rnd={f1_rnd:.3f}) "
        f"Precision={p:.3f} (rnd={p_rnd:.3f}) "
        f"Recall={r:.3f} (rnd={r_rnd:.3f})"
        f"Support={support}"
    )

# Confusion matrix
for k in range(1,3,1):
    y_true = df[f"habitats_code_lvl{k}"]
    y_pred = df[f"gpn_habitat_code_lvl{k}"]
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(
        ax=ax,
        cmap="plasma",
        xticks_rotation=90,
        colorbar=True,
        include_values=False,
    )
    plt.suptitle(f"Confusion matrix of EUNIS habitat lvl-{k} labels", fontsize=26)
    plt.title("GPN_v1 pseudo-labels (prediction) VS LUCAS_grasslands EUNIS labels (GT)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_lvl-{k}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion matrix saved as confusion_matrix_lvl-{k}.png")
