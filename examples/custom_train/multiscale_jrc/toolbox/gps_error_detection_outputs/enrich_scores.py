import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve

def best_threshold_roc(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Youden’s J statistic
    j_scores = tpr - fpr

    idx = np.argmax(j_scores)
    best_thresh = thresholds[idx]

    return best_thresh, tpr[idx], fpr[idx]

def accuracy_per_habitatslvl1(df):
    res = {}
    habitats_score_to_name = df.groupby('habitats_code_lvl1')['habitats'].first().to_dict()  # col "habitats" contains natural language habitats' names
    unique_habitats = df['habitats_code_lvl1'].unique()
    for uh in unique_habitats:
        df_slice = df[df['habitats_code_lvl1'] == uh]
        res[f'{uh} ({habitats_score_to_name[uh]})'] = [(df_slice['label'] == df_slice['prediction']).mean()]
    return res
    
def accuracy_per_noise_type(df):
    res_acc = {}
    nt_vc = df['noise_type'].value_counts()
    for u_nt in nt_vc.index:
        df_slice = df[df['noise_type'] == u_nt]
        accuracy = (df_slice['label'] == df_slice['prediction']).mean()
        mean_auc = df_slice['score'].mean()
        res_acc[u_nt] = [accuracy]
    df_acc = pd.DataFrame.from_dict(
        res_acc,
        orient="index",
        columns=["accuracy"]
    )
    count = nt_vc.to_frame(name="count")
    count_pct = (nt_vc / nt_vc.sum()).to_frame(name="count_pct")
    
    df_join = df_acc.join([count, count_pct])
    df_join.index.name = 'Noise type'
    return df_join


def main():
    parser = argparse.ArgumentParser(
        description="Process habitat scores and ground-truth labels."
    )
    parser.add_argument(
        "--scores",
        type=str,
        required=True,
        help="Path to the scores CSV file."
    )
    parser.add_argument(
        "--habitats",
        type=str,
        required=False,
        default='../../dataset/habitats_gpn25_eva/FloraVeg/metadata_labels_merged_gps_only_S2_encoded_test-0.00225deg.csv',
        help="Path to the ground-truth habitats CSV file."
    )
    parser.add_argument(
        "--out_name",
        type=str,
        required=False,
        default='',
        help="Path to the ground-truth habitats CSV file."
    )
    args = parser.parse_args()

    fp_scores = args.scores
    fp_habitats = args.habitats
    fn_out = args.out_name if args.out_name != '' else Path(fp_scores).stem

    # Find best rcoc threshold
    scores_cosine = pd.read_csv(fp_scores)
    threshold, tpr, fpr = best_threshold_roc(scores_cosine['label'], scores_cosine['score'])

    y_pred = (np.array(scores_cosine['score']) >= threshold).astype(int)
    y_true = scores_cosine['label']
    scores_cosine['prediction'] = y_pred
    scores_cosine['best_threshold_roc'] = threshold
    scores_cosine['tpr'] = tpr
    scores_cosine['fpr'] = fpr
    scores_cosine['accuracy'] = (y_pred == y_true).mean()

    # Adding habitats
    habitats = pd.read_csv(fp_habitats)

    assert habitats['noise_type'].equals(scores_cosine['noise_type'])
    scores_cosine_with_habitats = scores_cosine.merge(habitats.drop(columns=['noise_type']), left_on='id', right_on='id_floraveg')
    scores_cosine_with_habitats.to_csv(f'best_roc_threshold_and_habitats_{fn_out}.csv', index=False)

    # Computing average per habitats-lvl1
    df_acc_per_hlvl1 = pd.DataFrame(accuracy_per_habitatslvl1(scores_cosine_with_habitats))
    df_acc_per_hlvl1.to_csv(f'acc-best-roc-threshold_per_habitatslvl1_{fn_out}.csv', index=False)
    df_acc_per_noise_type = accuracy_per_noise_type(scores_cosine_with_habitats)
    df_acc_per_noise_type.to_csv(f'acc-best-roc-threshold_per_noise_type_{fn_out}.csv', index=True)

if __name__ == '__main__':
    main()
