import os
import pandas as pd
import numpy as np

def best_threshold_roc(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Youden’s J statistic
    j_scores = tpr - fpr

    idx = np.argmax(j_scores)
    best_thresh = thresholds[idx]

    return best_thresh, tpr[idx], fpr[idx]

def accuracy_per_habitatslvl1(df):
    res = {}
    habitats_score_to_name = df.groupby('habitats_lvl1')['habitats_lvl1_name'].first().to_dict()
    unique_habitats = df['habitats_lvl1'].unique()
    for uh in unique_habitats:
        df_slice = df[df['habitats_lvl1'] == uh]
        res[f'{uh} ({habitats_score_to_name[uh]})'] = [(df_slice['label'] == df_slice['y_pred']).mean()]
    return res

if __name__ == '__main__':
    OUTPUT_PATH = 'outputs/Downstream_GPS_error_detection_multi-images-avg/Downstream_GPS_error_detection_LUCAS_noise_mixture/
    fp_scores = os.path.join(OUTPUT_PATH, 'scores_cosine.csv')
    fp_out_socres_enriched = os.path.join(OUTPUT_PATH, 'scores_cosine_best_roc_threshold_and_habitats.csv')
    fp_out_acc_per_hlvl1 = os.path.join(OUTPUT_PATH, 'accuracies-best-roc-threshold_per_habitatslvl1.csv')
    fp_habitats = 'dataset/scale_2_landscape/GPN_API_habitats/response.csv'

    # Find best rcoc threshold
    scores_cosine = pd.read_csv(fp_scores)
    threshold, tpr, fpr = best_threshold_roc(scores_cosine['label'], scores_cosine['score'])

    y_pred = (np.array(scores_cosine['score']) >= threshold).astype(int)
    y_true = scores_cosine['label']
    scores_cosine['y_pred'] = y_pred
    scores_cosine['best_threshold_roc'] = threshold
    scores_cosine['tpr'] = tpr
    scores_cosine['fpr'] = fpr
    scores_cosine['accuracy'] = (y_pred == y_true).mean()

    # Adding habitats
    habitats = pd.read_csv(fp_habitats)

    scores_cosine_with_habitats = scores_cosine.merge(habitats, left_on='id', right_on='surveyId')
    scores_cosine_with_habitats.to_csv(fp_out_socres_enriched, index=False)

    # Computing average per habitats-lvl1
    df_acc_per_hlvl1 = pd.DataFrame(accuracy_per_habitatslvl1(scores_cosine_with_habitats))
    df_acc_per_hlvl1.to_csv(fp_out_acc_per_hlvl1, index=False)
