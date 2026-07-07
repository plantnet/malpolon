import argparse
import glob
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
)


def evaluate(csv_file):
    df = pd.read_csv(csv_file)

    y_true = df["label"].values
    y_score = df["score"].values
    # print('\n', df["score"].describe())
    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # print('thresholds :', thresholds)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    # print('best_idx: ', best_idx)
    # print('tpr: ', tpr)
    # print('fpr: ', fpr)
    # print('j_scores: ', j_scores)

    best_threshold = thresholds[best_idx]
    #if not np.isfinite(best_threshold):
    #    best_threshold = thresholds[best_idx + 1]  # fall back to next one
    sensitivity = tpr[best_idx]
    specificity = 1.0 - fpr[best_idx]
    auc = roc_auc_score(y_true, y_score)
    

    # Predictions at optimal threshold
    y_pred = (y_score >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "ROC_auc": auc,
        "threshold": best_threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "j": j_scores[best_idx],
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def mean_std(values):
    values = np.asarray(values)
    return values.mean(), values.std(ddof=1)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Glob pattern matching prediction CSVs "
             "(e.g. results/seed*/predictions.csv)"
    )

    args = parser.parse_args()

    files = sorted(glob.glob(args.input))

    if len(files) == 0:
        raise RuntimeError(f"No files found matching '{args.input}'")

    print(f"Found {len(files)} prediction files\n")

    results = []

    for file in files:
        r = evaluate(file)
        results.append(r)

        print(
            f"{file:40s} "
            f"ROC_AUC={r['ROC_auc']:.4f} "
            f"Thr={r['threshold']:.4f}"
        )

    print("\n========== Average over seeds ==========\n")

    metrics = [
        "ROC_auc",
        "threshold",
        "sensitivity",
        "specificity",
        "j",
    ]

    for metric in metrics:
        m, s = mean_std([r[metric] for r in results])
        print(f"{metric:12s}: {m:.4f} ± {s:.4f}")

    # Average confusion matrix entries
    for metric in ["tp", "tn", "fp", "fn"]:
        m, s = mean_std([r[metric] for r in results])
        print(f"{metric:12s}: {m:.1f} ± {s:.1f}")


if __name__ == "__main__":
    main()
