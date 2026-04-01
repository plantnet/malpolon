import pandas as pd
from pathlib import Path
import numpy as np

def summarize_metrics(folder_path):
    folder = Path(folder_path)

    # Find all CSV files matching metrics*.csv
    csv_files = folder.glob("metrics*.csv")

    # Lists to collect values
    roc_list = []
    pr_list = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Check columns exist
        if "roc_auc" in df.columns:
            roc_list.extend(df["roc_auc"].dropna().values)
        if "pr_auc" in df.columns:
            pr_list.extend(df["pr_auc"].dropna().values)

    # Convert to numpy arrays
    roc_arr = np.array(roc_list)
    pr_arr = np.array(pr_list)

    # Compute mean and std
    summary = {
        "roc_auc_mean": np.mean(roc_arr),
        "roc_auc_std": np.std(roc_arr, ddof=1),  # sample std
        "pr_auc_mean": np.mean(pr_arr),
        "pr_auc_std": np.std(pr_arr, ddof=1)
    }

    return summary

if __name__ == "__main__":
    folder = "./outputs/Downstream_GPS_error_detection_LUCAS_noise_mixture"  # change to your folder path
    stats = summarize_metrics(folder)
    print("Metrics summary:")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
