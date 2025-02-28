"""
result_analysis.py

This script reads the files "prediction_results.txt" and "metrics_results.txt" to analyze 
the computed metrics for each meteor trail. Each line in "prediction_results.txt" has the form:
    file.fits x1 y1 x2 y2 model_score
and each line in "metrics_results.txt" has the form:
    file.fits x1 y1 x2 y2 fwhm_default fwhm_07 fwhm_095 compactness auc_med auc_peak auc_full kurtosis gaussian_kurtosis

The script then calculates:
  1. The overall average, median, standard deviation, and range for each metric.
  2. The average, median, standard deviation, and range for trails predicted as meteors (model_score > 0.8).
  3. The average, median, standard deviation, and range for trails predicted as non-meteors (model_score < 0.2).
  4. The average, median, standard deviation, and range for trails predicted as meteors (model_score > 0.5).
  5. The average, median, standard deviation, and range for trails predicted as non-meteors (model_score <= 0.5).

Additionally, the script prints the count of trails in each group.

Usage:
    python result_analysis.py /path/to/fits_folder
"""

import os
import sys
import numpy as np

def read_prediction_results(pred_file="prediction_results.txt"):
    """
    Reads the prediction_results.txt file.
    
    Returns:
        A dictionary mapping filename to prediction score (float).
    """
    predictions = {}
    if not os.path.exists(pred_file):
        print(f"[ERROR] Prediction file {pred_file} not found.")
        sys.exit(1)
    with open(pred_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                print(f"[WARNING] Skipping malformed prediction line: {line.strip()}")
                continue
            filename = parts[0]
            try:
                score = float(parts[5])
            except ValueError:
                print(f"[WARNING] Invalid score in line: {line.strip()}")
                continue
            predictions[filename] = score
    return predictions

def read_metrics_results(metrics_file="metrics_results.txt"):
    """
    Reads the metrics_results.txt file.
    
    Returns:
        A list of dictionaries, one per line, with keys:
         'filename', 'x1', 'y1', 'x2', 'y2',
         'fwhm_default', 'fwhm_07', 'fwhm_095', 'compactness',
         'auc_med', 'auc_peak', 'auc_full', 'kurtosis', 'gaussian_kurtosis'
    """
    metrics_list = []
    if not os.path.exists(metrics_file):
        print(f"[ERROR] Metrics file {metrics_file} not found.")
        sys.exit(1)
    with open(metrics_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 14:
                print(f"[WARNING] Skipping malformed metrics line: {line.strip()}")
                continue
            try:
                entry = {
                    "filename": parts[0],
                    "x1": float(parts[1]),
                    "y1": float(parts[2]),
                    "x2": float(parts[3]),
                    "y2": float(parts[4]),
                    "fwhm_default": float(parts[5]),
                    "fwhm_07": float(parts[6]),
                    "fwhm_095": float(parts[7]),
                    "compactness": float(parts[8]),
                    "auc_med": float(parts[9]),
                    "auc_peak": float(parts[10]),
                    "auc_full": float(parts[11]),
                    "kurtosis": float(parts[12]),
                    "gaussian_kurtosis": float(parts[13])
                }
            except ValueError:
                print(f"[WARNING] Skipping line with invalid numeric values: {line.strip()}")
                continue
            metrics_list.append(entry)
    return metrics_list

def filter_data_by_prediction(metrics_list, predictions, threshold_func):
    """
    Filters the metrics_list based on the prediction score in predictions dict,
    using threshold_func which takes the prediction score and returns True/False.
    
    Returns:
        A list of metric dictionaries for which threshold_func(score) is True.
    """
    return [entry for entry in metrics_list if entry["filename"] in predictions and threshold_func(predictions[entry["filename"]])]

def compute_statistics(metrics_list, metric_keys):
    """
    Computes the average, median, standard deviation, and range for specified metrics.

    Args:
        metrics_list: list of dictionaries with metric keys.
        metric_keys: list of keys for which to compute statistics.

    Returns:
        A dictionary with computed statistics.
    """
    stats = {key: {} for key in metric_keys}
    for key in metric_keys:
        values = [entry[key] for entry in metrics_list]
        if values:
            stats[key]["average"] = np.mean(values)
            stats[key]["median"] = np.median(values)
            stats[key]["std_dev"] = np.std(values)
            stats[key]["range"] = np.ptp(values)
        else:
            stats[key]["average"] = None
            stats[key]["median"] = None
            stats[key]["std_dev"] = None
            stats[key]["range"] = None
    return stats

def print_table(title, stats, metric_keys, count):
    """
    Prints a formatted table with count, average, median, standard deviation, and range for each metric.
    """
    print(f"\n{title} (Count: {count})")
    print("-" * 110)
    header = "Metric".ljust(20) + "".join(f"{key}".rjust(15) for key in metric_keys)
    print(header)
    for stat_type in ["average", "median", "std_dev", "range"]:
        row = stat_type.capitalize().ljust(20)
        for key in metric_keys:
            value = f"{stats[key][stat_type]:.2f}" if stats[key][stat_type] is not None else "N/A"
            row += value.rjust(15)
        print(row)
    print("-" * 110)

def main():
    predictions = read_prediction_results()
    metrics_list = read_metrics_results()

    if not metrics_list:
        print("[ERROR] No metrics data found for FITS files in the specified folder.")
        return

    metric_keys = [
        "fwhm_default", "fwhm_07", "fwhm_095", "compactness",
        "auc_med", "auc_peak", "auc_full", "kurtosis", "gaussian_kurtosis"
    ]

    groups = {
        "Overall Metrics": metrics_list,
        "Very Likely Meteors (score > 0.8)": filter_data_by_prediction(metrics_list, predictions, lambda s: s > 0.8),
        "Very Likely Non-Meteors (score < 0.2)": filter_data_by_prediction(metrics_list, predictions, lambda s: s < 0.2),
        "Meteors (score > 0.5)": filter_data_by_prediction(metrics_list, predictions, lambda s: s > 0.5),
        "Non-Meteors (score <= 0.5)": filter_data_by_prediction(metrics_list, predictions, lambda s: s <= 0.5),
    }

    print("\nTrail Counts per Group:")
    print("-" * 40)
    for group_name, data in groups.items():
        print(f"{group_name}: {len(data)}")
    print("-" * 40)

    for group_name, data in groups.items():
        if data:
            stats = compute_statistics(data, metric_keys)
            print_table(group_name, stats, metric_keys, len(data))
        else:
            print(f"[INFO] No trails found for {group_name}.")

if __name__ == "__main__":
    main()