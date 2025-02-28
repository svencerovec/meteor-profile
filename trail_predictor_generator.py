"""
train_full_model.py

This script trains an MLP model on all lines from:
- meteors.txt (label=1)
- not_meteors.txt (label=0)

No train/test split is performed: the entire dataset is used for training.
Lines producing invalid or empty profiles are skipped.

Usage:
    python train_full_model.py /path/to/fits/folder

The resulting model is saved as 'full_meteor_model.pkl'.
"""

import os
import sys
import argparse
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier

from trail_profiler import TrailProfiler

METEORS_TXT = "meteors.txt"
NOT_METEORS_TXT = "not_meteors.txt"
MODEL_FILENAME = "full_meteor_model.pkl"


def parse_arguments():
    """
    Parse command-line arguments to get the folder containing the FITS files.
    """
    parser = argparse.ArgumentParser(
        description="Train a full MLP model on lines from meteors.txt and not_meteors.txt, "
                    "using all data (no test split). Skips invalid lines."
    )
    parser.add_argument(
        "fits_folder",
        type=str,
        help="Path to the folder containing the FITS files."
    )
    return parser.parse_args()


def load_lines(txt_file, label, fits_folder):
    """
    Load lines from a text file of the format:
        file.fits x1 y1 x2 y2

    Args:
        txt_file (str): Path to the text file.
        label (int): 1 for meteors, 0 for not meteors.
        fits_folder (str): Folder path where the FITS files reside.

    Returns:
        list of tuples: (full_fits_path, x1, y1, x2, y2, label)
    """
    lines_data = []
    if not os.path.exists(txt_file):
        print(f"[ERROR] File not found: {txt_file}")
        return lines_data

    with open(txt_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"[WARNING] Skipping malformed line: {line}")
                continue

            filename, x1_str, y1_str, x2_str, y2_str = parts
            try:
                x1, y1, x2, y2 = map(float, [x1_str, y1_str, x2_str, y2_str])
            except ValueError:
                print(f"[WARNING] Skipping invalid numeric line: {line}")
                continue

            full_fits_path = os.path.join(fits_folder, filename)
            lines_data.append((full_fits_path, x1, y1, x2, y2, label))

    return lines_data


def build_dataset(lines):
    """
    For each line, use TrailProfiler to get a median profile. Skip invalid lines.

    Args:
        lines (list): Each element is (full_fits_path, x1, y1, x2, y2, label).

    Returns:
        X (list of np.ndarray): Brightness profiles
        y (list of int): Labels corresponding to each profile
    """
    X, y = [], []
    for (fits_path, x1, y1, x2, y2, lbl) in lines:
        if not os.path.exists(fits_path):
            print(f"[WARNING] FITS file not found: {fits_path}. Skipping.")
            continue

        try:
            profiler = TrailProfiler(fits_path, (x1, y1), (x2, y2))
            if not profiler.brightness_profiles:
                print(f"[WARNING] No valid perpendicular lines for {fits_path}. Skipping.")
                continue

            median_profile = profiler.calculate_median_profile()
            if median_profile is None or len(median_profile) == 0:
                print(f"[WARNING] Empty median profile for {fits_path}. Skipping.")
                continue

            X.append(median_profile)
            y.append(lbl)
        except Exception as e:
            print(f"[WARNING] Error profiling {fits_path}: {e}. Skipping.")
            continue

    return X, y


def main():
    args = parse_arguments()

    meteor_lines = load_lines(METEORS_TXT, label=1, fits_folder=args.fits_folder)
    not_meteor_lines = load_lines(NOT_METEORS_TXT, label=0, fits_folder=args.fits_folder)
    all_lines = meteor_lines + not_meteor_lines

    if not all_lines:
        print("[ERROR] No lines found. Exiting.")
        sys.exit(1)

    X_raw, y_raw = build_dataset(all_lines)
    if not X_raw:
        print("[ERROR] No valid profiles built. Exiting.")
        sys.exit(1)

    lengths = {len(p) for p in X_raw}
    if len(lengths) > 1:
        print(f"[WARNING] Multiple profile lengths detected: {lengths}. ")

    X_data = np.array(X_raw, dtype=object)
    y_data = np.array(y_raw)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    clf.fit(X_data.tolist(), y_data)  
    print("[INFO] Training complete. Model is now fit to the entire dataset.")

    with open(MODEL_FILENAME, "wb") as f:
        pickle.dump(clf, f)

    print(f"[INFO] Model saved to {MODEL_FILENAME}")


if __name__ == "__main__":
    main()