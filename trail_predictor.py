#!/usr/bin/env python3

"""
meteor_classifier.py

This script uses two text files (meteors.txt and not_meteors.txt), each with 100 lines of
the form "file.fits x1 y1 x2 y2", where each file has exactly 20 lines per filter (u,g,r,i,z).
We verify that each filter has exactly 40 lines total (20 meteors + 20 not meteors).

We then randomly select 10 lines per filter from each file for training, and the remaining
10 lines per filter for testing. Next, we use the TrailProfiler to extract brightness
profiles, compute the median profile, and feed these vectors into an MLP neural network
for classification. Finally, we evaluate the model on the test set with a classification
report.

Usage Example:
    python meteor_classifier.py /path/to/fits/folder

Requires:
- meteors.txt
- not_meteors.txt
- scikit-learn
- A defined TrailProfiler class with "standard settings" 
"""

import os
import sys
import random
import argparse
import numpy as np
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

from trail_profiler import TrailProfiler

def parse_args():
    """
    Parses command-line arguments to get the FITS folder path.
    """
    parser = argparse.ArgumentParser(
        description="Train a meteor classifier on lines from meteors.txt and not_meteors.txt, "
                    "using FITS files from a specified folder."
    )
    parser.add_argument(
        "fits_folder",
        type=str,
        help="Path to the folder containing the FITS files."
    )
    return parser.parse_args()

def load_lines_from_file(txt_file, label, fits_folder):
    """
    Reads a text file where each line has the format:
        file.fits x1 y1 x2 y2
    Prepares a tuple (full_fits_path, x1, y1, x2, y2, label).

    Args:
        txt_file (str): Path to the text file to parse.
        label (int): Label for these samples (1 for meteor, 0 for not meteor).
        fits_folder (str): The directory containing the FITS files.

    Returns:
        List[Tuple[str, float, float, float, float, int]]
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
                x1, y1, x2, y2 = map(float, (x1_str, y1_str, x2_str, y2_str))
            except ValueError:
                print(f"[WARNING] Skipping invalid numeric line: {line}")
                continue

            full_fits_path = os.path.join(fits_folder, filename)
            lines_data.append((full_fits_path, x1, y1, x2, y2, label))

    return lines_data

def extract_filter(fits_filepath):
    """
    Extracts the filter character (u, g, r, i, or z) from the FITS filename.
    Expects something like "...-u-..." or "...-g-..." in the base name.
    """
    possible_filters = ['u','g','r','i','z']
    base = os.path.basename(fits_filepath)
    for f in possible_filters:
        pattern = f"-{f}-"
        if pattern in base:
            return f
    return None

def main():
    args = parse_args()

    meteor_file = "meteors.txt"
    not_meteor_file = "not_meteors.txt"

    meteor_lines = load_lines_from_file(meteor_file, label=1, fits_folder=args.fits_folder)
    not_meteor_lines = load_lines_from_file(not_meteor_file, label=0, fits_folder=args.fits_folder)

    data_by_filter = defaultdict(list)
    for entry in (meteor_lines + not_meteor_lines):
        fits_file, x1, y1, x2, y2, label = entry
        flt = extract_filter(fits_file)
        if flt is None:
            print(f"[WARNING] Could not detect filter for {fits_file}. Skipping.")
            continue
        data_by_filter[flt].append(entry)

    for flt in ['u','g','r','i','z']:
        lines_for_filter = data_by_filter[flt]
        if len(lines_for_filter) != 40:
            print(f"[ERROR] Filter '{flt}' does not have exactly 40 lines. Found: {len(lines_for_filter)}")
            print("[INFO] Exiting.")
            sys.exit(1)

    train_data = []
    test_data = []
    for flt in ['u','g','r','i','z']:
        lines_for_filter = data_by_filter[flt]
        meteor_subset = [x for x in lines_for_filter if x[5] == 1]
        not_meteor_subset = [x for x in lines_for_filter if x[5] == 0]

        random.shuffle(meteor_subset)
        random.shuffle(not_meteor_subset)

        train_meteor = meteor_subset[:10]
        test_meteor = meteor_subset[10:]
        train_not_meteor = not_meteor_subset[:10]
        test_not_meteor = not_meteor_subset[10:]

        train_data.extend(train_meteor + train_not_meteor)
        test_data.extend(test_meteor + test_not_meteor)

    def get_median_profile(line_info):
        fits_path, x1, y1, x2, y2, label = line_info
        if not os.path.exists(fits_path):
            print(f"[WARNING] FITS file not found: {fits_path}. Skipping.")
            return None

        try:
            profiler = TrailProfiler(fits_path, (x1, y1), (x2, y2))
            if not profiler.brightness_profiles:
                print(f"[WARNING] No valid perpendicular lines for {fits_path}. Skipping.")
                return None
            return profiler.calculate_median_profile()
        except Exception as e:
            print(f"[WARNING] Error profiling {fits_path}: {e}. Skipping.")
            return None

    def build_dataset(lines):
        X_list, y_list = [], []
        for entry in lines:
            profile = get_median_profile(entry)
            if profile is None:
                continue
            X_list.append(profile)
            y_list.append(entry[5])
        return X_list, y_list

    X_train_raw, y_train = build_dataset(train_data)
    X_test_raw, y_test = build_dataset(test_data)

    if not X_train_raw:
        print("[ERROR] No training samples found. Exiting.")
        sys.exit(1)
    if not X_test_raw:
        print("[ERROR] No testing samples found. Exiting.")
        sys.exit(1)

    all_profiles = X_train_raw + X_test_raw
    max_len = max(len(p) for p in all_profiles)
    print(f"[INFO] Maximum profile length found among all samples: {max_len}")

    def pad_profiles(profiles, length):
        result = []
        for p in profiles:
            if len(p) > length:
                arr = p[:length]
            else:
                arr = np.zeros(length, dtype=p.dtype)
                arr[:len(p)] = p
            result.append(arr)
        return np.array(result)

    X_train = pad_profiles(X_train_raw, max_len)
    X_test = pad_profiles(X_test_raw, max_len)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42
    )

    clf.fit(X_train, y_train)
    print("[INFO] Finished training the MLP model on the training set.")

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Meteor", "Meteor"]))


if __name__ == "__main__":
    main()