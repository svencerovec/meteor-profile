#!/usr/bin/env python3

"""
predict_meteors_resume.py

Script that:
1. Loads an MLP model from 'full_meteor_model.pkl' in the current folder.
2. Looks for 'prediction_results.txt' in the current folder and collects
   the filenames that have already been processed successfully.
3. Iterates over all FITS files in a given folder, skipping any that already
   appear in 'prediction_results.txt'.
4. Uses TrailDetector to find the "best" (longest) line in each file.
5. Profiles that line with TrailProfiler to get a median brightness profile.
6. Predicts a probability that the line is a meteor and appends the result to
   'prediction_results.txt' in the format:
       file.fits x1 y1 x2 y2 model_score
7. Handles exceptions in detection/profiling, logs minimal warnings, and continues.

Usage:
    python predict_meteors_resume.py /path/to/folder
"""

import os
import sys
import glob
import pickle
import numpy as np

# Adjust imports for your environment:
from trail_detector_3 import TrailDetector   # or your local name
from trail_profiler_2 import TrailProfiler   # or your local name

from sklearn.neural_network import MLPClassifier

MODEL_FILENAME = "full_meteor_model.pkl"
OUTPUT_FILENAME = "prediction_results.txt"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/fits_folder")
        sys.exit(1)

    fits_folder = sys.argv[1]
    if not os.path.isdir(fits_folder):
        print(f"[ERROR] Provided path is not a directory: {fits_folder}")
        sys.exit(1)

    # 1) Load existing results (if any) to build a set of completed file basenames
    completed_files = set()
    if os.path.exists(OUTPUT_FILENAME):
        with open(OUTPUT_FILENAME, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Expect a format like:  file.fits x1 y1 x2 y2 model_score
                parts = line.split()
                if len(parts) >= 1:
                    completed_files.add(parts[0])  # The first token is "file.fits"

    # 2) Load the pickled model
    if not os.path.exists(MODEL_FILENAME):
        print(f"[ERROR] Model file not found: {MODEL_FILENAME}")
        sys.exit(1)

    with open(MODEL_FILENAME, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, MLPClassifier):
        print(f"[ERROR] Loaded object is not an MLPClassifier. Check {MODEL_FILENAME} contents.")
        sys.exit(1)

    # 3) Gather all FITS files in the specified folder
    fits_files = glob.glob(os.path.join(fits_folder, "*.fits"))
    if not fits_files:
        print(f"[INFO] No FITS files found in {fits_folder}. Exiting.")
        sys.exit(0)

    # 4) We'll open output in append mode for new results
    out_path = os.path.join(os.getcwd(), OUTPUT_FILENAME)
    with open(out_path, "a") as out_file:
        for fits_path in fits_files:
            base_name = os.path.basename(fits_path)
            # Skip if already completed
            if base_name in completed_files:
                # minimal log, if any
                continue

            # 5) Detect lines with TrailDetector
            detector = TrailDetector()
            try:
                lines = detector.detect_trails(fits_path)
            except Exception as e:
                print(f"[WARNING] Detection error on {base_name}: {e}. Skipping.")
                continue

            if not lines or detector.best_line is None:
                # minimal log
                # no lines found, skip
                continue

            x1, y1, x2, y2 = detector.best_line

            # 6) Profile the line
            try:
                profiler = TrailProfiler(fits_path, (x1, y1), (x2, y2))
                if not profiler.brightness_profiles:
                    # no valid brightness lines
                    continue

                median_profile = profiler.calculate_median_profile()
                if median_profile is None or len(median_profile) == 0:
                    continue
            except Exception as e:
                print(f"[WARNING] Profiling error on {base_name}: {e}. Skipping.")
                continue

            # 7) Model prediction
            X_input = np.array(median_profile).reshape(1, -1)
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_input)[0][1]
                    model_score = round(proba, 3)
                else:
                    # fallback to predict => 0 or 1
                    label = model.predict(X_input)[0]
                    model_score = float(label)
            except Exception as e:
                print(f"[WARNING] Model prediction error on {base_name}: {e}. Skipping.")
                continue

            # 8) Write result line: "file.fits x1 y1 x2 y2 model_score"
            line_text = f"{base_name} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {model_score}\n"
            out_file.write(line_text)

    print("[INFO] Done. New predictions appended to", out_path)


if __name__ == "__main__":
    main()
