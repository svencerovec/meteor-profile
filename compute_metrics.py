#!/usr/bin/env python3
"""
compute_metrics.py

This script processes all lines in prediction_results.txt (which are of the form:
"file.fits x1 y1 x2 y2 model_score") and for each FITS file (located in a user-specified folder)
it uses the filename and line coordinates to instantiate a TrailProfiler. The script then computes
the following metrics from the median brightness profile:
  - FWHM (default, 0.5 factor)
  - FWHM at 0.7 (Spread)
  - FWHM at 0.95 (Extension)
  - Compactness (FWHM0.7 / FWHM0.95)
  - AUC_med (normalized by the median of the FITS image)
  - AUC_peak (normalized by the peak of the median profile)
  - AUC_full (normalized by the global maximum of the FITS image)
  - Kurtosis and Gaussian Kurtosis

The results are saved in a new file "metrics_results.txt".
"""

import os
import sys
import argparse
import numpy as np

from trail_profiler_2 import TrailProfiler

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute metrics for predicted trails using the median profile."
    )
    parser.add_argument(
        "fits_folder",
        type=str,
        help="Path to the folder containing the FITS files."
    )
    return parser.parse_args()

def process_prediction_file(fits_folder, pred_file="prediction_results.txt", out_file="metrics_results.txt"):
    if not os.path.exists(pred_file):
        print(f"[ERROR] Prediction file {pred_file} not found.")
        sys.exit(1)
    
    with open(pred_file, "r") as f:
        lines = f.readlines()
    
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # Expecting: filename x1 y1 x2 y2 model_score (we use only the first five tokens)
        if len(parts) < 5:
            print(f"[WARNING] Skipping malformed line: {line}")
            continue
        
        filename = parts[0]
        try:
            x1, y1, x2, y2 = map(float, parts[1:5])
        except ValueError:
            print(f"[WARNING] Skipping line with invalid coordinates: {line}")
            continue
        
        full_path = os.path.join(fits_folder, filename)
        if not os.path.exists(full_path):
            print(f"[WARNING] FITS file not found: {full_path}. Skipping.")
            continue
        
        # Instantiate the profiler and calculate the median profile.
        try:
            profiler = TrailProfiler(full_path, (x1, y1), (x2, y2))
            if not profiler.brightness_profiles:
                print(f"[WARNING] No brightness profiles for {full_path}. Skipping.")
                continue
            median_profile = profiler.calculate_median_profile()
        except Exception as e:
            print(f"[WARNING] Error processing {full_path}: {e}. Skipping.")
            continue
        
        # Compute metrics using the new methods.
        try:
            fwhm_default = profiler.calculate_fwhm_default(median_profile)
            fwhm_07 = profiler.calculate_fwhm_07(median_profile)
            fwhm_095 = profiler.calculate_fwhm_095(median_profile)
            compactness = fwhm_07 / fwhm_095 if fwhm_095 != 0 else 0
            auc_med = profiler.calculate_auc_med(median_profile)
            auc_peak = profiler.calculate_auc_peak(median_profile)
            auc_full = profiler.calculate_auc_full(median_profile)
            kurtosis_val = profiler.calculate_kurtosis(median_profile)
            gaussian_kurtosis_val = profiler.calculate_gaussian_kurtosis(median_profile)
        except Exception as e:
            print(f"[WARNING] Error computing metrics for {full_path}: {e}. Skipping.")
            continue

        gaussian_kurtosis_str = f"{gaussian_kurtosis_val:.2f}" if gaussian_kurtosis_val is not None else "N/A"

        result_line = (
            f"{filename} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} "
            f"{fwhm_default:.2f} {fwhm_07:.2f} {fwhm_095:.2f} {compactness:.2f} "
            f"{auc_med:.2f} {auc_peak:.2f} {auc_full:.2f} {kurtosis_val:.2f} {gaussian_kurtosis_str}"
        )
        results.append(result_line)
        print(f"[INFO] Processed {filename}")
    
    # Write all results to the output file in write mode.
    with open(out_file, "w") as f:
        for res in results:
            f.write(res + "\n")
    print(f"[INFO] Metrics written to {out_file}")

def main():
    args = parse_args()
    process_prediction_file(args.fits_folder)

if __name__ == "__main__":
    main()