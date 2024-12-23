#!/usr/bin/env python3
"""
Detect and plot best trails for all FITS files in a given directory.
"""

import os
import glob
from trail_detector_3 import TrailDetector  # Import the TrailDetector class
from trail_profiler import TrailProfiler    # Import the TrailProfiler class (if needed)


def main():
    # Parameters for star removal (even if remove_stars_before=False, this can be set)
    remove_stars_params={
        "defaultxy": 20,
        "maxxy": 80,
        "pixscale": 0.396,
        "filter_caps": {"u": 30.0, "g": 30.0, "r": 30.0, "i": 30.0, "z": 30.0},
        "magcount": 1,
        "maxmagdiff": 99,
    }

    # Step 1: Detect trails using TrailDetector
    detector = TrailDetector(
        remove_stars_before=False,
        remove_stars_params=remove_stars_params
    )

    # Create an output directory for plots of the best lines
    output_plots_dir = "detected_line_plots"
    os.makedirs(output_plots_dir, exist_ok=True)

    # Iterate through all FITS files in the testing directory
    fits_dir = "testing/"
    fits_files = glob.glob(os.path.join(fits_dir, "*.fits"))
    print(f"Found {len(fits_files)} FITS files in '{fits_dir}'.")

    for fits_file in fits_files:
        print(f"Processing {fits_file}")

        # Detect trails for the current file
        merged_lines = detector.detect_trails(
            fits_file,
            save_processed=True,              # This triggers saving the pre-processed image
            processed_dir="processed_images"  # Saved PNGs go in this folder
        )
        if not merged_lines:
            print("No trails detected, skipping analysis for this file.")
            continue

        # ----------------------------------------------------
        # Save a plot of the best detected line
        # ----------------------------------------------------
        base_name = os.path.splitext(os.path.basename(fits_file))[0]
        output_file = os.path.join(output_plots_dir, f"{base_name}_best_line.png")

        # Plot and save the best line
        detector.plot_trails(fits_file, output_file=output_file)
        print(f"Best line plot saved to: {output_file}")

        # ----------------------------------------------------
        # OPTIONAL: If you want to profile the lines,
        # uncomment and adapt the following:
        # ----------------------------------------------------
        # profiler = TrailProfiler(fits_file)
        # for i, line in enumerate(merged_lines):
        #     # Analyze brightness profiles
        #     brightness_profiles, line_coordinates = profiler.analyze_perpendicular_lines(
        #         line, num_perpendicular_lines=10
        #     )
        #     median_profile = profiler.get_combined_median_profile(brightness_profiles, fits_file)
        #     # ... do something with 'median_profile' ...
        # ----------------------------------------------------


if __name__ == "__main__":
    main()


