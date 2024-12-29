import os
import glob
from trail_detector_3 import TrailDetector  # Import the TrailDetector class
from trail_profiler import TrailProfiler  # Import the TrailProfiler class

def main():
    # Parameters for star removal (optional)
    remove_stars_params = {
        "defaultxy": 20,
        "maxxy": 80,
        "pixscale": 0.396,
        "filter_caps": {"u": 30.0, "g": 30.0, "r": 30.0, "i": 30.0, "z": 30.0},
        "magcount": 1,
        "maxmagdiff": 99,
    }

    # Initialize TrailDetector
    detector = TrailDetector(
        remove_stars_before=False,
        remove_stars_params=remove_stars_params
    )

    # Set input and output directories
    fits_dir = "/media/sf_SharedUbuntu/unzipped_frame_files/"
    output_file = "detected_trails_sorted.txt"

    fits_files = glob.glob(os.path.join(fits_dir, "*.fits"))
    print(f"Found {len(fits_files)} FITS files in '{fits_dir}'.")

    results = []

    # Iterate through all FITS files
    for fits_file in fits_files:
        print(f"Processing {fits_file}")

        try:
            # Detect trails for the current file
            merged_lines = detector.detect_trails(fits_file, save_processed=False)

            if merged_lines:
                # Use TrailProfiler to analyze the best line
                profiler = TrailProfiler(fits_file)
                brightness_profiles, _ = profiler.analyze_perpendicular_lines(detector.best_line)
                median_profile = profiler.get_combined_median_profile(brightness_profiles, os.path.basename(fits_file))

                # Calculate FWHM and AUC
                fwhm = profiler.calculate_fwhm(median_profile)
                auc = profiler.calculate_auc(median_profile)

                # Append results with the sum of FWHM and AUC for sorting
                x1, y1, x2, y2 = detector.best_line
                results.append((fits_file, x1, y1, x2, y2, fwhm, auc, fwhm + auc))
            else:
                print(f"No trails detected in {fits_file}")
        except Exception as e:
            print(f"Error processing {fits_file}: {e}")

    # Sort results by the sum of FWHM and AUC in descending order
    results.sort(key=lambda x: x[-1], reverse=True)

    # Write sorted results to the output file
    with open(output_file, "w") as out_file:
        out_file.write("Filename x1 y1 x2 y2 FWHM AUC Score\n")  # Header row
        for entry in results:
            out_file.write(f"{entry[0]} {entry[1]} {entry[2]} {entry[3]} {entry[4]} {entry[5]:.2f} {entry[6]:.2f} {entry[7]:.2f}\n")

    # Summary
    print(f"Detection complete. Results saved to {output_file}.")

if __name__ == "__main__":
    main()