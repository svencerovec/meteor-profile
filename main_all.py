from trail_detector_2 import TrailDetector  # Import the TrailDetector class
from trail_profiler import TrailProfiler  # Import the TrailProfiler class
import os
import glob

# Step 1: Detect trails using TrailDetector
detector = TrailDetector()

# Iterate through all FITS files in the testing directory
fits_dir = "testing"
fits_files = glob.glob(os.path.join(fits_dir, "*.fits"))

for fits_file in fits_files:
    print(f"Processing {fits_file}")

    # Detect trails for the current file
    merged_lines = detector.detect_trails(fits_file)
    if not merged_lines:
        print("No trails detected, skipping analysis for this file.")
        continue

    # Profile trails using TrailProfiler
    profiler = TrailProfiler(fits_file)

    for i, line in enumerate(merged_lines):
        # Analyze brightness profiles
        brightness_profiles, line_coordinates = profiler.analyze_perpendicular_lines(line, num_perpendicular_lines=10)
        median_profile = profiler.get_combined_median_profile(brightness_profiles, fits_file)
        # Save the combined median profile (this now only writes FITS filenames to text files)
        profiler.note_median_profile_type(median_profile)

        # Not saving the brightness profiles plot
        # profiler.plot_brightness_profiles(brightness_profiles, f"trail_{i + 1}_profiles")

        # Not saving the divided image
        # profiler.plot_divided_image(line_coordinates, f"trail_{i + 1}_divided")
