from trail_detector_2 import TrailDetector  # Import the TrailDetector class
from trail_profiler import TrailProfiler  # Import the TrailProfiler class
import argparse

# frame-i-002728-3-0430 meteor example

# Step 1: Detect trails using TrailDetector
detector = TrailDetector()
parser = argparse.ArgumentParser(description="Detect and group trails in a single FITS image.")
parser.add_argument('fits_file_path', type=str, help="Path to the FITS file")
args = parser.parse_args()
fits_file = args.fits_file_path
merged_lines = detector.detect_trails(fits_file)
#merged_lines = [[1820,244,2044,638]]
#merged_lines = [[1724, 1103, 1514, 3]]
if not merged_lines:
    print("No trails detected, skipping analysis.")
    exit()

# Step 2: Profile trails using TrailProfiler
profiler = TrailProfiler(fits_file)

for i, line in enumerate(merged_lines):
    print(f"Profiling trail {i + 1}: {line}")
    
    # Analyze brightness profiles and get perpendicular line coordinates
    brightness_profiles, line_coordinates = profiler.analyze_perpendicular_lines(line, num_perpendicular_lines=10)


     # Save the combined median profile and get it back
    median_profile = profiler.save_combined_median_profile(brightness_profiles, f"trail_{i+1}")

    # Evaluate the shape and print it
    shape = profiler.evaluate_profile_shape(median_profile)

    print(f"Trail {i+1} profile shape: {shape}")
    # Save brightness profiles
    profiler.plot_brightness_profiles(brightness_profiles, f"trail_{i + 1}_profiles")

    # Save image with overlaid perpendicular lines and enhanced brightness
    profiler.plot_divided_image(line_coordinates, f"trail_{i + 1}_divided")
