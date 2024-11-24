from trail_detector import TrailDetector  # Import the TrailDetector class
from trail_profiler import TrailProfiler  # Import the TrailProfiler class
import argparse

# Step 1: Detect trails using TrailDetector
detector = TrailDetector()
parser = argparse.ArgumentParser(description="Detect and group trails in a single FITS image.")
parser.add_argument('fits_file_path', type=str, help="Path to the FITS file")
args = parser.parse_args()
fits_file = args.fits_file_path
merged_lines = detector.detect_trails(fits_file)

if not merged_lines:
    print("No trails detected, skipping analysis.")
    exit()

# Step 2: Profile trails using TrailProfiler
profiler = TrailProfiler(fits_file)

for i, line in enumerate(merged_lines):
    profile_name = f"trail_profile_{i + 1}"
    print(f"Profiling trail {i + 1}: {line}")
    
    # Analyze brightness profiles and get perpendicular line coordinates
    brightness_profiles, line_coordinates = profiler.analyze_perpendicular_lines(line, num_perpendicular_lines=20)

    # Save brightness profiles
    profiler.plot_brightness_profiles(brightness_profiles, profile_name)

    # Save image with overlaid perpendicular lines and enhanced brightness
    profiler.plot_divided_image(line_coordinates, profile_name)
