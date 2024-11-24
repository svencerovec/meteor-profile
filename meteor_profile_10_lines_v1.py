import numpy as np
import argparse
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def extend_to_boundary(x, y, dx, dy, width, height):
    t1 = -x / dx if dx != 0 else float('inf')
    t2 = (width - x) / dx if dx != 0 else float('inf')
    t3 = -y / dy if dy != 0 else float('inf')
    t4 = (height - y) / dy if dy != 0 else float('inf')
    t_min = max(min(t1, t2), min(t3, t4))
    t_max = min(max(t1, t2), max(t3, t4))
    x_min, y_min = x + t_min * dx, y + t_min * dy
    x_max, y_max = x + t_max * dx, y + t_max * dy
    return x_min, y_min, x_max, y_max

parser = argparse.ArgumentParser(description="Detect and group trails in a single FITS image.")
parser.add_argument('fits_file_path', type=str, help="Path to the FITS file")
args = parser.parse_args()

# Load FITS file
fits_file = args.fits_file_path
with fits.open(fits_file) as hdul:
    data = hdul[0].data

# Define global minimum and maximum brightness values from the entire image for normalization
global_min_brightness = np.min(data)
global_max_brightness = np.max(data)

# Define original start and end points of the main line
#x0, y0 = 1639, 771
#x1, y1 = 1737, 1391

# filter i
# (742, 950, 1406, 1245)
x0, y0 = 742, 950
x1, y1 = 1406, 1245

#old values
#x0, y0 = 1133, 347
#x1, y1 = 1837, 576

# Calculate direction vector of the main line
dx = x1 - x0
dy = y1 - y0

# Extend the main line to the edges of the image
x0_ext, y0_ext, x1_ext, y1_ext = extend_to_boundary(x0, y0, dx, dy, data.shape[1], data.shape[0])
main_line_length = np.hypot(x1_ext - x0_ext, y1_ext - y0_ext)

# Calculate perpendicular direction vector (unit vector)
perp_dx = -dy / main_line_length
perp_dy = dx / main_line_length

# Number of perpendicular lines and step along the main line
num_perpendicular_lines = 20
step_along_main_line = main_line_length / (num_perpendicular_lines - 1)

# Define a finer step size for sampling along each perpendicular line
step_size = 0.1

# Define the shorter window size (in pixels) around the main line for perpendicular sampling
short_window_size = 100  # Sampling region length on each side of the main line, total 200 pixels

# Set up plot for the FITS image with all lines overlaid
fig, ax = plt.subplots(figsize=(10, 10))

# Display the FITS image with normalization
norm = ImageNormalize(data, interval=PercentileInterval(99.5), stretch=SqrtStretch())
ax.imshow(data, origin='lower', cmap='gray', norm=norm)

# Plot the extended main line
ax.plot([x0_ext, x1_ext], [y0_ext, y1_ext], color='yellow', linestyle='-', linewidth=2, label='Extended Main Line')

# Get a colormap and color range for unique line colors
colors = cm.viridis(np.linspace(0, 1, num_perpendicular_lines))

# Initialize figure for overlayed brightness profiles
plt.figure(figsize=(12, 8))

# Loop to generate, analyze, and plot each perpendicular line
for i in range(num_perpendicular_lines):
    # Calculate the point on the main line where the perpendicular line will be centered
    t = i * step_along_main_line / main_line_length
    x_center = x0_ext + t * (x1_ext - x0_ext)
    y_center = y0_ext + t * (y1_ext - y0_ext)

    # Define start and end points for the short perpendicular line centered at (x_center, y_center)
    x_perp_start = x_center - short_window_size * perp_dx
    y_perp_start = y_center - short_window_size * perp_dy
    x_perp_end = x_center + short_window_size * perp_dx
    y_perp_end = y_center + short_window_size * perp_dy

    # Plot the short perpendicular line with a unique color
    ax.plot([x_perp_start, x_perp_end], [y_perp_start, y_perp_end], color=colors[i], linestyle='--', linewidth=1, label=f'Perpendicular Line {i+1}')

    # Define coordinates along the short perpendicular line for sampling
    num_samples = int(2 * short_window_size / step_size)
    x_perpendicular_full = np.linspace(x_perp_start, x_perp_end, num_samples)
    y_perpendicular_full = np.linspace(y_perp_start, y_perp_end, num_samples)

    # Retrieve pixel values along the short perpendicular line
    perpendicular_coords = np.vstack((y_perpendicular_full, x_perpendicular_full))
    brightness_along_perpendicular = map_coordinates(data, perpendicular_coords, order=3)

    # Normalize brightness values using global min and max brightness
    normalized_brightness = (brightness_along_perpendicular - global_min_brightness) / (global_max_brightness - global_min_brightness)

    # Plot the normalized brightness profile on the same chart
    plt.plot(normalized_brightness, color=colors[i], label=f'Perpendicular Line {i+1}')

# Add labels and legend to the overlayed brightness profile plot
plt.axvline(num_samples // 2, color='red', linestyle='--', label='Intersection Point')  # Draw the intersection line
plt.xlabel('Position along short perpendicular line')
plt.ylabel('Normalized Pixel Brightness (0 to 1)')
plt.title(f'Overlayed Normalized Brightness Profiles for All Short Perpendicular Lines: {fits_file}')
plt.legend()
plt.savefig("1.png", dpi=300, bbox_inches='tight')
plt.show()
fig.savefig("2.png", dpi=300, bbox_inches='tight')
# Generate brightness profile along the extended main line
num_main_line_samples = int(main_line_length / step_size)
x_main_line = np.linspace(x0_ext, x1_ext, num_main_line_samples)
y_main_line = np.linspace(y0_ext, y1_ext, num_main_line_samples)
main_line_coords = np.vstack((y_main_line, x_main_line))
brightness_along_main_line = map_coordinates(data, main_line_coords, order=3)


