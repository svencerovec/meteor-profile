import numpy as np
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

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

# Load FITS file
with fits.open('testing/frame-g-000211-4-0364.fits') as hdul:
    data = hdul[0].data

# Define start and end points of the main line
x0, y0 = 1133, 347
x1, y1 = 1837, 576

# Calculate direction vector of the main line
dx = x1 - x0
dy = y1 - y0
main_line_length = np.hypot(dx, dy)

# Calculate perpendicular direction vector (unit vector)
perp_dx = -dy / main_line_length
perp_dy = dx / main_line_length

# Number of perpendicular lines and step along the main line
num_perpendicular_lines = 10
step_along_main_line = main_line_length / (num_perpendicular_lines - 1)



# Define the window size (in pixels) around the main line
window_size = 100  # This value defines how far out from the main line we sample on either side

# Loop to generate and analyze each perpendicular line
for i in range(num_perpendicular_lines):
    # Calculate the point on the main line where the perpendicular line will be centered
    t = i * step_along_main_line / main_line_length
    x_center = x0 + t * dx
    y_center = y0 + t * dy

    # Extend the perpendicular line from this center point to the image boundaries
    x_perp_start, y_perp_start, x_perp_end, y_perp_end = extend_to_boundary(x_center, y_center, perp_dx, perp_dy, data.shape[1], data.shape[0])

    # Calculate the length of the perpendicular line and number of samples for the entire line
    line_length = np.hypot(x_perp_end - x_perp_start, y_perp_end - y_perp_start)
    num_samples = int(line_length)
    # Define coordinates along the full perpendicular line for sampling
    x_perpendicular_full = np.linspace(x_perp_start, x_perp_end, num_samples)
    y_perpendicular_full = np.linspace(y_perp_start, y_perp_end, num_samples)

    # Calculate midpoint index for the perpendicular line
    midpoint_index = num_samples // 2

    # Calculate the range of indices around the midpoint to isolate the trail
    window_num_samples = int(window_size)  # Number of samples on each side of the midpoint
    start_index = max(midpoint_index - window_num_samples, 0)
    end_index = min(midpoint_index + window_num_samples, num_samples)

    # Define isolated coordinates around the midpoint
    x_perpendicular = x_perpendicular_full[start_index:end_index]
    y_perpendicular = y_perpendicular_full[start_index:end_index]

    # Retrieve pixel values along the isolated section of the perpendicular line
    perpendicular_coords = np.vstack((y_perpendicular, x_perpendicular))
    brightness_along_perpendicular = map_coordinates(data, perpendicular_coords, order=3)

    # Normalize brightness values to the range [0, 1]
    min_brightness = np.min(brightness_along_perpendicular)
    max_brightness = np.max(brightness_along_perpendicular)
    normalized_brightness = (brightness_along_perpendicular - min_brightness) / (max_brightness - min_brightness)

    # Plot the normalized brightness profile for this isolated section of the perpendicular line
    plt.figure(figsize=(10, 6))
    plt.plot(normalized_brightness, label=f'Normalized Brightness Along Perpendicular Line {i+1}')
    plt.axvline(window_num_samples, color='red', linestyle='--', label='Intersection Point')  # Draw the intersection line
    plt.xlabel('Position along isolated perpendicular line')
    plt.ylabel('Normalized Pixel Brightness (0 to 1)')
    plt.title(f'Isolated Normalized Brightness Along Perpendicular Line {i+1}')
    plt.legend()
    #plt.savefig(f"isolated_meteor_profile_{i+1}.png", dpi=300, bbox_inches='tight')
    plt.show()
