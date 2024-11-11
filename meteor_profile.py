import numpy as np
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

# Load FITS file
with fits.open('testing/frame-g-000211-4-0364.fits') as hdul:
    data = hdul[0].data

# Define start and end points of the main line
x0, y0 = 1133, 347
x1, y1 = 1837, 576

# Calculate midpoint of the main line
mid_x = (x0 + x1) / 2
mid_y = (y0 + y1) / 2

# Calculate direction vector of the main line
dx = x1 - x0
dy = y1 - y0

# Calculate perpendicular direction vector
perp_dx = -dy
perp_dy = dx
perp_length = np.hypot(perp_dx, perp_dy)
perp_dx /= perp_length
perp_dy /= perp_length

# Find the maximum extension of the perpendicular line within image boundaries
height, width = data.shape

# Function to extend a point in a direction until it hits the image boundary
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

# Get the start and end points of the longest perpendicular line
x_perp_start, y_perp_start, x_perp_end, y_perp_end = extend_to_boundary(mid_x, mid_y, perp_dx, perp_dy, width, height)

# Define coordinates along the perpendicular line for sampling
num_samples = int(np.hypot(x_perp_end - x_perp_start, y_perp_end - y_perp_end))
x_perpendicular = np.linspace(x_perp_start, x_perp_end, num_samples)
y_perpendicular = np.linspace(y_perp_start, y_perp_end, num_samples)

# Retrieve pixel values along the perpendicular line
perpendicular_coords = np.vstack((y_perpendicular, x_perpendicular))
brightness_along_perpendicular = map_coordinates(data, perpendicular_coords, order=1)

# Calculate the index of the midpoint along the perpendicular line
# This is the point where the main line intersects the perpendicular line
midpoint_index = num_samples // 2
midpoint_brightness = brightness_along_perpendicular[midpoint_index]

# Identify points along the perpendicular line where brightness is higher than at the intersection
bright_points_indices = np.where(brightness_along_perpendicular > midpoint_brightness)[0]
bright_x_coords = x_perpendicular[bright_points_indices]
bright_y_coords = y_perpendicular[bright_points_indices]

# Plot brightness data along the longest perpendicular line with midpoint marker
plt.figure(figsize=(10, 6))
plt.plot(brightness_along_perpendicular, label='Brightness Along Perpendicular Line')
plt.axvline(midpoint_index, color='red', linestyle='--', label='Intersection Point')
plt.xlabel('Position along perpendicular line')
plt.ylabel('Pixel Brightness')
plt.title('Brightness Along the Longest Perpendicular Line')
plt.legend()
plt.savefig("meteor_profile.png", dpi=300, bbox_inches='tight')
plt.show()

# Find the region of highest brightness to zoom in on
max_brightness = np.max(brightness_along_perpendicular)
threshold = 0.5 * max_brightness

# Find indices where brightness exceeds the threshold
bright_indices = np.where(brightness_along_perpendicular > threshold)[0]

# Determine the zoomed-in range
start_index = max(bright_indices[0] - 50, 0)
end_index = min(bright_indices[-1] + 50, len(brightness_along_perpendicular))

# Plot the brightness data, zoomed in on the brightest region with midpoint marker
plt.figure(figsize=(10, 6))
plt.plot(brightness_along_perpendicular, label='Brightness Along Perpendicular Line')
plt.axvline(midpoint_index, color='red', linestyle='--', label='Intersection Point')
plt.xlim(start_index, end_index)  # Set x-axis limits to zoom in
plt.xlabel('Position along perpendicular line')
plt.ylabel('Pixel Brightness')
plt.title('Zoomed-in Brightness Along the Longest Perpendicular Line')
plt.legend()


plt.show()

# Apply normalization to enhance visibility, similar to the previous script
norm = ImageNormalize(data, interval=PercentileInterval(1, 99), stretch=SqrtStretch())

# Plot and save the FITS image with the overlaid perpendicular line and bright points
plt.figure(figsize=(10, 10))
plt.imshow(data, cmap='gray', origin='lower', aspect='equal', norm=norm)
plt.colorbar(label='Pixel Brightness')
plt.plot([x_perp_start, x_perp_end], [y_perp_start, y_perp_end], color='cyan', linewidth=1.5, label='Perpendicular Line')
plt.scatter([mid_x], [mid_y], color='red', label='Intersection Point', zorder=5)
plt.scatter(bright_x_coords, bright_y_coords, color='green', s=10, label='Bright Points', zorder=5)
plt.legend()
plt.title("FITS Image with Overlaid Perpendicular Line and Bright Points")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")

# Save the plot with the perpendicular line and bright points as a PNG image
plt.savefig("edited.png", dpi=300, bbox_inches='tight')
plt.show()
