import os
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

# Set up argument parsing for the script
parser = argparse.ArgumentParser(description="Detect and group trails in a single FITS image.")
parser.add_argument('fits_file_path', type=str, help="Path to the FITS file")
parser.add_argument('output_dir', type=str, help="Path to the directory where trail PNG will be saved")
args = parser.parse_args()

# Hough and Canny edge detection parameters
canny_threshold1 = 6
canny_threshold2 = 18
hough_rho = 1
hough_theta = np.pi / 180
hough_threshold = 250
hough_minLineLength = 500
hough_maxLineGap = 30

# Ensure the output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Open the FITS file and load the image data
fits_file = args.fits_file_path
with fits.open(fits_file) as hdul:
    image_data = hdul[0].data

# Use PercentileInterval and SqrtStretch for normalization
norm = ImageNormalize(image_data, interval=PercentileInterval(1, 99), stretch=SqrtStretch())
normalized_image = norm(image_data)

# Convert normalized image to 8-bit grayscale
scaled_image = np.uint8(255 * (normalized_image - np.min(normalized_image)) / 
                        (np.max(normalized_image) - np.min(normalized_image)))

# Use Canny edge detection
edges = cv2.Canny(scaled_image, threshold1=canny_threshold1, threshold2=canny_threshold2)

# Use Hough Line Transform to detect lines
lines = cv2.HoughLinesP(edges, rho=hough_rho, theta=hough_theta, threshold=hough_threshold, 
                        minLineLength=hough_minLineLength, maxLineGap=hough_maxLineGap)

# Store starting and ending points with additional attributes for grouping
if lines is not None:
    lines_data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        angle = np.arctan2((y2 - y1), (x2 - x1))
        lines_data.append([x1, y1, x2, y2, length, angle])

    lines_data = np.array(lines_data)

    # First grouping with custom distance function
    def custom_distance(line1, line2):
        pos_dist = np.hypot(line1[0] - line2[0], line1[1] - line2[1])
        angle_dist = np.abs(line1[5] - line2[5])
        return pos_dist + 50 * angle_dist  # Adjust weight on angle difference

    dist_matrix = squareform(pdist(lines_data, metric=custom_distance))
    clustering = DBSCAN(eps=150, min_samples=2, metric="precomputed").fit(dist_matrix)

    # Compute initial middle lines for each cluster
    unique_labels = set(clustering.labels_)
    middle_lines = []
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise points
        cluster_lines = lines_data[clustering.labels_ == label]
        x1_mean = int(cluster_lines[:, 0].mean())
        y1_mean = int(cluster_lines[:, 1].mean())
        x2_mean = int(cluster_lines[:, 2].mean())
        y2_mean = int(cluster_lines[:, 3].mean())
        middle_lines.append([x1_mean, y1_mean, x2_mean, y2_mean])

    # Second grouping to merge nearly aligned "middle" lines
    merged_lines = []
    used = np.zeros(len(middle_lines), dtype=bool)
    for i, line1 in enumerate(middle_lines):
        if used[i]:
            continue
        x1, y1, x2, y2 = line1
        line_cluster = [[x1, y1, x2, y2]]
        for j, line2 in enumerate(middle_lines):
            if i != j and not used[j]:
                x3, y3, x4, y4 = line2
                # Check if line2 is close and aligned with line1
                dist_start = np.hypot(x1 - x3, y1 - y3)
                dist_end = np.hypot(x2 - x4, y2 - y4)
                angle1 = np.arctan2(y2 - y1, x2 - x1)
                angle2 = np.arctan2(y4 - y3, x4 - x3)
                if abs(angle1 - angle2) < np.pi / 18 and (dist_start < 200 or dist_end < 200):
                    line_cluster.append([x3, y3, x4, y4])
                    used[j] = True
        # Calculate the average start and end for merged lines
        line_cluster = np.array(line_cluster)
        x1_avg, y1_avg = int(line_cluster[:, 0].mean()), int(line_cluster[:, 1].mean())
        x2_avg, y2_avg = int(line_cluster[:, 2].mean()), int(line_cluster[:, 3].mean())
        merged_lines.append((x1_avg, y1_avg, x2_avg, y2_avg))
        used[i] = True

    # Plot the original image and overlay merged middle lines
    plt.figure(figsize=(10, 10))
    plt.imshow(image_data, cmap='gray', origin='lower', norm=norm)
    for line in merged_lines:
        x1, y1, x2, y2 = line
        plt.plot([x1, x2], [y1, y2], color='cyan', linewidth=2)

    print(f"{fits_file} - Merged middle lines: {merged_lines}")

else:
    print(f"No trails detected in {fits_file}.")

plt.title("Detected and Grouped Trails")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.colorbar()

# Save the plot as a PNG in the output directory
output_png_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(fits_file))[0]}_merged.png")
plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
plt.close()

# Summary of results
if lines is not None:
    print("Trail groups detected, merged, and plotted.")
else:
    print("No trail detected.")
