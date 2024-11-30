import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN


class TrailDetector:
    def __init__(self, canny_params=None, hough_params=None, dbscan_eps=150, dbscan_min_samples=2):
        # Default parameters
        self.canny_params = canny_params or {"threshold1": 6, "threshold2": 18}
        self.hough_params = hough_params or {
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 250,
            "minLineLength": 500,
            "maxLineGap": 30,
        }
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.merged_lines = []

    def detect_trails(self, fits_file):
        # Open and normalize the FITS file
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data

        # Normalize the image for processing
        norm = ImageNormalize(image_data, interval=PercentileInterval(1, 90), stretch=SqrtStretch())
        normalized_image = norm(image_data)

        # Convert normalized image to 8-bit grayscale
        scaled_image = np.uint8(255 * (normalized_image - np.min(normalized_image)) /
                                (np.max(normalized_image) - np.min(normalized_image)))

        # Detect edges using Canny
        edges = cv2.Canny(scaled_image, **self.canny_params)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, **self.hough_params)

        if lines is None:
            print(f"No trails detected in {fits_file}.")
            return []

        # Extract line data and cluster using DBSCAN
        lines_data = self._process_lines(lines)
        if len(lines_data) == 0:
            return []

        # Perform grouping and merging
        self.merged_lines = self._merge_lines(lines_data)
        return self.merged_lines

    def _process_lines(self, lines):
        lines_data = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            angle = np.arctan2((y2 - y1), (x2 - x1))
            lines_data.append([x1, y1, x2, y2, length, angle])
        return np.array(lines_data)

    def _merge_lines(self, lines_data):
        # Compute a custom distance matrix
        def custom_distance(line1, line2):
            pos_dist = np.hypot(line1[0] - line2[0], line1[1] - line2[1])
            angle_dist = np.abs(line1[5] - line2[5])
            return pos_dist + 50 * angle_dist  # Adjust weight on angle difference

        dist_matrix = squareform(pdist(lines_data, metric=custom_distance))
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric="precomputed").fit(dist_matrix)

        # Compute the middle lines for each cluster
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

        return middle_lines

    def plot_trails(self, fits_file, output_file=None):
        # Plot the original image with merged trails overlaid
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data

        plt.figure(figsize=(10, 10))
        norm = ImageNormalize(image_data, interval=PercentileInterval(1, 99), stretch=SqrtStretch())
        plt.imshow(image_data, cmap='gray', origin='lower', norm=norm)
        for line in self.merged_lines:
            x1, y1, x2, y2 = line
            plt.plot([x1, x2], [y1, y2], color='cyan', linewidth=2)

        plt.title("Detected and Grouped Trails")
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.colorbar()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

    def save_merged_lines(self, output_file):
        """
        Save the merged lines to a text file.
        :param output_file: Path to the output text file.
        """
        if not self.merged_lines:
            print("No merged lines to save.")
            return

        with open(output_file, 'w') as f:
            f.write("x1, y1, x2, y2\n")
            for line in self.merged_lines:
                f.write(f"{line[0]}, {line[1]}, {line[2]}, {line[3]}\n")
        print(f"Merged lines saved to {output_file}")
