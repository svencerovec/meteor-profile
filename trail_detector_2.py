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
        # Default parameters (original values)
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
        self.best_line = None  # Will store the single best candidate line

    def detect_trails(self, fits_file):
        # First attempt (bright-like conditions)
        lines_bright = self._attempt_detection(fits_file, interval=(1, 90))
        if lines_bright:
            self.merged_lines = lines_bright
            self.best_line = self._choose_best_line(self.merged_lines)
            return [self.best_line]

        # If no lines found, attempt with dim-like conditions
        lines_dim = self._attempt_detection(fits_file, interval=(5, 99))
        if lines_dim:
            self.merged_lines = lines_dim
            self.best_line = self._choose_best_line(self.merged_lines)
            return [self.best_line]

        print(f"No trails detected in {fits_file} after both attempts.")
        return []

    def _attempt_detection(self, fits_file, interval):
        # 1. Open and normalize the FITS file
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data

        norm = ImageNormalize(image_data, interval=PercentileInterval(*interval), stretch=SqrtStretch())
        normalized_image = norm(image_data)

        # 2. Convert to 8-bit grayscale
        scaled_image = np.uint8(
            255 * (normalized_image - np.min(normalized_image))
            / (np.max(normalized_image) - np.min(normalized_image))
        )

        # 3. Edge detection (Canny)
        edges = cv2.Canny(scaled_image, **self.canny_params)

        # ----------------------------------------------------
        # 4. Connected-Components Filtering to remove small blobs
        # ----------------------------------------------------
        # Make sure 'edges' is a binary image (0 or 255).
        # It's already mostly binary from Canny, but let's ensure it is
        _, bin_edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

        # Label each connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bin_edges, connectivity=8
        )

        # Choose a threshold based on how big your stars are vs. your line
        min_area = 50  # You might need to adjust this

        # Remove small components
        for label_idx in range(1, num_labels):  # label 0 is the background
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < min_area:
                # Set those pixels to black in the binary mask
                bin_edges[labels == label_idx] = 0

        # This cleaned mask is what we'll pass to Hough
        cleaned_edges = bin_edges.copy()

        # ----------------------------------------------------
        # 5. Hough Transform on the cleaned edge image
        # ----------------------------------------------------
        lines = cv2.HoughLinesP(
            cleaned_edges,
            **self.hough_params
        )
        if lines is None:
            return []

        # 6. Process, cluster, and merge lines
        lines_data = self._process_lines(lines)
        if len(lines_data) == 0:
            return []

        merged = self._merge_lines(lines_data)
        return merged



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
            return pos_dist + 50 * angle_dist  # weight angle difference

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

    def _choose_best_line(self, lines):
        # Choose the longest line as the best candidate
        if not lines:
            return None
        best = None
        best_length = 0
        for line in lines:
            x1, y1, x2, y2 = line
            length = np.hypot(x2 - x1, y2 - y1)
            if length > best_length:
                best_length = length
                best = line
        return best

    def plot_trails(self, fits_file, output_file=None):
        if not self.best_line:
            print("No best line to plot.")
            return

        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data

        plt.figure(figsize=(10, 10))
        norm = ImageNormalize(image_data, interval=PercentileInterval(1, 99), stretch=SqrtStretch())
        plt.imshow(image_data, cmap='gray', origin='lower', norm=norm)

        # Plot the single best line
        x1, y1, x2, y2 = self.best_line
        plt.plot([x1, x2], [y1, y2], color='cyan', linewidth=2)

        plt.title("Detected Trail")
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.colorbar()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

    def save_merged_lines(self, output_file):
        """
        Save the single best line to a text file.
        """
        if not self.best_line:
            print("No line to save.")
            return

        with open(output_file, 'w') as f:
            f.write("x1, y1, x2, y2\n")
            f.write(f"{self.best_line[0]}, {self.best_line[1]}, {self.best_line[2]}, {self.best_line[3]}\n")
        print(f"Best candidate line saved to {output_file}")