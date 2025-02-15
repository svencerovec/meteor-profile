import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import (
    ImageNormalize,
    PercentileInterval,
    SqrtStretch,
    ZScaleInterval,
)
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

# This class detects and processes linear trails in astronomical FITS images
class TrailDetector:
    # Initializes default parameters for edge detection, line detection, and clustering
    def __init__(
        self,
        canny_params=None,
        hough_params=None,
        dbscan_eps=150,
        dbscan_min_samples=2,
    ):
        self.canny_params = canny_params or {"threshold1": 6, "threshold2": 18}
        self.hough_params = hough_params or {
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 250,
            "minLineLength": 300,
            "maxLineGap": 150,
        }
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.merged_lines = []
        self.best_line = None

    # Attempts both "bright" and "dim" detections on a given FITS file
    def detect_trails(self, fits_file, save_processed=False, processed_dir="processed_images"):
        if save_processed:
            os.makedirs(processed_dir, exist_ok=True)

        lines_bright = self._attempt_detection(
            fits_file=fits_file,
            mode='bright',
            save_processed=save_processed,
            processed_dir=processed_dir
        )
        if lines_bright:
            self.merged_lines = lines_bright
            self.best_line = self._choose_best_line(lines_bright)
            return [self.best_line]

        lines_dim = self._attempt_detection(
            fits_file=fits_file,
            mode='dim',
            save_processed=save_processed,
            processed_dir=processed_dir
        )
        if lines_dim:
            self.merged_lines = lines_dim
            self.best_line = self._choose_best_line(lines_dim)
            return [self.best_line]

        print(f"No trails detected in {fits_file} after both attempts.")
        return []

    # Preprocessing and line detection for either bright or dim mode
    def _attempt_detection(
        self,
        fits_file,
        mode,
        save_processed=False,
        processed_dir="processed_images"
    ):
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data

        if mode == 'bright':
            processed = self._preprocess_bright(image_data)
        else:
            processed = self._preprocess_dim(image_data)

        if save_processed:
            base_no_ext = os.path.splitext(os.path.basename(fits_file))[0]
            out_name = f"{base_no_ext}_{mode}_processed.png"
            out_path = os.path.join(processed_dir, out_name)
            cv2.imwrite(out_path, processed)
            print(f"Saved pre-processed '{mode}' image to: {out_path}")

        edges = cv2.Canny(processed, **self.canny_params)
        lines = cv2.HoughLinesP(edges, **self.hough_params)
        if lines is None:
            return []

        lines_data = self._process_lines(lines)
        if len(lines_data) == 0:
            return []

        merged = self._merge_lines(lines_data)
        return merged

    # Handles bright-mode preprocessing, including intensity normalization, histogram equalization, and mild erosion/dilation
    def _preprocess_bright(self, image_data):
        norm = ImageNormalize(image_data, interval=ZScaleInterval())
        float_img = norm(image_data)
        gray_8u = cv2.convertScaleAbs(float_img)
        equ = cv2.equalizeHist(gray_8u)
        kernel = np.ones((4, 4), np.uint8)
        equ = cv2.erode(equ, kernel, iterations=1)
        equ = cv2.dilate(equ, kernel, iterations=1)
        return equ

    # Handles dim-mode preprocessing, including thresholding faint signals and applying morphological operations
    def _preprocess_dim(self, image_data):
        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        norm = ImageNormalize(
            image_data,
            interval=PercentileInterval(1, 99),
            stretch=SqrtStretch()
        )
        float_img = norm(image_data)
        gray_8u = cv2.convertScaleAbs(float_img)
        equ = cv2.equalizeHist(gray_8u)
        kernel = np.ones((3, 3), np.uint8)
        equ = cv2.erode(equ, kernel, iterations=1)
        kernel = np.ones((9, 9), np.uint8)
        equ = cv2.dilate(equ, kernel, iterations=1)
        return equ

    # Converts raw Hough lines into a structured array with line endpoints, length, and angle
    def _process_lines(self, lines):
        lines_data = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            angle = np.arctan2((y2 - y1), (x2 - x1))
            lines_data.append([x1, y1, x2, y2, length, angle])
        return np.array(lines_data)

    # Groups similar lines via DBSCAN clustering and merges them into single representative lines
    def _merge_lines(self, lines_data):
        def custom_distance(l1, l2):
            pos_dist = np.hypot(l1[0] - l2[0], l1[1] - l2[1])
            angle_dist = abs(l1[5] - l2[5])
            return pos_dist + 50 * angle_dist

        dist_matrix = squareform(pdist(lines_data, metric=custom_distance))
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="precomputed"
        ).fit(dist_matrix)

        merged_lines = []
        for label in set(clustering.labels_):
            if label == -1:
                continue
            cluster = lines_data[clustering.labels_ == label]
            x1_mean = int(cluster[:, 0].mean())
            y1_mean = int(cluster[:, 1].mean())
            x2_mean = int(cluster[:, 2].mean())
            y2_mean = int(cluster[:, 3].mean())
            merged_lines.append([x1_mean, y1_mean, x2_mean, y2_mean])
        return merged_lines

    # Chooses the longest line from a list of merged lines
    def _choose_best_line(self, lines):
        if not lines:
            return None
        best_line = None
        best_length = 0
        for line in lines:
            x1, y1, x2, y2 = line
            length = np.hypot(x2 - x1, y2 - y1)
            if length > best_length:
                best_length = length
                best_line = line
        return best_line

    # Displays the final detected line on the original FITS image
    def plot_trails(self, fits_file, output_file=None):
        if not self.best_line:
            print("No best line to plot.")
            return
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data
        plt.figure(figsize=(10, 10))
        norm = ImageNormalize(
            image_data,
            interval=PercentileInterval(1, 99),
            stretch=SqrtStretch()
        )
        plt.imshow(image_data, cmap='gray', origin='lower', norm=norm)
        x1, y1, x2, y2 = self.best_line
        plt.plot([x1, x2], [y1, y2], color='cyan', linewidth=2)
        plt.title("Detected Trail")
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.colorbar()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

    # Saves the best detected line to a text file
    def save_merged_lines(self, output_file):
        if not self.best_line:
            print("No line to save.")
            return
        with open(output_file, 'w') as f:
            f.write("x1, y1, x2, y2\n")
            f.write(f"{self.best_line[0]}, {self.best_line[1]}, "
                    f"{self.best_line[2]}, {self.best_line[3]}\n")
        print(f"Best candidate line saved to {output_file}")