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

"""
TrailDetector is a class for detecting linear trails, such as meteor streaks,
in astronomical FITS images. It employs edge detection, Hough line detection,
and DBSCAN clustering to identify candidate trails.
"""

class TrailDetector:
    """
    A class for detecting and analyzing linear trails in FITS images.

    This class processes astronomical FITS images using edge detection,
    Hough transforms, and DBSCAN clustering to extract linear features
    corresponding to potential meteor trails. It attempts two levels of
    detection: one for bright trails and another for dim ones.

    Attributes:
        canny_params (dict): Parameters for Canny edge detection.
        hough_params (dict): Parameters for Hough line detection.
        dbscan_eps (float): DBSCAN clustering epsilon value.
        dbscan_min_samples (int): Minimum samples required for DBSCAN clustering.
        merged_lines (list): List of detected and merged line segments.
        best_line (list): The most prominent detected trail.
    """

    def __init__(
        self,
        canny_params=None,
        hough_params=None,
        dbscan_eps=150,
        dbscan_min_samples=2,
    ):
        """
        Initializes the TrailDetector class with default or provided parameters.

        Args:
            canny_params (dict, optional): Parameters for Canny edge detection.
            hough_params (dict, optional): Parameters for Hough line detection.
            dbscan_eps (float, optional): DBSCAN epsilon value for clustering.
            dbscan_min_samples (int, optional): Minimum samples required for clustering.
        """
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

    def detect_trails(self, fits_file, save_processed=False, processed_dir="processed_images"):
        """
        Detects linear trails in a given FITS file using image processing.

        Args:
            fits_file (str): Path to the FITS file.
            save_processed (bool, optional): If True, saves preprocessed images.
            processed_dir (str, optional): Directory to store processed images.

        Returns:
            list: A list containing the best detected line, or an empty list if none are found.
        """
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

    def _attempt_detection(self, fits_file, mode, save_processed=False, processed_dir="processed_images"):
        """
        Attempts to detect trails in an image by preprocessing and applying edge detection.

        Args:
            fits_file (str): Path to the FITS file.
            mode (str): Either "bright" or "dim", determining preprocessing strategy.
            save_processed (bool, optional): If True, saves processed images.
            processed_dir (str, optional): Directory to store processed images.

        Returns:
            list: Merged detected line segments.
        """
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

    def _preprocess_bright(self, image_data):
        """
        Preprocesses for detection of bright trails by applying normalization and histogram equalization.

        Args:
            image_data (numpy.ndarray): FITS image data.

        Returns:
            numpy.ndarray: Processed image.
        """
        norm = ImageNormalize(image_data, interval=ZScaleInterval())
        float_img = norm(image_data)
        gray_8u = cv2.convertScaleAbs(float_img)
        equ = cv2.equalizeHist(gray_8u)
        kernel = np.ones((4, 4), np.uint8)
        equ = cv2.erode(equ, kernel, iterations=1)
        equ = cv2.dilate(equ, kernel, iterations=1)
        return equ
    
    def _preprocess_dim(self, image_data):
        """
        Preprocesses for detection dim trails by applying normalization and histogram equalization.

        Args:
            image_data (numpy.ndarray): FITS image data.

        Returns:
            numpy.ndarray: Processed image.
        """
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

    def _process_lines(self, lines):
        """
        Converts raw Hough transform line detections into structured data.

        Args:
            lines (numpy.ndarray): An array of detected lines from HoughLinesP.

        Returns:
            numpy.ndarray: An array containing structured line data, where each row consists of:
                - x1, y1, x2, y2: Start and end points of the line.
                - length: Euclidean distance between start and end points.
                - angle: Angle of the line with respect to the x-axis.
        """
        lines_data = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            angle = np.arctan2((y2 - y1), (x2 - x1))
            lines_data.append([x1, y1, x2, y2, length, angle])
        return np.array(lines_data)

    def _choose_best_line(self, lines):
        """
        Selects the longest detected line from the given list of merged lines.

        Args:
            lines (list): A list of detected and merged lines, where each line is
                          represented as [x1, y1, x2, y2].

        Returns:
            list or None: The longest detected line in the form [x1, y1, x2, y2],
                          or None if no lines are found.
        """
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

    def _merge_lines(self, lines_data):
        """
        Merges similar lines using DBSCAN clustering.

        Args:
            lines_data (numpy.ndarray): Array of detected lines.

        Returns:
            list: List of merged lines.
        """
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

    def plot_trails(self, fits_file, output_file=None):
        """
        Plots the detected trail on the original FITS image.

        Args:
            fits_file (str): Path to the FITS file.
            output_file (str, optional): If specified, saves the plot to the given file.
        """
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