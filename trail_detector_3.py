import os
import re
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

class TrailDetector:
    def __init__(
        self,
        canny_params=None,
        hough_params=None,
        dbscan_eps=150,
        dbscan_min_samples=2,
        remove_stars_before=False,
        remove_stars_params=None
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

        self.remove_stars_before = remove_stars_before
        self.remove_stars_params = remove_stars_params or {}

        self.merged_lines = []
        self.best_line = None

    def detect_trails(self, fits_file, save_processed=False, processed_dir="processed_images"):
        """
        1) Parse (run, camcol, filter, field) from filename
        2) Attempt 'bright' detection -> if found, return
        3) Attempt 'dim' detection -> if found, return
        4) Optionally save the final pre-processed image to disk
        """
        # Ensure directory exists if saving
        if save_processed:
            os.makedirs(processed_dir, exist_ok=True)

        run, camcol, filter_band, field = self._parse_filename(fits_file)

        # 1) BRIGHT pass
        lines_bright = self._attempt_detection(
            fits_file,
            mode='bright',
            run=run,
            camcol=camcol,
            filter_band=filter_band,
            field=field,
            save_processed=save_processed,
            processed_dir=processed_dir
        )
        if lines_bright:
            self.merged_lines = lines_bright
            self.best_line = self._choose_best_line(lines_bright)
            return [self.best_line]

        # 2) DIM pass
        lines_dim = self._attempt_detection(
            fits_file,
            mode='dim',
            run=run,
            camcol=camcol,
            filter_band=filter_band,
            field=field,
            save_processed=save_processed,
            processed_dir=processed_dir
        )
        if lines_dim:
            self.merged_lines = lines_dim
            self.best_line = self._choose_best_line(lines_dim)
            return [self.best_line]

        print(f"No trails detected in {fits_file} after both attempts.")
        return []

    def _parse_filename(self, fits_file):
        basename = os.path.basename(fits_file)
        base_no_ext, _ = os.path.splitext(basename)
        tokens = base_no_ext.split('-')  # e.g. ["frame", "g", "000250", "5", "0188"]
        filter_band = tokens[1]
        run = int(tokens[2])
        camcol = int(tokens[3])
        field = int(tokens[4])
        return run, camcol, filter_band, field

    def _attempt_detection(
        self, 
        fits_file, 
        mode, 
        run, 
        camcol, 
        filter_band, 
        field,
        save_processed=False,
        processed_dir="processed_images"
    ):
        # 1) Load FITS data
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data

        # 2) Optionally remove stars
        

        # 3) Pre-process (bright or dim)
        if mode == 'bright':
            processed = self._preprocess_bright(image_data)
        else:
            processed = self._preprocess_dim(image_data)

        # 4) (New) Save the final pre-processed image if requested
        if save_processed:
            base_no_ext = os.path.splitext(os.path.basename(fits_file))[0]
            out_name = f"{base_no_ext}_{mode}_processed.png"
            out_path = os.path.join(processed_dir, out_name)
            cv2.imwrite(out_path, processed)
            print(f"Saved pre-processed '{mode}' image to: {out_path}")

        # 5) Edge detection
        edges = cv2.Canny(processed, **self.canny_params)

        # 6) Hough transform
        lines = cv2.HoughLinesP(edges, **self.hough_params)
        if lines is None:
            return []

        # 7) Convert to [x1, y1, x2, y2, length, angle]
        lines_data = self._process_lines(lines)
        if len(lines_data) == 0:
            return []
        # 8) DBSCAN merging
        merged = self._merge_lines(lines_data)
        return merged

    def _preprocess_bright(self, image_data):
        """
        - ZScaleInterval => stretch
        - convertScaleAbs => 8-bit
        - equalizeHist
        - Erode & dilate (mild)
        """
        norm = ImageNormalize(image_data, interval=ZScaleInterval())
        float_img = norm(image_data)          # Normalized float
        gray_8u = cv2.convertScaleAbs(float_img)  # Convert to 8-bit
        equ = cv2.equalizeHist(gray_8u)

        kernel = np.ones((4, 4), np.uint8)
        equ = cv2.erode(equ, kernel, iterations=1)
        equ = cv2.dilate(equ, kernel, iterations=1)
        return equ
    
    def _preprocess_dim(self, image_data):
        """
        Dim preprocessing using ZScaleInterval and PowerStretch without erosion or dilation.
        """
        from astropy.visualization import ZScaleInterval, PowerStretch, ImageNormalize
        import numpy as np

        # Handle NaN and invalid values
        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply ZScaleInterval and PowerStretch
        norm = ImageNormalize(
            image_data,
            interval=PercentileInterval(1, 99),
            stretch=SqrtStretch()
        )

        # Normalize to 8-bit range
        float_img = norm(image_data)
        gray_8u = cv2.convertScaleAbs(float_img)  # Convert to 8-bit
        equ = cv2.equalizeHist(gray_8u)
        kernel = np.ones((3, 3), np.uint8)
        equ = cv2.erode(equ, kernel, iterations=1)
        kernel = np.ones((9, 9), np.uint8)
        equ = cv2.dilate(equ, kernel, iterations=1)
        return equ

    def _process_lines(self, lines):
        lines_data = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            angle = np.arctan2((y2 - y1), (x2 - x1))
            lines_data.append([x1, y1, x2, y2, length, angle])
        return np.array(lines_data)

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

    def save_merged_lines(self, output_file):
        if not self.best_line:
            print("No line to save.")
            return
        with open(output_file, 'w') as f:
            f.write("x1, y1, x2, y2\n")
            f.write(f"{self.best_line[0]}, {self.best_line[1]}, "
                    f"{self.best_line[2]}, {self.best_line[3]}\n")
        print(f"Best candidate line saved to {output_file}")
