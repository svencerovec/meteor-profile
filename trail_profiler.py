import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from scipy.ndimage import map_coordinates


class TrailProfiler:
    def __init__(self, fits_file, image_data=None, output_dir="trail_profiles", divided_dir="trails_divided"):
        """
        Initialize the TrailProfiler with a FITS file path or image data.
        :param fits_file: Path to the FITS file.
        :param image_data: Preloaded FITS image data (optional).
        :param output_dir: Directory to save trail profiles.
        :param divided_dir: Directory to save images with overlaid perpendicular lines.
        """
        self.fits_file = fits_file
        self.image_data = image_data
        self.normalized_data = None
        self.output_dir = output_dir
        self.divided_dir = divided_dir
        self._load_image_data()
        self._ensure_output_directories()

    def _load_image_data(self):
        """
        Load and normalize the FITS image data if not already provided.
        """
        if self.image_data is None:
            from astropy.io import fits
            with fits.open(self.fits_file) as hdul:
                self.image_data = hdul[0].data

        norm = ImageNormalize(self.image_data, interval=PercentileInterval(99.5), stretch=SqrtStretch())
        self.normalized_data = norm(self.image_data)

    def _ensure_output_directories(self):
        """
        Ensure that the output directories for trail profiles and divided images exist.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.divided_dir):
            os.makedirs(self.divided_dir)

    def extend_line(self, x0, y0, x1, y1):
        """
        Extend a line to the boundaries of the image.
        :param x0, y0: Start coordinates of the line.
        :param x1, y1: End coordinates of the line.
        :return: Extended line coordinates (x0_ext, y0_ext, x1_ext, y1_ext).
        """
        dx = x1 - x0
        dy = y1 - y0
        height, width = self.image_data.shape
        t1 = -x0 / dx if dx != 0 else float('inf')
        t2 = (width - x0) / dx if dx != 0 else float('inf')
        t3 = -y0 / dy if dy != 0 else float('inf')
        t4 = (height - y0) / dy if dy != 0 else float('inf')
        t_min = max(min(t1, t2), min(t3, t4))
        t_max = min(max(t1, t2), max(t3, t4))
        x0_ext, y0_ext = x0 + t_min * dx, y0 + t_min * dy
        x1_ext, y1_ext = x0 + t_max * dx, y0 + t_max * dy
        return x0_ext, y0_ext, x1_ext, y1_ext

    def analyze_perpendicular_lines(self, line, num_perpendicular_lines=10, short_window_size=100, step_size=0.1):
        """
        Analyze perpendicular brightness profiles along a given line.
        :param line: Line coordinates (x0, y0, x1, y1).
        :param num_perpendicular_lines: Number of perpendicular lines.
        :param short_window_size: Length of sampling region on each side of the perpendicular line.
        :param step_size: Step size for finer sampling along the perpendicular line.
        :return: Normalized brightness profiles for all perpendicular lines.
        """
        x0, y0, x1, y1 = line
        x0_ext, y0_ext, x1_ext, y1_ext = self.extend_line(x0, y0, x1, y1)
        main_line_length = np.hypot(x1_ext - x0_ext, y1_ext - y0_ext)

        # Perpendicular direction vector (unit vector)
        perp_dx = -(y1 - y0) / main_line_length
        perp_dy = (x1 - x0) / main_line_length
        step_along_main_line = main_line_length / (num_perpendicular_lines - 1)

        brightness_profiles = []
        line_coordinates = []  # Store coordinates of perpendicular lines for visualization

        for i in range(num_perpendicular_lines):
            t = i * step_along_main_line / main_line_length
            x_center = x0_ext + t * (x1_ext - x0_ext)
            y_center = y0_ext + t * (y1_ext - y0_ext)

            # Define perpendicular line
            x_perp_start = x_center - short_window_size * perp_dx
            y_perp_start = y_center - short_window_size * perp_dy
            x_perp_end = x_center + short_window_size * perp_dx
            y_perp_end = y_center + short_window_size * perp_dy

            # Sample brightness along the perpendicular line
            num_samples = int(2 * short_window_size / step_size)
            x_perpendicular_full = np.linspace(x_perp_start, x_perp_end, num_samples)
            y_perpendicular_full = np.linspace(y_perp_start, y_perp_end, num_samples)
            perpendicular_coords = np.vstack((y_perpendicular_full, x_perpendicular_full))
            brightness = map_coordinates(self.image_data, perpendicular_coords, order=3)

            # Normalize brightness
            global_min = np.min(self.image_data)
            global_max = np.max(self.image_data)
            normalized_brightness = (brightness - global_min) / (global_max - global_min)
            brightness_profiles.append(normalized_brightness)

            # Store coordinates for visualization
            line_coordinates.append(((x_perp_start, y_perp_start), (x_perp_end, y_perp_end)))

        return brightness_profiles, line_coordinates

    def plot_brightness_profiles(self, brightness_profiles, profile_name):
        """
        Plot and save normalized brightness profiles.
        :param brightness_profiles: List of brightness profiles for each perpendicular line.
        :param profile_name: Name for the saved profile file.
        """
        output_file = os.path.join(self.output_dir, f"{profile_name}.png")
        plt.figure(figsize=(12, 8))
        for i, profile in enumerate(brightness_profiles):
            plt.plot(profile, label=f'Perpendicular Line {i + 1}')
        plt.axvline(len(brightness_profiles[0]) // 2, color='red', linestyle='--', label='Intersection Point')
        plt.xlabel('Position along perpendicular line')
        plt.ylabel('Normalized Brightness (0-1)')
        plt.title(f'Brightness Profiles for Perpendicular Lines: {profile_name}')
        plt.legend()

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved profile plot to {output_file}")

    def plot_divided_image(self, line_coordinates, profile_name):
        """
        Plot and save the image with overlaid perpendicular lines, with enhanced brightness.
        :param line_coordinates: Coordinates of the perpendicular lines.
        :param profile_name: Name for the saved divided image file.
        """
        output_file = os.path.join(self.divided_dir, f"{profile_name}.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(self.normalized_data, cmap='gray', origin='lower')
        for (start, end) in line_coordinates:
            x_values = [start[0], end[0]]
            y_values = [start[1], end[1]]
            plt.plot(x_values, y_values, color='cyan', linestyle='--', linewidth=1)
        plt.title(f"Perpendicular Lines for: {profile_name}")
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved divided image to {output_file}")
