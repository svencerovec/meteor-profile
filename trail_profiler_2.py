import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from scipy.ndimage import map_coordinates

class TrailProfiler:
    def __init__(self, fits_file, point1, point2, output_dir="trail_profiles"):
        """
        Initialize the TrailProfiler with a FITS file and trail points.
        :param fits_file: Path to the FITS file.
        :param point1: First point of the trail (x0, y0).
        :param point2: Second point of the trail (x1, y1).
        :param output_dir: Directory to save trail profiles.
        """
        self.fits_file = fits_file
        self.point1 = point1
        self.point2 = point2
        self.output_dir = output_dir
        self.image_data = None
        self.normalized_data = None
        self.brightness_profiles = []
        self.line_coordinates = []
        self._load_image_data()
        self._ensure_output_directory()
        self._analyze_perpendicular_lines()

    def _load_image_data(self):
        """
        Load and normalize the FITS image data.
        """
        from astropy.io import fits
        with fits.open(self.fits_file) as hdul:
            self.image_data = hdul[0].data

        norm = ImageNormalize(self.image_data, interval=PercentileInterval(99.5), stretch=SqrtStretch())
        self.normalized_data = norm(self.image_data)

    def _ensure_output_directory(self):
        """
        Ensure that the output directory exists.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _analyze_perpendicular_lines(self, num_perpendicular_lines=10, short_window_size=100, step_size=0.1):
        """
        Analyze perpendicular brightness profiles along the defined trail line.
        Saves results to public variables.
        """
        x0, y0 = self.point1
        x1, y1 = self.point2
        main_line_length = np.hypot(x1 - x0, y1 - y0)

        # Perpendicular direction vector (unit vector)
        perp_dx = -(y1 - y0) / main_line_length
        perp_dy = (x1 - x0) / main_line_length
        step_along_main_line = main_line_length / (num_perpendicular_lines - 1)

        self.brightness_profiles = []
        self.line_coordinates = []

        for i in range(num_perpendicular_lines):
            t = i * step_along_main_line / main_line_length
            x_center = x0 + t * (x1 - x0)
            y_center = y0 + t * (y1 - y0)

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
            normalized_brightness = (brightness - np.min(self.image_data)) / (
                np.max(self.image_data) - np.min(self.image_data)
            )
            self.brightness_profiles.append(normalized_brightness)
            self.line_coordinates.append(((x_perp_start, y_perp_start), (x_perp_end, y_perp_end)))

    def plot_brightness_profiles(self, save_plot=False, profile_name="brightness_profiles"):
        """
        Plot brightness profiles with unique colors for each line.
        :param save_plot: If True, save the plot to a file.
        :param profile_name: Name for the saved plot file.
        """
        plt.figure(figsize=(12, 8))
        for i, profile in enumerate(self.brightness_profiles):
            plt.plot(profile, label=f'Profile {i + 1}')
        plt.axvline(len(self.brightness_profiles[0]) // 2, color='red', linestyle='--', label='Trail Center')
        plt.xlabel('Position along perpendicular line')
        plt.ylabel('Normalized Brightness')
        plt.title('Brightness Profiles')
        plt.legend()
        plt.grid()

        if save_plot:
            output_file = os.path.join(self.output_dir, f"{profile_name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved brightness profiles to {output_file}")

        plt.show()

    
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
    
    def get_combined_median_profile(self, brightness_profiles=None, profile_name="profile", save_median_profile=False):
        profiles_array = np.vstack(brightness_profiles if brightness_profiles else self.brightness_profiles)
        median_profile = np.median(profiles_array, axis=0)

        if save_median_profile:
            output_file = os.path.join(self.output_dir, f"{profile_name}_median.png")
            plt.figure(figsize=(12, 8))
            plt.plot(median_profile, label='Combined Median Profile', color='blue')
            plt.axvline(len(median_profile) // 2, color='red', linestyle='--', label='Intersection Point')
            plt.xlabel('Position along perpendicular line')
            plt.ylabel('Normalized Brightness (0-1)')
            plt.title(f'Combined Median Profile: {profile_name}')
            plt.legend()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved combined median profile plot to {output_file}")

        return median_profile
    
    def plot_median_profile(self):
        if not self.brightness_profiles:
            raise ValueError("No brightness profiles provided for median calculation.")

        profiles_array = np.vstack(self.brightness_profiles)
        median_profile = np.median(profiles_array, axis=0)

        plt.figure(figsize=(12, 8))
        plt.plot(median_profile, label='Combined Median Profile', color='blue')
        plt.axvline(len(median_profile) // 2, color='red', linestyle='--', label='Intersection Point')
        plt.xlabel('Position along perpendicular line')
        plt.ylabel('Normalized Brightness (0-1)')
        plt.title(f'Combined Median Profile')
        plt.show()

    def calculate_auc(self, profile):
        """
        Calculates the Area Under the Curve (AUC) of a normalized profile.
        
        Args:
            profile (numpy.ndarray): The intensity profile of the trail.

        Returns:
            float: The calculated AUC.
        """
        profile = profile / np.max(profile)
        return np.sum(profile)
    
    def calculate_auc_with_global_max(self, profile):
        """
        Calculates the Area Under the Curve (AUC) of a normalized profile,
        but normalizes using the maximum value of the entire image instead of the profile.

        Args:
            profile (numpy.ndarray): The intensity profile of the trail.

        Returns:
            float: The calculated AUC using the image's global maximum.
        """
        global_max = np.max(self.image_data)
        if global_max == 0:
            return 0
        normalized_profile = profile / global_max
        return np.sum(normalized_profile)

    def calculate_fwhm(self, profile, half_max_factor=0.5):
        """
        Calculates the Full Width at Half Maximum (FWHM) of a normalized profile.
        
        Args:
            profile (numpy.ndarray): The intensity profile of the trail.
            half_max_factor (float): The fraction of the maximum intensity to define "half max."

        Returns:
            float: The calculated FWHM.
        """
        profile = profile / np.max(profile)
        half_max = np.max(profile) * half_max_factor
        indices_above_half_max = np.where(profile >= half_max)[0]
        if len(indices_above_half_max) > 0:
            return indices_above_half_max[-1] - indices_above_half_max[0]
        return 0
    
    def extend_line(self, x0, y0, x1, y1):
        """
        Extend a line defined by (x0, y0) and (x1, y1) so that it touches the image boundaries.
        The resulting endpoints are clamped to the valid pixel indices based on the image dimensions.
        
        :param x0, y0: Start coordinates of the line.
        :param x1, y1: End coordinates of the line.
        :return: Extended line endpoints (x0_ext, y0_ext, x1_ext, y1_ext) clamped within image boundaries.
        """
        dx = x1 - x0
        dy = y1 - y0
        height, width = self.image_data.shape

        # Compute the t values for each image boundary.
        if dx != 0:
            t_left = -x0 / dx
            t_right = (width - 1 - x0) / dx
        else:
            t_left = -float('inf')
            t_right = float('inf')

        if dy != 0:
            t_top = -y0 / dy
            t_bottom = (height - 1 - y0) / dy
        else:
            t_top = -float('inf')
            t_bottom = float('inf')

        # Choose the t parameters that bring the line to the nearest boundaries.
        t_min = max(min(t_left, t_right), min(t_top, t_bottom))
        t_max = min(max(t_left, t_right), max(t_top, t_bottom))

        # Compute extended endpoints.
        x0_ext = x0 + t_min * dx
        y0_ext = y0 + t_min * dy
        x1_ext = x0 + t_max * dx
        y1_ext = y0 + t_max * dy

        # Clamp the endpoints so they don't exceed the image resolution.
        x0_ext = max(0, min(x0_ext, width - 1))
        x1_ext = max(0, min(x1_ext, width - 1))
        y0_ext = max(0, min(y0_ext, height - 1))
        y1_ext = max(0, min(y1_ext, height - 1))

        return x0_ext, y0_ext, x1_ext, y1_ext

    
    def remove_profile(self, index):
        """
        Remove the brightness profile and its corresponding line coordinates at the specified index.
        :param index: Index of the profile to remove (1-based index).
        """
        zero_based_index = index - 1  # Convert 1-based index to 0-based for internal use
        if zero_based_index < 0 or zero_based_index >= len(self.brightness_profiles):
            raise IndexError(f"Index {index} is out of range. Valid range: 1 to {len(self.brightness_profiles)}")

        # Remove the brightness profile and line coordinates
        del self.brightness_profiles[zero_based_index]
        del self.line_coordinates[zero_based_index]
        print(f"Removed profile and coordinates at index {index}.")
    
    def plot_profile_at_index(self, index, save_plot=False, profile_name="profile_at_index"):
        """
        Plot the brightness profile at the specified index.
        :param index: Index of the profile to plot (1-based index).
        :param save_plot: If True, save the plot to a file.
        :param profile_name: Name for the saved plot file.
        """
        zero_based_index = index - 1  # Convert 1-based index to 0-based for internal use
        if zero_based_index < 0 or zero_based_index >= len(self.brightness_profiles):
            raise IndexError(f"Index {index} is out of range. Valid range: 1 to {len(self.brightness_profiles)}")

        profile = self.brightness_profiles[zero_based_index]

        plt.figure(figsize=(10, 6))
        plt.plot(profile, label=f'Profile {index}', color='blue')
        plt.axvline(len(profile) // 2, color='red', linestyle='--', label='Trail Center')
        plt.xlabel('Position along perpendicular line')
        plt.ylabel('Normalized Brightness')
        plt.title(f'Brightness Profile at Index {index}')
        plt.legend()
        plt.grid()

        if save_plot:
            output_file = os.path.join(self.output_dir, f"{profile_name}_index_{index}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved profile plot for index {index} to {output_file}")

        plt.show()

    def plot_line_on_image(self, index, save_plot=False, profile_name="line_on_image"):
        """
        Plot the line corresponding to a brightness profile on the FITS image.
        :param index: Index of the line to plot (1-based index).
        :param save_plot: If True, save the plot to a file.
        :param profile_name: Name for the saved plot file.
        """
        zero_based_index = index - 1  # Convert 1-based index to 0-based for internal use
        if zero_based_index < 0 or zero_based_index >= len(self.line_coordinates):
            raise IndexError(f"Index {index} is out of range. Valid range: 1 to {len(self.line_coordinates)}")

        start, end = self.line_coordinates[zero_based_index]

        plt.figure(figsize=(10, 10))
        plt.imshow(self.normalized_data, cmap='gray', origin='lower')
        plt.plot(
            [start[0], end[0]], 
            [start[1], end[1]], 
            color='cyan', linestyle='-', linewidth=2, label=f'Line {index}'
        )
        plt.title(f"Line on Image for Profile {index}")
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.legend()

        if save_plot:
            output_file = os.path.join(self.output_dir, f"{profile_name}_index_{index}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved line plot for index {index} to {output_file}")

        plt.show()
    
    def plot_main_line(self, save_plot=False, profile_name="main_line_on_image"):
        """
        Plot the main line being analyzed on the FITS image.
        :param save_plot: If True, save the plot to a file.
        :param profile_name: Name for the saved plot file.
        """
        x0, y0 = self.point1
        x1, y1 = self.point2

        plt.figure(figsize=(10, 10))
        plt.imshow(self.normalized_data, cmap='gray', origin='lower')
        plt.plot(
            [x0, x1], 
            [y0, y1], 
            color='red', linestyle='-', linewidth=2, label='Main Line'
        )
        plt.title("Main Line on Image")
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.legend()

        if save_plot:
            output_file = os.path.join(self.output_dir, f"{profile_name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved main line plot to {output_file}")

        plt.show()
    
    def plot_all_perpendicular_lines(self, save_plot=False, profile_name="all_perpendicular_lines"):
        """
        Plot all perpendicular lines being analyzed on the FITS image.
        :param save_plot: If True, save the plot to a file.
        :param profile_name: Name for the saved plot file.
        """
        if not self.line_coordinates:
            raise ValueError("No perpendicular lines to plot. Run analyze_perpendicular_lines first.")

        plt.figure(figsize=(10, 10))
        plt.imshow(self.normalized_data, cmap='gray', origin='lower')

        for i, (start, end) in enumerate(self.line_coordinates):
            plt.plot(
                [start[0], end[0]], 
                [start[1], end[1]], 
                linestyle='--', linewidth=1, label=f'Perpendicular Line {i + 1}'
            )
        plt.title("All Perpendicular Lines on Image")
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.legend(loc='upper right', fontsize='small')

        if save_plot:
            output_file = os.path.join(self.output_dir, f"{profile_name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved all perpendicular lines plot to {output_file}")

        plt.show()