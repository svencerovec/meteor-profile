import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from scipy.ndimage import map_coordinates

class TrailProfiler:
    """
    The TrailProfiler class measures brightness distributions along lines
    perpendicular to a user-defined trail in an astronomical FITS image.
    It also provides utility methods to visualize and analyze these profiles,
    including quantitative metrics for further classification.
    """

    def __init__(self, fits_file, point1, point2, output_dir="trail_profiles"):
        """
        Initializes the profiler with a path to a FITS file and two points
        (x0, y0) and (x1, y1) defining the main trail.

        Args:
            fits_file (str): Path to the FITS file containing the image data.
            point1 (tuple): Coordinates of the first trail endpoint (x0, y0).
            point2 (tuple): Coordinates of the second trail endpoint (x1, y1).
            output_dir (str): Directory where plots and data will be saved.
        """
        self.fits_file = fits_file
        self.point1 = point1
        self.point2 = point2
        self.output_dir = output_dir

        self.image_data = None
        self.normalized_data = None
        self.brightness_profiles = []
        self.line_coordinates = []

        self._load_fits_data()
        self._create_output_dir()
        self._sample_perpendicular_profiles()

    def _load_fits_data(self):
        """
        Loads the FITS image data from disk and applies a brightness normalization.
        """
        from astropy.io import fits

        with fits.open(self.fits_file) as hdul:
            self.image_data = hdul[0].data

        norm = ImageNormalize(
            self.image_data,
            interval=PercentileInterval(99.5),
            stretch=SqrtStretch()
        )
        self.normalized_data = norm(self.image_data)

    def _create_output_dir(self):
        """
        Ensures that the directory for output files exists.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sample_perpendicular_profiles(self, num_perp_lines=10, half_line_length=100, sampling_step=0.1):
        """
        Draws multiple perpendicular lines along the main trail, then samples
        brightness values at regularly spaced points on each line. If any line
        would go outside the image boundaries, it is skipped entirely.

        Args:
            num_perp_lines (int): Number of perpendicular slices to take along the trail.
            half_line_length (float): Half the length of each perpendicular line in pixels.
            sampling_step (float): Step size (in pixels) used for sampling brightness.
        """
        x0, y0 = self.point1
        x1, y1 = self.point2
        main_length = np.hypot(x1 - x0, y1 - y0)

        # Unit vector perpendicular to the trail
        perp_dx = -(y1 - y0) / main_length
        perp_dy = (x1 - x0) / main_length

        # Distance between each perpendicular slice along the trail
        slice_spacing = main_length / (num_perp_lines - 1) if num_perp_lines > 1 else main_length

        height, width = self.image_data.shape
        self.brightness_profiles.clear()
        self.line_coordinates.clear()

        for i in range(num_perp_lines):
            t = i * slice_spacing / main_length
            x_center = x0 + t * (x1 - x0)
            y_center = y0 + t * (y1 - y0)

            x_start = x_center - half_line_length * perp_dx
            y_start = y_center - half_line_length * perp_dy
            x_end = x_center + half_line_length * perp_dx
            y_end = y_center + half_line_length * perp_dy

            if not (0 <= x_start <= (width - 1) and 0 <= x_end <= (width - 1) and
                    0 <= y_start <= (height - 1) and 0 <= y_end <= (height - 1)):
                continue

            num_samples = int(2 * half_line_length / sampling_step)
            x_coords = np.linspace(x_start, x_end, num_samples)
            y_coords = np.linspace(y_start, y_end, num_samples)
            coords_for_sampling = np.vstack((y_coords, x_coords))
            brightness_vals = map_coordinates(self.image_data, coords_for_sampling, order=3)

            data_min = np.min(self.image_data)
            data_max = np.max(self.image_data)
            if data_max == data_min:
                norm_vals = np.zeros_like(brightness_vals)
            else:
                norm_vals = (brightness_vals - data_min) / (data_max - data_min)

            self.brightness_profiles.append(norm_vals)
            self.line_coordinates.append(((x_start, y_start), (x_end, y_end)))

    def plot_perpendicular_profiles(self, save_plot=False, filename="perp_profiles"):
        """
        Plots all sampled brightness profiles in a single figure.

        Args:
            save_plot (bool): If True, saves the plot to disk.
            filename (str): Filename (without extension) for the saved plot.
        """
        plt.figure(figsize=(12, 8))
        for i, profile in enumerate(self.brightness_profiles):
            plt.plot(profile, label=f"Profile {i+1}")

        if self.brightness_profiles:
            center_index = len(self.brightness_profiles[0]) // 2
            plt.axvline(center_index, color='red', linestyle='--', label='Trail Center')

        plt.xlabel("Sample Index Along Perpendicular")
        plt.ylabel("Normalized Brightness")
        plt.title("Perpendicular Brightness Profiles")
        plt.legend()
        plt.grid()

        if save_plot:
            outpath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            print(f"Perpendicular profiles plot saved to {outpath}")

        plt.show()

    def plot_main_line(self, save_plot=False, filename="main_line"):
        """
        Displays the original image with the main trail (point1-point2) overlaid.

        Args:
            save_plot (bool): If True, saves the plot to disk.
            filename (str): Filename (without extension) for the saved plot.
        """
        x0, y0 = self.point1
        x1, y1 = self.point2

        plt.figure(figsize=(10, 10))
        plt.imshow(self.normalized_data, cmap="gray", origin="lower")
        plt.plot([x0, x1], [y0, y1], color="red", linestyle="-", linewidth=2, label="Main Trail")
        plt.title("Main Trail on Image")
        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")
        plt.legend()

        if save_plot:
            outpath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            print(f"Main line plot saved to {outpath}")

        plt.show()

    def plot_all_perp_lines(self, save_plot=False, filename="all_perp_lines"):
        """
        Displays the original image with all perpendicular lines overlaid.

        Args:
            save_plot (bool): If True, saves the plot to disk.
            filename (str): Filename (without extension) for the saved plot.
        """
        if not self.line_coordinates:
            raise ValueError("No perpendicular lines to plot. Run _sample_perpendicular_profiles first.")

        plt.figure(figsize=(10, 10))
        plt.imshow(self.normalized_data, cmap="gray", origin="lower")

        for i, (start, end) in enumerate(self.line_coordinates):
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            plt.plot(xs, ys, linestyle="--", linewidth=1, label=f"Line {i+1}")

        plt.title("All Perpendicular Lines on Image")
        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")
        plt.legend(loc="upper right", fontsize="small")

        if save_plot:
            outpath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            print(f"All perpendicular lines plot saved to {outpath}")

        plt.show()

    def calculate_median_profile(self, profiles=None):
        """
        Calculates the median brightness profile across multiple samples.

        Args:
            profiles (list or None): Optional list of arrays containing brightness profiles.
                                     Defaults to self.brightness_profiles if None.

        Returns:
            numpy.ndarray: Array of median brightness values.
        """
        chosen_profiles = profiles if profiles is not None else self.brightness_profiles
        if not chosen_profiles:
            raise ValueError("No brightness profiles available for median calculation.")
        stacked = np.vstack(chosen_profiles)
        median_vals = np.median(stacked, axis=0)
        return median_vals

    def plot_median_profile(self, profiles=None, save_plot=False, filename="median_profile"):
        """
        Displays and optionally saves the median brightness profile.

        Args:
            profiles (list or None): Optional list of arrays containing brightness profiles.
                                     Defaults to self.brightness_profiles if None.
            save_plot (bool): If True, saves the plot to disk.
            filename (str): Filename (without extension) for the saved plot.
        """
        median_vals = self.calculate_median_profile(profiles=profiles)

        plt.figure(figsize=(12, 8))
        plt.plot(median_vals, label="Median Brightness Profile", color="blue")
        center_idx = len(median_vals) // 2
        plt.axvline(center_idx, color="red", linestyle="--", label="Approx. Trail Center")
        plt.xlabel("Sample Index")
        plt.ylabel("Normalized Brightness")
        plt.title("Combined Median Profile")
        plt.legend()
        plt.grid()

        if save_plot:
            outpath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            print(f"Median profile plot saved to {outpath}")

        plt.show()

    def calculate_fwhm(self, profile, half_max_factor=0.5):
        """
        Determines the Full Width at Half Maximum (FWHM) of a profile.
        
        Args:
            profile (numpy.ndarray): Brightness profile of the trail.
            half_max_factor (float): Fraction of the max intensity defining "half max."
            
        Returns:
            float: The FWHM in index units.
        """
        peak = np.max(profile)
        if peak == 0:
            return 0
        half_max = peak * half_max_factor
        indices = np.where(profile >= half_max)[0]
        if len(indices) < 2:
            return 0
        return indices[-1] - indices[0]

    def calculate_fwhm_default(self, profile):
        """
        Computes the default FWHM using half_max_factor = 0.5.
        
        Returns:
            float: FWHM (default).
        """
        return self.calculate_fwhm(profile, half_max_factor=0.5)

    def calculate_fwhm_07(self, profile):
        """
        Computes the FWHM at 70\% of the peak intensity.
        
        Returns:
            float: FWHM with factor 0.7.
        """
        return self.calculate_fwhm(profile, half_max_factor=0.7)

    def calculate_fwhm_095(self, profile):
        """
        Computes the FWHM at 95\% of the peak intensity.
        
        Returns:
            float: FWHM with factor 0.95.
        """
        return self.calculate_fwhm(profile, half_max_factor=0.95)

    def calculate_auc_med(self, profile):
        """
        Computes the area under the curve (AUC) for a brightness profile normalized 
        by the median value of the entire FITS image.
        
        Returns:
            float: AUC with normalization by the median (CoverageMed).
        """
        m = np.median(self.image_data)
        if m == 0:
            return 0
        normalized_profile = profile / m
        return np.sum(normalized_profile)

    def calculate_auc_peak(self, profile):
        """
        Computes the area under the curve (AUC) for a brightness profile normalized 
        by the peak value of the median profile.
        
        Returns:
            float: AUC with normalization by the profile's peak (CoveragePeak).
        """
        M = np.max(profile)
        if M == 0:
            return 0
        normalized_profile = profile / M
        return np.sum(normalized_profile)

    def calculate_auc_full(self, profile):
        """
        Computes the area under the curve (AUC) for a brightness profile normalized 
        by the maximum pixel value of the entire FITS image.
        
        Returns:
            float: AUC with normalization by the global maximum (CoverageFull).
        """
        M_fits = np.max(self.image_data)
        if M_fits == 0:
            return 0
        normalized_profile = profile / M_fits
        return np.sum(normalized_profile)

    def calculate_kurtosis(self, profile):
        """
        Calculates the weighted kurtosis of a brightness profile.
        The brightness values serve as weights for the positional values.

        Returns:
            float: Kurtosis of the profile.
        """
        # Use indices as the x-values
        x = np.arange(len(profile))
        weights = profile
        total_weight = np.sum(weights)
        if total_weight == 0:
            return 0
        mu = np.sum(x * weights) / total_weight
        sigma2 = np.sum(((x - mu) ** 2) * weights) / total_weight
        if sigma2 == 0:
            return 0
        sigma = np.sqrt(sigma2)
        fourth_moment = np.sum(((x - mu) ** 4) * weights) / total_weight
        return fourth_moment / (sigma ** 4) - 3

    def calculate_gaussian_kurtosis(self, profile):
        """
        Fits the given brightness profile to a Gaussian function and computes the weighted kurtosis
        of the fitted profile.
        
        Args:
            profile (numpy.ndarray): The brightness profile to be fitted.
        
        Returns:
            float: The kurtosis of the fitted Gaussian profile, or None if the fitting fails.
        """
        from scipy.optimize import curve_fit

        # Define the Gaussian function.
        def gaussian(x, A, mu, sigma):
            return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        # Create an x-axis corresponding to profile indices.
        x = np.arange(len(profile))
        # Initial guess: Amplitude = max(profile), center = index of max, sigma = roughly a quarter of the profile length.
        initial_guess = [np.max(profile), np.argmax(profile), len(profile) / 4]

        try:
            params, _ = curve_fit(gaussian, x, profile, p0=initial_guess)
        except Exception as e:
            print(f"[WARNING] Gaussian fitting failed: {e}.")
            return None

        # Generate the fitted Gaussian profile.
        fitted_profile = gaussian(x, *params)
        
        # Compute weighted kurtosis for the fitted profile.
        total_weight = np.sum(fitted_profile)
        if total_weight == 0:
            return 0
        mu_weighted = np.sum(x * fitted_profile) / total_weight
        sigma2 = np.sum(((x - mu_weighted) ** 2) * fitted_profile) / total_weight
        if sigma2 == 0:
            return 0
        sigma_val = np.sqrt(sigma2)
        fourth_moment = np.sum(((x - mu_weighted) ** 4) * fitted_profile) / total_weight
        gaussian_kurtosis = fourth_moment / (sigma_val ** 4) - 3
        return gaussian_kurtosis


    def remove_profile_by_index(self, index):
        """
        Removes one brightness profile and its associated line coordinates.

        Args:
            index (int): 1-based index of the profile to remove.
        """
        idx = index - 1
        if idx < 0 or idx >= len(self.brightness_profiles):
            raise IndexError(f"Index {index} out of range. Valid: 1 to {len(self.brightness_profiles)}")
        del self.brightness_profiles[idx]
        del self.line_coordinates[idx]
        print(f"Removed brightness profile and line at index {index}.")

    def plot_profile_by_index(self, index, save_plot=False, filename="profile_by_index"):
        """
        Plots a single brightness profile from the stored list by index.

        Args:
            index (int): 1-based index of the profile to plot.
            save_plot (bool): If True, saves the plot to disk.
            filename (str): Filename (without extension) for the saved plot.
        """
        idx = index - 1
        if idx < 0 or idx >= len(self.brightness_profiles):
            raise IndexError(f"Index {index} out of range. Valid: 1 to {len(self.brightness_profiles)}")
        profile = self.brightness_profiles[idx]
        plt.figure(figsize=(10, 6))
        plt.plot(profile, label=f"Profile {index}", color="blue")
        plt.axvline(len(profile) // 2, color="red", linestyle="--", label="Approx. Trail Center")
        plt.xlabel("Sample Index Along Perpendicular")
        plt.ylabel("Normalized Brightness")
        plt.title(f"Brightness Profile {index}")
        plt.legend()
        plt.grid()
        if save_plot:
            outpath = os.path.join(self.output_dir, f"{filename}_index_{index}.png")
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            print(f"Profile {index} plot saved to {outpath}")
        plt.show()

    def plot_line_by_index(self, index, save_plot=False, filename="line_by_index"):
        """
        Overlays a single perpendicular line on the normalized image.

        Args:
            index (int): 1-based index of the line to plot.
            save_plot (bool): If True, saves the plot to disk.
            filename (str): Filename (without extension) for the saved plot.
        """
        idx = index - 1
        if idx < 0 or idx >= len(self.line_coordinates):
            raise IndexError(f"Index {index} out of range. Valid: 1 to {len(self.line_coordinates)}")
        start, end = self.line_coordinates[idx]
        plt.figure(figsize=(10, 10))
        plt.imshow(self.normalized_data, cmap="gray", origin="lower")
        plt.plot([start[0], end[0]], [start[1], end[1]], color="cyan", linestyle="-", linewidth=2, label=f"Line {index}")
        plt.title(f"Perpendicular Line for Profile {index}")
        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")
        plt.legend()
        if save_plot:
            outpath = os.path.join(self.output_dir, f"{filename}_index_{index}.png")
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            print(f"Line {index} plot saved to {outpath}")
        plt.show()
    
        def extend_line_to_image_edges(self, x0, y0, x1, y1):
            """
            Extends a line segment so that it intersects the image boundaries,
            ensuring the endpoints remain within valid pixel coordinates.

            Args:
                x0, y0 (float): Start of the line.
                x1, y1 (float): End of the line.

            Returns:
                tuple: (x0_extended, y0_extended, x1_extended, y1_extended)
            """
            dx, dy = x1 - x0, y1 - y0
            height, width = self.image_data.shape

            # Calculate intersection parameters for each edge
            if dx != 0:
                t_left = -x0 / dx
                t_right = (width - 1 - x0) / dx
            else:
                t_left = -np.inf
                t_right = np.inf

            if dy != 0:
                t_top = -y0 / dy
                t_bottom = (height - 1 - y0) / dy
            else:
                t_top = -np.inf
                t_bottom = np.inf

            # Determine how far to extend to reach the nearest edges
            t_min = max(min(t_left, t_right), min(t_top, t_bottom))
            t_max = min(max(t_left, t_right), max(t_top, t_bottom))

            # Compute extended coordinates
            x0_ext = x0 + t_min * dx
            y0_ext = y0 + t_min * dy
            x1_ext = x0 + t_max * dx
            y1_ext = y0 + t_max * dy

            # Clamp to image dimensions
            x0_ext = np.clip(x0_ext, 0, width - 1)
            x1_ext = np.clip(x1_ext, 0, width - 1)
            y0_ext = np.clip(y0_ext, 0, height - 1)
            y1_ext = np.clip(y1_ext, 0, height - 1)

            return x0_ext, y0_ext, x1_ext, y1_ext