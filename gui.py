import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch

from trail_detector_3 import TrailDetector
from trail_profiler_2 import TrailProfiler

class TrailProfilerGUI:
    """
    A simple GUI that demonstrates how to load a FITS file, detect trails, and profile them
    using the TrailProfiler class. Users can either select lines automatically (through
    TrailDetector) or enter coordinates manually, then visualize and manipulate the
    perpendicular brightness profiles in the image.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Trail Profiler Application")
        self.root.geometry("1100x700")
        
        self.fits_file = None
        self.image_data = None
        self.norm = None
        
        # Detector for automatic line finding
        self.detector = TrailDetector()

        # Profiler for analyzing lines
        self.profiler = None

        # Main UI frames
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(
            self.main_frame, text="Trail Profiler Application",
            font=("Helvetica", 18, "bold")
        )
        title_label.pack(pady=10)

        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # File Controls
        file_controls = ttk.LabelFrame(self.controls_frame, text="File Controls", padding=(10, 5))
        file_controls.pack(fill=tk.X, pady=5)
        self.load_button = ttk.Button(file_controls, text="Load FITS File", command=self.load_file)
        self.load_button.pack(fill=tk.X, pady=2)

        # Detection Controls
        detection_controls = ttk.LabelFrame(self.controls_frame, text="Detection", padding=(10, 5))
        detection_controls.pack(fill=tk.X, pady=5)
        self.detect_button = ttk.Button(
            detection_controls, text="Detect Lines Automatically",
            command=self.detect_lines, state=tk.DISABLED
        )
        self.detect_button.pack(fill=tk.X, pady=2)
        self.manual_line_button = ttk.Button(
            detection_controls, text="Select Points Manually",
            command=self.manual_line, state=tk.DISABLED
        )
        self.manual_line_button.pack(fill=tk.X, pady=2)

        # Profiling Controls
        profiling_controls = ttk.LabelFrame(self.controls_frame, text="Profiling", padding=(10, 5))
        profiling_controls.pack(fill=tk.X, pady=5)
        self.analyze_button = ttk.Button(
            profiling_controls, text="Reanalyze Perpendicular Lines",
            command=self.reanalyze_lines, state=tk.DISABLED
        )
        self.analyze_button.pack(fill=tk.X, pady=2)
        self.plot_main_line_button = ttk.Button(
            profiling_controls, text="Plot Main Line",
            command=self.plot_main_line, state=tk.DISABLED
        )
        self.plot_main_line_button.pack(fill=tk.X, pady=2)
        self.plot_all_lines_button = ttk.Button(
            profiling_controls, text="Plot All Perpendicular Lines",
            command=self.plot_all_lines, state=tk.DISABLED
        )
        self.plot_all_lines_button.pack(fill=tk.X, pady=2)
        
        # Additional Controls
        extra_controls = ttk.LabelFrame(self.controls_frame, text="Additional Controls", padding=(10, 5))
        extra_controls.pack(fill=tk.X, pady=5)
        self.index_entry = ttk.Entry(extra_controls)
        self.index_entry.pack(fill=tk.X, pady=2)
        self.index_entry.insert(0, "Enter Profile Index")

        self.plot_profile_button = ttk.Button(
            extra_controls, text="Plot Profile at Index", command=self.plot_profile, state=tk.DISABLED
        )
        self.plot_profile_button.pack(fill=tk.X, pady=2)
        self.remove_profile_button = ttk.Button(
            extra_controls, text="Remove Profile at Index", command=self.remove_profile, state=tk.DISABLED
        )
        self.remove_profile_button.pack(fill=tk.X, pady=2)
        self.plot_median_profile_button = ttk.Button(
            extra_controls, text="Plot Median Profile", command=self.plot_median_profile, state=tk.DISABLED
        )
        self.plot_median_profile_button.pack(fill=tk.X, pady=2)
        self.analyze_profile_button = ttk.Button(
            extra_controls, text="Analyze Median Profile", command=self.analyze_profile, state=tk.DISABLED
        )
        self.analyze_profile_button.pack(fill=tk.X, pady=2)
        self.plot_all_profiles_button = ttk.Button(
            extra_controls, text="Plot All Profiles", command=self.plot_all_profiles, state=tk.DISABLED
        )
        self.plot_all_profiles_button.pack(fill=tk.X, pady=2)
        
        # Checkbuttons for toggling overlays
        self.show_line_var = tk.BooleanVar(value=True)
        self.show_line_check = ttk.Checkbutton(
            extra_controls, text="Show line", variable=self.show_line_var,
            command=self.update_main_line_plot
        )
        self.show_line_check.pack(fill=tk.X, pady=2)
        
        self.show_perp_var = tk.BooleanVar(value=False)
        self.show_perp_check = ttk.Checkbutton(
            extra_controls, text="Show perpendicular lines", variable=self.show_perp_var,
            command=self.update_main_line_plot
        )
        self.show_perp_check.pack(fill=tk.X, pady=2)

    def disable_profiling_controls(self):
        """
        Disable all buttons outside of the detection block.
        """
        self.analyze_button.config(state=tk.DISABLED)
        self.plot_main_line_button.config(state=tk.DISABLED)
        self.plot_all_lines_button.config(state=tk.DISABLED)
        self.plot_profile_button.config(state=tk.DISABLED)
        self.remove_profile_button.config(state=tk.DISABLED)
        self.plot_median_profile_button.config(state=tk.DISABLED)
        self.analyze_profile_button.config(state=tk.DISABLED)
        self.plot_all_profiles_button.config(state=tk.DISABLED)
        self.show_line_var.set(False)
        self.show_perp_var.set(False)

    def load_file(self):
        """
        Load a FITS file, cache the image and its normalized version, and display it.
        Also, reset any existing profile information and disable all profiling controls.
        """
        self.fits_file = filedialog.askopenfilename(filetypes=[("FITS files", "*.fits")])
        if self.fits_file:
            print(f"Loaded file: {self.fits_file}")
            with fits.open(self.fits_file) as hdul:
                self.image_data = hdul[0].data
            self.norm = ImageNormalize(self.image_data, interval=PercentileInterval(1, 99), stretch=SqrtStretch())
            self.profiler = None
            self.detect_button.config(state=tk.NORMAL)
            self.manual_line_button.config(state=tk.NORMAL)
            self.disable_profiling_controls()
            self.display_fits_image()

    def display_fits_image(self):
        """
        Display the cached FITS image with its normalization in the canvas frame with a border.
        """
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.imshow(self.image_data, cmap="gray", origin="lower", norm=self.norm)
        ax.set_title("Loaded FITS Image")
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")

        border_frame = ttk.Frame(self.canvas_frame, borderwidth=2, relief="solid")
        border_frame.pack(fill=tk.BOTH, expand=True)
        canvas = FigureCanvasTkAgg(fig, master=border_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_main_line_plot(self):
        """
        Update the main canvas to show the cached FITS image with overlays based on the checkbutton settings.
        Overlays include the selected line (in red) and, if available, the perpendicular lines (in dashed cyan).
        """
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.imshow(self.image_data, cmap="gray", origin="lower", norm=self.norm)
        
        if self.profiler and self.show_line_var.get():
            x1, y1 = self.profiler.point1
            x2, y2 = self.profiler.point2
            ax.plot([x1, x2], [y1, y2], color="red", linestyle="-", linewidth=3, label="Selected Line")
        
        if self.profiler and self.show_perp_var.get() and self.profiler.line_coordinates:
            for coords in self.profiler.line_coordinates:
                start, end = coords
                ax.plot([start[0], end[0]], [start[1], end[1]], color="cyan", linestyle="--", linewidth=1)
        
        if self.profiler and (self.show_line_var.get() or self.show_perp_var.get()):
            ax.legend()
        ax.set_title("FITS Image with Overlays")
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")

        border_frame = ttk.Frame(self.canvas_frame, borderwidth=2, relief="solid")
        border_frame.pack(fill=tk.BOTH, expand=True)
        canvas = FigureCanvasTkAgg(fig, master=border_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def detect_lines(self):
        """
        Use the TrailDetector class to auto-detect trail lines.
        For each detected line, show a popup with its coordinates and options.
        A "View Plot" button is provided to see a detailed plot of the line.
        """
        if not self.fits_file:
            return
        detected = self.detector.detect_trails(self.fits_file)
        if not self.detector.merged_lines:
            messagebox.showinfo("No Lines Detected", "No trail lines were detected in the image.")
            return

        popup = tk.Toplevel(self.root)
        popup.title("Select a Line for Profiling")
        label = ttk.Label(popup, text="Detected Lines:")
        label.pack(pady=5)

        for i, line in enumerate(self.detector.merged_lines):
            frame = ttk.Frame(popup, relief="solid", padding=5)
            frame.pack(fill="x", padx=5, pady=5)

            x1, y1, x2, y2 = line
            coords_text = f"Line {i+1}: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})"
            coords_label = ttk.Label(frame, text=coords_text)
            coords_label.pack(anchor="w")

            extend_var = tk.BooleanVar()
            extend_check = ttk.Checkbutton(frame, text="Extend line", variable=extend_var)
            extend_check.pack(anchor="w", pady=2)

            def view_plot(selected_line=line, extend_flag=extend_var):
                view_popup = tk.Toplevel(self.root)
                view_popup.title("Detailed Line View")
                fig = plt.Figure(figsize=(6, 6))
                ax = fig.add_subplot(111)
                ax.imshow(self.image_data, cmap="gray", origin="lower", norm=self.norm)

                if extend_flag.get():
                    # Use extend_line_to_image_edges if we have a profiler; otherwise create a temp profiler
                    if self.profiler:
                        extended_line = self.profiler.extend_line_to_image_edges(
                            selected_line[0], selected_line[1],
                            selected_line[2], selected_line[3]
                        )
                    else:
                        temp_prof = TrailProfiler(
                            fits_file=self.fits_file,
                            point1=(selected_line[0], selected_line[1]),
                            point2=(selected_line[2], selected_line[3])
                        )
                        extended_line = temp_prof.extend_line_to_image_edges(
                            selected_line[0], selected_line[1],
                            selected_line[2], selected_line[3]
                        )
                    ax.plot(
                        [extended_line[0], extended_line[2]],
                        [extended_line[1], extended_line[3]],
                        color="red", linestyle="-", linewidth=3, label="Extended Line"
                    )
                else:
                    ax.plot(
                        [selected_line[0], selected_line[2]],
                        [selected_line[1], selected_line[3]],
                        color="red", linestyle="-", linewidth=3, label="Selected Line"
                    )
                ax.legend()
                ax.set_title("Detailed Line View")
                ax.set_xlabel("X pixel")
                ax.set_ylabel("Y pixel")

                border_frame = ttk.Frame(view_popup, borderwidth=2, relief="solid")
                border_frame.pack(fill=tk.BOTH, expand=True)
                canvas = FigureCanvasTkAgg(fig, master=border_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            view_button = ttk.Button(frame, text="View Plot", command=view_plot)
            view_button.pack(pady=2)

            def use_line(selected_line=line, extend_flag=extend_var):
                if extend_flag.get():
                    if not self.profiler:
                        temp_prof = TrailProfiler(
                            fits_file=self.fits_file, 
                            point1=(selected_line[0], selected_line[1]),
                            point2=(selected_line[2], selected_line[3])
                        )
                        chosen_line = temp_prof.extend_line_to_image_edges(
                            selected_line[0], selected_line[1],
                            selected_line[2], selected_line[3]
                        )
                    else:
                        chosen_line = self.profiler.extend_line_to_image_edges(
                            selected_line[0], selected_line[1],
                            selected_line[2], selected_line[3]
                        )
                else:
                    chosen_line = selected_line

                self.profiler = TrailProfiler(
                    fits_file=self.fits_file,
                    point1=(chosen_line[0], chosen_line[1]),
                    point2=(chosen_line[2], chosen_line[3])
                )

                # Enable profiling controls
                self.analyze_button.config(state=tk.NORMAL)
                self.plot_main_line_button.config(state=tk.NORMAL)
                self.plot_all_lines_button.config(state=tk.NORMAL)
                self.plot_profile_button.config(state=tk.NORMAL)
                self.remove_profile_button.config(state=tk.NORMAL)
                self.plot_median_profile_button.config(state=tk.NORMAL)
                self.analyze_profile_button.config(state=tk.NORMAL)
                self.plot_all_profiles_button.config(state=tk.NORMAL)

                self.update_main_line_plot()
                popup.destroy()

            use_button = ttk.Button(frame, text="Use this line for profiling", command=use_line)
            use_button.pack(pady=2)
    
    def manual_line(self):
        """
        Prompt the user for line endpoint coordinates and create a TrailProfiler.
        """
        if not self.fits_file:
            return
        coords_str = simpledialog.askstring("Enter Coordinates", "Enter the coordinates (x1 y1 x2 y2):")
        if not coords_str:
            return
        coords = coords_str.split()
        if len(coords) != 4:
            messagebox.showerror("Invalid Input", "Please provide exactly four values.")
            return
        try:
            x1, y1, x2, y2 = map(float, coords)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please provide valid numeric values.")
            return

        self.profiler = TrailProfiler(
            fits_file=self.fits_file,
            point1=(x1, y1),
            point2=(x2, y2)
        )
        print("Trail analysis complete.")
        self.analyze_button.config(state=tk.NORMAL)
        self.plot_main_line_button.config(state=tk.NORMAL)
        self.plot_all_lines_button.config(state=tk.NORMAL)
        self.plot_profile_button.config(state=tk.NORMAL)
        self.remove_profile_button.config(state=tk.NORMAL)
        self.plot_median_profile_button.config(state=tk.NORMAL)
        self.analyze_profile_button.config(state=tk.NORMAL)
        self.plot_all_profiles_button.config(state=tk.NORMAL)
        self.update_main_line_plot()

    def reanalyze_lines(self):
        """
        Re-sample the perpendicular lines by prompting the user for parameters,
        then calling the new _sample_perpendicular_profiles(...) method.
        """
        if self.profiler:
            num_lines = simpledialog.askinteger("Reanalyze", "Enter the number of perpendicular lines:", initialvalue=10)
            short_window = simpledialog.askinteger("Reanalyze", "Enter the half-line length:", initialvalue=100)
            step_size = simpledialog.askfloat("Reanalyze", "Enter the step size:", initialvalue=0.1)
            self.profiler.brightness_profiles.clear()
            self.profiler.line_coordinates.clear()
            self.profiler._sample_perpendicular_profiles(
                num_perp_lines=num_lines,
                half_line_length=short_window,
                sampling_step=step_size
            )
            print("Reanalysis complete.")
            self.update_main_line_plot()

    def plot_main_line(self):
        """
        Plot the main line from the profiler's data in a separate matplotlib window.
        """
        if self.profiler:
            self.profiler.plot_main_line()

    def plot_all_lines(self):
        """
        Plot all perpendicular lines from the profiler's data in a separate matplotlib window.
        """
        if self.profiler:
            self.profiler.plot_all_perp_lines()

    def plot_profile(self):
        """
        Plot a single profile at the given index (1-based) in a new window.
        """
        if self.profiler:
            try:
                index = int(self.index_entry.get())
                self.profiler.plot_profile_by_index(index)
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number.")

    def remove_profile(self):
        """
        Remove the brightness profile at the specified index and update the main canvas.
        """
        if self.profiler:
            try:
                index = int(self.index_entry.get())
                self.profiler.remove_profile_by_index(index)
                print(f"Removed profile at index {index}.")
                self.update_main_line_plot()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number.")
            except IndexError as e:
                messagebox.showerror("Index Error", str(e))

    def plot_median_profile(self):
        """
        Plot the median profile of all currently stored brightness profiles.
        """
        if self.profiler:
            self.profiler.plot_median_profile()

    def analyze_profile(self):
        """
        Analyze the median brightness profile by computing:
        - AUC using the profile's max,
        - AUC using the global image max,
        - FWHM with half max factors of 0.5 and 0.95.

        Displays the results in a popup.
        """
        if not self.profiler or not self.profiler.brightness_profiles:
            messagebox.showerror("Error", "No brightness profiles available for analysis.")
            return
        
        # Retrieve the median profile from the new class
        median_profile = self.profiler.calculate_median_profile()

        # Calculate metrics
        auc_local = self.profiler.calculate_auc(median_profile)
        auc_global = self.profiler.calculate_auc_global_max(median_profile)
        fwhm_05 = self.profiler.calculate_fwhm(median_profile, half_max_factor=0.5)
        fwhm_095 = self.profiler.calculate_fwhm(median_profile, half_max_factor=0.95)

        message = (
            f"AUC (using profile max): {auc_local:.2f}\n"
            f"AUC (using image global max): {auc_global:.2f}\n"
            f"FWHM (0.5 factor): {fwhm_05:.2f}\n"
            f"FWHM (0.95 factor): {fwhm_095:.2f}"
        )
        messagebox.showinfo("Profile Analysis", message)

    def plot_all_profiles(self):
        """
        Plot all perpendicular brightness profiles (already stored in the profiler)
        in one matplotlib window.
        """
        if not self.profiler or not self.profiler.brightness_profiles:
            messagebox.showerror("Error", "No brightness profiles available for plotting.")
            return

        fig = plt.Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        for i, profile in enumerate(self.profiler.brightness_profiles):
            ax.plot(profile, label=f'Profile {i+1}')
        if len(self.profiler.brightness_profiles[0]) > 0:
            center = len(self.profiler.brightness_profiles[0]) // 2
            ax.axvline(center, color='red', linestyle='--', label='Trail Center')
        ax.set_title("All Perpendicular Brightness Profiles")
        ax.set_xlabel("Position along perpendicular line")
        ax.set_ylabel("Normalized Brightness")
        ax.legend()

        plot_window = tk.Toplevel(self.root)
        plot_window.title("All Perpendicular Profiles")
        border_frame = ttk.Frame(plot_window, borderwidth=2, relief="solid")
        border_frame.pack(fill=tk.BOTH, expand=True)
        canvas = FigureCanvasTkAgg(fig, master=border_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrailProfilerGUI(root)
    root.mainloop()