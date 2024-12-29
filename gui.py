import tkinter as tk
from tkinter import filedialog, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from trail_profiler_2 import TrailProfiler  # Import your updated TrailProfiler class


class TrailProfilerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trail Profiler GUI")
        self.profiler = None

        # Buttons and input fields
        self.load_button = tk.Button(root, text="Load FITS File", command=self.load_file)
        self.load_button.pack(pady=5)

        self.analyze_button = tk.Button(root, text="Reanalyze Perpendicular Lines", command=self.reanalyze_lines, state=tk.DISABLED)
        self.analyze_button.pack(pady=5)

        self.plot_main_line_button = tk.Button(root, text="Plot Main Line", command=self.plot_main_line, state=tk.DISABLED)
        self.plot_main_line_button.pack(pady=5)

        self.plot_all_lines_button = tk.Button(root, text="Plot All Perpendicular Lines", command=self.plot_all_lines, state=tk.DISABLED)
        self.plot_all_lines_button.pack(pady=5)

        self.index_entry = tk.Entry(root)
        self.index_entry.pack(pady=5)
        self.index_entry.insert(0, "Enter Profile Index")

        self.plot_profile_button = tk.Button(root, text="Plot Profile at Index", command=self.plot_profile, state=tk.DISABLED)
        self.plot_profile_button.pack(pady=5)

        self.remove_profile_button = tk.Button(root, text="Remove Profile at Index", command=self.remove_profile, state=tk.DISABLED)
        self.remove_profile_button.pack(pady=5)

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("FITS files", "*.fits")])
        if filepath:
            # Ask user for the trail points
            point1_x = simpledialog.askfloat("Trail Point 1", "Enter X coordinate of the first point:")
            point1_y = simpledialog.askfloat("Trail Point 1", "Enter Y coordinate of the first point:")
            point2_x = simpledialog.askfloat("Trail Point 2", "Enter X coordinate of the second point:")
            point2_y = simpledialog.askfloat("Trail Point 2", "Enter Y coordinate of the second point:")

            if None in [point1_x, point1_y, point2_x, point2_y]:
                print("Trail points input was canceled.")
                return

            self.profiler = TrailProfiler(
                fits_file=filepath,
                point1=(point1_x, point1_y),
                point2=(point2_x, point2_y),
            )
            print(f"Loaded file: {filepath}")
            print("Trail analysis complete.")

            # Enable buttons
            self.analyze_button.config(state=tk.NORMAL)
            self.plot_main_line_button.config(state=tk.NORMAL)
            self.plot_all_lines_button.config(state=tk.NORMAL)
            self.plot_profile_button.config(state=tk.NORMAL)
            self.remove_profile_button.config(state=tk.NORMAL)

    def reanalyze_lines(self):
        if self.profiler:
            num_lines = simpledialog.askinteger("Reanalyze", "Enter the number of perpendicular lines:", initialvalue=10)
            short_window = simpledialog.askinteger("Reanalyze", "Enter the short window size:", initialvalue=100)
            step_size = simpledialog.askfloat("Reanalyze", "Enter the step size:", initialvalue=0.1)

            self.profiler._analyze_perpendicular_lines(
                num_perpendicular_lines=num_lines,
                short_window_size=short_window,
                step_size=step_size,
            )
            print("Reanalysis complete.")

    def plot_main_line(self):
        if self.profiler:
            self.show_plot(self.profiler.plot_main_line)

    def plot_all_lines(self):
        if self.profiler:
            self.show_plot(self.profiler.plot_all_perpendicular_lines)

    def plot_profile(self):
        if self.profiler:
            try:
                index = int(self.index_entry.get())
                self.show_plot(lambda: self.profiler.plot_profile_at_index(index))
            except ValueError:
                print("Invalid index. Please enter a valid number.")

    def remove_profile(self):
        if self.profiler:
            try:
                index = int(self.index_entry.get())
                self.profiler.remove_profile(index)
                print(f"Removed profile at index {index}.")
            except ValueError:
                print("Invalid index. Please enter a valid number.")
            except IndexError as e:
                print(e)

    def show_plot(self, plot_func):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig = plt.figure(figsize=(5, 4))
        plot_func()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrailProfilerGUI(root)
    root.mainloop()