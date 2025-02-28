# Meteor Trail Detection and Analysis

This repository contains a pipeline for detecting, profiling, and classifying meteor trails in astronomical FITS images. It includes detection algorithms, profiling tools, a graphical user interface (GUI), and a trained neural network for classification.

## Repository Structure

### **Main Scripts**
- **`compute_metrics.py`** - Computes various metrics related to detected trails, including FWHM, AUC, and kurtosis.
- **`result_analysis.py`** - Analyzes the results of computed metrics and classification predictions.
- **`test_predictions.py`** - Validates the performance of the classification model using test datasets.

### **Detection & Profiling**
- **`trail_detector.py`** - Implements the **TrailDetector** class, which detects linear artifacts in FITS images using edge detection and clustering.
- **`trail_profiler.py`** - Implements the **TrailProfiler** class, which extracts brightness profiles from detected trails and computes relevant metrics.
- **`trail_classifier.py`** - Trains and tests a simple classification model and prints the results.
- **`trail_classifier_advanced.py`** - Uses a neural network model to run classification on a folder containing fits files.
- **`trail_classifier_generator.py`** - Handles the training of the neural network with the complete training dataset.

### **Data & Model Files**
- **`full_meteor_model.pkl`** - Pickled neural network model trained to classify meteor trails.
- **`metrics_results.txt`** - Stores computed metrics for detected trails.
- **`prediction_results.txt`** - Contains predictions made by the classification model.
- **`meteors.txt`** - A list of trails manually identified as meteors.
- **`not_meteors.txt`** - A list of trails manually identified as non-meteors.

### **Graphical User Interface (GUI)**
- **`gui.py`** - Provides a GUI for visualizing detected trails, profiling data, and interacting with the pipeline.

### **Miscellaneous**
- **`README.md`** - This file, describing the contents of the repository.
- **`LICENSE`** - Specifies the licensing terms for using and modifying this code.
- **`.gitignore`** - Defines files and directories to be ignored by Git.
- **`testing/`** - A directory containing several fits files for testing.

## Usage

1. **Detect trails in a FITS image** using `trail_detector.py`.
2. **Profile detected trails** using `trail_profiler.py`.
3. **Classify detected trails** using `trail_predictor.py` or `trail_predictor_advanced.py`.
4. **Compute and analyze trail metrics** using `compute_metrics.py` and `result_analysis.py`.
5. **Use the GUI** (`gui.py`) for interactive visualization.

## Dependencies

- Python 3.x
- `numpy`, `opencv-python`, `matplotlib`, `astropy`, `scipy`, `sklearn`

## License

Refer to `LICENSE` for licensing details.