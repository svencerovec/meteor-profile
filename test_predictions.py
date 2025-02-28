"""
test_predictions.py

This script verifies how well the neural network's predictions align with the test dataset.
It checks:
- If trails from `meteors.txt` and `not_meteors.txt` were detected.
- If detected trails were classified correctly.

A correct classification is:
- `model_score > 0.5` for meteors
- `model_score <= 0.5` for non-meteors

At the end, it prints:
- Correctly predicted meteors
- Correctly predicted non-meteors
- Missed meteors
- Missed non-meteors

Usage:
    python check_test_predictions.py
"""

import os

METEOR_FILE = "meteors.txt"
NON_METEOR_FILE = "not_meteors.txt"
PREDICTION_FILE = "prediction_results.txt"

def load_ground_truth(file_path, label):
    """
    Load ground truth data from the given file.

    Args:
        file_path (str): Path to the file (meteors.txt or not_meteors.txt)
        label (int): 1 for meteors, 0 for non-meteors

    Returns:
        dict: Mapping from filename to list of coordinates and label
    """
    ground_truth = {}
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return ground_truth

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[WARNING] Skipping malformed line: {line.strip()}")
                continue
            
            filename = parts[0]
            coords = tuple(map(float, parts[1:5]))
            ground_truth[filename] = (coords, label)

    return ground_truth

def load_predictions(file_path):
    """
    Load predictions from prediction_results.txt.

    Returns:
        dict: Mapping from filename to (coords, model_score)
    """
    predictions = {}
    if not os.path.exists(file_path):
        print(f"[ERROR] Prediction file not found: {file_path}")
        return predictions

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 6:
                print(f"[WARNING] Skipping malformed prediction line: {line.strip()}")
                continue
            
            filename = parts[0]
            coords = tuple(map(float, parts[1:5]))
            try:
                model_score = float(parts[5])
            except ValueError:
                print(f"[WARNING] Invalid model score in line: {line.strip()}")
                continue

            predictions[filename] = (coords, model_score)

    return predictions

def evaluate_predictions(meteors, non_meteors, predictions):
    """
    Compare ground truth with predictions and count correct classifications.

    Args:
        meteors (dict): Ground truth meteors {filename: (coords, 1)}
        non_meteors (dict): Ground truth non-meteors {filename: (coords, 0)}
        predictions (dict): Model predictions {filename: (coords, model_score)}
    """
    correct_meteor_predictions = 0
    correct_non_meteor_predictions = 0
    missed_meteors = 0
    missed_non_meteors = 0

    for filename, (coords, label) in meteors.items():
        if filename in predictions:
            _, model_score = predictions[filename]
            if model_score > 0.5:
                correct_meteor_predictions += 1
        else:
            missed_meteors += 1

    for filename, (coords, label) in non_meteors.items():
        if filename in predictions:
            _, model_score = predictions[filename]
            if model_score <= 0.5:
                correct_non_meteor_predictions += 1
        else:
            missed_non_meteors += 1

    print("\n=== Prediction Analysis ===")
    print(f"Correctly predicted test meteors: {correct_meteor_predictions} / {len(meteors)}")
    print(f"Correctly predicted test non-meteors: {correct_non_meteor_predictions} / {len(non_meteors)}")
    print(f"Missed test meteors: {missed_meteors}")
    print(f"Missed test non-meteors: {missed_non_meteors}")
    print("===========================")

def main():
    meteors = load_ground_truth(METEOR_FILE, label=1)
    non_meteors = load_ground_truth(NON_METEOR_FILE, label=0)
    predictions = load_predictions(PREDICTION_FILE)

    if not meteors and not non_meteors:
        print("[ERROR] No test data found. Ensure meteors.txt and not_meteors.txt exist and are correctly formatted.")
        return

    if not predictions:
        print("[ERROR] No predictions found. Ensure prediction_results.txt exists and is correctly formatted.")
        return

    evaluate_predictions(meteors, non_meteors, predictions)

if __name__ == "__main__":
    main()