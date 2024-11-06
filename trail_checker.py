import os
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, MinMaxInterval, PercentileInterval, SqrtStretch, LogStretch, AsinhStretch

# Set up argument parsing for the script
parser = argparse.ArgumentParser(description="Detect trails in FITS images within a specified directory.")
parser.add_argument('fits_dir_path', type=str, help="Path to the directory containing FITS files")
parser.add_argument('output_dir', type=str, help="Path to the directory where trail PNGs will be saved")
args = parser.parse_args()

#canny_threshold1 = 7
#canny_threshold2 = 20
#hough_rho = 1
#hough_theta = np.pi / 180
#hough_threshold = 250
#hough_minLineLength = 400
#hough_maxLineGap = 100

canny_threshold1 = 6
canny_threshold2 = 18
hough_rho = 1
hough_theta = np.pi / 180
hough_threshold = 250
hough_minLineLength = 500
hough_maxLineGap = 30

# Initialize counters
trail_count = 0
no_trail_count = 0

# Ensure the output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Iterate through all files in the specified directory
for fits_file in os.listdir(args.fits_dir_path):
    if fits_file.endswith('.fits'):
        fits_file_path = os.path.join(args.fits_dir_path, fits_file)
        
        # Open the FITS file and load the image data
        with fits.open(fits_file_path) as hdul:
            image_data = hdul[0].data
        
        # Use ZScaleInterval for normalization
        norm = ImageNormalize(image_data, interval=PercentileInterval(1, 99), stretch=SqrtStretch())
        normalized_image = norm(image_data)
        
        #norm = ImageNormalize(image_data, interval=ZScaleInterval())
        #normalized_image = norm(image_data)
        
        #norm = ImageNormalize(normalized_image, interval=MinMaxInterval())
        #normalized_image = norm(normalized_image)

        # Convert normalized image to 8-bit grayscale
        scaled_image = np.uint8(255 * (normalized_image - np.min(normalized_image)) / 
                                (np.max(normalized_image) - np.min(normalized_image)))

        # Apply Gaussian blur to reduce noise
        #scaled_image = cv2.GaussianBlur(scaled_image, (1, 1), 0)
        
        # Use Canny edge detection
        edges = cv2.Canny(scaled_image, threshold1=canny_threshold1, threshold2=canny_threshold2)
        
        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, rho=hough_rho, theta=hough_theta, threshold=hough_threshold, 
                                minLineLength=hough_minLineLength, maxLineGap=hough_maxLineGap)
            
            # Plot the original image and overlay detected trails
        plt.figure(figsize=(10, 10))
        plt.imshow(image_data, cmap='gray', origin='lower', norm=norm)
        if lines is not None:
            trail_count += 1
            for line in lines:
                x1, y1, x2, y2 = line[0]
                plt.plot([x1, x2], [y1, y2], color='red', linewidth=1)
        
        plt.title("Detected Trail")
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.colorbar()
        
        # Save the plot as a PNG in the output directory
        output_png_path = os.path.join(args.output_dir, f"{os.path.splitext(fits_file)[0]}_edited.png")

        plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
        plt.close()

            #plt.figure(figsize=(10, 10))
            #plt.imshow(image_data, cmap='gray', origin='lower', norm=norm)
            #plt.title("Detected Trail")
            #plt.xlabel("X pixel")
            #plt.ylabel("Y pixel")
            #plt.colorbar()
            #output_original_path = os.path.join(args.output_dir, f"{os.path.splitext(fits_file)[0]}_original.png")
            #plt.savefig(output_original_path, dpi=300, bbox_inches='tight')
            #plt.close()
        if lines is not None:
            print(f"{fits_file} n. = {len(lines)}")

# Print the summary of the results
print(f"Trail Detection Summary:")
print(f" - Images with trails detected: {trail_count}")
print(f" - Images without trails detected: {no_trail_count}")