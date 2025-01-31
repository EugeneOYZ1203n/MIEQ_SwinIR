import cv2
import numpy as np
from PIL import Image

def detect_rotation_angle(img, edge_thresholds=(50, 150), hough_threshold=100):
    """Detects the rotation angle of an image based on its black edge using Hough Transform."""
    # Load image in grayscale
    image = np.array(img)
    
    # Apply edge detection
    edges = cv2.Canny(image, edge_thresholds[0], edge_thresholds[1])

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)

    if lines is None:
        print("No prominent edges detected.")
        return 0  # No rotation detected

    # Extract angles from detected lines
    angles = []
    for rho, theta in lines[:, 0]:  # Hough transform returns rho, theta
        angle = np.degrees(theta) - 90  # Convert to a more intuitive angle
        if -45 < angle < 45:  # Filter extreme angles
            angles.append(angle)

    if not angles:
        return 0  # No valid angle found

    # Compute the most frequent angle
    rotation_angle = np.median(angles)  # Using median for robustness
    return rotation_angle

def rotate_image(img):
    """Rotates an image based on the detected rotation angle and saves the corrected image."""
    angle = detect_rotation_angle(img)
    print(f"Detected Rotation Angle: {angle:.2f} degrees")
    
    # Rotate and save
    rotated_image = img.rotate(-angle, expand=True, fillcolor=(0, 0, 0))
    return rotated_image