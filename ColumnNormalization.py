from PIL import Image, ImageOps
import numpy as np

def columnwise_normalization(img, window):
    img_array = np.array(img)
    
    col_means = np.mean(img_array, axis = 0)

    smoothed_means = np.convolve(col_means, np.ones(window)/window, mode='same')

    normalized_img = img_array - smoothed_means

    normalized_img = normalized_img - normalized_img.min()
    normalized_img = (normalized_img / normalized_img.max()) * 255
    normalized_img = Image.fromarray(normalized_img.astype(np.uint8))

    return normalized_img

def gamma_correction(image_array, gamma):
    """Applies gamma correction to an image array with a given gamma value."""
    max_val = np.max(image_array)
    normalized = image_array / max_val  # Normalize to [0,1] range
    corrected = np.power(normalized, gamma) * max_val  # Apply gamma and rescale
    return np.clip(corrected, 0, 255).astype(np.uint8)

def optimize_gamma_for_column(col, gamma_range=(0.5, 2.0), steps=20):
    """Finds the best gamma value for a column to minimize adjacent pixel differences."""
    best_gamma = 1.0  # Default gamma
    min_diff = np.inf
    
    for gamma in np.linspace(*gamma_range, steps):
        corrected_col = gamma_correction(col, gamma)
        diff = np.abs(np.diff(corrected_col)).sum()  # Sum of differences between adjacent pixels
        
        if diff < min_diff:
            min_diff = diff
            best_gamma = gamma
    
    return gamma_correction(col, best_gamma)

def remove_column_noise_gamma(img, gamma_range=(0.5, 2.0), steps=20):
    # Load image and convert to grayscale
    img_array = np.array(img, dtype=np.float32)
    
    # Process each column independently
    for col_idx in range(img_array.shape[1]):
        img_array[:, col_idx] = optimize_gamma_for_column(img_array[:, col_idx], gamma_range, steps)
    
    # Convert back to image and save
    denoised_image = Image.fromarray(img_array.astype(np.uint8))
    
    return denoised_image