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

def optimize_gamma_for_column(image_array, col_idx, num_neighbors = 3, gamma_range=(0.5, 2.0), steps=20):
    """Finds the best gamma value for a column to minimize adjacent pixel differences."""
    best_gamma = 1.0
    min_diff = np.inf
    column = image_array[:, col_idx].copy()  # Copy to avoid modifying original
    
    left_idx = max(0, col_idx - num_neighbors)
    right_idx = min(image_array.shape[1] - 1, col_idx + num_neighbors)

    # Compute the reference neighboring column as the mean of selected neighbors
    if left_idx == right_idx:
        neighbor_col = image_array[:, left_idx]  # Edge case: only one column available
    else:
        neighbor_col = np.mean(image_array[:, left_idx:right_idx + 1], axis=1)

    for gamma in np.linspace(*gamma_range, steps):
        corrected_col = gamma_correction(column, gamma)

        # Compute absolute differences with horizontally adjacent pixels
        diff = np.abs(corrected_col - neighbor_col).sum()

        if diff < min_diff:
            min_diff = diff
            best_gamma = gamma
    
    return gamma_correction(column, best_gamma)

def remove_column_noise_gamma(img, num_neighbors =3,  gamma_range=(0.5, 2.0), steps=20):
    # Load image and convert to grayscale
    img_array = np.array(img, dtype=np.float32)
    
    # Process each column independently
    for col_idx in range(img_array.shape[1]):
        img_array[:, col_idx] = optimize_gamma_for_column(img_array, col_idx, num_neighbors, gamma_range, steps)
    
    # Convert back to image and save
    denoised_image = Image.fromarray(img_array.astype(np.uint8))
    
    return denoised_image