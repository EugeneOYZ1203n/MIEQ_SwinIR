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

