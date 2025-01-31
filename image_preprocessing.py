from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

def denoise_gaussianBlur(radius=2):
    def processing(img):
        img_denoised = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img_denoised
    return processing

def denoise_fastN1MeansDenoising(templateWindowSize=7, searchWindowSize=21):
    def processing(img):
        img_np = np.array(img)
        denoised = cv2.fastNlMeansDenoising(img_np, None, h=10, 
            templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
        output = Image.fromarray(denoised)
        return output
    return processing

def denoise_medianBlur(kernel_size=3):
    def processing(img):
        img_np = np.array(img)
        median_filtered = cv2.medianBlur(img_np, kernel_size)
        output = Image.fromarray(median_filtered)
        return output
    return processing

def denoise_bilateralFilter():
    def processing(img):
        img_np = np.array(img)
        smoothed = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
        output = Image.fromarray(smoothed)
        return output
    return processing

def edge_unsharpMask(radius=2, percent=150, threshold=3):
    def processing(img):
        return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    return processing

def edge_grayscale_erosion(kernel_size=3, iterations=1, custom_kernel_shape=None):
    def processing(img):
        img_np = np.array(img)
        
        kernel_shape = custom_kernel_shape
        if kernel_shape == None:
            kernel_shape = cv2.MORPH_RECT

        kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
        eroded_image = cv2.erode(img_np, kernel, iterations=iterations)
        output = Image.fromarray(eroded_image)
        return output
    return processing

def edge_mask_laplacian():
    def processing(img):
        img_np = np.array(img)
        laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        output = Image.fromarray(laplacian)
        return output
    return processing

def edge_mask_binary_erosion(kernel_size=5, iterations=1, custom_kernel_shape=None):
    def processing(img):
        img_np = np.array(img)
        
        _, binary_image = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)

        kernel_shape = custom_kernel_shape
        if kernel_shape == None:
            kernel_shape = cv2.MORPH_RECT

        kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
        eroded_image = cv2.erode(binary_image, kernel, iterations=iterations)
        enhanced_edges = cv2.subtract(binary_image, eroded_image)
        enhanced_edges = cv2.convertScaleAbs(enhanced_edges)
        output = Image.fromarray(enhanced_edges)
        return output
    return processing

def edge_mask_sobel_filter(kernel_size=3):
    def processing(img):
        img_np = np.array(img)

        sobel_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=kernel_size)  
        sobel_x = cv2.convertScaleAbs(sobel_x)

        sobel_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=kernel_size)
        sobel_y = cv2.convertScaleAbs(sobel_y)

        combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        output = Image.fromarray(combined)
        return output
    return processing

def edge_mask_scharr_filter():
    def processing(img):
        img_np = np.array(img)

        scharr_x = cv2.Scharr(img_np, cv2.CV_64F, 1, 0)
        scharr_x = cv2.convertScaleAbs(scharr_x)

        scharr_y = cv2.Scharr(img_np, cv2.CV_64F, 0, 1)
        scharr_y = cv2.convertScaleAbs(scharr_y)

        combined = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)
        output = Image.fromarray(combined)
        return output
    return processing

def contrast_clahe(clipLimit=2.0, tileGridSize=8):
    def processing(img):
        img_np = np.array(img)
        
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize,tileGridSize))
        clahe_img = clahe.apply(img_np)
        
        output = Image.fromarray(clahe_img)

        return output
    return processing

def contrast_gamma_correction(gamma = 0.8):
    def processing(img):
        lut = [int((i / 255.0) ** gamma * 255) for i in range(256)]
        img_out = img.point(lut)
        return img_out
    return processing

def mask_combine(mask):
    def processing(img_mask, img_unmask):
        mask_np = np.array(mask)
        img_mask_np = np.array(img_mask)
        img_unmask_np = np.array(img_unmask)
        masked_edges = cv2.bitwise_and(img_mask_np, mask_np)
        unmasked_edges = cv2.bitwise_and(img_unmask_np, ~mask_np)
        final_image = cv2.add(masked_edges, unmasked_edges)
        output = Image.fromarray(final_image)
        return output
    return processing

def misc_convert_binary(threshold=50):
    def processing(img):
        img_np = np.array(img)
        _, edge_mask = cv2.threshold(img_np, threshold, 255, cv2.THRESH_BINARY)
        output = Image.fromarray(edge_mask)
        return output
    return processing

def misc_normalise_img(target_min = 0, target_max = 255):
    def processing(img):
        img_np = np.array(img)
        img_np = img_np.astype(np.float32)
    
        # Find minimum and maximum pixel values in the image
        img_min = np.min(img_np)
        img_max = np.max(img_np)
        
        # Avoid division by zero for uniform images
        if img_max == img_min:
            return np.full_like(img_np, target_min, dtype=np.float32)  # Uniform image

        # Apply min-max normalization
        normalized_image = target_min + (img_np - img_min) * (target_max - target_min) / (img_max - img_min)
        normalized_image = normalized_image.astype(np.uint8)
        output = Image.fromarray(normalized_image)
        return output
    return processing