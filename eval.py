import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def normalized_mse(image_true, image_test):
    # would need to call function for each pair if evaluating a batch
    mse = np.mean((image_true - image_test) ** 2)
    nmse = mse / np.var(image_true)
    return nmse

def calculate_metrics(real_images, generated_images):
    psnr_values = []
    ssim_values = []
    nmse_values = []

    for real_img, gen_img in zip(real_images, generated_images):
        
        # Ensure the images are in the correct range and format
        real_img = real_img.squeeze().numpy()  # Convert from tensor to numpy
        gen_img = gen_img.squeeze().numpy()  # Convert from tensor to numpy
        
        # If images are in range [-1, 1], convert to [0, 255]
        real_img = ((real_img + 1) * 127.5).astype(np.uint8)
        gen_img = ((gen_img + 1) * 127.5).astype(np.uint8)

        # Calculate PSNR
        psnr = peak_signal_noise_ratio(real_img, gen_img, data_range=255)
        psnr_values.append(psnr)

        # Calculate SSIM
        ssim = structural_similarity(real_img, gen_img, data_range=255, multichannel=True)
        ssim_values.append(ssim)

        # Calculate NMSE
        mse = mean_squared_error(real_img, gen_img)
        nmse = mse / np.mean(real_img**2)
        nmse_values.append(nmse)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_nmse = np.mean(nmse_values)
    
    return avg_psnr, avg_ssim, avg_nmse