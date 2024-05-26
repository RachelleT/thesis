import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_fid import fid_score

def psnr(original, synthesize): # ndarray
    psnr_score = peak_signal_noise_ratio(original, synthesize)
    # psnr_scores = peak_signal_noise_ratio(batch_true, batch_test) 
    return psnr_score

def ssim(original, synthesize): # ndarray
    # need to loop over the pairs and call structural_similarity for each pair when evaluating a batch
    ssim_score, _ = structural_similarity(original, synthesize, full=True)
    return ssim_score

def normalized_mse(image_true, image_test):
    # would need to call function for each pair if evaluating a batch
    mse = np.mean((image_true - image_test) ** 2)
    nmse = mse / np.var(image_true)
    return nmse

def fid(paths, batch_size, device):
    # paths = ['path of source image dir', 'path of target image dir']
    fid_value = fid_score.calculate_fid_given_paths(paths, batch_size, device)
    return fid_value