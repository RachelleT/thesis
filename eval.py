import torch

from piqa import SSIM, HaarPSI, PSNR, MS_GMSD, MDSI

def compute_metrics(real, fakes):
    p, s, h, ms, md = [], [], [], [], []

    ssim = SSIM(data_range=1).cpu()
    psnr = PSNR(data_range=1)
    haar = HaarPSI(data_range=1)
    ms_gmsd = MS_GMSD(data_range=1)
    mdsi = MDSI(data_range=1)

    # Choose the minimum number of images to compare
    thres = min(len(real), len(fakes))

    for i in range(thres):
        f = fakes[i].unsqueeze(1).unsqueeze(0)  # Add batch and channel dimensions
        r = real[i].unsqueeze(1).unsqueeze(0)   # Add batch and channel dimensions
        r_norm = (r - r.min()) / (r.max() - r.min())
        f_norm = (f - f.min()) / (f.max() - f.min())

        p.append(psnr(r_norm, f_norm))
        s.append(ssim(r_norm, f_norm))
        h.append(haar(r_norm, f_norm))
        ms.append(ms_gmsd(r_norm, f_norm))
        md.append(mdsi(r_norm, f_norm))

    avg_psnr = sum(p) / len(p) if p else 0
    avg_ssim = sum(s) / len(s) if s else 0
    avg_haar = sum(h) / len(h) if h else 0
    avg_ms_gmsd = sum(ms) / len(ms) if ms else 0
    avg_mdsi = sum(md) / len(md) if md else 0

    print('PSNR: {}, SSIM: {}, HAAR: {}, MSGMSD: {}, MDSI: {}'.format(
        avg_psnr, avg_ssim, avg_haar, avg_ms_gmsd, avg_mdsi))

    return avg_psnr, avg_ssim, avg_haar, avg_ms_gmsd, avg_mdsi

def compute_metrics_old(real, fakes, size):

    p, s, h, ms, md = [], [], [], [], []

    ssim = SSIM().cpu()
    psnr = PSNR()
    haar = HaarPSI()
    ms_gmsd = MS_GMSD()
    mdsi = MDSI()

    if len(real[0]) > len(fakes[-1]):
        thres = len(fakes[-1])

    elif len(real[0]) < len(fakes[-1]):
        thres = len(real[0])

    else:
        thres = len(fakes[-1])

    for i in range(0, thres-1):
        f = torch.reshape(fakes[-1][i], (-1, 1, size, size))
        r = torch.reshape(real[0][i], (-1, 1, size, size))
        r_norm = (r - r.min()) / (r.max() - r.min())
        f_norm = (f - f.min()) / (f.max() - f.min())

        print(r_norm.size())
        print(f_norm.size())

        p.append(psnr(r_norm, f_norm))
        s.append(ssim(r_norm, f_norm))
        h.append(haar(r_norm, f_norm))
        ms.append(ms_gmsd(r_norm, f_norm))
        md.append(mdsi(r_norm, f_norm))

    print('PSNR: {}, SSIM: {}, HAAR: {}, MSGMSD: {}, MDSI: {}'.format(
        sum(p)/(len(p)), sum(s)/(len(s)), sum(h)/(len(h)), sum(ms)/(len(ms)), sum(md)/(len(md))))

    return 0