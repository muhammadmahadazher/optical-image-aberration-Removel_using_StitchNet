import torch
import torch.nn.functional as F


def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1, mu2 = img1.mean(dim=[2, 3], keepdim=True), img2.mean(dim=[2, 3], keepdim=True)
    s1, s2 = img1.var(dim=[2, 3], keepdim=True), img2.var(dim=[2, 3], keepdim=True)
    cov = ((img1 - mu1) * (img2 - mu2)).mean(dim=[2, 3], keepdim=True)
    num = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return torch.clamp((1 - num / den) / 2, 0, 1)


def total_loss(warp, gt, flow, lam_ssim=1.0, lam_smooth=0.1):
    l1 = F.l1_loss(warp, gt)
    lssim = ssim(warp, gt).mean()
    smooth = (flow[:, :, 1:] - flow[:, :, :-1]).abs().mean() + \
             (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
    return l1 + lam_ssim * lssim + lam_smooth * smooth
