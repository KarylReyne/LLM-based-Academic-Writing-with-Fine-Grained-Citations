import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return th.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)  

class L1SSIMLoss(nn.Module):
    def __init__(self, ssim_factor = 0.85):
        super(L1SSIMLoss, self).__init__()

        self.ssim = SSIM()
        self.ssim_factor = ssim_factor

    def forward(self, output, target):
        
        l1    = th.abs(output - target)
        ssim  = self.ssim(output, target)

        f = self.ssim_factor

        return th.mean(l1 * (1 - f) + ssim * f), th.mean(l1), th.mean(ssim)

class MaskedL1SSIMLoss(nn.Module):
    def __init__(self, ssim_factor = 0.85):
        super(MaskedL1SSIMLoss, self).__init__()

        self.ssim = SSIM()
        self.ssim_factor = ssim_factor

    def forward(self, output, target, mask):
        
        l1    = th.abs(output - target) * mask
        ssim  = self.ssim(output, target) * mask

        numel = th.sum(mask, dim=(1, 2, 3)) + 1e-7

        l1 = th.sum(l1, dim=(1, 2, 3)) / numel
        ssim = th.sum(ssim, dim=(1, 2, 3)) / numel

        f = self.ssim_factor

        return th.mean(l1 * (1 - f) + ssim * f), th.mean(l1), th.mean(ssim)
