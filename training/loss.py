# training/loss.py
# Copyright (c) 2025, Your Name
#
# Licensed under the Apache License, Version 2.0

import torch
from torch_utils import persistence

"""
Loss functions used in the Denoising Score Distillation (DSD) project.
Based on the Score Identity Distillation loss,稍作改动以适应 DSD 框架。
"""

@persistence.persistent_class
class DSDLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, beta_d=19.9, beta_min=0.1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.beta_d = beta_d
        self.beta_min = beta_min

    def generator_loss(self, true_score, fake_score, images, labels=None, augment_pipe=None, alpha=1.2, tmax=800):
        # 一步噪声反演的生成器损失
        sigma_min = 0.002
        sigma_max = 80
        rho = 7.0
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        rnd_t = torch.rand([images.shape[0], 1, 1, 1], device=images.device) * tmax / 1000
        sigma = (max_inv_rho + (1 - rnd_t) * (min_inv_rho - max_inv_rho)) ** rho

        # 可选的数据增强
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, torch.zeros(images.shape[0], 9, device=images.device))
        n = torch.randn_like(y) * sigma

        # 真分数与伪分数
        y_real = true_score(y + n, sigma, labels, augment_labels=augment_labels)
        y_fake = fake_score(y + n, sigma, labels, augment_labels=augment_labels)

        # 去除 NaN
        nan_mask = torch.isnan(y).flatten(1).any(1) | \
                   torch.isnan(y_real).flatten(1).any(1) | \
                   torch.isnan(y_fake).flatten(1).any(1)
        if nan_mask.any():
            keep = ~nan_mask
            y, y_real, y_fake = y[keep], y_real[keep], y_fake[keep]

        # 动态归一化权重
        with torch.no_grad():
            weight = (y - y_real).abs().mean(dim=[1,2,3], keepdim=True).clamp(min=1e-5)

        loss = (y_real - y_fake) * ((y_real - y) - alpha * (y_real - y_fake)) / weight
        return loss

    def __call__(self, fake_score, images, labels=None, augment_pipe=None):
        # 伪分数网络的常规模拟训练损失
        rnd = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        y_fake = fake_score(y + n, sigma, labels, augment_labels=augment_labels)

        nan_mask = torch.isnan(y).flatten(1).any(1) | torch.isnan(y_fake).flatten(1).any(1)
        if nan_mask.any():
            keep = ~nan_mask
            y, y_fake, weight = y[keep], y_fake[keep], weight[keep]

        return weight * (y_fake - y).square()
