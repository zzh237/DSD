# training/unet_score_wrapper.py
import torch
import torch.nn as nn
from training.networks import SongUNet

class UNetScoreWrapper(nn.Module):
    def __init__(self, img_resolution=32, model_channels=16):
        super().__init__()
        self.img_resolution = img_resolution
        self.net = SongUNet(
            img_resolution=img_resolution,
            in_channels=1, out_channels=1,
            model_channels=model_channels,      # ↓ 从 64 → 16
            channel_mult=[1, 1],                # ↓ 更少的通道
            num_blocks=1,                       # ↓ 只有一个 block
            attn_resolutions=[],
            embedding_type='positional'
        )

    def forward(self, x, t):
        """
        x: (B, 2) 里的坐标
        t: (B, 1) 里的 time embedding (sigma)
        """
        B = x.shape[0]
        H = self.img_resolution

        # 把 (B,2) 坐标稀疏映射到 (B,1,H,H) 的 one-hot 图上
        img = torch.zeros(B, 1, H, H, device=x.device)
        # coord[:,0] -> x 方向，coord[:,1] -> y 方向
        coord = ((x + 6) / 12 * H).long().clamp(0, H - 1)
        img[torch.arange(B), 0, coord[:,1], coord[:,0]] = 1.0

        # U-Net 要求的 inputs
        sigma_embed = t.view(B)  # 变成 (B,)
        out = self.net(img, noise_labels=sigma_embed, class_labels=None)  # (B,1,H,W)

        # 把输出聚合成一个标量，再扩回 (B,2) 以匹配原始输入维度
        m = out.view(B, -1).sum(dim=1, keepdim=True)  # (B,1)
        return m.repeat(1, x.shape[1])               # (B,2)



