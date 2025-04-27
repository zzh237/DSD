import torch
import numpy as np
from torch import nn


class Config:
    n_samples       = 1000
    noise_sigma     = 0.1
    latent_dim      = 32
    hidden_dim      = 64
    low_rank_dim    = 1
    batch_size      = 128
    pretrain_epochs = 400
    distill_epochs  = 300
    lr              = 1e-3
    device          = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed            = 42
    plot_size       = (10,5)
    data_dim        = 2

cfg = Config()
torch.manual_seed(cfg.seed)



# ========= Reusable MLP =========
class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.Tanh):
        """
        layer_sizes: e.g. [in_dim, hid1, hid2, ..., out_dim]
        activation:  隐藏层激活函数
        """
        super().__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.net(x)

# ========= Models =========
class LowRankGenerator(nn.Module):
    """
    低秩线性生成器 G(z)=U (V^T z)，U∈R^{2×r}, V∈R^{latent_dim×r}
    """
    def __init__(self):
        super().__init__()
        # proj 相当于 V^T：从 latent_dim 投到 r
        self.proj   = nn.Linear(cfg.data_dim, cfg.low_rank_dim, bias=False)
        # expand 相当于 U：从 r 投到 2
        self.expand = nn.Linear(cfg.low_rank_dim, 2,                 bias=False)
        nn.init.orthogonal_(self.proj.weight)
        nn.init.orthogonal_(self.expand.weight)

    def forward(self, z, t):
        h = self.proj(z)      # [B, r]
        return self.expand(h) # [B, 2]

class FreeMLPGenerator(nn.Module):
    """不带约束的 MLP 生成器"""
    def __init__(self):
        super().__init__()
        self.mlp = MLP([cfg.data_dim, cfg.hidden_dim, 2], activation=nn.Tanh)

    def forward(self, z, t):
        return self.mlp(z)

class DiffusionMLP(nn.Module):
    """伪扩散模型 f(x,t) → posterior mean 或 score"""
    def __init__(self):
        super().__init__()
        # 把标量 t embed 到 hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, cfg.hidden_dim),
            nn.Tanh()
        )
        # 主 MLP：输入 = x_dim + hidden_dim, 输出 = 2
        self.mlp = MLP([2 + cfg.hidden_dim, cfg.hidden_dim, 2], activation=nn.Tanh)

    def forward(self, x, t):
        # 规范化 t 到 (B,1)
        if not torch.is_tensor(t):
            t = x.new_tensor([t]).expand(x.shape[0])
        else:
            t = t.view(-1)
            if t.numel() == 1:
                t = t.expand(x.shape[0])
        t = t.unsqueeze(-1)  # [B,1]

        te = self.time_embed(t)       # [B, hidden_dim]
        inp = torch.cat([x, te], dim=1)  # [B, 2+hidden_dim]
        return self.mlp(inp)  # [B,2]