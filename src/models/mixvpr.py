import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from utils import print_nb_params


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        out_channels=512,
        mix_depth=1,
        mlp_ratio=1,
        out_rows=4,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels  # depth of input feature maps

        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimesion

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = (
            mlp_ratio  # ratio of the mid projection layer in the mixer block
        )
        self.mix = None

    def create_mix_layer(self, in_h, in_w):
        hw = in_h * in_w
        self.mix = nn.Sequential(
            *[
                FeatureMixerLayer(in_dim=hw, mlp_ratio=self.mlp_ratio)
                for _ in range(self.mix_depth)
            ]
        )
        self.channel_proj = nn.Linear(self.in_channels, self.out_channels)
        self.row_proj = nn.Linear(hw, self.out_rows)

    def forward(self, x):
        in_h, in_w = x.shape[-2:]
        if self.mix is None:
            self.create_mix_layer(in_h, in_w)
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x
