import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, SubsetRandomSampler

import models.functional as LF
import models.normalization as normalization
from models.netvlad import NetVLAD


class MAC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return LF.mac(x)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SPoC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return LF.spoc(x)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.work_with_tokens = work_with_tokens

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class RMAC(nn.Module):
    def __init__(self, L=3, eps=1e-6):
        super().__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        return LF.rmac(x, L=self.L, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + "(" + "L=" + "{}".format(self.L) + ")"


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


class RRM(nn.Module):
    """Residual Retrieval Module as described in the paper
    `Leveraging EfficientNet and Contrastive Learning for AccurateGlobal-scale
    Location Estimation <https://arxiv.org/pdf/2105.07645.pdf>`
    """

    def __init__(self, dim):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = Flatten()
        self.ln1 = nn.LayerNorm(normalized_shape=dim)
        self.fc1 = nn.Linear(in_features=dim, out_features=dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=dim, out_features=dim)
        self.ln2 = nn.LayerNorm(normalized_shape=dim)
        self.l2 = normalization.L2Norm()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.ln1(x)
        identity = x
        out = self.fc2(self.relu(self.fc1(x)))
        out += identity
        out = self.l2(self.ln2(out))
        return out


class CRNModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Downsample pooling
        self.downsample_pool = nn.AvgPool2d(
            kernel_size=3, stride=(2, 2), padding=0, ceil_mode=True
        )

        # Multiscale Context Filters
        self.filter_3_3 = nn.Conv2d(
            in_channels=dim, out_channels=32, kernel_size=(3, 3), padding=1
        )
        self.filter_5_5 = nn.Conv2d(
            in_channels=dim, out_channels=32, kernel_size=(5, 5), padding=2
        )
        self.filter_7_7 = nn.Conv2d(
            in_channels=dim, out_channels=20, kernel_size=(7, 7), padding=3
        )

        # Accumulation weight
        self.acc_w = nn.Conv2d(in_channels=84, out_channels=1, kernel_size=(1, 1))
        # Upsampling
        self.upsample = F.interpolate

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize Context Filters
        torch.nn.init.xavier_normal_(self.filter_3_3.weight)
        torch.nn.init.constant_(self.filter_3_3.bias, 0.0)
        torch.nn.init.xavier_normal_(self.filter_5_5.weight)
        torch.nn.init.constant_(self.filter_5_5.bias, 0.0)
        torch.nn.init.xavier_normal_(self.filter_7_7.weight)
        torch.nn.init.constant_(self.filter_7_7.bias, 0.0)

        torch.nn.init.constant_(self.acc_w.weight, 1.0)
        torch.nn.init.constant_(self.acc_w.bias, 0.0)
        self.acc_w.weight.requires_grad = False
        self.acc_w.bias.requires_grad = False

    def forward(self, x):
        # Contextual Reweighting Network
        x_crn = self.downsample_pool(x)

        # Compute multiscale context filters g_n
        g_3 = self.filter_3_3(x_crn)
        g_5 = self.filter_5_5(x_crn)
        g_7 = self.filter_7_7(x_crn)
        g = torch.cat((g_3, g_5, g_7), dim=1)
        g = F.relu(g)

        w = F.relu(self.acc_w(g))  # Accumulation weight
        mask = self.upsample(w, scale_factor=2, mode="bilinear")  # Reweighting Mask

        return mask


class CRN(NetVLAD):
    def __init__(self, clusters_num=64, dim=128, normalize_input=True):
        super().__init__(clusters_num, dim, normalize_input)
        self.crn = CRNModule(dim)

    def forward(self, x):
        N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim

        mask = self.crn(x)

        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # Weight soft_assign using CRN's mask
        soft_assign = soft_assign * mask.view(N, 1, H * W)

        vlad = torch.zeros(
            [N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device
        )
        for D in range(
            self.clusters_num
        ):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[
                D : D + 1, :
            ].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:, D : D + 1, :].unsqueeze(2)
            vlad[:, D : D + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad
