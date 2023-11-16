import unittest
from typing import List, Tuple
import torch
from src.models import netvlad, cosplace, mixvpr, network


class TestNetwork(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.n_clusters: int = 64
        self.dim: int = 1024
        self.batch_size: int = 32
        self.input_dim: int = 1024
        self.output_dim: int = 512
        self.input_height: int = 20
        self.input_width: int = 20
        # input format: (batch_size, channels, height, width)
        self.x: torch.Tensor = torch.randn(
            self.batch_size, self.input_dim, self.input_height, self.input_width
        )

    def test_netvlad(self) -> None:
        model: netvlad.NetVLAD = netvlad.NetVLAD(
            num_clusters=self.n_clusters, dim=self.dim
        )
        out: torch.Tensor = model(self.x)
        assert out.shape == (self.batch_size, self.n_clusters * self.dim)

    def test_cosplace(self) -> None:
        model: cosplace.CosPlace = cosplace.CosPlace(self.input_dim, self.output_dim)
        out: torch.Tensor = model(self.x)
        assert out.shape == (self.batch_size, self.output_dim)

    def test_mixvpr(self) -> None:
        out_rows = 4
        model: mixvpr.MixVPR = mixvpr.MixVPR(
            in_channels=self.input_dim,
            in_h=20,
            in_w=20,
            out_channels=self.output_dim,
            mix_depth=4,
            mlp_ratio=1,
            out_rows=out_rows,
        )
        out: torch.Tensor = model(self.x)
        assert out.shape == (self.batch_size, self.output_dim * out_rows)
