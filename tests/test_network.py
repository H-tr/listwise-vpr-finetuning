import unittest
from typing import List, Tuple
import torch
from src.models import netvlad, cosplace, mixvpr, network
import unittest
from typing import List, Tuple
import torch
from src.models import netvlad, cosplace, mixvpr, network


# List of backbones
BACKBONES: List[str] = ["ResNet18", "ResNet50", "ResNet101", "ResNet152", "VGG16"]


class TestAggregation(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.n_clusters: int = 64
        self.batch_size: int = 32
        self.input_dim: int = 1024
        self.output_dim: int = 512
        self.input_height: int = 20
        self.input_width: int = 20
        # input format: (batch_size, channels, height, width)
        self.x: torch.Tensor = torch.randn(
            self.batch_size, self.input_dim, self.input_height, self.input_width
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x = self.x.to(self.device)

    def test_netvlad(self) -> None:
        model: netvlad.NetVLAD = netvlad.NetVLAD(
            num_clusters=self.n_clusters, dim=self.input_dim
        )
        model = model.to(self.device)
        out: torch.Tensor = model(self.x)
        assert out.shape == (self.batch_size, self.n_clusters * self.input_dim)

    def test_cosplace(self) -> None:
        model: cosplace.CosPlace = cosplace.CosPlace(self.input_dim, self.output_dim)
        model = model.to(self.device)
        out: torch.Tensor = model(self.x)
        assert out.shape == (self.batch_size, self.output_dim)

    def test_mixvpr(self) -> None:
        out_rows = 4
        model: mixvpr.MixVPR = mixvpr.MixVPR(
            in_channels=self.input_dim,
            out_channels=self.output_dim,
            mix_depth=4,
            mlp_ratio=1,
            out_rows=out_rows,
        )
        model = model.to(self.device)
        out: torch.Tensor = model(self.x)
        assert out.shape == (self.batch_size, self.output_dim * out_rows)


class TestNetwork(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.n_clusters: int = 64
        self.batch_size: int = 32
        self.input_dim: int = 3
        self.output_dim: int = 512
        self.input_height: int = 640
        self.input_width: int = 480
        # input format: (batch_size, channels, height, width)
        self.x: torch.Tensor = torch.randn(
            self.batch_size, self.input_dim, self.input_height, self.input_width
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x = self.x.to(self.device)

    def test_backbones_netvlad(self) -> None:
        for backbone in BACKBONES:
            aggregation: str = "NetVLAD"
            model: network.VPRNetwork = network.VPRNetwork(
                backbone, aggregation, n_clusters=self.n_clusters
            )
            model = model.to(self.device)
            out: torch.Tensor = model(self.x)
            assert out.shape == (
                self.batch_size,
                self.n_clusters * model.feature_dimention,
            )

    def test_backbones_cosplace(self) -> None:
        for backbone in BACKBONES:
            aggregation: str = "CosPlace"
            model: network.VPRNetwork = network.VPRNetwork(
                backbone, aggregation, output_dim=self.output_dim
            )
            model = model.to(self.device)
            out: torch.Tensor = model(self.x)
            assert out.shape == (self.batch_size, self.output_dim)

    def test_backbones_mixvpr(self) -> None:
        for backbone in BACKBONES:
            aggregation: str = "MixVPR"
            out_rows = 4
            model: network.VPRNetwork = network.VPRNetwork(
                backbone,
                aggregation,
                output_dim=self.output_dim,
                out_rows=out_rows,
            )
            model = model.to(self.device)
            out: torch.Tensor = model(self.x)
            assert out.shape == (self.batch_size, self.output_dim * out_rows)
