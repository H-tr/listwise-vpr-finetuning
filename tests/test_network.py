import unittest
import torch
from src.models import netvlad, cosplace, mixvpr, network


class TestNetwork(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.n_clusters = 64
        self.dim = 1024
        self.batch_size = 32
        self.input_dim = 1024
        self.output_dim = 512
        # input format: (batch_size, channels, height, width)
        self.x = torch.randn(self.batch_size, self.input_dim, 20, 20)

    def test_netvlad(self):
        model = netvlad.NetVLAD(num_clusters=self.n_clusters, dim=self.dim)
        out = model(self.x)
        assert out.shape == (self.batch_size, self.n_clusters * self.dim)

    def test_cosplace(self):
        model = cosplace.CosPlace(self.input_dim, self.output_dim)
        out = model(self.x)
        assert out.shape == (self.batch_size, self.output_dim)
