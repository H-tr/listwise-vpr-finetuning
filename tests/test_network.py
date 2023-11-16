import unittest
import torch
from src.models import netvlad, cosplace, mixvpr, network


class TestNetwork(unittest.TestCase):
    def test_netvlad(self):
        # input format: (batch_size, channels, height, width)
        n_clusters, dim, batch_size = 64, 1024, 32
        x = torch.randn(batch_size, 1024, 20, 20)
        model = netvlad.NetVLAD(num_clusters=n_clusters, dim=dim)
        out = model(x)
        assert out.shape == (batch_size, n_clusters * dim)