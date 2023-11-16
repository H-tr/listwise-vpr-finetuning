import unittest
import torch
from src.models import netvlad, cosplace, mixvpr, network


class TestNetwork(unittest.TestCase):
    def test_netvlad(self):
        x = torch.randn(1, 1024, 20, 20)
        netvlad = netvlad.NetVLAD(num_clusters=64, dim=1024)
        print(netvlad(x).shape)