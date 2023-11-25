import unittest
import torch
from src.models.loss import APLoss, TAPLoss, TripletLoss
import unittest
import torch
from src.models.loss import APLoss, TAPLoss, TripletLoss


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        self.ap_loss = APLoss().cuda()  # Move to GPU
        self.tap_loss = TAPLoss().cuda()  # Move to GPU
        self.triplet_loss = TripletLoss().cuda()  # Move to GPU

    def test_ap_loss(self):
        x = torch.randn(10, 5).cuda()  # Move to GPU
        label = torch.randint(0, 2, (10, 5)).cuda()  # Move to GPU
        with torch.no_grad():
            loss = self.ap_loss(x, label)
        self.assertIsInstance(loss, torch.Tensor)

    def test_tap_loss(self):
        x = torch.randn(10, 5).cuda()  # Move to GPU
        label = torch.randint(0, 2, (10, 5)).cuda()  # Move to GPU
        with torch.no_grad():
            loss = self.tap_loss(x, label)
        self.assertIsInstance(loss, torch.Tensor)

    def test_triplet_loss(self):
        anchor = torch.randn(10, 128).cuda()  # Move to GPU
        positive = torch.randn(10, 128).cuda()  # Move to GPU
        negative = torch.randn(10, 128).cuda()  # Move to GPU
        with torch.no_grad():
            loss = self.triplet_loss(anchor, positive, negative)
        self.assertIsInstance(loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
