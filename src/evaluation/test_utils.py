import torch
from torch.utils.data import Dataset
from torch import nn
import logging


def test(eval_ds: Dataset, model: nn.Module):
    model = model.eval()
    with torch.no_grad():
        logging.info("Extracting database descriptors...")
