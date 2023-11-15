from torch.utils.data import Dataset
from torch import nn 
import random


def generate_list(length: int, seed: int) -> list:
    random.seed(seed)
    return random.sample(range(0, 100), length) / 100

test_sample = generate_list(10, 42)

def test(
    eval_ds: Dataset,
    model: nn.Module
):
    pass
    