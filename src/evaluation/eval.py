import torch
import logging
import multiprocessing

import test
from models import network
from dataset.seq_dataset import SeqDataset


def evaluate_model(
    backbone: str, aggregation: str, resume_model: str, test_set_folder: str
):
    #### Model
    model = network.VPRNetwork(backbone, aggregation)

    logging.info(
        f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs."
    )

    if resume_model is not None:
        logging.info(f"Loading model from {resume_model}")
        model_state_dict = torch.load(resume_model)
        model.load_state_dict(model_state_dict)
    else:
        logging.info(
            "WARNING: You didn't provide a path to resume the model (--resume_model parameter). "
            + "Evaluation will be computed using randomly initialized weights."
        )

    model = model.to("cuda")

    test_ds = SeqDataset(test_set_folder)

    recalls, recalls_str = test.test(test_ds, model)
    logging.info(f"{test_ds}: {recalls_str}")


evaluate_model(
    "ResNet50",
    "NetVLAD",
    "experiments/saved_models/ResNet50_512.pth",
    "data/processed/pioneer_hall",
)
