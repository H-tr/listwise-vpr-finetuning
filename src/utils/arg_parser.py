import argparse
from os import environ

parser = argparse.ArgumentParser(description="pytorch-NetVlad")
parser.add_argument(
    "--mode",
    type=str,
    default="train",
    help="Mode",
    choices=["train", "test", "cluster"],
)
parser.add_argument(
    "--batchSize",
    type=int,
    default=4,
    help="Number of triplets (query, pos, negs). Each triplet consists of 12 images.",
)
parser.add_argument(
    "--cacheBatchSize", type=int, default=24, help="Batch size for caching and testing"
)
parser.add_argument(
    "--cacheRefreshRate",
    type=int,
    default=1000,
    help="How often to refresh cache, in number of queries. 0 for off",
)
parser.add_argument(
    "--nEpochs", type=int, default=30, help="number of epochs to train for"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument("--nGPU", type=int, default=1, help="number of GPU to use.")
parser.add_argument(
    "--optim", type=str, default="SGD", help="optimizer to use", choices=["SGD", "ADAM"]
)
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate.")
parser.add_argument("--lrStep", type=float, default=5, help="Decay LR ever N steps.")
parser.add_argument(
    "--lrGamma", type=float, default=0.5, help="Multiply LR by Gamma for decaying."
)
parser.add_argument(
    "--weightDecay", type=float, default=0.001, help="Weight decay for SGD."
)
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
parser.add_argument("--nocuda", action="store_true", help="Dont use cuda")
parser.add_argument(
    "--threads",
    type=int,
    default=8,
    help="Number of threads for each data loader to use",
)
parser.add_argument("--seed", type=int, default=123, help="Random seed to use.")
parser.add_argument(
    "--dataPath",
    type=str,
    default="/nfs/ibrahimi/data/",
    help="Path for centroid data.",
)
parser.add_argument(
    "--runsPath", type=str, default="/nfs/ibrahimi/runs/", help="Path to save runs to."
)
parser.add_argument(
    "--savePath",
    type=str,
    default="checkpoints",
    help="Path to save checkpoints to in logdir. Default=checkpoints/",
)
parser.add_argument(
    "--cachePath", type=str, default=environ["TMPDIR"], help="Path to save cache to."
)
parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="Path to load checkpoint from, for resuming training or testing.",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="latest",
    help="Resume from latest or best checkpoint.",
    choices=["latest", "best"],
)
parser.add_argument(
    "--evalEvery",
    type=int,
    default=1,
    help="Do a validation set run, and save, every N epochs.",
)
parser.add_argument(
    "--patience", type=int, default=10, help="Patience for early stopping. 0 is off."
)
parser.add_argument(
    "--dataset",
    type=str,
    default="pittsburgh",
    help="Dataset to use",
    choices=["pittsburgh"],
)
parser.add_argument(
    "--arch",
    type=str,
    default="vgg16",
    help="basenetwork to use",
    choices=["vgg16", "alexnet"],
)
parser.add_argument("--vladv2", action="store_true", help="Use VLAD v2")
parser.add_argument(
    "--pooling",
    type=str,
    default="netvlad",
    help="type of pooling to use",
    choices=["netvlad", "max", "avg"],
)
parser.add_argument(
    "--num_clusters",
    type=int,
    default=64,
    help="Number of NetVlad clusters. Default=64",
)
parser.add_argument(
    "--margin", type=float, default=0.1, help="Margin for triplet loss. Default=0.1"
)
parser.add_argument(
    "--split",
    type=str,
    default="val",
    help="Data split to use for testing. Default is val",
    choices=["test", "test250k", "train", "val"],
)
parser.add_argument(
    "--fromscratch",
    action="store_true",
    help="Train from scratch rather than using pretrained models",
)
