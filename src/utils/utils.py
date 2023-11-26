import numpy as np
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
import torch
import shutil


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Trainable parameters: {params/1e6:.3}M")


def save_checkpoint(opt, state, is_best, filename="checkpoint.pth.tar"):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, "model_best.pth.tar"))
