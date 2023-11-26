import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader
from models.network import VPRNetwork
from src.models.loss import TripletLoss
from dataset.msls_dataloader import MSLSDataset
from utils.arg_parser import parser
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
import random
import numpy as np


def main():
    opt = parser.parse_args()

    restore_var = [
        "lr",
        "lrStep",
        "lrGamma",
        "weightDecay",
        "momentum",
        "runsPath",
        "savePath",
        "arch",
        "num_clusters",
        "pooling",
        "optim",
        "margin",
        "seed",
        "patience",
    ]
    if opt.resume:
        flag_file = join(opt.resume, "checkpoints", "flags.json")
        if exists(flag_file):
            with open(flag_file, "r") as f:
                stored_flags = {
                    "--" + k: str(v)
                    for k, v in json.load(f).items()
                    if k in restore_var
                }
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ""
                for flag in to_del:
                    del stored_flags[flag]

                train_flags = [
                    x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0
                ]
                print("Restored flags:", train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    print(opt)

    if opt.dataset.lower() == "pittsburgh":
        import pittsburgh as dataset
    else:
        raise Exception("Unknown dataset")

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print("===> Loading dataset(s)")
    if opt.mode.lower() == "train":
        whole_train_set = dataset.get_whole_training_set()
        whole_training_data_loader = DataLoader(
            dataset=whole_train_set,
            num_workers=opt.threads,
            batch_size=opt.cacheBatchSize,
            shuffle=False,
            pin_memory=cuda,
        )

        train_set = dataset.get_training_query_set(opt.margin)

        print("====> Training query set:", len(train_set))
        whole_test_set = dataset.get_whole_val_set()
        print("===> Evaluating on val set, query count:", whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == "test":
        if opt.split.lower() == "test":
            whole_test_set = dataset.get_whole_test_set()
            print("===> Evaluating on test set")
        elif opt.split.lower() == "test250k":
            whole_test_set = dataset.get_250k_test_set()
            print("===> Evaluating on test250k set")
        elif opt.split.lower() == "train":
            whole_test_set = dataset.get_whole_training_set()
            print("===> Evaluating on train set")
        elif opt.split.lower() == "val":
            whole_test_set = dataset.get_whole_val_set()
            print("===> Evaluating on val set")
        else:
            raise ValueError("Unknown dataset split: " + opt.split)
        print("====> Query count:", whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == "cluster":
        whole_train_set = dataset.get_whole_training_set(onlyDB=True)

    print("===> Building model")

    pretrained = not opt.fromscratch
    if opt.arch.lower() == "alexnet":
        encoder_dim = 256
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == "vgg16":
        encoder_dim = 512
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False

    if opt.mode.lower() == "cluster" and not opt.vladv2:
        layers.append(L2Norm())

    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module("encoder", encoder)

    if opt.mode.lower() != "cluster":
        if opt.pooling.lower() == "netvlad":
            net_vlad = netvlad.NetVLAD(
                num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2
            )
            if not opt.resume:
                if opt.mode.lower() == "train":
                    initcache = join(
                        opt.dataPath,
                        "centroids",
                        opt.arch
                        + "_"
                        + train_set.dataset
                        + "_"
                        + str(opt.num_clusters)
                        + "_desc_cen.hdf5",
                    )
                else:
                    initcache = join(
                        opt.dataPath,
                        "centroids",
                        opt.arch
                        + "_"
                        + whole_test_set.dataset
                        + "_"
                        + str(opt.num_clusters)
                        + "_desc_cen.hdf5",
                    )

                if not exists(initcache):
                    raise FileNotFoundError(
                        "Could not find clusters, please run with --mode=cluster before proceeding"
                    )

                with h5py.File(initcache, mode="r") as h5:
                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                    net_vlad.init_params(clsts, traindescs)
                    del clsts, traindescs

            model.add_module("pool", net_vlad)
        elif opt.pooling.lower() == "max":
            global_pool = nn.AdaptiveMaxPool2d((1, 1))
            model.add_module("pool", nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        elif opt.pooling.lower() == "avg":
            global_pool = nn.AdaptiveAvgPool2d((1, 1))
            model.add_module("pool", nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        else:
            raise ValueError("Unknown pooling type: " + opt.pooling)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        if opt.mode.lower() != "cluster":
            model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if not opt.resume:
        model = model.to(device)

    if opt.mode.lower() == "train":
        if opt.optim.upper() == "ADAM":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr
            )  # , betas=(0,0.9))
        elif opt.optim.upper() == "SGD":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay,
            )

            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=opt.lrStep, gamma=opt.lrGamma
            )
        else:
            raise ValueError("Unknown optimizer: " + opt.optim)

        # original paper/code doesn't sqrt() the distances, we do, so sqrt() the margin, I think :D
        criterion = nn.TripletMarginLoss(
            margin=opt.margin**0.5, p=2, reduction="sum"
        ).to(device)

    if opt.resume:
        if opt.ckpt.lower() == "latest":
            resume_ckpt = join(opt.resume, "checkpoints", "checkpoint.pth.tar")
        elif opt.ckpt.lower() == "best":
            resume_ckpt = join(opt.resume, "checkpoints", "model_best.pth.tar")

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(
                resume_ckpt, map_location=lambda storage, loc: storage
            )
            opt.start_epoch = checkpoint["epoch"]
            best_metric = checkpoint["best_score"]
            model.load_state_dict(checkpoint["state_dict"])
            model = model.to(device)
            if opt.mode == "train":
                optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume_ckpt, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    if opt.mode.lower() == "test":
        print("===> Running evaluation step")
        epoch = 1
        recalls = test(whole_test_set, epoch, write_tboard=False)
    elif opt.mode.lower() == "cluster":
        print("===> Calculating descriptors and clusters")
        get_clusters(whole_train_set)
    elif opt.mode.lower() == "train":
        print("===> Training model")
        writer = SummaryWriter(
            log_dir=join(
                opt.runsPath,
                datetime.now().strftime("%b%d_%H-%M-%S")
                + "_"
                + opt.arch
                + "_"
                + opt.pooling,
            )
        )

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        opt.savePath = join(logdir, opt.savePath)
        if not opt.resume:
            makedirs(opt.savePath)

        with open(join(opt.savePath, "flags.json"), "w") as f:
            f.write(json.dumps({k: v for k, v in vars(opt).items()}))
        print("===> Saving state to:", logdir)

        not_improved = 0
        best_score = 0
        for epoch in range(opt.start_epoch + 1, opt.nEpochs + 1):
            if opt.optim.upper() == "SGD":
                scheduler.step(epoch)
            train(epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(whole_test_set, epoch, write_tboard=True)
                is_best = recalls[5] > best_score
                if is_best:
                    not_improved = 0
                    best_score = recalls[5]
                else:
                    not_improved += 1

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "recalls": recalls,
                        "best_score": best_score,
                        "optimizer": optimizer.state_dict(),
                        "parallel": isParallel,
                    },
                    is_best,
                )

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print(
                        "Performance did not improve for",
                        opt.patience,
                        "epochs. Stopping.",
                    )
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()


if __name__ == "__main__":
    main()
