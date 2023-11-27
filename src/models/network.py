import torch
import logging
import torchvision
import torch.nn as nn

from models.netvlad import NetVLAD
from models.cosplace import CosPlace
from models.mixvpr import MixVPR


# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
}


class VPRNetwork(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.backbone = get_backbone(args)
        self.aggregation = get_aggregation(args)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def get_pretrained_torchvision_model(backbone_name: str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(
            __import__("torchvision.models", fromlist=[f"{backbone_name}_Weights"]),
            f"{backbone_name}_Weights",
        )
        model = getattr(torchvision.models, backbone_name.lower())(
            weights=weights_module.DEFAULT
        )
    except (
        ImportError,
        AttributeError,
    ):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model


def get_backbone(args):
    backbone = get_pretrained_torchvision_model(args.backbone)
    if args.backbone.startswith("ResNet"):
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(
            f"Train only layer3 and layer4 of the {args.backbone}, freeze the previous ones"
        )
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    elif args.backbone == "VGG16":
        layers = list(backbone.features.children())[
            :-2
        ]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")

    backbone = torch.nn.Sequential(*layers)

    args.features_dim = CHANNELS_NUM_IN_LAST_CONV[args.backbone]

    return backbone


def get_aggregation(
    args,
    out_dim: int = 512,
    out_rows: int = 4,
) -> torch.nn.Module:
    if args.aggregation == "NetVLAD":
        return NetVLAD(clusters_num=args.netvlad_clusters, dim=args.features_dim)
    elif args.aggregation == "CosPlace":
        return CosPlace(in_dim=args.features_dim, out_dim=out_dim)
    elif args.aggregation == "MixVPR":
        return MixVPR(
            in_channels=args.features_dim,
            out_channels=out_dim,
            mix_depth=4,
            mlp_ratio=1,
            out_rows=out_rows,
        )
    else:
        raise ValueError(f"Unknown aggregation: {args.aggregation}")
