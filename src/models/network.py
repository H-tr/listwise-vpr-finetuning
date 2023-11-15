import torch
import logging
import torchvision
import torch.nn as nn

from netvlad import NetVLAD
from cosplace import CosPlace
from mixvpr import MixVPR


# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
}


class VPRNetwork(nn.Module):
    def __init__(self, backbone: str, aggregation: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = get_backbone(backbone)
        self.aggregation = get_aggregation(aggregation)

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


def get_backbone(backbone_name: str):
    backbone = get_pretrained_torchvision_model(backbone_name)
    if backbone_name.startswith("ResNet"):
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(
            f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones"
        )
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[
            :-2
        ]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")

    backbone = torch.nn.Sequential(*layers)

    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]

    return backbone, features_dim


def get_aggregation(aggregation_name: str):
    if aggregation_name == "NetVLAD":
        return NetVLAD(num_clusters=64, dim=512, alpha=1.0)
    elif aggregation_name == "CosPlace":
        return CosPlace(in_dim=512, out_dim=512)
    elif aggregation_name == "MixVPR":
        return MixVPR()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation_name}")
