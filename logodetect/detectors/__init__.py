"Collection of models for detection."
from .faster_rcnn import *

# Pip packages:
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ALL_ARCHITECTURES = [
    "fasterrcnn_resnet50",
    "binary_fasterrcnn_resnet50",
]


def get(function_name: str):
    """Get a model architecture by function name.

    :param function_name: function name to call
    :return: the function created from the name.
    """
    if function_name not in ALL_ARCHITECTURES:
        raise ValueError(
            f"{function_name} is not a valid model architecture, choose from {ALL_ARCHITECTURES}."
        )
    return globals()[f"_{function_name}"]


def _fasterrcnn_resnet50(device: str, model_weights: str = None):
    """Creates a FasterRCNN with a ResNet50 backbone.
    """
    model = fasterrcnn_resnet50_fpn()

    if model_weights:
        checkpoint = torch.load(model_weights, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    return model.eval().to(device)


def _binary_fasterrcnn_resnet50(device: str, model_weights: str):
    """Loads a pre-trained FasterRCNN with a ResNet50 backbone and 2-class output.
    """
    model = fasterrcnn_resnet50_fpn()

    # Get the number of input features for the top layer:
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the head with a new one for just 2 classes: background and logo
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    # Define the computing device explicitly:
    checkpoint = torch.load(model_weights, map_location=device)
    # checkpoint = torch.load(model_weights)
    model.load_state_dict(checkpoint["state_dict"])

    return model.eval().to(device)
