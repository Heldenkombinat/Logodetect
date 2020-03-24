"""
All "stacked" models will take as input the channel-wise
concatenation of two images and output whether it's the same class or not.
Input should always be exemplar ON TOP of detection, i.e.
    input[:3, :, :] is the detection and input[3:, :, :] is the exemplar
"""

import torch
import torch.nn as nn
import torchvision
from sklearn.neighbors import KNeighborsClassifier

ALL_ARCHITECTURES = [
    "binary_stacked_resnet50",
    "binary_stacked_resnet18",
    "siamese_resnet18",
    "knn",
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


def _knn(n_neighbors, metric, *args, **kwargs) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, *args, **kwargs)


def load_model_weights(model, device: str, model_weights: str) -> nn.Sequential:
    checkpoint = torch.load(model_weights, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return nn.Sequential(model, nn.Sigmoid())


def _binary_stacked_resnet50(device: str, model_weights: str):
    """Creates a ResNet50 with a 6-channel input.
    """
    # Load standard, built-in ResNet 50 architecture:
    model = torchvision.models.resnet50(pretrained=False)

    # Get pre-trained weights of first layer:
    conv1_weight = model.conv1.weight
    # Duplicate array:
    duplicated_weights = torch.cat((conv1_weight, conv1_weight), dim=1)

    # Replace the 3-channel input with 6-channel input:
    model.conv1 = nn.Conv2d(
        6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    # Update layer with duplicated pre-trained weights:
    model.conv1.weight = torch.nn.Parameter(data=duplicated_weights)

    # Replace the final layer and add a sigmoid on top:
    model.fc = nn.Linear(model.fc.in_features, out_features=1, bias=True)

    model = load_model_weights(model, device, model_weights)

    return model.eval().to(device)


def _binary_stacked_resnet18(device: str, model_weights: str):
    """Creates a ResNet18 with a 6-channel input.
    """
    # Load standard architecture:
    model = torchvision.models.resnet18(pretrained=False)
    return _load_binary_stacked_net(model, device, model_weights)


def _load_binary_stacked_net(model, device: str, model_weights: str):
    """Creates a ResNet18 with a 6-channel input.
    """
    # Replace the 3-channel input with 6-channel input:
    model.conv1 = nn.Conv2d(
        6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    # Replace the final layer and add a sigmoid on top:
    model.fc = nn.Linear(model.fc.in_features, out_features=1, bias=True)

    model = load_model_weights(model, device, model_weights)
    return model.eval().to(device)


def _siamese_resnet18(device: str, model_weights: str, model_out: int = 345):
    """Loads a pre-trained ResNet18 for Siamese network.
    """
    # Load standard architecture:
    model = torchvision.models.resnet18(pretrained=False)

    # Replace the final layer:
    model.fc = nn.Linear(model.fc.in_features, model_out)

    checkpoint = torch.load(model_weights, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    return model.eval().to(device)
