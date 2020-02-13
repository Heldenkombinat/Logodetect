"""
All "stacked" models will take as input the channel-wise
concatenation of two images and output whether it's the same class or not.
Input should always be exemplar ON TOP of detection, i.e.
    input[:3, :, :] is the detection and input[3:, :, :] is the exemplar
"""

# Pip packages:
import torch
import torch.nn as nn
import torchvision
from sklearn.neighbors import KNeighborsClassifier


def binary_stacked_resnet50(device, model_weights):
    """
    Creates a ResNet50 with a 6-channel input.
    """
    # Load standard architecture:
    model = torchvision.models.resnet50(pretrained=False)

    # MODIFY ARCHITECTURE #
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

    # LOAD WEIGHTS #
    # Define the computing device explicitly:
    checkpoint = torch.load(model_weights, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = nn.Sequential(model, nn.Sigmoid())

    return model.eval().to(device)


def binary_stacked_resnet18(device, model_weights):
    """
    Creates a ResNet18 with a 6-channel input.
    """
    # Load standard architecture:
    model = torchvision.models.resnet18(pretrained=False)
    return load_binary_stacked_net(model, device, model_weights)


# def binary_stacked_resnet50(device, model_weights):
#     '''
#     Creates a ResNet50 with a 6-channel input.
#     '''
#     # Load standard architecture:
#     model = torchvision.models.resnet50(pretrained=False)
#     return load_binary_stacked_net(model, device, model_weights)


def load_binary_stacked_net(model, device, model_weights):
    """
    Creates a ResNet18 with a 6-channel input.
    """
    # MODIFY ARCHITECTURE #
    # Replace the 3-channel input with 6-channel input:
    model.conv1 = nn.Conv2d(
        6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    # Replace the final layer and add a sigmoid on top:
    model.fc = nn.Linear(model.fc.in_features, out_features=1, bias=True)

    # LOAD WEIGHTS #
    # Define the computing device explicitly:
    checkpoint = torch.load(model_weights, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = nn.Sequential(model, nn.Sigmoid())

    return model.eval().to(device)


def siamese_resnet18(device, model_weights, model_out=345):
    """
    Loads a pre-trained ResNet18 for Siamese network.
    """
    # Load standard architecture:
    model = torchvision.models.resnet18(pretrained=False)

    # MODIFY ARCHITECTURE #
    # Replace the final layer:
    model.fc = nn.Linear(model.fc.in_features, model_out)

    # LOAD WEIGHTS #
    # Define the computing device explicitly:
    checkpoint = torch.load(model_weights, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    return model.eval().to(device)
