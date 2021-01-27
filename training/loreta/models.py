"""
Models for logo detection.

All "stacked" models will take as input the channel-wise
concatenation of two images and output whether it's the same class or not.
Input should always START with exemplar END with detection, i.e.
    input[:3, :, :] is the detection and input[3:, :, :] is the exemplar

Functions return both model and loss criterion to be used.

NOTE:
    For binary classification we remove the last sigmoid. See:
    https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss
"""

# Standard library:

# Pip packages:
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Local:


def stacked_resnet(model_type, pretrained=False, frozen_backbone=False):
    """
    Creates a ResNet with a 6-channel input.
    Also returns the loss function.
    """
    # Load model:
    model = torchvision.models.__dict__[model_type](pretrained=pretrained)
    # Freeze backbone?
    model = freeze_backbone(model) if frozen_backbone else model
    if pretrained:
        # Get pre-trained weights of first layer:
        conv1_weight = model.conv1.weight
        # Duplicate array:
        duplicated_weights = torch.cat((conv1_weight, conv1_weight), dim=1)
    # Replace the 3-channel input with 6-channel input:
    model.conv1 = nn.Conv2d(
        6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    if pretrained:
        # Update layer with duplicated pre-trained weights:
        model.conv1.weight = torch.nn.Parameter(data=duplicated_weights)
    # Replace the final layer but DO NOT add a sigmoid on top:
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1, bias=True)

    # Create the loss criterion which INCLUDES the sigmoid:
    criterion = nn.BCEWithLogitsLoss()

    return model, criterion


def cifar10_resnet18():
    """
    Creates a test ResNet18 for CIFAR10
    """
    model = torchvision.models.resnet18()
    # replace the final layer but DO NOT add a sigmoid on top
    model.fc = nn.Linear(in_features=512, out_features=10, bias=True)

    # create the loss criterion which INCLUDES the sigmoid
    criterion = nn.CrossEntropyLoss()

    return model, criterion


def fasterrcnn_resnet50():
    """
    Creates a fasterrcnn_resnet50 with a 6-channel input.
    Also returns the loss function.
    """
    # Load model:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 1 class + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one:
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Freeze backbone:
    model = freeze_backbone(model)

    # Criterion is not necessary for this object detection architecture:
    criterion = None

    return model, criterion


def freeze_backbone(pretrained_model, train_layers=None):
    "Freeze all layers in a model except the last one."
    all_layers = [len([l for l in pretrained_model.children()]) - 1]
    train_layers = train_layers if train_layers else all_layers
    for idx, child in enumerate(pretrained_model.children()):
        for param in child.parameters():
            if idx in train_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False
    print("[INFO] Backbone freezed. Training layers: {}".format(train_layers))
    return pretrained_model
