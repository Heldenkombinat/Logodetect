"Collection of models for detection."

# Pip packages:
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



def fasterrcnn_resnet50(device, model_weights=None):
    '''
    Creates a FasterRCNN with a ResNet50 backbone.
    '''
    model = fasterrcnn_resnet50_fpn()

    if model_weights:
        # Define the computing device explicitly:
        # checkpoint = torch.load(model_weights, map_location=device)
        checkpoint = torch.load(model_weights)
        model.load_state_dict(checkpoint['state_dict'])

    return model.eval().to(device)


def binary_fasterrcnn_resnet50(device, model_weights):
    '''
    Loads a pre-trained FasterRCNN with a ResNet50 backbone and 2-class output.
    '''
    model = fasterrcnn_resnet50_fpn()

    # Get the number of input features for the top layer:
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the head with a new one for just 2 classes: background and logo
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    # Define the computing device explicitly:
    checkpoint = torch.load(model_weights, map_location=device)
    # checkpoint = torch.load(model_weights)
    model.load_state_dict(checkpoint['state_dict'])

    return model.eval().to(device)
