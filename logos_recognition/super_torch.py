"Parent class for Torch models"

# Standard library:
import os
import abc

# Pip packages:
import torch

# Logos-Recognition:
from logos_recognition.constants import DEVICE


class SuperTorch(abc.ABC):
    "Agency template for all custom agencies."

    def _load(self, model_base, model_weights, model_top=None):
        "Add documentation."
        # `model_base` says what kind of architecture we want
        model = model_base()

        if model_top:
            # Get the number of input features for the classifier:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # Replace the pre-trained head with a new one for
            # just 2 classes: background and logo
            model.roi_heads.box_predictor = model_top(in_features, 2)

        # Load weights if they exist:
        if os.path.isfile(model_weights):
            # we need to give the map_location argument
            checkpoint = torch.load(model_weights, map_location=DEVICE)
            # there is other stuff saved in the checkpoint also
            model.load_state_dict(checkpoint['state_dict'])

        # we move it to the device
        model.to(DEVICE)
        # and set the model in evaluation mode
        model.eval()
        return model

    def _select_detections(self, detections, selections):
        "Add documentation."
        detections['boxes'] = detections['boxes'][selections]
        detections['labels'] = detections['labels'][selections]
        detections['scores'] = detections['scores'][selections]
        if 'brands' in detections.keys():
            detections['brands'] = detections['brands'][selections]
        return detections
