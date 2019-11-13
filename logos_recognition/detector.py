"Detector module."

# Standard library:
import os

# Pip packages:
import numpy as np
import torch

# Github repos:

# Logos-Recognition:
from logos_recognition.super_torch import SuperTorch
from logos_recognition.utils import image_to_gpu_tensor
from logos_recognition.constants import (DETECTOR_BASE, DETECTOR_WEIGHTS,
                                         DETECTOR_TOP, DEVICE)


class Detector(SuperTorch):
    "Add documentation."

    def __init__(self):
        "Add documentation."
        self.model = self._load(DETECTOR_BASE, DETECTOR_WEIGHTS, DETECTOR_TOP)

    def predict(self, image, confidence_threshold=0.3):
        "Add documentation."
        image = image_to_gpu_tensor(image)
        with torch.no_grad():
            # for some reason it's a list with one element
            detections = self.model(image)[0]
            detections = self._process_detections(
                detections, confidence_threshold)
        return detections

    def _process_detections(self, detections, confidence_threshold):
        "Add documentation."
        # move to cpu
        detections = self._detections_to_cpu(detections)
        # keep the ones above some (potential) threshold
        selections = np.array(detections["scores"] > confidence_threshold)
        detections = self._select_detections(detections, selections)
        return detections

    def _detections_to_cpu(self, detections):
        "Moves all the fields of `detections` to the CPU."
        detections["boxes"] = detections["boxes"].cpu().numpy()
        detections["labels"] = detections["labels"].cpu().numpy()
        detections["scores"] = detections["scores"].cpu().numpy()
        if "masks" in detections:
            detections["masks"] = detections["masks"].cpu().numpy()
        return detections
