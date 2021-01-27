"""Detector module."""

# Standard library:
import os

# Pip packages:
import numpy as np
import torch

# Current library:
from logodetect import detectors
from logodetect.utils import image_to_gpu_tensor
from constants import (
    DETECTOR_DEVICE,
    DETECTOR_ALG,
    DETECTOR_WEIGHTS,
    MIN_CONFIDENCE,
)


class Detector:
    def __init__(self):
        model = detectors.get(DETECTOR_ALG)
        self.model = model(DETECTOR_DEVICE, DETECTOR_WEIGHTS)

    @torch.no_grad()
    def predict(self, image):
        image = image_to_gpu_tensor(image, DETECTOR_DEVICE)
        # Always returns list with one element:
        detections = self.model(image)[0]
        return self._process_detections(detections)

    def _process_detections(self, detections):
        # Move to cpu:
        detections = self._detections_to_cpu(detections)
        # keep the ones above some (potential) threshold
        selections = np.array(detections["scores"] > MIN_CONFIDENCE)
        return self._select_detections(detections, selections)

    def _detections_to_cpu(self, detections):
        """Moves all the fields of 'detections' to the CPU."""
        detections["boxes"] = detections["boxes"].cpu().numpy()
        detections["labels"] = detections["labels"].cpu().numpy()
        detections["scores"] = detections["scores"].cpu().numpy()
        return detections

    def _select_detections(self, detections, selections):
        detections["boxes"] = detections["boxes"][selections]
        detections["labels"] = detections["labels"][selections]
        detections["scores"] = detections["scores"][selections]
        detections["brands"] = []
        return detections
