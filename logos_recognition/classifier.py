"Detector module."

# Standard library:
import os

# Pip packages:
import numpy as np
from PIL import Image
import torch

# Github repos:

# Logos-Recognition:
from logos_recognition.super_torch import SuperTorch
from logos_recognition.utils import image_to_gpu_tensor
from logos_recognition.constants import (CLASSIFIER_BASE, CLASSIFIER_WEIGHTS,
                                         DEVICE, IMAGE_RESIZE)


class Classifier(SuperTorch):
    "Add documentation."

    def __init__(self):
        "Add documentation."
        self.model = self._load(CLASSIFIER_BASE, CLASSIFIER_WEIGHTS)

    def predict(self, detections, image, query_logos):
        "Add documentation."
        image = Image.fromarray(image)
        # we compare each detection to each exemplar but do only one forward pass
        comb_input, comb_info = self._create_comb_input(
            image, detections, query_logos)
        output = self.model(comb_input).detach().cpu().numpy()
        detections = self._process_output(output, detections, query_logos)
        return detections

    def _create_comb_input(self, image, detections, query_logos):
        "Add documentation."
        # comb_input will be a batch storing all combinations
        # of detections and exemplars and will have a size of
        # (n_detections * n_exemplars, 6, H, W)
        comb_input = []
        # comb_info will store the combination indices so we know what's what
        comb_info = {"order": "Detection-Exemplar", "indices": []}
        for i, box in enumerate(detections["boxes"]):
            for j, exemplar in enumerate(query_logos):
                # extract the detection from the image and convert
                detection = image_to_gpu_tensor(
                    image.crop(box).resize(IMAGE_RESIZE))
                # now concatenate with the exemplar
                comb_input.append(torch.cat((detection, exemplar), 1))
                comb_info["indices"].append([i, j])
        comb_input = torch.cat(comb_input).to(DEVICE)
        return comb_input, comb_info

    def _process_output(self, output, detections, query_logos):
        "Add documentation."
        # process the output
        n_logos = len(query_logos)
        n_detections = len(detections["boxes"])
        scores = self._create_output_mat(output, n_detections, n_logos)
        # go through each detection
        selection = np.ones(n_detections).astype(np.bool)
        for i, row in enumerate(scores):
            if np.sum(np.round(row)) >= 1:
                # we have at least a match
                detections["labels"][i] = np.argmax(row)
                detections["scores"][i] = np.max(row)
            else:
                # no match
                selection[i] = False
        detections = self._select_detections(detections, selection)
        return detections

    def _create_output_mat(self, output, n_detections, n_logos):
        "Add documentation."
        scores = np.zeros((n_detections, n_logos))
        idx = 0
        for i in range(n_detections):
            for j in range(n_logos):
                # extract the detection from the image and convert
                scores[i, j] = output[idx]
                idx += 1
        return scores
