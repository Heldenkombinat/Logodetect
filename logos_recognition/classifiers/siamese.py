"Classifier module."

# Standard library:
import os
import sys

# Pip packages:
import numpy as np
from PIL import Image
import torch

# Current library:
from logos_recognition import classifiers
from logos_recognition.augmenters.outdoors import get_augmentations
from logos_recognition.utils import clean_name, open_and_resize, image_to_gpu_tensor
from logos_recognition.constants import (
    CLASSIFIER_ALG,
    CLASSIFIER_DEVICE,
    CLASSIFIER_WEIGHTS,
    IMAGE_RESIZE,
    AUGMENTER_PARAMS,
    MIN_CONFIDENCE,
)


class Classifier:
    "Add documentation."

    def __init__(self, exemplar_paths):
        "Add documentation."
        # Define class variables:
        self.exemplars_imgs = None
        self.exemplars_brands = None

        # Set the network to classify the detections:
        self.load_exemplars(exemplar_paths)
        self.classifier = classifiers.__dict__[CLASSIFIER_ALG](
            CLASSIFIER_DEVICE, CLASSIFIER_WEIGHTS
        )

    def load_exemplars(self, exemplars_paths):
        "Add documentation."
        self.exemplars_imgs = []
        self.exemplars_brands = []
        for path in exemplars_paths:
            brand = clean_name(path)
            image = open_and_resize(path, IMAGE_RESIZE)

            # Store clean image:
            image_gpu = image_to_gpu_tensor(image, CLASSIFIER_DEVICE)
            self.exemplars_imgs.append(image_gpu)
            self.exemplars_brands.append(brand)

            # Store augmented image:
            for aug_image in get_augmentations(image):
                aug_image_gpu = image_to_gpu_tensor(aug_image, CLASSIFIER_DEVICE)
                self.exemplars_imgs.append((aug_image_gpu))
                self.exemplars_brands.append(brand)

    def predict(self, detections, image):
        "Add documentation."
        if len(detections["boxes"]) != 0:
            image = Image.fromarray(image)
            # Compare each detection to each exemplar in one forward pass:
            comb_images = self._create_comb_images(image, detections)
            comb_scores = self.classifier(comb_images).detach().cpu().numpy()
            detections = self._process_scores(comb_scores, detections)
        return detections

    def _create_comb_images(self, image, detections):
        """
        comb_images: Is a batch storing all combinations
                     of detections and exemplars and will have a size of
                     (n_detections * n_exemplars, 6, H, W)
        """
        comb_images = []
        for box in detections["boxes"]:
            # Extract the detection from the image:
            crop = image.crop(box).resize(IMAGE_RESIZE)
            detection = image_to_gpu_tensor(crop, CLASSIFIER_DEVICE)

            for exemplar in self.exemplars_imgs:
                # Concatenate with the exemplar:
                comb_images.append(torch.cat((detection, exemplar), 1))

        return torch.cat(comb_images).to(CLASSIFIER_DEVICE)

    def _process_scores(self, comb_scores, detections):
        "Add documentation."
        n_detections = len(detections["boxes"])
        n_logos = len(self.exemplars_imgs)
        selections = np.zeros(n_detections).astype(np.bool)

        for n_det in range(n_detections):
            a = n_det * n_logos
            b = a + n_logos
            scores = comb_scores[a:b]

            if (scores >= MIN_CONFIDENCE).any():
                detections["labels"][n_det] = np.argmax(scores)
                detections["scores"][n_det] = np.max(scores)
                selections[n_det] = True

        brands = [self.exemplars_brands[idx] for idx in detections["labels"]]
        detections["brands"] = np.array(brands)
        return self._select_detections(detections, selections)

    def _select_detections(self, detections, selections):
        "Add documentation."
        detections["boxes"] = detections["boxes"][selections]
        detections["labels"] = detections["labels"][selections]
        detections["scores"] = detections["scores"][selections]
        detections["brands"] = detections["brands"][selections]
        return detections
