import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional
import torchvision

from logodetect import classifiers
from logodetect.augmenters import get_augmentations
from logodetect.utils import (
    open_and_resize,
    clean_name,
)
from constants import (
    CLASSIFIER_ALG,
    PATH_EXEMPLARS_EMBEDDINGS,
    EMBEDDER_ALG,
    EMBEDDER_WEIGHTS,
    EMBEDDER_DEVICE,
    BRAND_LOGOS,
    IMAGE_RESIZE,
    LOAD_EMBEDDINGS,
    EMBEDDING_SIZE,
    DISTANCE,
    MAX_DISTANCE,
)


class Classifier:
    """KNN Classifier
    """

    def __init__(self, exemplar_paths: str) -> None:

        self.transform = self._compute_transform()
        self.embedder = classifiers.get(EMBEDDER_ALG)(EMBEDDER_DEVICE, EMBEDDER_WEIGHTS)
        self._load_exemplars(exemplar_paths)
        self.classifier = self._set_classifier()

    @staticmethod
    def _compute_transform() -> torchvision.transforms.Compose:
        """Compute image transform for inference.

        :return: torchvision transform
        """
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(IMAGE_RESIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def _load_exemplars(self, exemplars_paths: str) -> None:
        """Load exemplars from file

        :param exemplars_paths: path to exemplars folder
        :return: None
        """
        if LOAD_EMBEDDINGS:
            # Load embeddings and names of exemplars:
            exemplars = pd.read_pickle(PATH_EXEMPLARS_EMBEDDINGS)
            mask = exemplars["brand"].isin(BRAND_LOGOS)
            self.exemplars_vectors = list(exemplars["img_vec"][mask])
            self.exemplars_brands = list(exemplars["brand"][mask])
        else:
            self.exemplars_vectors = []
            self.exemplars_brands = []
            for path in exemplars_paths:
                brand = clean_name(path)
                image = open_and_resize(path, IMAGE_RESIZE)

                # Store clean image:
                embedding = self.embed_image(image)
                self.exemplars_vectors.append(embedding)
                self.exemplars_brands.append(brand)

                # Store augmented image:
                for aug_image in get_augmentations(image):
                    embedding = self.embed_image(aug_image)
                    self.exemplars_vectors.append(embedding)
                    self.exemplars_brands.append(brand)

    def _set_classifier(self):
        """Determine parameters for scikit-learn KNeighborsClassifier, fit
        data to it and return it.

        :return: trained classifier model
        """
        if CLASSIFIER_ALG is not "knn":
            raise ValueError(
                f"A classifiers.knn.Classifier can only be run with CLASSIFIER_ALG='knn', got {CLASSIFIER_ALG}."
            )
        if DISTANCE.lower() == "minkowski_1":
            model = classifiers.get(CLASSIFIER_ALG)(
                n_neighbors=1, metric="minkowski", p="1"
            )
        elif DISTANCE.lower() == "minkowski_2":
            model = classifiers.get(CLASSIFIER_ALG)(
                n_neighbors=1, metric="minkowski", p="2"
            )
        elif DISTANCE.lower() == "cosine":
            model = classifiers.get(CLASSIFIER_ALG)(n_neighbors=1, metric="cosine")
        else:
            print("{} is not a valid distance.".format(DISTANCE))
            sys.exit()
        # TODO: code smell here, setting a classifier does not suggest training (or returning anything)
        return model.fit(self.exemplars_vectors, self.exemplars_brands)

    def predict(self, detections: dict, image: np.ndarray) -> dict:
        """Classify an image given precomputed detection objects

        :param detections: detections dictionary with overlay boxes
        :param image: input image
        :return: augmented detections
        """
        if len(detections["boxes"]) != 0:
            image = Image.fromarray(image)
            detections = self.load_detections(image, detections)
            detections = self.classify_embeddings(detections)
        return detections

    def load_detections(self, image: Image.Image, detections: dict) -> dict:
        """Load all detections specific to the given input image

        :param image: input image
        :param detections: precomputed detections
        :return: detections
        """
        detections_mat = np.zeros((len(detections["boxes"]), EMBEDDING_SIZE))
        for idx, box in enumerate(detections["boxes"]):
            crop_image = image.crop(box).resize(IMAGE_RESIZE)
            embedding = self.embed_image(crop_image)
            detections_mat[idx, :] = embedding
        detections["embeddings"] = detections_mat
        return detections

    def embed_image(self, image):
        """Embeds an image using the embedding algorithm

        :param image: input image
        :return: normalized embedding
        """
        image = (
            self.transform(image).unsqueeze(0).to(EMBEDDER_DEVICE, dtype=torch.float)
        )
        embedding = self.embedder(image)
        normalized_embedding = functional.normalize(embedding, p=2, dim=1)
        return normalized_embedding.squeeze().detach().cpu().numpy()

    def classify_embeddings(self, detections: dict) -> dict:
        """Use the classifier algorithm to predict classes of
        the given detections.

        :param detections:
        :return:
        """
        dists, classes = self.classifier.kneighbors(detections["embeddings"])
        brands = [self.exemplars_brands[idx] for idx in classes.flatten()]
        detections["brands"] = np.array(brands)
        selections = (dists < MAX_DISTANCE).flatten()
        return self._select_detections(detections, selections)

    @staticmethod
    def _select_detections(detections, selections):
        """Specify detections for the given selections.

        :param detections:
        :param selections:
        :return:
        """
        detections["boxes"] = detections["boxes"][selections]
        detections["labels"] = detections["labels"][selections]
        detections["scores"] = detections["scores"][selections]
        detections["brands"] = detections["brands"][selections]
        return detections
