"Classifier module."

# Standard library:
import sys

# Pip packages:
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as nn_F
import torchvision

from logos_recognition import classifiers
from logos_recognition.augmenters.outdoors import get_augmentations
from logos_recognition.utils import (
    open_and_resize,
    clean_name,
)
from logos_recognition.constants import (
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
    "Add documentation."

    def __init__(self, exemplar_paths):
        "Add documentation."
        # Define class variables:
        self.exemplars_vectors = None
        self.exemplars_brands = None

        # Set image transforms for inference:
        self.transform = self.set_transform()
        # Set the network to perform image embeddings:
        self.representer = classifiers.get(EMBEDDER_ALG)(
            EMBEDDER_DEVICE, EMBEDDER_WEIGHTS
        )

        # Set the network to classify the detections:
        self.load_exemplars(exemplar_paths)
        self.classifier = self.set_classifier()

    @staticmethod
    def set_transform():
        "Add documentation."
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(IMAGE_RESIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def load_exemplars(self, exemplars_paths):
        "Add documentation."
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

    def set_classifier(self):
        "Add documentation."
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
        # TODO: code smell here, setting a classifier does not suggest training
        return model.fit(self.exemplars_vectors, self.exemplars_brands)

    def predict(self, detections, image):
        "Add documentation."
        if len(detections["boxes"]) != 0:
            image = Image.fromarray(image)
            detections = self.load_detections(image, detections)
            detections = self.classify_embeddings(detections)
        return detections

    def load_detections(self, image, detections):
        "Add documentation."
        detections_mat = np.zeros((len(detections["boxes"]), EMBEDDING_SIZE))
        for idx, box in enumerate(detections["boxes"]):
            # Extract the detection from the image:
            image = image.crop(box).resize(IMAGE_RESIZE)
            embedding = self.embed_image(image)
            detections_mat[idx, :] = embedding
        detections["embeddings"] = detections_mat
        return detections

    def embed_image(self, image):
        "Add documentation."
        image = (
            self.transform(image).unsqueeze(0).to(EMBEDDER_DEVICE, dtype=torch.float)
        )
        embedding = self.representer(image)
        embedding = nn_F.normalize(embedding, p=2, dim=1)
        return embedding.squeeze().detach().cpu().numpy()

    def classify_embeddings(self, detections):
        "Add documentation."
        dists, classes = self.classifier.kneighbors(detections["embeddings"])
        brands = [self.exemplars_brands[idx] for idx in classes.flatten()]
        detections["brands"] = np.array(brands)
        selections = (dists < MAX_DISTANCE).flatten()
        return self._select_detections(detections, selections)

    def _select_detections(self, detections, selections):
        "Add documentation."
        detections["boxes"] = detections["boxes"][selections]
        detections["labels"] = detections["labels"][selections]
        detections["scores"] = detections["scores"][selections]
        detections["brands"] = detections["brands"][selections]
        return detections
