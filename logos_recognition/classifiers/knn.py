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

# Github packages:
import imgaug.augmenters as ia

# Current library:
from logos_recognition import classifiers
from logos_recognition.utils import (open_resize_and_load_gpu, open_and_resize,
                                     get_class_name)
from logos_recognition.constants import (CLASSIFIER_ALG, PATH_EXEMPLARS_EMBEDDINGS,
                                         REPRESENTER_ALG, REPRESENTER_WEIGHTS,
                                         REPRESENTER_DEVICE, BRAND_LOGOS, IMAGE_RESIZE,
                                         LOAD_EMBEDDINGS, EMBEDDING_SIZE,
                                         AUGMENTER_PARAMS, DISTANCE, MAX_DISTANCE)



class Classifier():
    "Add documentation."

    def __init__(self, exemplar_paths):
        "Add documentation."
        # Define class variables:
        self.exemplars_vecs = None
        self.exemplars_brands = None
        
        # Set image transforms for inference:
        self.transform = self.set_transform()
        # Set the network to perform image embeddings:
        self.representer = classifiers.__dict__[
            REPRESENTER_ALG](REPRESENTER_DEVICE, REPRESENTER_WEIGHTS)
        # Set the network to classify the detections:
        self.load_exemplars(exemplar_paths)
        self.classifier = self.set_classifier()
        
    def set_transform(self):
        "Add documentation."
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(IMAGE_RESIZE),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010))])

    def set_classifier(self):
        "Add documentation."
        if DISTANCE.lower() == 'minkowski_1':
            model = classifiers.__dict__[
                CLASSIFIER_ALG](n_neighbors=1, metric='minkowski', p='1')
        elif DISTANCE.lower() == 'minkowski_2':
            model = classifiers.__dict__[
                CLASSIFIER_ALG](n_neighbors=1, metric='minkowski', p='2')
        elif DISTANCE.lower() == 'cosine':
            model = classifiers.__dict__[
                CLASSIFIER_ALG](n_neighbors=1, metric='cosine')
        else:
            print('{} is not a valid distance.'.format(DISTANCE))
            sys.exit()
        return model.fit(self.exemplars_vecs, self.exemplars_brands)

    def predict(self, detections, image):
        "Add documentation."
        if len(detections['boxes']) != 0:
            image = Image.fromarray(image)
            detections = self.load_detections(image, detections)
            return self.classify_embeddings(detections)
        else:
            return detections

    def load_exemplars(self, exemplars_paths):
        "Add documentation."
        if LOAD_EMBEDDINGS:
            # Load embeddings and names of exemplars:
            exemplars = pd.read_pickle(PATH_EXEMPLARS_EMBEDDINGS)
            mask = exemplars['brand'].isin(BRAND_LOGOS)
            self.exemplars_vecs = list(exemplars['img_vec'][mask])
            self.exemplars_brands = list(exemplars['brand'][mask])
        else:
            self.exemplars_vecs = []
            self.exemplars_brands = []
            for path in exemplars_paths:
                brand = get_class_name(path)
                image = open_and_resize(path, IMAGE_RESIZE)
                for aug_image in self.get_augmentations(image):
                    embedding = self.embed_image(aug_image)
                    self.exemplars_vecs.append(embedding)
                    self.exemplars_brands.append(brand)

    def get_augmentations(self, image):
        "Add documentation."
        augmented_images = []
        # For each combination:
        for mu in AUGMENTER_PARAMS['Multiply']:
            for gabl in AUGMENTER_PARAMS['GaussianBlur']:
                for adga in AUGMENTER_PARAMS['AdditiveGaussianNoise']:
                    for afsh in AUGMENTER_PARAMS['AffineShear']:
                        for afro in AUGMENTER_PARAMS['AffineRotate']:
                            # Process image:
                            image_aug = self.augment_image(
                                image, mu, gabl, adga, afsh, afro)
                            augmented_images.append(image_aug)
        return augmented_images

    def augment_image(self, image, mu, gabl, adga, afsh, afro):
        "Add documentation."
        augmenter = ia.Sequential([
            ia.Multiply(mul=mu),
            ia.GaussianBlur(sigma=gabl),
            ia.AdditiveGaussianNoise(scale=adga),
            ia.Affine(rotate=afro, shear=afsh),
        ])
        # Process image:
        image_arr = augmenter.augment_images(np.array(image))
        return Image.fromarray(image_arr)

    def load_detections(self, image, detections):
        "Add documentation."
        detections_mat = np.zeros((len(detections['boxes']), EMBEDDING_SIZE))
        for idx, box in enumerate(detections['boxes']):
            # Extract the detection from the image:
            image = image.crop(box).resize(IMAGE_RESIZE)
            embedding = self.embed_image(image)
            detections_mat[idx, :] = embedding
        detections['embeddings'] = detections_mat
        return detections

    def embed_image(self, image):
        "Add documentation."
        image = self.transform(image).unsqueeze(0).to(
            REPRESENTER_DEVICE, dtype=torch.float)
        embedding = self.representer(image)
        embedding = nn_F.normalize(embedding, p=2, dim=1)
        return embedding.squeeze().detach().cpu().numpy()

    def classify_embeddings(self, detections):
        "Add documentation."
        dists, classes = self.classifier.kneighbors(detections['embeddings'])
        brands = [self.exemplars_brands[idx] for idx in classes.flatten()]
        detections['brands'] = np.array(brands)
        selections = (dists < MAX_DISTANCE).flatten()
        return self._select_detections(detections, selections)

    def _select_detections(self, detections, selections):
        "Add documentation."
        detections['boxes'] = detections['boxes'][selections]
        detections['labels'] = detections['labels'][selections]
        detections['scores'] = detections['scores'][selections]
        detections['brands'] = detections['brands'][selections]
        return detections
