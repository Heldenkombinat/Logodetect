"Detector module."

# Standard library:
import os
import sys

# Pip packages:
import numpy as np
from PIL import Image
import torch

# Github repos:
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import torchvision
from torch.nn import functional as nn_F

# Logos-Recognition:
from logos_recognition.classifiers import stacked_resnet18, siamese_resnet18
from logos_recognition.super_torch import SuperTorch
from logos_recognition.utils import image_to_gpu_tensor
from logos_recognition.constants import (CLASSIFIER_WEIGHTS, EXEMPLARS_PATH, DEVICE,
                                         IMAGE_RESIZE, REPRESENTER_WEIGHTS,
                                         REPRESENTER_DEVICE, IMG_SIZE, BRAND_LOGOS,
                                         EMBEDDING_SIZE, DISTANCE, MAX_DISTANCE)


class Classifier(SuperTorch):
    "Add documentation."

    def __init__(self):
        "Add documentation."
        CLASSIFIER_BASE = stacked_resnet18
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
        for i, box in enumerate(detections['boxes']):
            # extract the detection from the image and convert
            detection = image_to_gpu_tensor(
                image.crop(box).resize(IMAGE_RESIZE))
            for j, exemplar in enumerate(query_logos):
                # now concatenate with the exemplar
                comb_input.append(torch.cat((detection, exemplar), 1))
                comb_info['indices'].append([i, j])
        comb_input = torch.cat(comb_input).to(DEVICE)
        return comb_input, comb_info

    def _process_output(self, output, detections, query_logos):
        "Add documentation."
        # process the output
        n_logos = len(query_logos)
        n_detections = len(detections['boxes'])
        scores = self._create_output_mat(output, n_detections, n_logos)
        # go through each detection
        selection = np.ones(n_detections).astype(np.bool)
        for i, row in enumerate(scores):
            if np.sum(np.round(row)) >= 1:
                # we have at least a match
                detections['labels'][i] = np.argmax(row)
                detections['scores'][i] = np.max(row)
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


class KNNClassifier(SuperTorch):
    "Add documentation."

    def __init__(self):
        "Add documentation."
        self.transform = self.set_transform()
        
        self.representer = siamese_resnet18(
            REPRESENTER_WEIGHTS, REPRESENTER_DEVICE)
        
        exemplars = pd.read_pickle(EXEMPLARS_PATH)
        mask = exemplars['brand'].isin(BRAND_LOGOS)
        self.exemplars_vecs = list(exemplars['img_vec'][mask])
        self.exemplars_brands = list(exemplars['brand'][mask])

        self.classifier = self.set_classifier()
        
    def predict(self, detections, image, query_logos):
        "Add documentation."
        image = Image.fromarray(image)
        # we compare each detection to each exemplar but do only one forward pass
        detections = self.embed_detections(image, detections)
        return self.classify_embeddings(detections)

    def embed_detections(self, image, detections):
        "Add documentation."
        detections_mat = np.zeros((len(detections['boxes']), EMBEDDING_SIZE))
        for idx, box in enumerate(detections['boxes']):
            # Extract the detection from the image:
            crop = image.crop(box).resize(IMAGE_RESIZE)
            crop = self.transform(crop).unsqueeze(0).to(
                REPRESENTER_DEVICE, dtype=torch.float)
            features = self.representer(crop)
            features = nn_F.normalize(features, p=2, dim=1)
            features =  features.squeeze().detach().cpu().numpy()
            detections_mat[idx, :] = features
        detections['embeddings'] = detections_mat
        return detections
    
    def classify_embeddings(self, detections):
        "Add documentation."
        dists, classes = self.classifier.kneighbors(detections['embeddings'])
        brands = [self.exemplars_brands[idx] for idx in classes.flatten()]
        detections['brands'] = np.array(brands)
        selections = (dists < MAX_DISTANCE).flatten()
        return self._select_detections(detections, selections)

    def set_classifier(self):
        "Add documentation."
        if DISTANCE.lower() == 'minkowski_1':
            model = KNeighborsClassifier(
                n_neighbors=1, metric='minkowski', p='1')
        elif DISTANCE.lower() == 'minkowski_2':
            model = KNeighborsClassifier(
                n_neighbors=1, metric='minkowski', p='2')
        elif DISTANCE.lower() == 'cosine':
            model = KNeighborsClassifier(
                n_neighbors=1, metric='cosine')
        else:
            print('{} is not a usable distance.'.format(DISTANCE))
            sys.exit()
        return model.fit(self.exemplars_vecs, self.exemplars_brands)

    def set_transform(self):
        "Add documentation."
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010))])
