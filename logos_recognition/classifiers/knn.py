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

# Current library:
from logos_recognition import classifiers
from logos_recognition.constants import (CLASSIFIER_ALG, PATH_EXEMPLARS_EMBEDDINGS,
                                         REPRESENTER_ALG, REPRESENTER_WEIGHTS,
                                         REPRESENTER_DEVICE, BRAND_LOGOS, IMAGE_RESIZE,
                                         EMBEDDING_SIZE, DISTANCE, MAX_DISTANCE)



class Classifier():
    "Add documentation."

    def __init__(self):
        "Add documentation."
        self.transform = self.set_transform()
        
        self.representer = classifiers.__dict__[
            REPRESENTER_ALG](REPRESENTER_DEVICE, REPRESENTER_WEIGHTS)

        exemplars = pd.read_pickle(PATH_EXEMPLARS_EMBEDDINGS)
        mask = exemplars['brand'].isin(BRAND_LOGOS)
        self.exemplars_vecs = list(exemplars['img_vec'][mask])
        self.exemplars_brands = list(exemplars['brand'][mask])

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

    def predict(self, detections, image, query_logos):
        "Add documentation."
        if len(detections['boxes']) != 0:
            image = Image.fromarray(image)
            detections = self.embed_detections(image, detections)
            return self.classify_embeddings(detections)
        else:
            return detections

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

    def _select_detections(self, detections, selections):
        "Add documentation."
        detections['boxes'] = detections['boxes'][selections]
        detections['labels'] = detections['labels'][selections]
        detections['scores'] = detections['scores'][selections]
        detections['brands'] = detections['brands'][selections]
        return detections
