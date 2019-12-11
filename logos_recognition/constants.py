"Constants for logos recognition."

# Standard library:
import os

# Pip packages:
import torch


##########
# Global #
##########

PATH_GLOB = os.path.join(os.environ['HKT'], 'Logos-Recognition')
PATH_DATA = os.path.join(os.environ['DATASETS'], 'logos')
VIDEO_FILENAME = os.path.join(PATH_GLOB, 'data', 'test_video_small.mp4')
PATH_EXEMPLARS = os.path.join(PATH_DATA, 'exemplars')
# BRAND_LOGOS = ['pepsi', 'redbull', 'heineken', 'stellaartois']
BRAND_LOGOS = ['redbull']
IMAGE_RESIZE = (100, 100)
USE_CLASSIFIER = True

############
# Detector #
############

DETECTOR = 'detectors.faster_rcnn'
DETECTOR_ALG = 'binary_fasterrcnn_resnet50'
DETECTOR_WEIGHTS = os.path.join(PATH_GLOB, 'models', 'detector.pth')
DETECTOR_DEVICE = 'cuda:1'
MIN_CONFIDENCE = 0.9

##############
# Classifier #
##############

REPRESENTER_ALG = 'siamese_resnet18'
REPRESENTER_WEIGHTS = os.path.join(PATH_GLOB, 'models', 'siamese_embedding.pth')
REPRESENTER_DEVICE = 'cuda:2'
REPRESENTER_IMG_SIZE = 100

CLASSIFIER = 'classifiers.knn'
CLASSIFIER_ALG = 'KNeighborsClassifier'
CLASSIFIER_WEIGHTS = os.path.join(PATH_GLOB, 'models', 'classifier.pth')
CLASSIFIER_DEVICE = 'cuda:3'

PATH_EXEMPLARS_EMBEDDINGS = os.path.join(PATH_DATA, 'exemplars_siamese.zip')
LOAD_EMBEDDINGS = True
EMBEDDING_SIZE = 345
DISTANCE = 'cosine'  # {cosine, minkowski_1, minkowski_2}
MAX_DISTANCE = 0.015  # {siamese-cosine: 0.013}
