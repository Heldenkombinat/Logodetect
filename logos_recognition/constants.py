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

# VIDEO_FILENAME = os.path.join(PATH_GLOB, 'data', 'test_video.mp4')
# BRAND_LOGOS = ['pepsi', 'redbull', 'heineken', 'stellaartois']

# VIDEO_FILENAME = os.path.join(PATH_GLOB, 'data', 'test_video_small.mp4')
# BRAND_LOGOS = ['pepsi']

VIDEO_FILENAME = os.path.join(PATH_GLOB, 'data', 'football_redbull_small.mp4')
BRAND_LOGOS = ['redbull']

PATH_EXEMPLARS_EMBEDDINGS = os.path.join(PATH_DATA, 'exemplars_siamese.zip')
LOAD_EMBEDDINGS = False

# PATH_EXEMPLARS = os.path.join(PATH_DATA, 'exemplars')
PATH_EXEMPLARS = os.path.join(PATH_DATA, 'exemplars_hq')

EXEMPLARS_FORMAT = 'jpg'
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

AUGMENTER_PARAMS = {
    'Multiply': [0.5, 1.5],
    'GaussianBlur': [0.4],
    'AdditiveGaussianNoise': [0.2*255],
    'AffineShear': [-25, 25],
    'AffineRotate': [-25, 25],
}
REPRESENTER_ALG = 'siamese_resnet18'
REPRESENTER_WEIGHTS = os.path.join(PATH_GLOB, 'models', 'siamese_embedding.pth')
REPRESENTER_DEVICE = 'cuda:2'
REPRESENTER_IMG_SIZE = 100

CLASSIFIER = 'classifiers.knn'
CLASSIFIER_ALG = 'KNeighborsClassifier'
CLASSIFIER_WEIGHTS = os.path.join(PATH_GLOB, 'models', 'classifier.pth')
CLASSIFIER_DEVICE = 'cuda:3'

EMBEDDING_SIZE = 345
DISTANCE = 'cosine'  # {cosine, minkowski_1, minkowski_2}
MAX_DISTANCE = 0.010  # {siamese-cosine: 0.013}
