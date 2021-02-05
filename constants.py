"""Configuration file for `logodetect`.

Note: leave the base configuration untouched, unless you know what you're doing.
The parameters that can be tuned concern the
- Detector (first phase)
- Classifier (second phase)
- Embeddings used
- Data augmentation
- and the devices you want to use (CPU, GPU, etc.)
"""

import os

# Base configuration start
if "LOGOS_RECOGNITION" in os.environ:
    PATH_GLOB = os.environ["LOGOS_RECOGNITION"]
else:
    PATH_GLOB = os.path.expanduser(
        os.path.join(os.environ["HOME"], ".hkt", "logodetect")
    )

PATH_MODELS = os.path.join(PATH_GLOB, "models")
PATH_DATA = os.path.join(PATH_GLOB, "data")
PATH_VIDEO = os.path.join(PATH_DATA, "test_videos")
PATH_IMAGE = os.path.join(PATH_DATA, "test_images")
PATH_EXEMPLARS = os.path.join(PATH_DATA, "exemplars")

SAMPLE_VIDEOS = [
    "test_video_small.mp4",
    "test_video_small_50ms.mp4",
    "football_redbull_small.mp4",
]
SAMPLE_BRANDS = [
    ["pepsi", "redbull", "heineken", "stellaartois"],
    ["pepsi"],
    ["redbull"],
]
CHOICE = 1

VIDEO_FILENAME = os.path.join(PATH_VIDEO, SAMPLE_VIDEOS[CHOICE])
IMAGE_FILENAME = os.path.join(PATH_IMAGE, "test_image_small.png")
BRAND_LOGOS = SAMPLE_BRANDS[CHOICE]

TEST_VIDEO = os.path.join(PATH_VIDEO, "test_video_small_50ms.mp4")
TEST_IMAGE = os.path.join(PATH_IMAGE, "test_image_small.png")
# Base configuration end


"""Detector: first phase of logodetect (object detection)

We provide an architecture and weights for Faster-RCNN here, but
you can provide your own models by changing the settings below.
"""
DETECTOR = "detectors.faster_rcnn"
DETECTOR_ALG = "binary_fasterrcnn_resnet50"
DETECTOR_WEIGHTS = os.path.join(PATH_GLOB, "models", "detector.pth")


"""Classifier: first phase of logodetect (object recognition)

We provide two classifiers, namely simple distance measure
(cosine or Eudlidean metrics) via a KNN classifier, or a more
sophisticated approach with Siamese networks. To bring your own
algorithm, modify the weights path accordingly (and potentially
provide a new `Classifier` implementation).
"""
# Use a classifier by default, not recommended to turn off
USE_CLASSIFIER = True

CLASSIFIER = "knn"  # choose between "knn" and "siamese"
CLASSIFIER_ALG = "knn" if CLASSIFIER == "knn" else "binary_stacked_resnet18"
CLASSIFIER_WEIGHTS = os.path.join(PATH_MODELS, "classifier_resnet18.pth")

# Distance measure used. Choose from: cosine, minkowski_1, or minkowski_2
DISTANCE = "cosine"

# How far apart are detections and exemplars allowed to be to count as recognized?
MAX_DISTANCE = 0.010

###############
# Embeddings  #
###############

# Load pre-computed embeddings, if you have them available
PATH_EXEMPLARS_EMBEDDINGS = os.path.join(PATH_DATA, "exemplars_siamese.pkl")
LOAD_EMBEDDINGS = False

EMBEDDING_SIZE = 345  # Embedding vector length

# Change this to use your own embedding algorithm
EMBEDDER_ALG = "siamese_resnet18"
EMBEDDER_WEIGHTS = os.path.join(PATH_MODELS, "embedder.pth")
EMBEDDER_IMG_SIZE = 100  # Pixel height or width of a square image


#########################
# Data and augmentation #
#########################

# Which file format do the exemplars have
EXEMPLARS_FORMAT = "jpg"

# Resize exemplars to this size
IMAGE_RESIZE = (100, 100)

# Detection confidence, e.g. 0.9 means that only objects p>=0.9 get detected for the next phase
MIN_CONFIDENCE = 0.9

# Exemplar augmentation parameters. See `logodetect/augmenters.py` for more examples.
AUGMENTER_PARAMS = {
    "Multiply": [0.5, 1.5],  # mu
    "GaussianBlur": [0.4],  # sigma
    "AdditiveGaussianNoise": [0.2 * 255],  # scale
    "AffineShear": [-25, 25],  # shear
    "AffineRotate": [-25, 25],  # rotate
}

#########################
# Device management     #
#########################

DEVICE = "cpu"  # {cpu, cuda:1, cuda:2, ...}
EMBEDDER_DEVICE = DEVICE
DETECTOR_DEVICE = DEVICE
CLASSIFIER_DEVICE = DEVICE
