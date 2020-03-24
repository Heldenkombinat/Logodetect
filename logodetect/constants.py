"""Constants for logos recognition.
"""

import os


PATH_GLOB = os.environ["LOGOS_RECOGNITION"]
PATH_DATA = os.path.join(PATH_GLOB, "data")
PATH_VIDEO = os.path.join(PATH_DATA, "test_videos")
PATH_IMAGE = os.path.join(PATH_DATA, "test_images")
PATH_MODELS = os.path.join(PATH_GLOB, "models")

IMAGE_RESIZE = (100, 100)
MIN_CONFIDENCE = 0.9

###############
# Input video #
###############

SAMPLE_VIDEOS = [
    "test_video.mp4",
    "test_video_small.mp4",
    "test_video_small_50ms.mp4",
    "football_redbull_small.mp4",
]
SAMPLE_BRANDS = [
    ["pepsi", "redbull", "heineken", "stellaartois"],
    ["pepsi"],
    ["redbull"],
]
CHOICE = 2

VIDEO_FILENAME = os.path.join(PATH_VIDEO, SAMPLE_VIDEOS[CHOICE])
IMAGE_FILENAME = os.path.join(PATH_IMAGE, "test_image_small.png")
BRAND_LOGOS = SAMPLE_BRANDS[CHOICE]

EXEMPLARS = ["exemplars", "exemplars_100x100", "exemplars_100x100_aug", "exemplars_hq"]
PATH_EXEMPLARS = os.path.join(PATH_DATA, EXEMPLARS[3])

############
# Detector #
############

DETECTOR = "detectors.faster_rcnn"
DETECTOR_ALG = "binary_fasterrcnn_resnet50"
DETECTOR_WEIGHTS = os.path.join(PATH_GLOB, "models", "detector.pth")

##############
# Classifier #
##############

USE_CLASSIFIER = True

PATH_EXEMPLARS_EMBEDDINGS = os.path.join(PATH_DATA, "exemplars_siamese.zip")
LOAD_EMBEDDINGS = False
EXEMPLARS_FORMAT = "jpg"

# Keep embedder algorithm and weights as is
EMBEDDER_ALG = "siamese_resnet18"
EMBEDDER_WEIGHTS = os.path.join(PATH_MODELS, "embedder.pth")
EMBEDDER_IMG_SIZE = 100

# choose between "knn" and "siamese"
CLASSIFIER = "knn"
CLASSIFIER_ALG = "knn" if CLASSIFIER == "knn" else "binary_stacked_resnet18"

# TODO: currently only resnet18 seems to work, resnet50 throws a shape error
CLASSIFIER_WEIGHTS = os.path.join(PATH_MODELS, "classifier_resnet18.pth")

EMBEDDING_SIZE = 345
DISTANCE = "cosine"  # {cosine, minkowski_1, minkowski_2}
MAX_DISTANCE = 0.010  # {siamese-cosine: 0.013}

#############
# Augmenter #
#############

AUGMENTER_PARAMS = {
    "Multiply": [0.5, 1.5],  # mu
    "GaussianBlur": [0.4],  # sigma
    "AdditiveGaussianNoise": [0.2 * 255],  # scale
    "AffineShear": [-25, 25],  # shear
    "AffineRotate": [-25, 25],  # rotate
}


# Device management
DEVICE = "cpu"  # 'cuda:1' etc.
EMBEDDER_DEVICE = DEVICE
DETECTOR_DEVICE = DEVICE
CLASSIFIER_DEVICE = DEVICE
