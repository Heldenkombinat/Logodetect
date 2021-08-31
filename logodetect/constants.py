"""Configuration file for `logodetect`.

Note: leave this configuration untouched, unless you know what you're doing.
The parameters that can be tuned in the "get_recognizer_config" function below
all concern the
- Detector (first phase)
- Classifier (second phase)
- Data augmentation
- and the devices you want to use (CPU, GPU, etc.)
"""

import os

# Base configuration start
if "LOGOS_RECOGNITION" in os.environ:
    PATH_GLOB = os.environ["LOGOS_RECOGNITION"]
else:
    PATH_GLOB = os.path.expanduser(os.path.join("~", ".hkt", "logodetect"))

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


def get_recognizer_config(config: dict = None) -> dict:
    """If no config is provided, the default values are chosen,
    otherwise get overwritten.

    :param config: A config dictionary with keys a subset of the below base_config
    :return: the updated (or base) config dictionary
    """

    base_config = {
        "DETECTOR": "detectors.faster_rcnn",
        "DETECTOR_ALG": "binary_fasterrcnn_resnet50",
        "DETECTOR_WEIGHTS": os.path.join(PATH_GLOB, "models", "detector.pth"),
        "USE_CLASSIFIER": True,
        "CLASSIFIER": "knn",  # choose between "knn" and "siamese"
        "CLASSIFIER_WEIGHTS": os.path.join(PATH_MODELS, "classifier_resnet18.pth"),
        # Distance measure used. Choose from: cosine, minkowski_1, or minkowski_2
        "DISTANCE": "cosine",
        # How far apart are detections and exemplars allowed to be to count as recognized?
        "MAX_DISTANCE": 0.010,
        "PATH_EXEMPLARS_EMBEDDINGS": os.path.join(PATH_DATA, "exemplars_siamese.pkl"),
        "LOAD_EMBEDDINGS": False,
        "EMBEDDING_SIZE": 345,  # Embedding vector length
        # Change this to use your own embedding algorithm
        "EMBEDDER_ALG": "siamese_resnet18",
        "EMBEDDER_WEIGHTS": os.path.join(PATH_MODELS, "embedder.pth"),
        "EMBEDDER_IMG_SIZE": 100,  # Pixel height or width of a square image
        # Which file format do the exemplars have
        "EXEMPLARS_FORMAT": "jpg",
        # Resize exemplars to this size
        "IMAGE_RESIZE": (100, 100),
        # Detection confidence, e.g. 0.9 means that only objects p>=0.9 get detected for the next phase
        "MIN_CONFIDENCE": 0.9,
        # Exemplar augmentation parameters. See `logodetect/augmenters.py` for more examples.
        "AUGMENTER_PARAMS": {
            "Multiply": [0.5, 1.5],  # mu
            "GaussianBlur": [0.4],  # sigma
            "AdditiveGaussianNoise": [0.2 * 255],  # scale
            "AffineShear": [-25, 25],  # shear
            "AffineRotate": [-25, 25],  # rotate
        },
        "DEVICE": "cpu",  # {cpu, cuda:1, cuda:2, ...}
        "EMBEDDER_DEVICE": "cpu",
        "DETECTOR_DEVICE": "cpu",
        "CLASSIFIER_DEVICE": "cpu",
    }

    classifier = base_config.get("CLASSIFIER")
    base_config["CLASSIFIER_ALG"] = (
        "knn" if classifier == "knn" else "binary_stacked_resnet18"
    )

    if config:
        base_config.update(config)
    return base_config
