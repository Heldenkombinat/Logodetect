from logodetect.utils import open_and_resize
from logodetect.augmenters import augment_image, get_augmentations
from logodetect.constants import get_recognizer_config
import itertools
from PIL import Image
import pytest


@pytest.mark.unit
def test_augmentation():
    path = "data/exemplars/adidas.jpg"
    IMAGE_RESIZE = (100, 100)
    image = open_and_resize(path, IMAGE_RESIZE)

    all_augmentations = get_augmentations(image, config=None)

    augmented_single = augment_image(
        image, mu=0.0, sigma=1.0, scale=1.0, shear=0, rotate=0
    )
    assert type(augmented_single) == Image.Image

    config = get_recognizer_config(config=None)
    augmenter_params = config.get("AUGMENTER_PARAMS")

    total_param_combinations = len(list(itertools.product(*augmenter_params.values())))
    assert total_param_combinations == len(all_augmentations)
