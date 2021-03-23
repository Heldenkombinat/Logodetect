from logodetect.utils import open_and_resize
from logodetect.augmenters import augment_image, get_augmentations
from constants import AUGMENTER_PARAMS
import itertools
from PIL import Image
import pytest


@pytest.mark.unit
def test_augmentation():
    path = "data/exemplars/adidas.jpg"
    IMAGE_RESIZE = (100, 100)
    image = open_and_resize(path, IMAGE_RESIZE)

    all_augmentations = get_augmentations(image, AUGMENTER_PARAMS)

    augmented_single = augment_image(
        image, mu=0.0, sigma=1.0, scale=1.0, shear=0, rotate=0
    )
    assert type(augmented_single) == Image.Image

    total_param_combinations = len(list(itertools.product(*AUGMENTER_PARAMS.values())))
    assert total_param_combinations == len(all_augmentations)
