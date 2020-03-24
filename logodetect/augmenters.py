"""Augmentations for outdoor environments.
"""

import itertools
import numpy as np
from PIL import Image
from imgaug import augmenters

from logodetect.constants import AUGMENTER_PARAMS


def get_augmentations(image: Image.Image, params: dict = AUGMENTER_PARAMS) -> list:
    """Get a list of augmented images for an image,
    according to the specified augmentation

    :param image: a PIL.Image instance
    :param params: dictionary of augmenter params.
    :return: list of augmented PIL.Image images
    """
    param_product = itertools.product(*params.values())
    return [augment_image(image, *param) for param in param_product]


def augment_image(
    image: Image.Image, mu: float, sigma: float, scale: float, shear: int, rotate: int
) -> Image.Image:
    """Augment an image using the 'imgaug' library.

    :param image: a PIL.image instance
    :param mu: Gaussian blur mu
    :param sigma: Gaussian blur sigma
    :param scale: Additive Gaussian noise
    :param shear: Affine shear
    :param rotate: Affine rotation
    :return: Augmented PIL.Image
    """
    augmenter = augmenters.Sequential(
        [
            augmenters.Multiply(mul=mu),
            augmenters.GaussianBlur(sigma=sigma),
            augmenters.AdditiveGaussianNoise(scale=scale),
            augmenters.Affine(rotate=rotate, shear=shear),
        ]
    )
    image_arr = augmenter.augment_image(np.array(image))
    return Image.fromarray(image_arr)
