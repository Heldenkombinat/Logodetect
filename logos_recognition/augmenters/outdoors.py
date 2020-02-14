"Augmentations for outdoor environments."

# Pip packages:
import numpy as np
from PIL import Image

# Github packages:
import imgaug.augmenters as ia

# Current library:
from logos_recognition.constants import AUGMENTER_PARAMS


def get_augmentations(image):
    "Add documentation."
    augmented_images = []
    # For each combination:
    for mu in AUGMENTER_PARAMS["Multiply"]:
        for gabl in AUGMENTER_PARAMS["GaussianBlur"]:
            for adga in AUGMENTER_PARAMS["AdditiveGaussianNoise"]:
                for afsh in AUGMENTER_PARAMS["AffineShear"]:
                    for afro in AUGMENTER_PARAMS["AffineRotate"]:
                        # Process image:
                        image_aug = augment_image(image, mu, gabl, adga, afsh, afro)
                        augmented_images.append(image_aug)
    return augmented_images


def augment_image(image, mu, gabl, adga, afsh, afro):
    "Add documentation."
    augmenter = ia.Sequential(
        [
            ia.Multiply(mul=mu),
            ia.GaussianBlur(sigma=gabl),
            ia.AdditiveGaussianNoise(scale=adga),
            ia.Affine(rotate=afro, shear=afsh),
        ]
    )
    # Process image:
    image_arr = augmenter.augment_images(np.array(image))
    return Image.fromarray(image_arr)
