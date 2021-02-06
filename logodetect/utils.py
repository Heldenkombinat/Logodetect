"""Global utilities."""

# Standard library:
import os

# Pip packages:
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as F
import torch
from typing import Tuple


def open_and_resize(path: str, image_resize: Tuple[int, int]) -> Image.Image:
    """Checks if image is valid and moves it to the GPU.

    :param path: path to image
    :param image_resize: tuple of two integers
    :return: resized PIL.Image
    """
    return Image.open(path).convert("RGB").resize(image_resize)


def image_to_gpu_tensor(image: Image.Image, device: str) -> torch.Tensor:
    """Checks if image is valid and moves it to the GPU.

    :param image: PIL.Image
    :param device: device to run op on
    :return:
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise Exception(
            "'predict' method takes a 3D image as input \
            of shape (H, W, 3). Instead got {}".format(
                image.shape
            )
        )
    return F.to_tensor(image).unsqueeze(0).to(device)


def clean_name(filename: str) -> str:
    """Clean file name

    :param filename: name of the file you want to clean.

    Example:
    >> ' '.join(sorted(set(''.join(list(set(brands))))))
    >> "& ' + - 1 2 3 4 ? a b c d e f g h i j kl m n
        o p q r s t u v w x y z \udcbc \udcc3 \udcfc"
    """
    name, extension = os.path.splitext(os.path.basename(filename))
    brand = name.split("_")[0]
    return brand.encode("ascii", "replace").decode()


def save_df(vectors, file_names, path, net_type="") -> None:
    """Save image vectors and brands stored in file
    names as pandas DataFrame. Only used for visualisation, e.g. in notebooks.

    :param vectors:
    :param file_names:
    :param path:
    :param net_type:
    :return:
    """
    vectors_list = [v for v in vectors]
    brands = [clean_name(n) for n in file_names]
    logos_df = pd.DataFrame({"brand": brands, "img_vec": vectors_list})
    logos_df.to_pickle(path + "{}.pkl".format(net_type))
