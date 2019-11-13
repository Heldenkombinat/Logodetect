"Global utilities."

# Standard library:
import os

# Pip packages:
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F

# Github repos:

# Logos-Recognition:
from logos_recognition.constants import DEVICE, IMAGE_RESIZE


def image_to_gpu_tensor(image, device=None):
    "Checks if image is valid and moves it to the GPU."
    device = device if device else DEVICE
    # convert to numpy if input is PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise Exception("'predict' method takes a 3D image as input \
            of shape (H, W, 3). Instead got {}".format(image.shape))
    return F.to_tensor(image).unsqueeze(0).to(device)


def open_resize_and_load_gpu(path, device=None):
    "Add documentation."
    # load image and resize it
    image = Image.open(path).convert("RGB").resize(IMAGE_RESIZE)
    # convert it to gpu tensor
    return image_to_gpu_tensor(image)

def get_class_name(path):
    "Add documentation."
    return os.path.split(path)[1].split(".")[0]
