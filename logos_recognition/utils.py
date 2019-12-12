"Global utilities."

# Standard library:
import os

# Pip packages:
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as F



def open_resize_and_load_gpu(path, device, image_resize):
    "Add documentation."
    # Load image and resize it:
    image = open_and_resize(path, image_resize)
    # Convert it to gpu tensor:
    return image_to_gpu_tensor(image, device)


def open_and_resize(path, image_resize):
    "Checks if image is valid and moves it to the GPU."
    return Image.open(path).convert('RGB').resize(image_resize)


def image_to_gpu_tensor(image, device):
    "Checks if image is valid and moves it to the GPU."
    # Convert to numpy if input is PIL image:
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise Exception("'predict' method takes a 3D image as input \
            of shape (H, W, 3). Instead got {}".format(image.shape))
    return F.to_tensor(image).unsqueeze(0).to(device)


def clean_name(filename):
    '''
    >> ' '.join(sorted(set(''.join(list(set(brands))))))
    >> "& ' + - 1 2 3 4 ? a b c d e f g h i j kl m n
        o p q r s t u v w x y z \udcbc \udcc3 \udcfc"
    '''
    name, extension = os.path.splitext(os.path.basename(filename))
    brand = name.split('_')[0]
    return brand.encode('ascii', 'replace').decode()


def save_df(vectors, filenames, path, net_type=''):
    vectors_list = [v for v in vectors]
    brands = [clean_name(n) for n in filenames]
    logos_df = pd.DataFrame({'brand': brands, 'img_vec': vectors_list})
    # Save data:
    logos_df.to_pickle(path + '{}.zip'.format(net_type))
