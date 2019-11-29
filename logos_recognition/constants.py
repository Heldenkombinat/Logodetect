"Constants for logos recognition."

# Standard library:
import os

# Pip packages:
import torch


##########
# Global #
##########

PATH = os.path.join(os.environ['HKT'], 'Logos-Recognition')
PATH_DATA = os.path.join(os.environ['DATASETS'], 'logos')
PATH_EXEMPLARS = os.path.join(PATH_DATA, 'exemplars')
LOAD_NAME = os.path.join(PATH, 'data', 'football_redbull_small.mp4')
name, extension = os.path.splitext(LOAD_NAME)  # extension includes the '.'
OUTPUT_NAME = LOAD_NAME.replace(extension, '_output.mp4')

############
# Detector #
############

DETECTOR_WEIGHTS = os.path.join(PATH, 'models', 'detector.pth')
DETECTOR_DEVICE = 'cuda:2'
MIN_CONFIDENCE = 0.9
IMG_SIZE = 100

##############
# Classifier #
##############

CLASSIFIER_WEIGHTS = os.path.join(PATH, 'models', 'classifier.pth')
EXEMPLARS_PATH = os.path.join(PATH_DATA, 'exemplars_siamese.zip')
REPRESENTER_WEIGHTS = os.path.join(PATH, 'models', 'siamese_embedding.pth')
REPRESENTER_DEVICE = 'cuda:3'
EMBEDDING_SIZE = 345
DISTANCE = 'cosine'  # {cosine, minkowski_1, minkowski_2}
MAX_DISTANCE = 0.015  # {siamese-cosine: 0.013}

###########
# PyTorch #
###########

DEVICE = torch.device('cuda:1') if os.uname()[1] == 'Shannon' \
    else torch.device('cuda:0')

#########
# Logos #
#########

# BRAND_LOGOS = ['pepsi', 'redbull', 'heineken', 'stellaartois']
BRAND_LOGOS = ['redbull']
QUERY_LOGOS = [os.path.join(PATH_EXEMPLARS, logo + '.jpg')
               for logo in BRAND_LOGOS]
IMAGE_RESIZE = (100, 100)
