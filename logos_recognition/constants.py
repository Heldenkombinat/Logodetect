"Constants for logos recognition."

# Standard library:
import os

# Pip packages:
import torch


##########
# Global #
##########

PATH = os.path.join(os.environ['HKT'], 'Logos-Recognition')
PATH_EXEMPLARS = os.path.join(os.environ['DATASETS'], 'logos', 'exemplars')
LOAD_NAME = os.path.join(PATH, 'data', 'test_video_small.mp4')
name, extension = os.path.splitext(LOAD_NAME)  # extension includes the '.'
OUTPUT_NAME = LOAD_NAME.replace(extension, '_output.mp4')

###########
# PyTorch #
###########

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
DETECTOR_BASE = fasterrcnn_resnet50_fpn
DETECTOR_WEIGHTS = os.path.join(PATH, 'models', 'detector_old.pth')
DETECTOR_TOP = FastRCNNPredictor

from logos_recognition.classifiers import stacked_resnet18
CLASSIFIER_BASE = stacked_resnet18
CLASSIFIER_WEIGHTS = os.path.join(PATH, 'models', 'classifier.pth')

DEVICE = torch.device('cuda:1') if \
    os.uname()[1] == 'Shannon' else torch.device('cuda:0')

#########
# Logos #
#########

BRAND_LOGOS = ['pepsi', 'redbull', 'heineken', 'stellaartois']
QUERY_LOGOS = [os.path.join(PATH_EXEMPLARS, logo + '.jpg')
               for logo in BRAND_LOGOS]
IMAGE_RESIZE = (100, 100)
