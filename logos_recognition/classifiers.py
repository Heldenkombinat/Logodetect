import torch.nn as nn
import torchvision

'''
All "stacked" models will take as input the channel-wise
concatenation of two images and output whether it's the same class or not.
Input should always be exemplar ON TOP of detection, i.e.
    input[:3, :, :] is the detection and input[3:, :, :] is the exemplar
'''


def stacked_resnet18():
    '''
    Creates a ResNet18 with a 6-channel input.
    '''
    model = torchvision.models.resnet18()
    # replace the 3-channel input with 6-channel input
    model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=False)
    # replace the final layer and add a sigmoid on top
    model.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=1, bias=True),
        nn.Sigmoid())
    return model
