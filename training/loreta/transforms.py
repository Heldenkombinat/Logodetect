"""Custom transforms for logos recognition."""

# Standard library:

# Pip packages:
from torchvision import transforms as T

# Local:


def transform_small_crop(resize=100):
    transform = T.Compose(
        [
            T.Resize((resize, resize)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomResizedCrop(resize, scale=(0.08, 1), ratio=(1, 1)),
            T.RandomRotation(20),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform


def transform_big_crop(resize=100):
    transform = T.Compose(
        [
            T.Resize((resize, resize)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomResizedCrop(resize, scale=(0.7, 1), ratio=(1, 1)),
            T.RandomRotation(20),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform


def transform_no_crop(resize=100):
    transform = T.Compose(
        [
            T.Resize((resize, resize)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(20),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform


def transform_no_aug(resize=100):
    transform = T.Compose(
        [
            T.Resize((resize, resize)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform


def cifar10_train(resize=100):
    transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.Resize((resize, resize)),
            T.RandomResizedCrop(resize, scale=(0.08, 1), ratio=(1, 1)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform


def cifar10_test(resize=100):
    transform = T.Compose(
        [
            T.Resize((resize, resize)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform


def logos_train(resize=100):
    transform = T.Compose(
        [
            T.Resize((resize, resize)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(20),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform


def logos_test(resize=100):
    transform = T.Compose(
        [
            T.Resize((resize, resize)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform
