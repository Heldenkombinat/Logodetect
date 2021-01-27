"""Utilities for loading all required modules."""

# Standard library:

# Pip packages:
import torch
from torch.utils.data import DataLoader, RandomSampler

# Local:
from .pytorch.detect_utils import collate_fn
from . import transforms as module_transforms
from . import datasets as module_datasets
from . import models as module_models
from . import loggers as module_loggers
from . import trainers as module_trainers


def load_modules(config_file):
    "Add documentation."
    # create transforms
    transform_train = create_transform(
        module_transforms, config_file["transform"]["train"]
    )
    transform_valid = create_transform(
        module_transforms, config_file["transform"]["valid"]
    )

    # create datasets
    dataset_train = create_dataset(
        module_datasets, config_file["dataset"]["train"], transform_train
    )
    dataset_valid = create_dataset(
        module_datasets, config_file["dataset"]["valid"], transform_valid
    )

    # create data loaders
    loader_train = create_loader(
        dataset_train,
        config_file["general"],
        config_file["general"]["training_detector"],
    )
    loader_valid = create_loader(
        dataset_valid,
        config_file["general"],
        config_file["general"]["training_detector"],
    )

    # create model & co
    model, criterion = create_model(module_models, config_file["model"])
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.__dict__[config_file["optimizer"]["type"]](
        model_parameters, **config_file["optimizer"]["args"]
    )

    # create logger
    logger = create_logger(module_loggers, config_file)

    # put everything in one place
    config = {
        "dataset_train": dataset_train,
        "dataset_valid": dataset_valid,
        "loader_train": loader_train,
        "loader_valid": loader_valid,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "logger": logger,
        "general": config_file["general"],
    }

    # create the trainer
    trainer = create_trainer(module_trainers, config, config_file["trainer"])
    return trainer


def create_transform(module, config_file):
    "Add documentation."
    transform_class = getattr(module, config_file["type"])
    transform_instance = transform_class(**config_file["args"])
    return transform_instance


def create_dataset(module, config_file, transform):
    "Add documentation."
    dataset_class = getattr(module, config_file["type"])
    dataset_instance = dataset_class(transform=transform, **config_file["args"])
    return dataset_instance


def create_loader(dataset, config_file, training_detector):
    "Add documentation."
    used_collate_fn = collate_fn if training_detector else None
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=config_file["batchsize"],
        sampler=sampler,
        num_workers=config_file["num_workers"],
        pin_memory=True,
        collate_fn=used_collate_fn,
    )
    return data_loader


def create_model(module, config_file):
    "Add documentation."
    model_class = getattr(module, config_file["type"])
    model_instance, criterion = model_class(**config_file["args"])
    return model_instance, criterion


def create_logger(module, config_file):
    "Add documentation."
    logger_class = getattr(module, config_file["logger"]["type"])
    logger_instance = logger_class(config_file)
    return logger_instance


def create_trainer(module, config, config_file):
    "Add documentation."
    trainer_class = getattr(module, config_file["type"])
    trainer_instance = trainer_class(config, **config_file["args"])
    return trainer_instance
