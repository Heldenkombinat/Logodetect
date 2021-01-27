"""Global utilities."""

# Standard library:
import os
import csv
import json
import argparse
from copy import deepcopy

# Pip packages:
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Local:
from .constants import CONFIG_FILE


##########
# Images #
##########


def image_to_gpu_tensor(image, device=None):
    "Checks if image is valid and moves it to the GPU."
    device = device if device else DEVICE
    # convert to numpy if input is PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise Exception(
            "`predict` method takes a 3D image as input \
            of shape (H, W, 3). Instead got {}".format(
                image.shape
            )
        )
    return F.to_tensor(image).unsqueeze(0).to(device)


def extract_annotation(txt_path):
    "Add documentation."
    csv_reader = csv.reader(open(txt_path, "rt"), delimiter=" ")
    classes = []
    boxes = []
    for row in csv_reader:
        # Row convention:
        # (Left, Top, Right, Bottom, Class)
        x1, y1, x2, y2, brand = row
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        classes.append(brand)
        boxes.append([x1, y1, x2, y2])
    return np.array(classes), np.array(boxes)


def extract_crop(image_path, box):
    "Add documentation."
    return Image.open(image_path).convert("RGB").crop(box)


###############
# Directories #
###############


def get_file_info(path):
    "Add documentation."
    root, file = os.path.split(path)
    try:
        name, ext = file.split(".")
        ext = "." + ext
    # If no '.' in the file name:
    except ValueError:
        name = file
        ext = ""
    return root, file, name, ext


def get_class_name(path):
    "Add documentation."
    return os.path.split(path)[1].split(".")[0]


########
# Logs #
########


def read_tensorboard_log(path):
    "Reads the .tfevents file and returns the scalars."
    # First let's read the config file
    config = json.load(open(os.path.join(path, "config.json")))
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    # print(event_acc.Tags())  # shows all tags in the log file
    wall_times, step_nums, loss_train = zip(*event_acc.Scalars("Loss/train"))
    # We assume that all scalars are recorded at each step
    _, _, loss_valid = zip(*event_acc.Scalars("Loss/valid"))
    _, _, acc_train = zip(*event_acc.Scalars("Accuracy/train"))
    _, _, acc_valid = zip(*event_acc.Scalars("Accuracy/valid"))
    info = {
        "loss_train": loss_train,
        "loss_valid": loss_valid,
        "acc_train": acc_train,
        "acc_valid": acc_valid,
        "step_nums": step_nums,
        "wall_times": wall_times,
        "config": config,
    }
    return info


###########
# Configs #
###########


def load_config(config_file):
    "Loads the config file from the path given."

    # Read config file path:
    parser = argparse.ArgumentParser(description="PyTorch Detection Training")
    parser.add_argument("--config", help="JSON config file path", default=config_file)
    args = parser.parse_args()

    # Return JSON file:
    return json.load(open(args.config))


def create_configs(config):
    "Creates multiple configs if hyperparameter search is desired."
    if ("hyperparam_fields" not in config["general"]) or (
        len(config["general"]["hyperparam_fields"]) == 0
    ):
        return [config]

    param_names = []
    param_data = []
    for param in config["general"]["hyperparam_fields"]:
        param_names.append(param)
        param_data.append(get_config_data(config, param))

    combs = np.meshgrid(*param_data, indexing="ij")
    combs = np.array(combs).T.reshape(-1, len(param_names))
    configs = []

    for i in range(len(combs)):
        new_config = deepcopy(config)
        for j in range(len(combs[i])):
            new_config = set_config_data(
                new_config, config["general"]["hyperparam_fields"][j], combs[i, j]
            )
        configs.append(new_config)

    return configs


def get_config_data(config, param):
    "Returns values of field 'param'"
    fields = param.split("/")
    data = config[fields[0]]
    for field in fields[1:]:
        data = data[field]
    return data


def set_config_data(config, param, value):
    "Sets the value of field 'param'"
    try:
        if "seed" in param:
            value = int(value)
        else:
            value = float(value)
    except ValueError:
        value = value

    fields = param.split("/")
    eval_str = "config"
    for field in fields:
        eval_str += "['{}']".format(field)
    eval_str += " = value"
    exec(eval_str)
    return config


if __name__ == "__main__":
    CONFIG = load_config(CONFIG_FILE)
    DEVICE = CONFIG["general"]["device"]  # {'cuda:1', 'cpu'}
