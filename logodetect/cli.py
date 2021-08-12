"""Command line interface
"""
import os
import click
from functools import partial
import requests
import zipfile
import json

from logodetect import constants
from logodetect.recognizer import Recognizer


out = partial(click.secho, bold=True, err=True)

if "LOGOS_RECOGNITION" in os.environ:
    BASE_PATH = os.environ["LOGOS_RECOGNITION"]
else:
    BASE_PATH = os.path.expanduser(os.path.join("~", ".hkt", "logodetect"))
DATA_PATH = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(BASE_PATH, "models")
BASE_URL = "https://hkt-logodetect.s3.eu-central-1.amazonaws.com"


def common_options(function):
    function = click.option(
        "-e",
        "--exemplars",
        default=constants.PATH_EXEMPLARS,
        required=False,
        help="path to your exemplars folder",
    )(function)
    function = click.option(
        "-o",
        "--output_appendix",
        default="_output",
        required=False,
        help="string appended to your resulting file",
    )(function)
    function = click.option(
        "-c",
        "--config_file",
        default=None,
        required=False,
        help="path to file containing a logodetect config JSON",
    )(function)
    return function


@click.command()
@click.option(
    "-i",
    "--image_filename",
    default=constants.IMAGE_FILENAME,
    required=False,
    help="path to your input image",
)
@common_options
def image(image_filename, exemplars, output_appendix, config_file):
    config = None
    if config_file:
        with open(config_file, "r") as f:
            config = json.load(f)
    recognizer = Recognizer(exemplars, config)
    recognizer.predict_image(
        image_filename=image_filename, output_appendix=output_appendix
    )
    out("All done! ✨ 🍰 ✨")


@click.command()
@click.option(
    "-v",
    "--video_filename",
    default=constants.VIDEO_FILENAME,
    required=False,
    help="path to your input video",
)
@common_options
def video(video_filename, exemplars, output_appendix, config_file):
    config = None
    if config_file:
        with open(config_file, "r") as f:
            config = json.load(f)
    recognizer = Recognizer(exemplars, config)

    recognizer.predict(video_filename=video_filename, output_appendix=output_appendix)
    out("All done! ✨ 🍰 ✨")


@click.command()
def init():
    os.makedirs(MODEL_PATH, exist_ok=True)
    print(
        ">>> Note that downloading all model and data files might take a few minutes!"
    )
    download("detector.pth", "model")
    download("embedder.pth", "model")
    download("classifier_resnet18.pth", "model")
    download("data.zip", "data")


def download(file_name: str, data_type: str):
    print(f">>> Downloading file {file_name} now, please wait...")

    if data_type == "model":
        local_path = os.path.join(MODEL_PATH, file_name)
    elif data_type == "data" or data_type == "base":
        local_path = os.path.join(BASE_PATH, file_name)
    else:
        raise ValueError(
            f"Data type {data_type} not allowed for downloads, pick from 'model', 'data' or "
            f"'base'."
        )

    # only download data if it does not already exist
    if not (
        (data_type == "data" and os.path.exists(DATA_PATH))
        or (data_type != "data" and os.path.exists(local_path))
    ):
        url = os.path.join(BASE_URL, file_name)
        get_request = requests.get(url)
        open(local_path, "wb").write(get_request.content)
        if data_type == "data":
            with zipfile.ZipFile(local_path, "r") as zip_ref:
                zip_ref.extractall(DATA_PATH)
            os.remove(local_path)
        print(">>> Download complete!")
    else:
        print(">>> Skipping download, file already exists.")


@click.group()
def cli():
    pass


cli.add_command(init)
cli.add_command(image)
cli.add_command(video)
