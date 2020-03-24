"""Command line interface
"""
import click
from logodetect import PATH_EXEMPLARS, Recognizer, VIDEO_FILENAME, IMAGE_FILENAME
from functools import partial


out = partial(click.secho, bold=True, err=True)


def common_options(function):
    function = click.option(
        "-e",
        "--exemplars",
        default=PATH_EXEMPLARS,
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
    return function


@click.command()
@click.option(
    "-i",
    "--image_filename",
    default=IMAGE_FILENAME,
    required=False,
    help="path to your input image",
)
@common_options
def image(image_filename, exemplars, output_appendix):
    recognizer = Recognizer(exemplars)
    recognizer.predict_image(
        image_filename=image_filename, output_appendix=output_appendix
    )
    out("All done! ✨ 🍰 ✨")


@click.command()
@click.option(
    "-v",
    "--video_filename",
    default=VIDEO_FILENAME,
    required=False,
    help="path to your input video",
)
@common_options
def video(video_filename, exemplars, output_appendix):
    recognizer = Recognizer(exemplars)
    recognizer.predict(video_filename=video_filename, output_appendix=output_appendix)
    out("All done! ✨ 🍰 ✨")


@click.group()
def cli():
    pass


cli.add_command(image)
cli.add_command(video)
