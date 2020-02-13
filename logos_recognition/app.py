"Top module for client-specific application."

# Standard library:
import argparse

# Current library:
from logos_recognition.recognizer import Recognizer
from logos_recognition.constants import VIDEO_FILENAME, PATH_EXEMPLARS


class App(object):
    "Add documentation."

    def __init__(self, exemplars_path):
        "Add documentation."
        self.recognizer = Recognizer(exemplars_path)

    def run(self, video_filename):
        "Add documentation."
        self.recognizer.predict(video_filename)


if __name__ == "__main__":

    # Set argument parser:
    parser = argparse.ArgumentParser(description="One-shot object detector.")
    parser.add_argument(
        "-inp",
        "--video_filename",
        type=str,
        help="Filename of the video to process.",
        default=VIDEO_FILENAME,
    )
    parser.add_argument(
        "-exm",
        "--exemplars_path",
        type=str,
        help="Directory of the exemplars to detect.",
        default=PATH_EXEMPLARS,
    )
    args = parser.parse_args()

    # Create instance of application:
    APP = App(args.exemplars_path)

    # Process video:
    APP.run(args.video_filename)
