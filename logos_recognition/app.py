"""Top module for client-specific application.
"""

import argparse
from logos_recognition.recognizer import Recognizer
from logos_recognition.constants import VIDEO_FILENAME, PATH_EXEMPLARS


if __name__ == "__main__":

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

    app = Recognizer(args.exemplars_path)
    app.predict(args.video_filename)
