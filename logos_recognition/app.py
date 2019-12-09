"Top module for client-specific application."

# Standard library:
import argparse

# Current library:
from logos_recognition.recognizer import Recognizer
from logos_recognition.constants import (VIDEO_FILENAME, OUTPUT_NAME,
                                         EXEMPLARS_PATHS)



class App(object):
    "Add documentation."

    def __init__(self):
        "Add documentation."
        self.recognizer = Recognizer()

    def run(self, input_filename, output_name, exemplars_paths):
        "Add documentation."
        self.recognizer.recognize(input_filename, output_name, exemplars_paths)


if __name__ == "__main__":

    # Set argument parser:
    parser = argparse.ArgumentParser(description='One-shot object detector.')
    parser.add_argument('-inp', '--input_filename', type=str,
                        help='Filename of video to process.',
                        default=VIDEO_FILENAME)
    parser.add_argument('-out', '--output_filename', type=str,
                        help='Savename of video to process.',
                        default=OUTPUT_NAME)
    parser.add_argument('-exm', '--exemplars_paths', type=str,
                        help='Filenames of exemplars to detect.',
                        default=EXEMPLARS_PATHS)
    args = parser.parse_args()

    # Create instance of application:
    APP = App()

    # Process video:
    APP.run(args.input_filename, args.output_filename, args.exemplars_paths)
