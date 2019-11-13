"Top module for client-specific application."

# Standard library:

# Pip packages:

# Github repos:

# Local:
from logos_recognition.constants import LOAD_NAME, OUTPUT_NAME, QUERY_LOGOS
from logos_recognition.recognizer import Recognizer


class App(object):
    "Add documentation."

    def __init__(self):
        "Add documentation."
        self.recognizer = Recognizer()

    def run(self, load_name=None, output_name=None, query_logos=None):
        "Add documentation."

        # Video paths
        load_name = load_name if load_name else LOAD_NAME
        output_name = output_name if output_name else OUTPUT_NAME
        query_logos = query_logos if query_logos else QUERY_LOGOS

        # Analyze video
        self.recognizer.recognize(load_name, output_name, query_logos)


if __name__ == "__main__":

    # Create instance of application:
    APP = App()

    # Process video:
    APP.run()
