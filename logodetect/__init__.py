import os
import sys

if "LOGOS_RECOGNITION" in os.environ:
    PATH_GLOB = os.environ["LOGOS_RECOGNITION"]
else:
    PATH_GLOB = os.path.expanduser("~/.hkt/logodetect")

# We append the base path for constants.py to the Python path dynamically,
# so we can import it project-wide.
sys.path.append(os.path.abspath(PATH_GLOB))
