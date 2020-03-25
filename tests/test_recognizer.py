import pytest
import os

# "constants.py" is only known to the library code within logodetect at runtime,
# so we use the backup here for simplicity
from backup_constants import PATH_EXEMPLARS, IMAGE_FILENAME, VIDEO_FILENAME
from logodetect.recognizer import Recognizer


@pytest.mark.unit
def test_recognizer_image():
    app = Recognizer(PATH_EXEMPLARS)
    output_appendix = "_foo"
    app.predict_image(IMAGE_FILENAME, output_appendix=output_appendix)
    name, extension = os.path.splitext(IMAGE_FILENAME)
    output_filename = IMAGE_FILENAME.replace(extension, f"{output_appendix}{extension}")
    assert os.path.exists(output_filename)
    os.remove(output_filename)


@pytest.mark.integration
def test_recognizer_video():
    app = Recognizer(PATH_EXEMPLARS)
    output_appendix = "_foo"
    app.predict(VIDEO_FILENAME, output_appendix=output_appendix)
    name, extension = os.path.splitext(VIDEO_FILENAME)
    output_filename = VIDEO_FILENAME.replace(extension, f"{output_appendix}{extension}")
    assert os.path.exists(output_filename)
    os.remove(output_filename)
