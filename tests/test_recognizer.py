import pytest
import os
from logodetect import PATH_EXEMPLARS, Recognizer, IMAGE_FILENAME, VIDEO_FILENAME


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
