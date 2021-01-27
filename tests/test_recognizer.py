import pytest
import os

from constants import PATH_EXEMPLARS, IMAGE_FILENAME, VIDEO_FILENAME
from logodetect.recognizer import Recognizer, append_to_file_name

import cv2


@pytest.mark.unit
def test_file_appender():
    output_appendix = "_foo"
    output_filename = append_to_file_name(VIDEO_FILENAME, output_appendix)
    assert "test_video_small_50ms_foo.mp4" in output_filename


@pytest.mark.unit
def test_recognizer_basics():
    reco = Recognizer(PATH_EXEMPLARS)

    reco.set_exemplars(PATH_EXEMPLARS)
    assert reco.exemplars_path == PATH_EXEMPLARS

    reco.set_video_source(VIDEO_FILENAME)

    output_appendix = "_foo"
    image = cv2.imread(IMAGE_FILENAME)
    reco.save_image(
        image=image, image_filename=IMAGE_FILENAME, output_appendix=output_appendix
    )

    output_filename = append_to_file_name(IMAGE_FILENAME, output_appendix)
    assert os.path.exists(output_filename)
    os.remove(output_filename)


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
