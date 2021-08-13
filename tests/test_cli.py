from click.testing import CliRunner
from logodetect.cli import image, video, init
import pytest


@pytest.mark.unit
def test_init():
    runner = CliRunner()
    result = runner.invoke(init)
    assert not result.exception


@pytest.mark.integration
def test_image():
    runner = CliRunner()
    result = runner.invoke(image, ["-i", "./data/test_images/test_image_small.png"])
    assert not result.exception


@pytest.mark.integration
def test_image_config():
    runner = CliRunner()
    result = runner.invoke(
        image, ["-i", "./data/test_images/test_image_small.png", "-c", "./config.json"]
    )
    assert not result.exception


@pytest.mark.integration
def test_video():
    runner = CliRunner()
    result = runner.invoke(
        video, ["-v", "./data/test_videos/test_video_small_50ms.mp4"]
    )
    assert not result.exception
