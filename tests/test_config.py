import pytest
from logodetect.constants import get_recognizer_config


@pytest.mark.unit
def test_config():
    config = get_recognizer_config()
    assert config is not None
    assert "DETECTOR" in config.keys()
    assert config.get("DETECTOR") is not None
    assert config.get("CLASSIFIER") == "knn"
    assert config.get("CLASSIFIER_ALG") == "knn"
