from logodetect.detectors.faster_rcnn import Detector
from logodetect.utils import open_and_resize
import pytest


@pytest.mark.unit
def test_detectors():
    detector = Detector()
    img = open_and_resize("./data/exemplars/pepsi_1.jpg", (200, 200))
    detections = detector.predict(img)
    assert sorted(list(detections.keys())) == ["boxes", "brands", "labels", "scores"]
