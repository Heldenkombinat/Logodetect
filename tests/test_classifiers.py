from logodetect.classifiers import knn, siamese
from logodetect.recognizer import Recognizer
from logodetect.detectors.faster_rcnn import Detector
import constants

import cv2
import pytest


@pytest.mark.integration
def test_siamese():
    # Note that `Classifier`s are not generally initialized directly. We're initializing the `Recognizer` first
    # to set the correct `exemplar_paths` for us.
    reco = Recognizer(constants.PATH_EXEMPLARS)
    classifier = siamese.Classifier(
        exemplar_paths=reco.exemplar_paths, classifier_algo="binary_stacked_resnet18"
    )

    detector = Detector()
    image = cv2.imread("./data/exemplars/3m.jpg")
    detections = detector.predict(image)

    final_detections = classifier.predict(detections, image)
    assert type(final_detections) == dict


@pytest.mark.integration
def test_knn():
    reco = Recognizer(constants.PATH_EXEMPLARS)
    classifier = knn.Classifier(
        exemplar_paths=reco.exemplar_paths, classifier_algo="knn"
    )

    detector = Detector()
    image = cv2.imread("./data/exemplars/3m.jpg")
    detections = detector.predict(image)

    final_detections = classifier.predict(detections, image)
    assert type(final_detections) == dict
