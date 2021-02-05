from logodetect.classifiers import knn, siamese
from logodetect.recognizer import Recognizer
import constants


def test_siamese():
    # Note that `Classifier`s are not generally initialized directly. We're initializing the `Recognizer` first
    # to set the correct `exemplar_paths` for us.
    reco = Recognizer(constants.PATH_EXEMPLARS)
    classifier = siamese.Classifier(
        exemplar_paths=reco.exemplar_paths, classifier_algo="binary_stacked_resnet18"
    )


def test_knn():
    reco = Recognizer(constants.PATH_EXEMPLARS)
    classifier = knn.Classifier(
        exemplar_paths=reco.exemplar_paths, classifier_algo="knn"
    )
