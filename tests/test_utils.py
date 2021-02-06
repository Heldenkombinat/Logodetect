import pytest
from logodetect.utils import open_and_resize, image_to_gpu_tensor, clean_name, save_df
import os


@pytest.mark.unit
def test_open_and_resize():
    resize = (100, 100)
    img = open_and_resize("./data/exemplars/3m.jpg", resize)
    assert img.size == resize


def test_image_to_gpu():
    img = open_and_resize("./data/exemplars/3m.jpg", (200, 200))
    image_to_gpu_tensor(
        img, "cpu"
    )  # can't use cuda etc. here, as not every system will have it


def test_clean_name():
    file_name = "./data/exemplars/bankofamerica_text.jpg"
    name = clean_name(file_name)
    assert name == "bankofamerica"


def test_save_df():
    vectors = [[1, 2, 3], [4, 5, 6]]
    files = ["./data/exemplars/bankofamerica_text.jpg", "./data/exemplars/3m.jpg"]
    save_df(vectors, files, "./test_df")
    assert os.path.exists("./test_df.pkl")
    os.remove("./test_df.pkl")
