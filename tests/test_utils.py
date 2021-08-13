import pytest
from logodetect.utils import open_and_resize, image_to_gpu_tensor, clean_name, save_df
import os


@pytest.mark.unit
def test_open_and_resize():
    resize = (100, 100)
    img = open_and_resize("./data/exemplars/pepsi_1.jpg", resize)
    assert img.size == resize


@pytest.mark.unit
def test_image_to_gpu():
    img = open_and_resize("./data/exemplars/pepsi_1.jpg", (200, 200))
    image_to_gpu_tensor(
        img, "cpu"
    )  # can't use cuda etc. here, as not every system will have it


@pytest.mark.unit
def test_clean_name():
    file_name = "./data/exemplars/redbull_1.jpg"
    name = clean_name(file_name)
    assert name == "redbull"


@pytest.mark.unit
def test_save_df():
    vectors = [[1, 2, 3], [4, 5, 6]]
    files = ["./data/exemplars/redbull_1.jpg", "./data/exemplars/pepsi_1.jpg"]
    save_df(vectors, files, "./test_df")
    assert os.path.exists("./test_df.pkl")
    os.remove("./test_df.pkl")
