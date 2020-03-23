<p align="center">
  <img src="https://github.com/Heldenkombinat/Logos-Recognition/blob/master/hkt_logo_detect.png">
</p>

---

## Installation

- (Optional) Create a new virtual environment and activate it:

```bash_script
conda create -n test python=3
conda activate test
```

- Clone repository
- Install with `pip install -e .` from the `Logos-Recognition` folder.

## Usage

TODO: when installing the library, download and unzip models and videos automatically

Videos: https://drive.google.com/a/heldenkombinat.com/file/d/1Htp0qGsp2IufaeVSQe40iBaroDtlbIbJ/view?usp=sharing


Download data and models from the [Google Drive folder](https://drive.google.com/a/heldenkombinat.com/file/d/17yi4J8YFRSkdsUOMpqpBgMGnqOjvugnq/view?usp=sharing)
- Run test case with `python logos_recognition/app.py`
- By default, output video will be saved as `Logos-Recognition/data/test_video_output.mp4`

## Automatic code linting with `black`

This project uses `black` for code linting. To install the git pre-commit hook for `black`,
simply run

```bash_script
pre-commit install
```

from the base of this repository. This will run (and fail in case of grave errors) black each time you make a commit. Once CI is up for this project, we will ensure this hook runs on each CI pass. To manually use `black` 
on a file, use `black <path-to-file>`.

## Running tests

Run all tests with `pytest`, or just run the quicker unit test suite with 

```bash_script
pytest -m unit
```

or all longer-running integration tests with

```bash_script
pytest -m integration
```

## Modules and concepts

### Recognizer

### Classifier

### Detector

### Augmenter

## Current capabilities

- Load video
- Load detector model
- Detect logos of any kind
- Save output video
