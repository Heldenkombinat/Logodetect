<p align="center">
  <img src="https://github.com/Heldenkombinat/Logos-Recognition/blob/master/hkt_logo_detect.png">
</p>

[![PyPI version](https://badge.fury.io/py/logodetect.svg)](https://badge.fury.io/py/logodetect)
---

`logodetect` is a one-shot detection library to find logos of any kind in video and image data. Go check out
[our demo if you want to see it in action right away.](https://logodetect.netlify.com/)

Here's a quick example of live football footage that can detect the logos on jerseys and the sports fields.

![one-shot logo detection on video footage](demo.gif)

## Installation

This library is available on PyPI, so you can simply run `pip install logodetect` to install it.

If you want to build `logodetect` from source, run

```bash_script
git clone git@github.com:Heldenkombinat/Logos-Recognition.git
cd Logos-Recognition
pip install -e ".[tests, dev]"
```

## Usage

After successful installation, a CLI tool called `logodetect` becomes available to you. If you invoke `logodetect`
without any arguments, you will get help on how to use it. To automatically download all models and data needed
to test the application first run:

```bash_script
logodetect init
```

which will download all files to `~/.hkt/logodetect`. Note that if you prefer another folder to download the data,
please use the environment variable `LOGOS_RECOGNITION`. For instance, if you want to install models and data relative
to your clone of this repository, use

```bash_script
export LOGOS_RECOGNITION=path/to/this/folder
```

before running `logodetect init`, or consider putting this variable in your `.bash_rc`, `.zshrc` or an equivalent
configuration file on your system.

The `logodetect` CLI tool comes with two main commands, namely `video`
and `image`, both of which work fairly similar. In each case you need to provide the input data for which you would
like to detect logos, and the logo exemplars that you want to detect in the footage. To get you started, we've provided
some demo data that you can use out of the box. That means you can simply run:

```bash_script
logodetect video
```

to run one-shot detection on an example video, or you can run

```bash_script
logodetect image
```

to do so for an example image. If you want to use another video, you can do so with the `-v` option. Images can be provided
with the `-i` option and custom exemplars are configured with the `-e` option. That means, if you want to run detection
for custom video data with custom exemplars, you should use

```bash_script
logodetect video -v <path-to-video> -e <path-to-exemplars-folder>
```

## Docker support

If you prefer docker, build an image and run it like this:

```bash_script
docker build . -t logodetect
docker run -e LOGOS_RECOGNITION=/app -p 5000:5000 -t logodetect
```

**Important**: this assumes that you have previously downloaded all `data` and `models` right next to
the `Dockerfile` in the local copy of this repo. 

## Automatic code linting with `black`

This project uses `black` for code linting. To install the git pre-commit hook for `black`,
simply run

```bash_script
pre-commit install
```

from the base of this repository. This will run (and fail in case of grave errors) black each time you make a commit. 
Once CI is up for this project, we will ensure this hook runs on each CI pass. To manually use `black` on a file,
use `black <path-to-file>`.

## Running tests

Run all tests with `pytest`, or just run the quicker unit test suite with 

```bash_script
pytest -m unit
```

or all longer-running integration tests with

```bash_script
pytest -m integration
```
