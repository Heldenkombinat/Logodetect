<p align="center">
  <img src="https://github.com/Heldenkombinat/Logos-Recognition/blob/master/docs/hkt_logo_detect.png">
</p>

[![PyPI version](https://badge.fury.io/py/logodetect.svg)](https://badge.fury.io/py/logodetect)

Never be embarrassed again to say you have no big data! `logodetect` is a one-shot
detection library to find logos of any kind in video and image data.

Here's a quick example of football footage that can detect all logos on jerseys and the sports field.
Go check out [our demo if you want to see it in action right away](https://logodetect.netlify.com/demo).

<p align="center">
  <img src="https://github.com/Heldenkombinat/Logos-Recognition/blob/master/docs/demo.gif">
</p>

## Introduction

There is plenty of literature on the use of deep-learning for detecting logos, so, 
additionally to sharing with you a couple of algorithms to get started with one-shot logo-detection, 
the aim of this project is to develop a flexible architecture to facilitate the comparison of 
different algorithms for one-shot object recognition.

The pipeline supports one or two stages. It is possible to only perform *object-recognition*, 
or to first perform *object-detection* and then *object-recognition* in a second stage.

The idea is that you can use a generic detector for a single class of objects (e.g. logos, traffic signs or
faces) and then compare each of its detections with the *exemplar*, i.e., the sub-class that you are trying
to recognize, to determine if both belong to the same sub-class (e.g. a concrete brand, a stop sign or the face of a
loved one).
To get started, we include two algorithms that you can play with. Both have a Faster-RCNN [1] in
the first stage that performs object-detection and they differ in the second stage that performs
object-recognition.

As a **baseline**, we bring the exemplars and the detections from the first stage to the same latent space
(this reduces the course of dimensionality) and then simply measure the Euclidian or the cosine distance
between both embeddings for object-recognition. Both inputs are considered to belong to the same sub-class
if their distance is below a threshold determined in a preliminary analysis of the training dataset.
The code also provides functionality to add various transformations, so you have the option to augment
each exemplar with different transformations if you want. Simply add one or more exemplars into
the `data/exemplars` folder that is generated after you've followed the installation instructions below,
and you are good to go.

<p align="center">
  <img src="https://github.com/Heldenkombinat/logodetect/blob/master/docs/distances_analysis.png">
</p>

As a **first reference** against the baseline, we also provide a modified ResNet [2] for object-recognition
that directly takes the exemplars and the detections from the first stage and predicts if both belong to
the same sub-class. Similarly to [3], this network infers a distance metric after being trained with
examples of different sub-classes, but instead of sharing the same weights and processing each input in a
separate pass as in [4], it concatenates both inputs and processes them in one pass. This concept follows
more closely the architecture proposed by [5], where the assumption is that the exemplars often have
more high-frequency components than the detections, and therefore the model can increase its accuracy
by learning a separate set of weights for each input.

The models that we are including in the repo achieved a reasonable performance after a few training epochs.
However, if you would like to improve their performance you can find pointers to various datasets in [6],
which can be used in the `training` part of this project.

## References
[1] Ren et. al. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf) (2016)\
[2] He et. al. [Deep Residual Learning for Image Recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (2016)\
[3] Hsieh et. al. [One-Shot Object Detection with Co-Attention and Co-Excitation](https://papers.nips.cc/paper/8540-one-shot-object-detection-with-co-attention-and-co-excitation.pdf) (2019)\
[4] Koch et. al. [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) (2015)\
[5] Bhunia et. al. [A Deep One-Shot Network for Query-based Logo Retrieval](https://arxiv.org/pdf/1811.01395.pdf) (2019)\
[6] Hoi et. al. [LOGO-Net: Large-scale Deep Logo Detection and Brand Recognition with Deep Region-based Convolutional Networks](https://arxiv.org/pdf/1511.02462.pdf) (2015)

## Installation

This library is intended fo Linux-based OS and it's available on PyPI, so you can simply run `pip install logodetect` to install it.
Make sure you have Python version 3.7 or later installed and to have an up-to-date `pip` version
by running `pip install -U pip`. Also, we recommend working with virtual environments, but that
is ultimately up to you.

If you want to build `logodetect` from source, run

```bash_script
git clone git@github.com:Heldenkombinat/logodetect.git
cd logodetect
pip install -e ".[tests, dev]"
```

Depending on your system and setup, you might have to run the install command as `sudo`.

## Usage

After successful installation, a CLI tool called `logodetect` becomes available to you. If you invoke `logodetect`
without any arguments, you will get help on how to use it. To automatically download all models and data needed
to test the application first run the following command in your clone of this repository:

```bash_script
export LOGOS_RECOGNITION=$(pwd)
logodetect init
```

which will download all files to the current working directory. Note that if you prefer another folder to download the data,
please use the environment variable `LOGOS_RECOGNITION` accordingly. Consider putting this variable in your `.bash_rc`, `.zshrc` or an equivalent
configuration file on your system. If you don't specify a folder, it will default to `~/.hkt/logodetect`.

After running the `logodetect init` CLI, you'll find data and models relative to the specified folder in the  following
structure:

```text
data/
    exemplars/
    exemplars_100x100/
    exemplars_100x100_aug/
    exemplars_hq/
    test_images/
    test_videos/
models/
    classifier_resnet18.pth
    detector.pth
    embedder.pth
```

If you're interested in training your own algorithms, it's a good idea to have a look at how the exemplar data is
structured. For more on training, see the `training` folder and its readme.

The `logodetect` CLI tool comes with two main commands, namely `video` and `image`, both of which work
fairly similarly. In each case you need to provide the input data for which you would  like to detect logos,
and the logo exemplars that you want to detect in the footage. To get you started, we've provided
some demo data that you can use out of the box. That means you can simply run:

```bash_script
logodetect video
```

which should output the following text:

```text
Rendering video: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17/17 [00:00<00:00, 707.42it/s]
Moviepy - Building video /path/data/test_videos/test_video_small_50ms_output.mp4.
Moviepy - Writing video /path/data/test_videos/test_video_small_50ms_output.mp4

Moviepy - Done !
Moviepy - video ready /path/data/test_videos/test_video_small_50ms_output.mp4
All done! ✨ 🍰 ✨
```

to run one-shot detection on an example video, or you can run

```bash_script
logodetect image
```

to do so for an example image, which results in the following output:

```text
Saved resulting image as /path/data/test_images/test_image_small_output.png.
All done! ✨ 🍰 ✨
```

If you want to use another video, you can do so with the `-v` option. Images can be provided
with the `-i` option and custom exemplars are configured with the `-e` option. If you want to run `logodetect` with your
own, custom configuration, please provide a JSON file (like the `config.json` in this repo) with the `-c` option.
That means, if you want to run detection for custom video data with custom exemplars, you should use

```bash_script
logodetect video -v <path-to-video> -e <path-to-exemplars-folder> -c <path-to-custom-config-json>
```

### Minimal web application for image recognition

To run a small web app locally in your browser to upload images to recognize, simply run

```commandline
python app.py
```

and navigate to `https://localhost:5000` in the browser of your choice. Also, we've hosted an online
demo for you [here](https://logodetect.netlify.app/demo/).

On top of that, the `aws` folder explains in detail how to host this application yourself on Amazon Web
Services. This minimalistic application can of course be extended to your own needs at any point.

### Full set of CLI commands and help pages

In the last section we have already discussed the three commands exposed to users through the `logodetect`
CLI tool, namely `init`, `image`, and `video`. While `init` does not take any parameters, the other two
need a bit more explanation. Below you find the complete API reference from the respective help pages
of our CLI.

#### Images

```commandline
logodetect image --help
```

```text
Usage: logodetect image [OPTIONS]

Options:
  -i, --image_filename TEXT   path to your input image
  -c, --config_file TEXT      path to file containing a logodetect config JSON
  -o, --output_appendix TEXT  string appended to your resulting file
  -e, --exemplars TEXT        path to your exemplars folder
  --help                      Show this message and exit.
```

#### Videos

```commandline
logodetect video --help
```

```text
Usage: logodetect video [OPTIONS]

Options:
  -v, --video_filename TEXT   path to your input video
  -c, --config_file TEXT      path to file containing a logodetect config JSON
  -o, --output_appendix TEXT  string appended to your resulting file
  -e, --exemplars TEXT        path to your exemplars folder
  --help                      Show this message and exit.
```

## Core abstractions

`logodetect` works with a two-phased approach. In the first phase, objects get detected with 
a [`Detector`](https://github.com/Heldenkombinat/Logodetect/blob/master/logodetect/detectors/faster_rcnn.py#L21),
and then get compared to and identified with exemplars in a [`Classifier`](https://github.com/Heldenkombinat/Logodetect/blob/master/logodetect/classifiers/siamese.py#L19).
Both phases get integrated into the inference pipeline using a single [`Recognizer`](https://github.com/Heldenkombinat/Logodetect/blob/master/logodetect/recognizer.py#L27),
in which we detect potential overlay boxes in video frames or images and then the detected boxes
get labeled according to their classification.

## Configuration

The specific parameter settings of the algorithms used in `logodetect`, i.e. options for all of our detectors, 
classifiers, data augmenters, and system devices used, can be changed by providing a `config.json` file with the
`-c` flag in the main CLI commands explained above. The example `config.json` file explains the options you have
and what exactly you can modify in `logodetect`.

## Notebooks

You can find exemplary jupyter notebooks from the `logodetect` project in the `notebooks/` folder. If
you're interested in training new models, then `training/notebooks/` might interest you.

## Docker support

If you prefer to work with Docker, build an image and run it like this:

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

## Building the paper locally

```commandline
docker run --rm \
    --volume $PWD:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/paperdraft
```

