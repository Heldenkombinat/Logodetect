<p align="center">
  <img src="https://github.com/Heldenkombinat/Logos-Recognition/blob/master/docs/hkt_logo_detect.png">
</p>

[![PyPI version](https://badge.fury.io/py/logodetect.svg)](https://badge.fury.io/py/logodetect)

Never be embarrassed again to say that you have a small data situation! `logodetect` is a one-shot detection library to find logos of any kind in video and image data.

Here's a quick example of football footage that can detect all logos on jerseys and the sports field.
Go check out [our demo if you want to see it in action right away](https://logodetect.netlify.com/).

<p align="center">
  <img src="https://github.com/Heldenkombinat/Logos-Recognition/blob/master/docs/demo.gif">
</p>

## Introduction

There is plenty of literature on the use of deep-learning for detecting logos, so, additionally to sharing with you a couple of algorithms to get started with one-shot logo-detection, the aim of this project is to develop a flexible architecture to facilitate the comparison of different algorithms for one-shot object recognition.

The pipeline supports one or two stages, so it is possible to directly perform *object-recognition*, or to first perform *object-detection* and then *object-recognition*.
The idea is that you can use a generic detector for a single class of objects (e.g. logos, traffic signs or faces) and then compare each of its detections with the *exemplar* -the sub-class that you are trying to recognize- to determine if both belong to the same sub-class (e.g. a brand, a stop sign or your mom's face).
To get started, we include two algorithms that you can play with. Both have a Faster-RCNN [1] in the first stage that performs object-detection and they differ in the second stage that performs object-recognition.

As a **baseline**, we bring the exemplars and the detections from the first stage to the same latent space (this reduces the course of dimensionality) and then simply measure the euclidian or the cosine distance between both embeddings for object-recognition. Both inputs are considered to belong to the same sub-class if their distance is below a threshold determined in a preliminary analysis of the training dataset. The code also provides functionality to add various transformations, so you have the option to augment each exemplar with different transformations if you want. Simply add one or more exemplars into the `exemplars` folder and you are good to go.

<p align="center">
  <img src="https://github.com/Heldenkombinat/logodetect/blob/master/docs/distances_analysis.png">
</p>

As a **first reference** against the baseline, we also provide a modified ResNet [2] for object-recognition that directly takes the exemplars and the detections from the first stage and predicts if both belong tho the same sub-class. Similarly to [3], this network infers a distance metric after being trained with examples of different sub-classes, but instead of sharing the same weights and processing each input in a separate pass as in [4], it concatenates both inputs and processes them in one pass. This concept follows more closely the architecture proposed by [5], where the assumption is that the exemplars often have more high-frequency components than the detections, and therefore the model can increase its accuracy by learning a separate set of weights for each input.

The models that we are including in the repo achieved a reasonable performance after a few training epochs. However, if you would like to improve their performance you can find pointers to various datasets in [6]. Enjoy the code!

## References
[1] Ren et. al. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf) (2016)\
[2] He et. al. [Deep Residual Learning for Image Recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (2016)\
[3] Hsieh et. al. [One-Shot Object Detection with Co-Attention and Co-Excitation](https://papers.nips.cc/paper/8540-one-shot-object-detection-with-co-attention-and-co-excitation.pdf) (2019)\
[4] Koch et. al. [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) (2015)\
[5] Bhunia et. al. [A Deep One-Shot Network for Query-based Logo Retrieval](https://arxiv.org/pdf/1811.01395.pdf) (2019)\
[6] Hoi et. al. [LOGO-Net: Large-scale Deep Logo Detection and Brand Recognition with Deep Region-based Convolutional Networks](https://arxiv.org/pdf/1511.02462.pdf) (2015)

## Installation

This library is available on PyPI, so you can simply run `pip install logodetect` to install it.

If you want to build `logodetect` from source, run

```bash_script
git clone git@github.com:Heldenkombinat/logodetect.git
cd logodetect
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

