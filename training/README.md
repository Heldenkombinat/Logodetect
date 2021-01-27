# Logo recognition training

The training part of `logodetect`, a package called `loreta`, is structurally based
on [this template](https://github.com/NullConvergence/mtorch).

## Installation
First, create a new virtual environment and activate it (we're using `conda` here, but you could
also use `virtualenv`s or similar):
```
conda create -n test python=3.7
conda activate test
```
Then clone this repository and install `loreta` with
```
pip install -e .
pip install -r requirements.txt
```

Also, make sure the right libraries are installed on your underlying system (dependencies
shown for Linux):

`sudo apt-get install libjpeg-dev zlib1g-dev`

## Usage

Choose a JSON file from the `configs` folder and run training with 

```
python loreta/app.py --config configs/default.json
```

However, note that the data this project is based on is not available in the public domain,
which is why it is not provided in this repository. This means that the above training script
won't work, unless you provide your own `data` first, e.g. obtained from pointers in [1].

## Data Sources

[1] Hoi et. al. [LOGO-Net: Large-scale Deep Logo Detection and Brand Recognition with Deep Region-based Convolutional Networks](https://arxiv.org/pdf/1511.02462.pdf) (2015)
