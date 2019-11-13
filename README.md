# Logo recognition

### Installation
- (Optional) Create a new virtual environment and activate it:
```
conda create -n test python=3
conda activate test
```
- Clone repository
- Install with `pip install -e .` from the `Logos-Recognition` folder.

### Usage
Friendly reminder: always use virtual environments! Otherwise replace `python` with `python3`.
- Download data and models from the [Google Drive folder](https://drive.google.com/drive/u/1/folders/12CQkp3K4QdzfNRtsLjkc_iqbT-IZVwAq)
- Run test case with `python Logos-Recognition/logos_recognition/app.py`
- By default, output video will be saved as `Logos-Recognition/data/test_video_output.mp4`

### Current capabilities
- Load video
- Load detector model
- Detect logos of any kind
- Save output video