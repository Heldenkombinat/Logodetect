from setuptools import setup
from setuptools import find_packages

setup(
    name="logodetection",
    version="0.1",
    description="One-shot logo detection for videos and images.",
    long_description="One-shot logo detection for videos and images.",
    url="https://github.com/Heldenkombinat/Logos-Recognition",
    download_url="https://github.com/Heldenkombinat/Logos-Recognition/tarball/0.1",
    author="Jorge Davila Chacon",
    author_email="jorge@heldenkombinat.com",
    install_requires=[
        "Cython>=0.29.15",
        "imgaug>=0.4.0",
        "matplotlib>=3.2.0rc3",
        "moviepy>=1.0.1",
        "numpy>=1.18.1",
        "opencv-python>=4.2.0.32",
        "pandas>=1.0.1",
        "scipy>=1.4.1",
        "sklearn",
        "torch>=1.4.0",
        "torchvision>=0.5.0",
        "tqdm>=4.42.1",
    ],
    packages=find_packages(),
    license="GNU AGPLv3",
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Environment :: Console",
        "License :: OSI Approved :: GNU Affero General Public License v3.0",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)
