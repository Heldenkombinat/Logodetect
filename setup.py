from setuptools import setup
from setuptools import find_packages

setup(
    name="logodetect",
    version="1.1.0",
    description="One-shot logo detection for videos and images.",
    long_description="One-shot logo detection for videos and images.",
    url="https://github.com/Heldenkombinat/logodetect",
    download_url="https://github.com/Heldenkombinat/logodetect/tarball/0.1",
    author="Jorge Davila Chacon",
    author_email="jorge@heldenkombinat.com",
    install_requires=[
        "Cython>=0.29.15",
        "click>=7.1.1",
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
    extras_require={
        "tests": ["pytest", "pytest-pep8", "pytest-cov", "mock"],
        "dev": ["black", "pre-commit", "flask", "gunicorn"],
    },
    entry_points={"console_scripts": ["logodetect=logodetect.cli:cli"]},
    packages=find_packages(),
    license="GNU AGPLv3",
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Environment :: Console",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)
