#!/bin/bash

# remove old wheels
sudo rm -rf dist/*

# Build Python 2 & 3 wheels for current version
sudo python3 setup.py sdist bdist_wheel

twine upload dist/*
