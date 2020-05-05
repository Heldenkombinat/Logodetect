#!/bin/bash
mkdir logo

sudo apt-get update
sudo apt-get install -y apache2 apache2-dev
sudo apt-get install -y libapache2-mod-wsgi-py3
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y libsm6 libxext6 libxrender-dev ffmpeg

cd logo
sudo pip3 install -y virtualenv
virtualenv venv -p python3.6 && source venv/bin/activate

pip3 install -e ."[dev]"
pip install pyopenssl


export LOGOS_RECOGNITION="/home/ubuntu/logo"

# Flask on the nose
# python app.py

# Gunicorn (slightly better)
# gunicorn --bind 0.0.0.0:5000 wsgi:app

# Apache2 reverse proxy fro gunicorn (much better)
sudo ln -sT ~/logo /var/www/html/logo
sudo cp apache.conf /etc/apache2/sites-enabled/000-default.conf
sudo service apache2 restart
