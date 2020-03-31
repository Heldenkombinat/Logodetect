#!/bin/bash

scp ../backup_constants.py  aws_logodetect:~/logo/
scp ../backup_constants.py  aws_logodetect:~/logo/constants.py # for good measure
scp ../setup.py  aws_logodetect:~/logo/
scp ../requirements.txt  aws_logodetect:~/logo/
scp ../app.py  aws_logodetect:~/logo/
scp wsgi.py  aws_logodetect:~/logo/
scp logo.wsgi  aws_logodetect:~/logo/

scp -r ../logodetect aws_logodetect:~/logo/
scp -r ../models aws_logodetect:~/logo/
scp -r ../data aws_logodetect:~/logo/

scp aws_install.sh aws_logodetect:~/