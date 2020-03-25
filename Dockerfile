FROM python:3.7

MAINTAINER Max Pumperla "max.pumperla@gmail.com"

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app
COPY backup_constants.py /app/constants.py

EXPOSE 5000
ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
