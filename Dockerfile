FROM pytorch/pytorch

MAINTAINER Max Pumperla "max.pumperla@gmail.com"

RUN apt update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 5000
ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
