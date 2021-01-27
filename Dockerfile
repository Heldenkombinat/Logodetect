FROM pytorch/pytorch

MAINTAINER Max Pumperla "max.pumperla@gmail.com"

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app
COPY constants.py /app/constants.py

EXPOSE 5000
ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
