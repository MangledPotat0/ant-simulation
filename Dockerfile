FROM tensorflow/tensorflow:2.7.0-gpu

ENV DEBIAN_FRONTEND=noninteractive

RUN adduser --quiet --disabled-password qtuser && usermod -a -G audio qtuser

ENV LIBGL_ALWAYS_INDIRECT=1

WORKDIR /app/antproject/codebase

COPY . .

COPY requirements.txt .

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y python3-pyqt5 ffmpeg

RUN pip install -r requirements.txt

WORKDIR /app/antproject

RUN mkdir data

#COPY inference.py /usr/local/lib/python3.8/dist-packages/sleap/nn/.

CMD ["/bin/bash"]
