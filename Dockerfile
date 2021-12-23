FROM tensorflow/tensorflow:2.7.0-gpu

ENV DEBIAN_FRONTEND=noninteractive

RUN adduser --quiet --disabled-password qtuser && usermod -a -G audio qtuser

ENV LIBGL_ALWAYS_INDIRECT=1

WORKDIR /app/antproject/codebase

COPY . .

COPY requirements.txt .

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y python3-pyqt5 git

RUN git clone https://github.com/murthylab/sleap.git

RUN pip install ./sleap

RUN pip install -r requirements.txt

WORKDIR /app/antproject

RUN mkdir data

CMD ["/bin/bash"]
