#!/bin/bash

# FOR LINUX

cd ~/Documents/antproject/codebase/

# build docker image

sudo docker build -t antproject .

# run docker image and launch shell

sudo docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /home/maelstrom/Documents/antproject/data:/app/antproject/data/ \
       	-e DISPLAY=:0 \
       	-u qtuser --rm --gpus all \
       	--name antprojectproc \
	antproject
