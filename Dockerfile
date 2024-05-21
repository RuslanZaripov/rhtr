FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git python3-pip cmake

RUN pip3 install torch torchvision torchaudio

COPY /backend/worker/requirements.txt /rhtr/requirements.txt
WORKDIR /rhtr
RUN pip3 install -r requirements.txt

COPY /backend/worker/download_models.py /rhtr/worker/download_models.py
RUN python3 ./worker/download_models.py

RUN pip3 uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
RUN pip3 install opencv-python-headless==4.8.0.74

RUN pip3 install flower

# Remember to copy necessary models inside docker container
COPY /models/segmentation/linknet_12_2.onnx /rhtr/models/segmentation/linknet_12_2.onnx

COPY /backend/worker /rhtr/worker

COPY /src/pipeline /rhtr/src/pipeline
COPY /src/segmentation /rhtr/src/segmentation

RUN touch /rhtr/src/__init__.py

RUN mkdir "images" # create directory to store images for debugging
