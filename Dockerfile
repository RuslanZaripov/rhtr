# FROM hdgigante/python-opencv:4.9.0-alpine
FROM python:3.10-slim
# FROM pachyderm/opencv:2.6.0

COPY /backend/worker/requirements.txt /rhtr/requirements.txt

WORKDIR /rhtr

# RUN apk add --no-cache gcc libc-dev geos-dev
# RUN apk add gcc libc-dev geos-dev
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get -y --no-install-recommends install \
        ffmpeg libsm6 libxext6 -y

# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir --compile -r requirements.txt && \
#     rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY /backend/worker/download_models.py /rhtr/worker/download_models.py

RUN python ./worker/download_models.py

COPY /models/segmentation/linknet-7.onnx /rhtr/models/segmentation/linknet-7.onnx

COPY /backend/worker /rhtr/worker

COPY /src/pipeline /rhtr/src/pipeline

RUN touch /rhtr/src/__init__.py

RUN pip install flower