# FROM hdgigante/python-opencv:4.9.0-alpine
FROM python:3.10-slim
# FROM pachyderm/opencv:2.6.0

COPY /backend/api/requirements.txt /rhtr/requirements.txt

COPY /backend/api/download_models.py /rhtr/download_models.py

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

RUN python download_models.py

COPY /backend/api/main.py /rhtr/api/main.py

COPY /src/pipeline /rhtr/src/pipeline

RUN touch /rhtr/src/__init__.py

COPY /models/segmentation/linknet-7.onnx /rhtr/models/segmentation/linknet-7.onnx

# EXPOSE <port> [<port>/<protocol>...]
EXPOSE 8000

CMD uvicorn api.main:rhtr_api --port 8000 --host 0.0.0.0
