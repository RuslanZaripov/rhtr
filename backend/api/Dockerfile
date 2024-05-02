FROM python:3.10-slim

COPY requirements.txt /api/requirements.txt

WORKDIR /api

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --compile -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

COPY main.py /api/main.py

EXPOSE 8000

CMD uvicorn main:rhtr_api --port 8000 --host 0.0.0.0
