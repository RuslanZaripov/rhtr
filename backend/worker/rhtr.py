import os
from os import getenv

import cv2
import numpy as np
from celery import Celery, states

import src.pipeline.pipeline_predictor
from .backend import is_backend_running, get_backend_url
from .broker import is_broker_running, get_broker_url
from .config import (
    TASK_QUEUES,
    WORKER_PREFETCH_MULTIPLIER,
    TASK_ACKS_LATE,
    RESULT_EXPIRES
)

if not is_broker_running():
    exit()

if not is_backend_running():
    exit()

rhtr_celery = Celery("rhtr", broker=get_broker_url(), backend=get_backend_url())
rhtr_celery.conf.task_queues = TASK_QUEUES
rhtr_celery.conf.worker_prefetch_multiplier = WORKER_PREFETCH_MULTIPLIER
rhtr_celery.conf.task_acks_late = TASK_ACKS_LATE
rhtr_celery.conf.result_expires = RESULT_EXPIRES

predictor = src.pipeline.pipeline_predictor.PipelinePredictor(
    config_path='src/pipeline/scripts/pipeline_config.json'
)

IMAGES_SAVE_DIR = getenv("IMAGES_SAVE_DIR", "/rhtr/shared")
os.makedirs(IMAGES_SAVE_DIR, exist_ok=True)


@rhtr_celery.task(bind=True, name="process_image")
def process_image(self, uuid):
    file_path = f'{IMAGES_SAVE_DIR}/{uuid}.txt'
    with open(file_path, 'rb') as file:
        data = file.read()

    if os.path.exists(file_path):
        os.remove(file_path)
        print("Delete image file")

    def bbox2xywh(bbox):
        x1, y1, x2, y2 = bbox
        return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    print(f"Image of size {image.shape=} accepted by worker")

    rotated_image, data = predictor.predict(image)

    filtered_words = filter(
        lambda prediction: prediction['class_name'] in predictor.get_prediction_classes(),
        data['predictions']
    )

    # print(f"{data['predictions'][0].keys()=}")

    # # Sorting by word_idx
    # sorted_predictions = sorted(filtered_words, key=lambda prediction: prediction['word_idx'])

    result = {
        "words": [{"word": prediction['text'], "rect": bbox2xywh(prediction['bbox'])}
                  for prediction in filtered_words]
    }
    print(f"{result=}")

    self.update_state(state=states.SUCCESS, meta={'words': result['words']})

    return result
