import os
from os import getenv

import cv2
import numpy as np
from celery import Celery, states
from kombu import Queue

import src.pipeline.pipelinepredictor
from .backend import is_backend_running, get_backend_url
from .broker import is_broker_running, get_broker_url

if not is_broker_running():
    exit()

if not is_backend_running():
    exit()

predictor = src.pipeline.pipelinepredictor.PipelinePredictor(
    config_path='src/pipeline/scripts/pipeline_config.json'
)

rhtr_celery = Celery("rhtr", broker=get_broker_url(), backend=get_backend_url())
rhtr_celery.conf.task_queues = (
    Queue(name="rhtr"),
)
rhtr_celery.conf.worker_prefetch_multiplier = 1
rhtr_celery.conf.task_acks_late = True
rhtr_celery.conf.result_expires = 60 * 60 * 48  # 48 hours in seconds

IMAGES_DIR = getenv("IMAGES_DIR", "/rhtr/shared")
os.makedirs(IMAGES_DIR, exist_ok=True)


@rhtr_celery.task(bind=True, name="process_image")
def process_image(self, uuid):

    file_path = f'{IMAGES_DIR}/{uuid}.txt'
    with open(file_path, 'rb') as file:
        data = file.read()

    if os.path.exists(file_path):
        os.remove(file_path)
        print("Delete image file")

    def bbox2xywh(bbox):
        x1, y1, x2, y2 = bbox
        return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    print(f"Input image: {image.shape=}")
    rotated_image, data = predictor.predict(image)

    filtered_words = filter(
        lambda prediction: prediction['class_name'] in predictor.get_prediction_classes(),
        data['predictions']
    )

    # print(f"{data['predictions'][0].keys()=}")
    # sorted_predictions = sorted(filtered_words, key=lambda prediction: prediction['word_idx'])
    # print(f"{len(sorted_predictions)=}")

    result = {
        "words": [{"word": prediction['text'], "rect": bbox2xywh(prediction['bbox'])}
                  for prediction in filtered_words]
    }
    print(f"{result=}")

    self.update_state(state=states.SUCCESS, meta={'words': result['words']})

    return result
