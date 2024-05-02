import os

import cv2
import numpy as np
from celery import Celery, states

import src.pipeline.pipelinepredictor
from os import getenv

from redis import Redis
from redis.exceptions import ConnectionError

from kombu import Connection
from kombu.exceptions import OperationalError

from kombu import Queue


# --------------------------- REDIS ---------------------------

def get_redis_password() -> str:
    return getenv("REDIS_PASS", "password")


def get_redis_port() -> str:
    return getenv("REDIS_PORT", "6379")


def get_redis_dbnum() -> str:
    return getenv("REDIS_DB", "0")


def get_redis_host() -> str:
    return getenv("REDIS_HOST", "127.0.0.1")


def get_backend_url() -> str:
    pw = get_redis_password()
    port = get_redis_port()
    db = get_redis_dbnum()
    return "redis://{password}{hostname}{port}{db}".format(
        hostname=get_redis_host(),
        password=':' + pw + '@' if len(pw) != 0 else '',
        port=':' + port if len(port) != 0 else '',
        db='/' + db if len(db) != 0 else ''
    )


def is_backend_running() -> bool:
    try:
        conn = Redis(
            host=get_redis_host(),
            port=int(get_redis_port()),
            db=int(get_redis_dbnum()),
            password=get_redis_password()
        )
        conn.client_list()  # Must perform an operation to check connection.
    except ConnectionError as e:
        print("Failed to connect to Redis instance at %s", get_redis_host())
        print(repr(e))
        return False

    conn.close()  # type: ignore

    return True


# --------------------------- RABBIT ---------------------------

def get_rabbitmq_userpass() -> str:
    RABBITMQ_USER = getenv("RABBITMQ_USER", "guest")
    RABBITMQ_PASS = getenv("RABBITMQ_PASS", "guest")
    return RABBITMQ_USER + ":" + RABBITMQ_PASS + "@" if RABBITMQ_USER else ""


def get_rabbitmq_port() -> str:
    RABBITMQ_PORT = getenv("RABBITMQ_PORT", "5672")
    return ":" + RABBITMQ_PORT if RABBITMQ_PORT else ""


def get_rabbitmq_vhost() -> str:
    RABBITMQ_VHOST = getenv("RABBITMQ_VHOST", "")
    return "/" + RABBITMQ_VHOST if RABBITMQ_VHOST else ""


def get_rabbitmq_host() -> str:
    return getenv("RABBITMQ_HOST", "127.0.0.1")


def get_broker_url() -> str:
    return "amqp://{userpass}{hostname}{port}{vhost}".format(
        hostname=get_rabbitmq_host(),
        userpass=get_rabbitmq_userpass(),
        port=get_rabbitmq_port(),
        vhost=get_rabbitmq_vhost()
    )


def is_broker_running(retries: int = 3) -> bool:
    try:
        conn = Connection(get_broker_url())
        conn.ensure_connection(max_retries=retries)
    except OperationalError as e:
        print("Failed to connect to RabbitMQ instance at %s", get_rabbitmq_host())
        print(str(e))
        return False

    conn.close()
    return True


# --------------------------- WORKER ---------------------------

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


@rhtr_celery.task(bind=True, name="process_image")
def process_image(self, uuid):

    file_path = f'/rhtr/shared/{uuid}.txt'
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

    self.update_state(state=states.SUCCESS, meta={'custom': 'Image processed'})

    # print(f"{data['predictions'][0].keys()=}")
    # sorted_predictions = sorted(filtered_words, key=lambda prediction: prediction['word_idx'])
    # print(f"{len(sorted_predictions)=}")

    result = {
        "words": [{"word": prediction['text'], "rect": bbox2xywh(prediction['bbox'])}
                  for prediction in filtered_words]
    }
    print(f"{result=}")
    return result
