import random
import time
import requests
import os

from locust import HttpUser, task, between, events
from celery import states

from backend.client.client import make_image_post
from backend.client.config import IMAGE_PATHS, IMAGE_DIR


class MyUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def hello(self):
        self.client.get("/hello")

    def get_result(self, task_id):
        return requests.get(f"{self.client.base_url}/task/{task_id}").json()

    @task
    def recognize(self):
        image_file = random.choice(IMAGE_PATHS)
        image_path = os.path.join(IMAGE_DIR, image_file)

        status, task_id = make_image_post(str(image_path))

        start = time.time()

        response = self.get_result(task_id)
        while response['status'] != states.SUCCESS:
            response = self.get_result(task_id)

        end = time.time()

        events.request.fire(
            request_type="POST",
            name="recognize",
            response_time=int((end - start) * 1000),
            response_length=0,
        )
