import random
import time
import requests

from locust import HttpUser, task, between, events
import os

images_root = 'data/handwritten_text_images'
image_files = os.listdir(images_root)
image_files = [image for image in image_files if image.endswith('.jpg')]


class MyUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def hello(self):
        self.client.get("/hello")

    def get_result(self, task_id):
        return requests.get(f"{self.client.base_url}/task/{task_id}").json()

    def uploadfile(self, image_path):
        return requests.post(f"{self.client.base_url}/uploadfile",
                             files={'file': open(image_path, mode='rb')}).json()

    @task
    def recognize(self):
        image_file = random.choice(image_files)
        image_path = os.path.join(images_root, image_file)

        response = self.uploadfile(image_path)
        task_id = response['task_id']

        start = time.time()

        response = self.get_result(task_id)

        while response['status'] != 'SUCCESS':
            response = self.get_result(task_id)

        end = time.time()

        events.request.fire(
            request_type="POST",
            name="recognize",
            response_time=int((end - start) * 1000),
            response_length=0,
        )
