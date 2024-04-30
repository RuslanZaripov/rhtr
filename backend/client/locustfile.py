import random

from locust import HttpUser, task, between
import os

images_root = 'C:\\Users\\rusla\\Desktop\\ITMO\\7-sem\\thesis\\rhtr\\data\\handwritten_text_images'
image_files = os.listdir(images_root)
image_files = [image for image in image_files if image.endswith('.jpg')][:2]


class MyUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def hello(self):
        self.client.get("/hello")

    @task
    def recognize(self):
        image_file = random.choice(image_files)
        image_path = os.path.join(images_root, image_file)
        self.client.post("/uploadfile", files={'file': open(image_path, mode='rb')})
