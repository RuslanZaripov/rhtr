import os

IMAGE_DIR = 'data/handwritten_text_images'
IMAGE_PATHS = os.listdir(IMAGE_DIR)
IMAGE_PATHS = [image for image in IMAGE_PATHS if image.endswith('.jpg')]

API_URL = 'http://api.statanly.com:9023'
