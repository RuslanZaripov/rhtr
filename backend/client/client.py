import os
import requests

from celery import states
from starlette.status import HTTP_201_CREATED

from joblib import Parallel, delayed
from retrying import retry

from typing import (
    Any,
    Dict,
    Tuple
)

IMAGE_DIR = 'data/handwritten_text_images'
IMAGE_PATHS = os.listdir(IMAGE_DIR)
IMAGE_PATHS = [image for image in IMAGE_PATHS if image.endswith('.jpg')]

API_URL = 'http://api.statanly.com:9023'

ENDPOINT_RECOGNIZE = API_URL + "/uploadfile"

ENDPOINT_TASK_RESULT = API_URL + "/task/{}"


def make_image_post(image_path: str) -> Tuple[int, str]:
    response = requests.post(ENDPOINT_RECOGNIZE, files={'file': open(image_path, mode='rb')})
    task_id = response.json()['task_id'] if response.status_code == HTTP_201_CREATED else None
    return response.status_code, task_id


@retry(stop_max_attempt_number=10, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def get_result(task_id: str) -> Dict[Any, Any]:
    response = requests.get(ENDPOINT_TASK_RESULT.format(task_id))
    status = response.json()['status']

    if status == states.PENDING:
        raise Exception("Task on progress")

    print(f"Task {task_id} processed")

    return response.json()


if __name__ == "__main__":
    print("Sending recognize")
    recognize_tasks = Parallel(n_jobs=2, prefer="threads")(
        delayed(make_image_post)(f'{IMAGE_DIR}/{image_path}')
        for image_path in IMAGE_PATHS
    )

    tasks = recognize_tasks

    input("Press Enter to get the results...")

    print("Geting results")

    results = Parallel(n_jobs=2, prefer="threads")(
        delayed(get_result)(task_id)
        for (status, task_id) in tasks if status == HTTP_201_CREATED
    )

    for idx, data in enumerate(results):
        output = data['result'] if data['status'] == 'SUCCESS' else data['error']
        print(f"{idx + 1:2d} - {data['id']} - {data['status']} - {output}")
        print()
