import os
import shutil
from os import getenv
from typing import Optional, Any

from celery import Celery, states
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.status import HTTP_200_OK, HTTP_201_CREATED

rhtr_api = FastAPI()

rhtr_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RABBITMQ_HOST = getenv("RABBITMQ_HOST", "127.0.0.1")
RABBITMQ_PORT = getenv("RABBITMQ_PORT", "5672")
RABBITMQ_USER = getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = getenv("RABBITMQ_PASS", "guest")
RABBITMQ_VHOST = getenv("RABBITMQ_VHOST", "")

# RabbitMQ's connection string: amqp://user:pass@localhost:5672/myvhost
BROKER = "amqp://{userpass}{hostname}{port}{vhost}".format(
    hostname=RABBITMQ_HOST,
    userpass=RABBITMQ_USER + ":" + RABBITMQ_PASS + "@" if RABBITMQ_USER else "",
    port=":" + RABBITMQ_PORT if RABBITMQ_PORT else "",
    vhost="/" + RABBITMQ_VHOST if RABBITMQ_VHOST else ""
)

REDIS_HOST = getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = getenv("REDIS_PORT", "6379")
REDIS_PASS = getenv("REDIS_PASS", "password")
REDIS_DB = getenv("REDIS_DB_BACKEND", "0")

BACKEND = "redis://{password}{hostname}{port}{db}".format(
    hostname=REDIS_HOST,
    password=':' + REDIS_PASS + '@' if REDIS_PASS else '',
    port=":" + REDIS_PORT if REDIS_PORT else "",
    db="/" + REDIS_DB if REDIS_DB else ""
)

app_celery = Celery(broker=BROKER, backend=BACKEND)

IMAGES_SAVE_DIR = getenv("IMAGES_SAVE_DIR", "/api/shared")
os.makedirs(IMAGES_SAVE_DIR, exist_ok=True)


def send_result(task_id):
    while True:
        result = app_celery.AsyncResult(task_id)
        if result.state in states.READY_STATES:
            break

    output = TaskResult(
        id=task_id,
        status=result.state,
        error=str(result.info) if result.failed() else None,
        result=result.get() if result.state == states.SUCCESS else None
    )

    print(output)


@rhtr_api.post("/uploadfile", status_code=HTTP_201_CREATED)
def celery_upload_file(file: UploadFile, queue: BackgroundTasks):
    import uuid
    uuid = uuid.uuid4()

    content = file.file.read()
    with open(f"{IMAGES_SAVE_DIR}/{uuid}.txt", "wb") as binary_file:
        binary_file.write(content)

    task = app_celery.send_task(
        name="process_image",
        args=[uuid],
        queue='rhtr'
    )

    print(f"Created task with {task.id}")

    queue.add_task(send_result, task.id)

    return {"task_id": task.id, "file": file.filename}


class TaskResult(BaseModel):
    id: str
    status: str
    error: Optional[str] = None
    result: Optional[Any] = None


@rhtr_api.get("/task/{task_id}")
def celery_get_task(task_id: str):
    result = app_celery.AsyncResult(task_id)

    output = TaskResult(
        id=task_id,
        status=result.state,
        error=str(result.info) if result.failed() else None,
        result=result.get() if result.state == states.SUCCESS else None
    )

    return JSONResponse(
        status_code=HTTP_200_OK,
        content=output.dict()
    )


@rhtr_api.get("/hello")
def home():
    return {"msg": "Hello from the rhtr backend!"}


if __name__ == "__main__":
    """
    uvicorn api.main:rhtr_api --reload
    """
    import uvicorn

    uvicorn.run(rhtr_api, host="127.0.0.1", port=8000)
