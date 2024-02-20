import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import src.pipeline.pipelinepredictor

rhtr_api = FastAPI()

rhtr_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@rhtr_api.post("/uploadfile/")
async def create_upload_files(file: UploadFile):
    def bbox2xywh(bbox):
        x1, y1, x2, y2 = bbox
        return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}

    image = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    print(f"Input image: {image.shape=}")

    predictor = src.pipeline.pipelinepredictor.PipelinePredictor(
        config_path='src/pipeline/scripts/pipeline_config.json'
    )
    rotated_image, data = predictor.predict(image)

    result = {"filenames": file.filename,
              "words": [{"word": prediction['text'], "rect": bbox2xywh(prediction['bbox'])}
                        for prediction in data['predictions']]
              }
    print(f"{result=}")
    return result


@rhtr_api.get("/hello")
def home():
    return {"msg": "Hello from the rhtr backend!"}


if __name__ == "__main__":
    """
    uvicorn api.main:rhtr_api --reload
    """
    import uvicorn

    uvicorn.run(rhtr_api, host="127.0.0.1", port=8000)
