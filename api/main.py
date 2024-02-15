import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse

from src.pipeline.pipelinepredictor import PipelinePredictor

rhtr_api = FastAPI()


@rhtr_api.post("/uploadfile/")
async def create_upload_files(file: UploadFile):
    image = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    print(f"Input image: {image.shape=}")

    predictor = PipelinePredictor(config_path='src/pipeline/scripts/pipeline_config.json')
    rotated_image, data = predictor.predict(image)

    text = " ".join([prediction['text'] for prediction in data['predictions']])
    return {"filenames": file.filename, "text": text}


@rhtr_api.get("/")
async def main():
    content = """
<body>
<form action="/uploadfile/" enctype="multipart/form-data" method="post">
<input name="file" type="file">
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
