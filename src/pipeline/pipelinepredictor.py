import numpy as np

import src.ocr as ocr
import src.pipeline.config
import src.pipeline.segmpostproc
import src.pipeline.anglerestorer
import src.segmentation.predictor
import src.pipeline.linefinder


from src.pipeline.segmentor_utils import UNet
from src.pipeline.word_recognition_utils import TrOCR


def visualize(image, pred_img):
    import cv2
    import matplotlib.pyplot as plt

    image_copy = image.copy()
    for prediction in pred_img['predictions']:
        if prediction["class_name"] != "handwritten_text_shrinked_mask1": continue
        class2color = {  # BGR
            "handwritten_text_shrinked_mask1": (0, 255, 0),  # green # text
            "class_name_2": (0, 0, 255),  # red # comments
            "class_name_3": (255, 0, 0),  # blue # text_line
        }
        polygon = [tuple(point) for point in prediction["polygon"]]
        polygon_np = np.array(polygon, np.int32)
        polygon_np = polygon_np.reshape((-1, 1, 2))
        cv2.polylines(image_copy, [polygon_np],
                      isClosed=True,
                      color=class2color[prediction["class_name"]],
                      thickness=3)

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image_copy)
    plt.show()


class WordSegmentation:
    def __init__(
            self,
            model_path: str,
            config_path: str,
            pipeline_config: src.pipeline.config.Config,
    ):
        # self.segm_predictor = UNet()
        self.segm_predictor = src.segmentation.predictor.SegmPredictor(
            model_path=model_path,
            config_path=config_path,
        )

    def __call__(self, image, pred_img):
        pred_img = self.segm_predictor([image])[0]

        # visualize(image, pred_img)

        return image, pred_img


class OpticalCharacterRecognition:
    def __init__(
            self,
            model_path: str,
            config_path: str,
            ocr_classes: list[str],
            pipeline_config: src.pipeline.config.Config,
    ):
        self.ocr_classes = ocr_classes
        # self.recognizer = ocr.OCRTorchModel(model_path, config_path)
        self.recognizer = TrOCR()

    def __call__(self, image: np.ndarray, data: dict) -> tuple[np.ndarray, dict]:
        crops = [prediction['crop']
                 for prediction in data['predictions']
                 if prediction['class_name'] in self.ocr_classes]
        text_predictions = self.recognizer(crops)
        for prediction, text in zip(data['predictions'], text_predictions):
            prediction['text'] = text
        return image, data


class PipelinePredictor:
    STEPS_DICT = {
        'WordSegmentation': WordSegmentation,
        'OpticalCharacterRecognition': OpticalCharacterRecognition,
        'ContourPostprocessors': src.pipeline.segmpostproc.ContourPostprocessors,
        'RestoreImageAngle': src.pipeline.anglerestorer.ImageAngleRestorer,
        'LineFinder': src.pipeline.linefinder.LineFinder
    }

    def __init__(self, config_path: str):
        self.config = src.pipeline.config.Config(config_path)
        self.steps = [
            self.STEPS_DICT[step_name](**args, pipeline_config=self.config)
            for step_name, args in self.config['pipeline'].items()
        ]

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        data = None
        for step in self.steps:
            image, data = step(image, data)
        return image, data

    def get_prediction_classes(self):
        return self.config['pipeline']['OpticalCharacterRecognition']['ocr_classes']
