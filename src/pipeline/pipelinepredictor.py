import numpy as np

import src.ocr as ocr
import src.pipeline.config
import src.pipeline.segmpostproc
import src.pipeline.anglerestorer
import src.segmentation.predictor
import src.pipeline.linefinder


class WordSegmentation:
    def __init__(
            self,
            model_path: str,
            config_path: str,
            pipeline_config: src.pipeline.config.Config,
    ):
        self.segm_predictor = src.segmentation.predictor.SegmPredictor(
            model_path=model_path,
            config_path=config_path,
        )

    def __call__(self, image, pred_img):
        pred_img = self.segm_predictor([image])[0]
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
        self.recognizer = ocr.OCRTorchModel(model_path, config_path)

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
