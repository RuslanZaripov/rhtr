import cv2
import numpy as np

import src.pipeline.config
from src.pipeline.anglerestorer import ImageAngleRestorer
from src.pipeline.linefinder import LineFinder
from src.pipeline.ocr_recognition import OpticalCharacterRecognition
from src.pipeline.segm_postprocessing import ContourPostprocessors
from src.pipeline.word_segmentation import WordSegmentation


class PipelinePredictor:
    STEPS_DICT = {
        'WordSegmentation': WordSegmentation,
        'OpticalCharacterRecognition': OpticalCharacterRecognition,
        'ContourPostprocessors': ContourPostprocessors,
        'RestoreImageAngle': ImageAngleRestorer,
        'LineFinder': LineFinder
    }

    def __init__(self, config_path: str):
        self.config = src.pipeline.config.Config(config_path)
        self.steps = [
            self.STEPS_DICT[step_name](args)
            for step_name, args in self.config['pipeline'].items()
        ]

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        data = None
        for step in self.steps:
            image, data = step(image, data)
        return image, data

    def get_prediction_classes(self):
        return self.config['pipeline']['OpticalCharacterRecognition']['ocr_classes']
