import numpy as np

from src.pipeline.abstract import Recognizer
from src.pipeline.tr_ocr import TrOCR
from src.pipeline.utils import get_constructor_params, timeit


def recognizer_factory(args: dict) -> Recognizer:
    source = args['model_name']
    factory = {
        'TrOCR': TrOCR,
    }
    if source in factory:
        recognoizer = factory[source]
        params = get_constructor_params(recognoizer)
        input_args = {
            param: args[param]
            if param in args.keys()
            else print(f"WARN: param {param} not found in args")
            for param in params
        }
        return recognoizer(**input_args)
    else:
        raise ValueError(f"source {source} is not supported. Please pass a valid source.")


class OpticalCharacterRecognition:
    def __init__(self, args):
        print(f"{self.__class__.__name__} input {args}")
        self.ocr_classes = args['ocr_classes']
        self.recognizer: Recognizer = recognizer_factory(args)

    @timeit
    def __call__(self, image: np.ndarray, data: dict) -> tuple[np.ndarray, dict]:
        # print(f"{data['predictions'][0].keys()=}")
        crops = [prediction['crop']
                 for prediction in data['predictions']
                 if prediction['class_name'] in self.ocr_classes]

        text_predictions = self.recognizer.predict(crops)

        for prediction, text in zip(data['predictions'], text_predictions):
            prediction['text'] = text
        return image, data
