import numpy as np

from src.pipeline.abstract import Segmentor
from src.pipeline.unet import UNet
from src.pipeline.utils import get_constructor_params, timeit, visualize


def segmentation_factory(args: dict) -> Segmentor:
    source = args['model_name']
    factory = {
        'UNet': UNet
    }
    if source in factory:
        segmentor = factory[source]
        params = get_constructor_params(segmentor)
        input_args = {
            param: args[param]
            if param in args.keys()
            else print(f"WARN: param {param} not found in args")
            for param in params
        }
        return segmentor(**input_args)
    else:
        raise ValueError(f"source {source} is not supported. Please pass a valid source.")


class WordSegmentation:
    def __init__(self, args):
        print(f"{self.__class__.__name__} input {args}")
        self.segm_predictor: Segmentor = segmentation_factory(args)

    @timeit
    def __call__(self, image: np.ndarray, data: dict) -> tuple[np.ndarray, dict]:
        """
        :return dict: { 'predictions': [
            { 'polygon': np.ndarray (N, 2),
              'bbox': tuple (4),
              'class_name': str
            }
        ] }
        """

        data = self.segm_predictor.predict([image])[0]
        print(f"{data['predictions'][0]['polygon'].shape=}")
        # visualize(image, data)
        return image, data
