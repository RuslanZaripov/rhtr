import numpy as np

from src.pipeline.abstract import Segmentor
from src.pipeline.unet import UNet
from src.pipeline.utils import get_constructor_params, timeit, visualize, collect_params_from_dict


def segmentation_factory(args: dict) -> Segmentor:
    source = args['model_name']
    factory = {
        'UNet': UNet
    }
    if source in factory:
        segmentor = factory[source]
        params = get_constructor_params(segmentor)
        input_args = collect_params_from_dict(params, args)
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
        :return dict: {
            'predictions': [
                {
                    'polygon': np.ndarray (N, 2),
                    'bbox': tuple (4),
                    'class_name': str
                }
            ]
        }
        """
        data = self.segm_predictor.predict([image])[0]
        # visualize(image, data)
        return image, data
