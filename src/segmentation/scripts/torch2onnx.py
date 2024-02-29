import torch

from pathlib import Path
import argparse

from src.segmentation.predictor import SegmTorchModel
from src.segmentation.config import Config
from src.segmentation.utils import configure_logging


def main(args):
    logger = configure_logging()

    segm_torch_model = SegmTorchModel(
        model_path=args.model_path,
        config_path=args.config_path,
        device='cpu'
    )
    config = Config(args.config_path)

    onnx_path = Path(args.model_path)
    onnx_path = onnx_path.parents[0] / onnx_path.stem
    onnx_path = str(onnx_path) + '.onnx'

    example_forward_input = torch.rand(
        1, 3, config.get_image('height'), config.get_image('width'))

    torch.onnx.export(segm_torch_model.model,
                      example_forward_input,
                      onnx_path,
                      opset_version=12,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}}
                      )
    logger.info(f"ONNX model was saved to '{onnx_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='/workdir/scripts/segm_config.json',
                        help='Path to config.json.')
    parser.add_argument('--model_path', type=str,
                        help='Path to torch model weights.')
    main(parser.parse_args())
