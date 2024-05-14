from itertools import chain

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.pipeline.abstract import Recognizer


def get_model(model_path):
    """Load a Hugging Face model and tokenizer from the specified directory"""
    processor = TrOCRProcessor.from_pretrained(model_path)
    trained_model = VisionEncoderDecoderModel.from_pretrained(model_path)
    return processor, trained_model


def batch(iterable, batch_size):
    """Yield successive batches of given size from the iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def flatten(matrix):
    return list(chain.from_iterable(matrix))


class TrOCR(Recognizer):
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = get_model(model_path)
        self.processor = model[0]
        self.trained_model = model[1].to(self.device)

    def predict(self, images):
        return flatten([self.ocr(b) for b in batch(images, 50)])

    def ocr(self, images):
        # We can directly perform OCR on cropped images.
        pixel_values = self.processor(images, return_tensors='pt').pixel_values.to(self.device)
        self.trained_model.eval()
        with torch.no_grad(), torch.inference_mode():
            generated_ids = self.trained_model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text
