from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from itertools import chain
import torch


def batch(iterable, batch_size):
    """Yield successive batches of given size from the iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def flatten(matrix):
    return list(chain.from_iterable(matrix))


class TrOCR:
    def __init__(self):
        # raxtemur/trocr-base-ru
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.processor = TrOCRProcessor.from_pretrained("raxtemur/trocr-base-ru")
        self.trained_model = VisionEncoderDecoderModel.from_pretrained("raxtemur/trocr-base-ru").to(self.device)

    def __call__(self, images):
        return flatten([self.ocr(b) for b in batch(images, 100)])

    def ocr(self, images):
        # We can directly perform OCR on cropped images.
        pixel_values = self.processor(images, return_tensors='pt').pixel_values.to(self.device)
        self.trained_model.eval()
        with torch.no_grad(), torch.inference_mode():
            generated_ids = self.trained_model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text
