from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


class TrOCR:
    def __init__(self):
        # raxtemur/trocr-base-ru
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.processor = TrOCRProcessor.from_pretrained("raxtemur/trocr-base-ru")
        self.trained_model = VisionEncoderDecoderModel.from_pretrained("raxtemur/trocr-base-ru").to(self.device)

    def __call__(self, images):
        return self.ocr(images)

    def ocr(self, images):
        # We can directly perform OCR on cropped images.
        pixel_values = self.processor(images, return_tensors='pt').pixel_values.to(self.device)
        generated_ids = self.trained_model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text
