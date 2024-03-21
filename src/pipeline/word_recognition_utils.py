from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


class TrOCR:
    def __init__(self):
        # raxtemur/trocr-base-ru
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.processor = TrOCRProcessor.from_pretrained("raxtemur/trocr-base-ru")
        self.trained_model = VisionEncoderDecoderModel.from_pretrained("raxtemur/trocr-base-ru").to(self.device)

    def __call__(self, images):
        texts = []
        for image in images:
            # plt.imshow(image)
            # plt.show()

            text = self.ocr(image)
            texts.append(text)
        return texts

    def ocr(self, image):
        # We can directly perform OCR on cropped images.
        pixel_values = self.processor(image, return_tensors='pt').pixel_values.to(self.device)
        generated_ids = self.trained_model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
