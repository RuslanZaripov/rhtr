import torch
import src.ocr


def predict(images, model, decoder, device):
    """
    Make model prediction.

    images (torch.Tensor):
        Batch with tensor images.
    model (ocr.src.models.CRNN):
        OCR model.
    decoder: (ocr.tokenizer.OCRDecoder)
    device (torch.device):
        Torch device.
    """
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    label_preds = decoder.decode(output)
    return label_preds


class OCRTorchModel:
    def __init__(self, model_path, config_path):
        self.config = src.ocr.Config(config_path)

        self.tokenizer = src.ocr.Tokenizer(self.config.get('alphabet'))

        self.decoder = src.ocr.BestPathDecoder(self.tokenizer)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = src.ocr.CRNN(1,
                                  self.config.get_image('height'),
                                  self.config.get_image('width'),
                                  self.tokenizer.get_num_chars(),
                                  map_to_seq_hidden=64,
                                  rnn_hidden=255,
                                  leaky_relu=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        self.transforms = src.ocr.DefaultBatchPreprocessor(
            height=self.config.get_image('height'),
            width=self.config.get_image('width'),
        )

    def __call__(self, images):
        return predict(
            self.transforms(images),
            self.model, self.decoder, self.device
        )
