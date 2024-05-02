from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os


def download_model(model_path, model_name):
    """Download a Hugging Face model and processor to the specified directory"""
    # Check if the directory already exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Save the model and processor to the specified directory
    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)


# For this demo, download the English-French and French-English models
download_model('models/tr_ocr/', 'raxtemur/trocr-base-ru')
