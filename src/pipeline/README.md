# Recognition process

- Read this in other languages: [Russian](README.ru.md)

## Structure description

- `abstract.py` - contains abstract classes of recognition modules to generalize the code structure
- `anglerestorer.py` - contains code for calculating the angle of rotation of the image
- `config.py` - contains the code necessary to process the configuration file
- `linefinder.py` - contains the code necessary for composing text fragments
- `ocr_recognition.py` - contains code to initialize the recognition stage
- the `recognizer_factory` method is responsible for selecting a model based on configuration files
- `word_segmentation.py` - contains code to initialize the segmentation stage
- the `segmentation_factory` method is responsible for selecting a model based on the configuration file
- `segm_postprocessing.py` - contains code for initializing the contour postprocessing stage
- `tr_ocr.py` - contains code for the recognition model trained by the `transformers` library
- `unet.py` - contains the code of the segmentation model trained in the `segmentation` module
- `utils.py` - contains auxiliary code for the recognition process (visualization, calculations of indicators)
- `scripts/pipeline_config.json` - example configuration file
- `scripts/evaluate.ipynb` - launch code example

An example of what the recognition process configuration file looks like:

- The `WordSegmentation` segmentation stage specifies the name of the model from the `segmentation_factory` method, 
  the path to the file with weights, and the path to the training process configuration file.
- After the segmentation stage, the parameters of the module for processing segmentation results are indicated.
  Postprocessing options can be found in the `segm_postprocessing.py` file. Below is an example of use. 
  The name of the mask class and processing modules for it are written. 
  It is important to maintain order.
- The `OpticalCharacterRecognition` recognition stage specifies the name of the model from the `recognizer_factory` method, 
  the path to the directory or file with the weights, and the name of the classes to be recognized.
- At the layout stage, the names of the word and line classes are specified.

```json
{
    "pipeline": {
        "WordSegmentation": {
            "model_name": "UNet",
            "model_path": "models/segmentation/${weights_filename}.onnx",
            "config_path": "src/segmentation/configs/${config_filename}.json"
        },
        "ContourPostprocessors": {
            "class2postprocessors": {
                "handwritten_text_mask": {
                    "BboxFromContour": {},
                    "UpscaleBbox": {
                        "upscale_bbox": [1, 1.5]
                    },
                    "CropByBbox": {}
                }
            }
        },
        "OpticalCharacterRecognition": {
            "model_name": "TrOCR",
            "model_path": "models/tr_ocr/",
            "ocr_classes": ["handwritten_text_mask"]
        },
        "LineFinder": {
            "line_classes": ["lines"],
            "text_classes": ["handwritten_text_mask"]
        }
    }
}
```

# How do I start the recognition process?

- An example of work is written in the file `src/pipeline/scripts/evaluate.ipynb`
- You can download the weights from the link `https://disk.yandex.ru/d/rxlpAgiTJYWrjA`
  (it is recommended to make the same folder structure as in the link above).
  The root directory is `/models` and the folders `ocr/` and `segmentation` inside.
- Put the weights in the folder `rhtr/models/segmentation` and specify the path in the
  config `src/pipeline/scripts/pipeline_config.json`
- It’s better to use a jupyter to start, opening it on the kaggle.com platform
