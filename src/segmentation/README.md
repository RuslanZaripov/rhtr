# Segmentation process

- Read this in other languages: [Russian](README.ru.md)

# Structure description

- `/configs` - folder where json files with configuration are stored
- `config.py` - code of the Config class for reading the json file with the training configuration
- `dataset.py` - code used for image preprocessing and mask generation
- If you need to use a different data set for training, then it is better to implement another Dataset class in this
  file
- `losses.py` - contains implemented loss functions for the segmentation module
- Since the model returns a dictionary of masks, it is important to respect the model output format to calculate the
  loss function
- `metrics.py` - contains metrics used to evaluate quality
- `models.py` - contains the implementation of models used for segmentation - the result of the model is a dictionary of
  predicted masks
- `predictor.py` - contains methods for extracting contours and algorithms for processing contours, the methods contain
  commented out code used to visualize the results
- `preprocessing.py` - contains image preprocessing methods for the segmentation model
- `utils.py` - contains auxiliary methods for visualization and calculation of additional indicators
- `train.ipynb` - contains the code used to train the models

Description of the current configuration files format:

```json
{
    "images": {
        "width": "int",
        "height": "int"
    },
    "masks": {
        "binary": "List[int]",
        "lines": "int",
        "border_mask": "List[int]",
        "watershed": "List[int]"
    }
}
```

The "image" parameter contains the values to which all incoming images will be converted

The "masks" parameter contains the masks that will be used in the model.
It is very important to maintain order, since a dictionary with a similar format is generated throughout the pipeline.
The name of the mask can be anything. The main thing is to remember how to address it.
Before editing the code, it is better to study how the entire learning process is carried out.

## How did I train the models?

- Since I don’t have a GPU locally, I used the kaggle.com platform
- On the kaggle.com platform, I opened the `train.ipynb` notebook and ran the cells
  (the current notebook contains some results from running the cells)
- I tried to generalize all the parameters in the laptop so that the model could also be run on other platforms
- I advise you to be careful and double-check everything if you use
- The parameters for saving the results are written in the laptop
  (if desired, they can be placed in the configuration file)
