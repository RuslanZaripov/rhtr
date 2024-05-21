import albumentations as A
import matplotlib.pyplot as plt
import onnxruntime
import torchvision
import torch

from src.pipeline.abstract import Segmentor
from src.segmentation.config import Config
from src.segmentation.predictor import get_watershed_predictions
from src.segmentation.preprocessing import *


class UNet(Segmentor):
    def __init__(self, model_path, config_path):
        self.ort_session = onnxruntime.InferenceSession(model_path)

        self.train_config = Config(config_path)

        ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
        ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
        self.train_transform = A.Compose([
            A.Resize(width=self.train_config.get_image('width'),
                     height=self.train_config.get_image('height')),
            A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ])

    @staticmethod
    def preprocessing(input_image):
        #     gray = grayscale(input_image)
        #     cl = clahe1channel(gray)
        #     g = gaussian(cl)
        b = bilateral(input_image)
        return b

    def predict(self, images):
        tensors = []
        for image in images:
            print(f"{image.shape=}")

            transformed_image = self.preprocessing(image)
            transformed_image = self.train_transform(image=transformed_image)['image']
            transformed_image = torchvision.transforms.ToTensor()(transformed_image)
            tensors.append(transformed_image)
        tensors = torch.stack(tensors)

        images = [np.transpose(image, (2, 0, 1)) for image in images]  # convert to (C, H, W) format
        outputs = self.ort_session.run(None, {'input': tensors.numpy()})

        class2outpus = {key: outputs[idx] for idx, key in enumerate(self.train_config.get_masks().keys())}

        preds = get_watershed_predictions(images, class2outpus)

        # self.draw_polygons(preds, images)

        return preds

    def draw_polygons(self, preds, images):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from shapely.geometry import Polygon

        def get_polygons(batch_pred, class_name):
            return [pred['polygon']
                    for pred in batch_pred['predictions']
                    if pred['class_name'] == class_name]

        predicted_word_polygons = list(map(
            lambda batch_pred: get_polygons(batch_pred, 'handwritten_text_shrinked_mask1'), preds))
        predicted_line_polygons = list(map(
            lambda batch_pred: get_polygons(batch_pred, 'lines'), preds))

        target_h = self.train_config.get_image('height')
        target_w = self.train_config.get_image('width')

        print(f"drawing polygons {images[0].shape=}")
        image = np.zeros((images[0].shape[1], images[0].shape[2]), dtype=np.uint8)

        fig = plt.figure(figsize=(30, 30))
        ax = fig.add_subplot(1, 1, 1)
        for polygon in predicted_line_polygons[0]:
            if len(polygon) == 0: continue
            poly = patches.Polygon(polygon, closed=True, edgecolor='r', alpha=0.5)
            ax.add_patch(poly)
        ax.imshow(image)

        plt.savefig('images/polygons.png')
        plt.clf()
