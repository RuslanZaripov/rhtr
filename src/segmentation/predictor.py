import torch
from enum import Enum

import cv2
import numpy as np

from src.segmentation.transforms import InferenceTransform
from src.segmentation.models import LinkResNet
from src.segmentation.config import Config


def predict(images, model, device, targets=None):
    """Make model prediction.
    Args:
        images (torch.Tensor): Batch with tensor images.
        model (ocr.src.models.CRNN): OCR model.
        device (torch.device): Torch device.
        targets (torch.Tensor): Batch with tensor masks. By default is None.
    """
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)

    if targets is not None:
        targets = targets.to(device)
        return output, targets
    return output


def contour2bbox(contour):
    """Get bbox from contour."""
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, x + w, y + h


def get_preds(images, preds, cls2params, config, cuda_torch_input=True):
    pred_data = []
    for image, pred in zip(images, preds):  # iterate through images
        img_h, img_w = image.shape[:2]
        pred_img = {'predictions': []}
        for cls_idx, cls_name in enumerate(cls2params):  # iter through classes
            pred_cls = pred[cls_idx]
            # thresholding works faster on cuda than on cpu
            pred_cls = \
                pred_cls > cls2params[cls_name]['postprocess']['threshold']
            if cuda_torch_input:
                pred_cls = pred_cls.cpu().numpy()

            contours = get_contours_from_mask(
                pred_cls, cls2params[cls_name]['postprocess']['min_area'])
            contours = rescale_contours(
                contours=contours,
                pred_height=config.get_image('height'),
                pred_width=config.get_image('width'),
                image_height=img_h,
                image_width=img_w
            )
            bboxes = [contour2bbox(contour) for contour in contours]
            contours = reduce_contours_dims(contours)

            for contour, bbox in zip(contours, bboxes):
                pred_img['predictions'].append(
                    {
                        'polygon': contour,
                        'bbox': bbox,
                        'class_name': cls_name
                    }
                )
        pred_data.append(pred_img)
    return pred_data


class SegmTorchModel:
    def __init__(self, model_path, config_path):
        self.config = Config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cls2params = self.config.get_classes()
        # load model
        self.model = LinkResNet(
            output_channels=len(self.cls2params),
            pretrained=False
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=self.config.get_image('height'),
            width=self.config.get_image('width'),
        )

    def predict(self, images):
        transformed_images = self.transforms(images)
        preds = predict(transformed_images, self.model, self.device)
        return preds

    def get_preds(self, images, preds):
        pred_data = get_preds(images, preds, self.cls2params, self.config)
        return pred_data


class SegmPredictor:
    """Make SEGM prediction.

    Args:
        model_path (str): The path to the model weights.
        config_path (str): The path to the model config.
    """

    def __init__(
            self, model_path, config_path
    ):
        self.model = SegmTorchModel(model_path, config_path)

    def __call__(self, images):
        """
        Args:
            images (list of np.ndarray): A list of images in BGR format.

        Returns:
            pred_data (dict or list of dicts): A result dict for one input
                image, and a list with dicts if there was a list with images.
            [
                {
                    'predictions': [
                        {
                            'polygon': polygon [[x1,y1], [x2,y2], ..., [xN,yN]]
                            'bbox': bounding box [x_min, y_min, x_max, y_max]
                            'class_name': str, class name of the polygon.
                        },
                        ...
                    ]

                },
                ...
            ]
        """
        preds = self.model.predict(images)
        pred_data = self.model.get_preds(images, preds)
        return pred_data


def get_contours_from_mask(mask, min_area=5):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            contour_list.append(contour)
    return contour_list


def rescale_contours(
        contours, pred_height, pred_width, image_height, image_width
):
    """Rescale contours from prediction mask shape to input image size."""
    y_ratio = image_height / pred_height
    x_ratio = image_width / pred_width
    scale = (x_ratio, y_ratio)
    for contour in contours:
        for i in range(2):
            contour[:, :, i] = contour[:, :, i] * scale[i]
    return contours


def reduce_contours_dims(contours):
    reduced_contours = []
    for contour in contours:
        contour = [[int(i[0][0]), int(i[0][1])] for i in contour]
        reduced_contours.append(contour)
    return reduced_contours
