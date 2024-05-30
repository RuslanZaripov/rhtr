import cv2
import numpy as np

from src.segmentation.predictor import upscale_contour


class BboxFromContour:
    @staticmethod
    def contour2bbox(contour):
        """Get bbox from contour."""
        x, y, w, h = cv2.boundingRect(contour.astype(np.float32))
        return x, y, x + w, y + h

    def __call__(self, image, crop, bbox, contour):
        bbox = self.contour2bbox(np.array(contour))
        return crop, bbox, contour


class BboxUpscaler:
    def __init__(self, upscale_bbox):
        self.upscale_bbox = upscale_bbox

    @staticmethod
    def get_upscaled_bbox(bbox, upscale_width_coef=1, upscale_height_coef=1):
        """Increase size of the bbox."""
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]
        height_change = (height * upscale_height_coef) - height
        width_change = (width * upscale_width_coef) - width
        width_r = int(width_change / 2)
        height_r = int(height_change / 2)
        x_min = max(0, bbox[0] - width_r)
        y_min = max(0, bbox[1] - height_r)
        x_max = bbox[2] + width_r
        y_max = bbox[3] + height_r
        return x_min, y_min, x_max, y_max

    def __call__(self, image, crop, bbox, contour):
        bbox = self.get_upscaled_bbox(
            bbox=bbox,
            upscale_width_coef=self.upscale_bbox[0],
            upscale_height_coef=self.upscale_bbox[1]
        )
        return crop, bbox, contour


class BboxCropper:
    @staticmethod
    def crop_image(image, bbox):
        return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def __call__(self, image, crop, bbox, contour):
        crop = self.crop_image(image, bbox)
        return crop, bbox, contour


class PolygonUpscaler:
    def __init__(self, upscale_ratio):
        self.upscale_ratio = upscale_ratio

    def __call__(self, image, crop, bbox, contour):
        contour = upscale_contour(contour, self.upscale_ratio)
        return crop, bbox, contour


class ContourPostprocessors:
    """Class to handle postprocess functions for bboxs and contours."""
    CONTOUR_PROCESS_DICT = {
        "BboxFromContour": BboxFromContour,
        "UpscaleBbox": BboxUpscaler,
        "CropByBbox": BboxCropper,
        "UpscaleContour": PolygonUpscaler,
    }

    def __init__(self, args):
        self.class2postprocessors = {
            class_name: [self.CONTOUR_PROCESS_DICT[postprocess_name](**args)
                         for postprocess_name, args in postprocess_names.items()]
            for class_name, postprocess_names in args['class2postprocessors'].items()
        }

    def __call__(self, image, data):
        for prediction in data['predictions']:
            if prediction['class_name'] in self.class2postprocessors:
                postprocessors = self.class2postprocessors[prediction['class_name']]

                bbox = None \
                    if 'bbox' not in prediction \
                    else prediction['bbox']
                contour = prediction['rotated_polygon'] \
                    if 'rotated_polygon' in prediction \
                    else prediction['polygon']
                crop = None

                for f in postprocessors:
                    crop, bbox, contour = f(image, crop, bbox, contour)

                prediction['rotated_polygon'] = contour
                prediction['rotated_bbox'] = bbox
                prediction['crop'] = crop

        return image, data
