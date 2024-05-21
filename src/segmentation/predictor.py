import cv2
import matplotlib.pyplot as plt
from src.segmentation.utils import colorize
import numpy as np
import pyclipper
import torch
from skimage.measure import label
from skimage.segmentation import watershed
import imutils
from shapely.geometry import Polygon


def is_valid_polygon(polygon):
    """Check if a polygon is valid. Return True if valid and False otherwise.

    Args:
        polygon (shapely.geometry.Polygon): The polygon.
    """
    return polygon.length >= 1 and polygon.area > 0


def calculate_energy(pred_output):
    pred_energy = np.mean(pred_output, axis=0)
    pred_mask = np.copy(pred_output[-1])
    return pred_mask, pred_energy


def get_contours_from_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(contour.astype(np.int32))
             for contour in contours]
    mean_area = np.mean(areas)

    result = [contour
              for contour, area in zip(contours, areas)
              if area > mean_area * 0.15]

    result = [np.squeeze(contour, axis=1)
              for contour in result]

    return result


def rescale_contour(contour, pred_height, pred_width, image_height, image_width):
    """Rescale contour from prediction mask shape to input image size."""
    y_ratio = image_height / pred_height
    x_ratio = image_width / pred_width
    scale = (x_ratio, y_ratio)
    for i in range(2):
        contour[:, i] = (contour[:, i] * scale[i]).astype(np.int64)
    return contour


def process_class_contours(pred_class, image, threshold):
    """image and pred_class format should be (C, H, W)"""
    assert image.shape[0] == 3 or image.shape[0] == 1, "Channel should be first in image"
    assert pred_class.shape[0] == 1, "Channel should be first in pred_class mask"

    img_h, img_w = image.shape[1:]
    pred_h, pred_w = pred_class.shape[1:]

    pred_class = np.squeeze(pred_class, axis=0)

    # plt.imshow(colorize(pred_class))
    # plt.savefig('images/lines.png')
    # plt.clf()

    pred_class = pred_class > threshold

    contours = get_contours_from_mask(pred_class)

    contours = [rescale_contour(contour,
                                pred_h, pred_w,
                                img_h, img_w)
                for contour in contours]

    boxes = [contour2bbox(contour)
             for contour in contours]

    return list(zip(contours, boxes))


def contour2bbox(contour):
    """Get bbox from contour."""
    assert len(contour) > 0, "Contour is empty."
    x, y, w, h = cv2.boundingRect(contour.astype(np.float32))
    return x, y, x + w, y + h


def get_predictions(images, predictions, threshold=0.5):
    pred_data = []

    for batch_idx in range(len(images)):
        pred_img = {'predictions': []}

        binary_contours = process_class_contours(
            predictions['binary'][batch_idx],
            images[batch_idx],
            threshold)

        for b_contour, b_bbox in binary_contours:
            pred_img['predictions'].append(
                {
                    'polygon': b_contour,
                    'bbox': b_bbox,
                    'class_name': 'handwritten_text'
                }
            )

        line_contours = process_class_contours(
            predictions['lines'][batch_idx],
            images[batch_idx],
            threshold)

        for l_contour, l_bbox in line_contours:
            pred_img['predictions'].append(
                {
                    'polygon': l_contour,
                    'bbox': l_bbox,
                    'class_name': 'lines'
                }
            )

        pred_data.append(pred_img)
    return pred_data


def energy_baseline(last_msk, energy):
    # fig = plt.figure(figsize=(10, 10))

    # fig.add_subplot(1, 3, 1)
    # plt.imshow(colorize(energy))

    msk_ths = (np.copy(energy) > 0.1) * 1
    # msk_ths = last_msk > 0.5

    # fig.add_subplot(1, 3, 2)
    # plt.imshow(msk_ths)

    energy_ths = (np.copy(energy) > 0.7) * 1

    # fig.add_subplot(1, 3, 3)
    # plt.imshow(energy_ths)

    # plt.savefig('images/energy_baseline.png')
    # plt.clf()

    # distance = ndi.distance_transform_edt(msk_ths)
    distance = energy

    # Marker labelling
    markers = label(energy_ths)

    labels = watershed(-distance,
                       markers,
                       mask=msk_ths)

    return labels


def process_watershed(watershed_masks, image):
    """watershed_masks and image format should be (C, H, W)"""
    assert image.shape[0] == 3 or image.shape[0] == 1, "Image should have 1 or 3 channels."
    # TODO: assert watershed_masks shape is (C, H, W)

    # image and watershed_masks should have following shape structure (C, H, W)
    img_h, img_w = image.shape[1:]
    pred_h, pred_w = watershed_masks.shape[1:]

    pred_mask, energy = calculate_energy(watershed_masks)

    # import matplotlib.pyplot as plt
    # plt.imshow(energy)
    # plt.savefig("images/energy.png")
    # plt.clf()

    labels = energy_baseline(pred_mask, energy)

    #     contours = []
    #     for labeled_matrix in np.unique(labels):
    #         contour = skimage.measure.find_contours(labels == labeled_matrix, level=0.5)[0]
    #         contour = np.flip(contour, axis=1)
    #         contours.append(contour)

    contours = []
    # loop over the unique labels, and append contours to all_cnts
    for lab in np.unique(labels):
        if lab == 0:
            continue

        mask = np.zeros((pred_h, pred_w), dtype="uint8")
        mask[labels == lab] = 255

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # hull = cv2.convexHull(c)
        c = np.squeeze(c, axis=1)

        contours.append(c)

    areas = [cv2.contourArea(contour.astype(np.int32)) for contour in contours]
    mean = np.mean(areas)
    contours = [contour
                for contour, area in zip(contours, areas)
                if area > mean * 0.15]

    to_remove = set()
    for i in range(len(contours)):
        c = contours[i].astype(np.float32)
        area = cv2.contourArea(c)

        # Iterate all contours from i+1 to end of list
        for j in range(i + 1, len(contours)):
            c2 = contours[j].astype(np.float32)
            area2 = cv2.contourArea(c2)

            polygon1 = Polygon(c)
            polygon2 = Polygon(c2)

            intersection = polygon1.intersection(polygon2)

            smaller_area = min(area, area2)

            if smaller_area * 0.5 < intersection.area:
                to_remove.add(j)
    contours = [contours[i] for i in range(len(contours)) if i not in to_remove]

    contours = [rescale_contour(contour,
                                pred_h, pred_w,
                                img_h, img_w)
                for contour in contours]

    boxes = [contour2bbox(contour)
             for contour in contours]

    return list(zip(contours, boxes))


def get_watershed_predictions(images, predictions, threshold=0.5):
    pred_data = []

    for batch_idx in range(len(images)):
        pred_img = {'predictions': []}

        word_contours = process_watershed(
            predictions['watershed'][batch_idx],
            images[batch_idx])

        for b_contour, b_bbox in word_contours:
            pred_img['predictions'].append(
                {
                    'polygon': b_contour,
                    'bbox': b_bbox,
                    'class_name': 'handwritten_text_shrinked_mask1'
                }
            )

        line_contours = process_class_contours(
            predictions['lines'][batch_idx],
            images[batch_idx],
            threshold=0.25)

        for l_contour, l_bbox in line_contours:
            pred_img['predictions'].append(
                {
                    'polygon': l_contour,
                    'bbox': l_bbox,
                    'class_name': 'lines'
                }
            )

        pred_data.append(pred_img)
    return pred_data


def change_contour_size(polygon, scale_ratio=1.2):
    poly = Polygon(polygon)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(polygon, pyclipper.JT_ROUND,
                pyclipper.ET_CLOSEDPOLYGON)
    if not is_valid_polygon(poly):
        return None
    distance = int(poly.area * (1 - scale_ratio ** 2) / poly.length)
    scaled_polygons = pco.Execute(-distance)
    return scaled_polygons


def upscale_contour(contour, scale_ratio):
    upscaled_contour = change_contour_size(contour, scale_ratio)
    if upscaled_contour is None: return contour
    # take zero contour (when upscaling only one contour could be returned)
    upscaled_contour = upscaled_contour[0]
    # coords shouldn't be outside image after upscaling
    upscaled_contour = [[max(0, i[0]), max(0, i[1])]
                        for i in upscaled_contour]
    return upscaled_contour
