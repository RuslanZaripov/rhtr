import matplotlib.pyplot as plt
import torch
from enum import Enum

import cv2
import numpy as np

from src.segmentation.transforms import InferenceTransform
from src.segmentation.models import LinkResNet
from src.segmentation.config import Config

from scipy import ndimage as ndi
from skimage.color import label2rgb
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
from skimage.segmentation import watershed
import skimage


def predict(images, model, device, targets=None):
    """Make model prediction.
    Args:
        images (torch.Tensor): Batch with tensor images.
        model (ocr.src.models.CRNN): OCR model.
        device (torch.device): Torch device.
        targets (torch.Tensor): Batch with tensor masks. By default, is None.
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
    assert len(contour) > 0, "Contour is empty."
    x, y, w, h = cv2.boundingRect(contour.astype(np.float32))
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


def get_preds_watershed(images, preds, cls2params, config, cuda_torch_input=True):
    pred_data = []
    for image, pred in zip(images, preds):
        img_h, img_w = image.shape[:2]
        pred_img = {'predictions': []}

        mask, energy = calculate_energy(pred)

        labels = energy_baseline(mask, energy, img_h, img_w)

        # labels_seed = cv2.applyColorMap((labels / labels.max() * 255).astype('uint8'), cv2.COLORMAP_JET)
        # plt.imshow(labels_seed)
        # plt.savefig(f'data/processed/labels.png')

        # plt.title('mask')
        # plt.imshow(mask)

        contours = []
        for labeled_matrix in np.unique(labels):
            contour = skimage.measure.find_contours(labels == labeled_matrix, level=0.5)[0]
            contour = np.flip(contour, axis=1)
            contours.append(contour)
            # mask = cv2.drawContours(mask, [contour.astype(np.int64)], -1, (255, 0, 0), 2)

        # plt.imshow(mask)
        # plt.show()

        contours = [rescale_contour(contour,
                                    config.get_image('height'),
                                    config.get_image('width'),
                                    img_h,
                                    img_w) for contour in contours]
        boxes = [contour2bbox(contour) for contour in contours]

        for contour, bbox in zip(contours, boxes):
            pred_img['predictions'].append(
                {
                    'polygon': contour,
                    'bbox': bbox,
                    'class_name': 'handwritten_text_shrinked_mask1'
                }
            )
        pred_data.append(pred_img)
    return pred_data


def calculate_energy(pred_output, or_h=None, or_w=None):
    m = torch.nn.Sigmoid()

    pred_mask = m(pred_output[0, :, :]).data.cpu().numpy()  # actual mask
    pred_mask1 = m(pred_output[1, :, :]).data.cpu().numpy()  # 0.7 shrinked mask
    pred_mask2 = m(pred_output[2, :, :]).data.cpu().numpy()  # 0.4 shrinked mask
    pred_mask3 = m(pred_output[3, :, :]).data.cpu().numpy()  # 0.8 word boundary mask
    pred_mask0 = m(pred_output[4, :, :]).data.cpu().numpy()  # distance mask
    pred_distance = m(pred_output[5, :, :]).data.cpu().numpy()  # thresh_binary mask

    # actual thin1 thin2 thin3 center distance
    # pred_mask = cv2.resize(pred_mask, (or_h, or_w), interpolation=cv2.INTER_LINEAR)
    # pred_mask1 = cv2.resize(pred_mask1, (or_h, or_w), interpolation=cv2.INTER_LINEAR)
    # pred_mask2 = cv2.resize(pred_mask2, (or_h, or_w), interpolation=cv2.INTER_LINEAR)
    # pred_mask3 = cv2.resize(pred_mask3, (or_h, or_w), interpolation=cv2.INTER_LINEAR)
    # pred_mask0 = cv2.resize(pred_mask0, (or_h, or_w), interpolation=cv2.INTER_LINEAR)
    # pred_distance = cv2.resize(pred_distance, (or_h, or_w), interpolation=cv2.INTER_LINEAR)

    # predict average energy by summing all the masks up
    pred_energy = (pred_mask + pred_mask1 + pred_mask2 + pred_mask0) / 4 * 255
    pred_mask = np.copy(pred_mask)

    return pred_mask, pred_energy


def energy_baseline(msk=None,
                    energy=None,
                    threshold=0.5,
                    thin_labels=False):
    msk_ths = (np.copy(msk) > 0.6) * 1

    # print(f"msk {np.min(msk)}-{np.max(msk)}")
    # plt.imshow(msk_ths)
    # plt.show()

    import seaborn as sns
    sns.heatmap(energy, cmap='rainbow', cbar=True, xticklabels=False, yticklabels=False)
    plt.savefig(f'data/processed/energy.png')

    energy_ths = (np.copy(energy) > 255 * 0.6) * 1

    # print(f"energy_ths: {np.min(energy_ths)}-{np.max(energy_ths)}")

    distance = ndi.distance_transform_edt(msk_ths)

    # Marker labelling
    markers = label(energy_ths)

    # image_label_overlay = label2rgb(markers, image=msk, bg_label=0)
    #
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)
    #
    # for region in regionprops(markers):
    #     # take regions with large enough areas
    #     if region.area >= 100:
    #         # draw rectangle around segmented coins
    #         minr, minc, maxr, maxc = region.bbox
    #         rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                                   fill=False, edgecolor='red', linewidth=2)
    #         ax.add_patch(rect)
    #
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.savefig("data/processed/image_label_overlay.png")

    labels = watershed(-distance,
                       markers,
                       mask=msk_ths)

    # plt.imshow(labels, cmap=plt.cm.nipy_spectral)
    # plt.savefig('data/processed/labels.png')

    return labels


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
        # use get_preds_watershed or get_preds depending on type you use
        pred_data = get_preds_watershed(images, preds, self.cls2params, self.config)
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


def rescale_contour(
        contour, pred_height, pred_width, image_height, image_width
):
    """Rescale contour from prediction mask shape to input image size."""
    y_ratio = image_height / pred_height
    x_ratio = image_width / pred_width
    scale = (x_ratio, y_ratio)
    for i in range(2):
        contour[:, i] = (contour[:, i] * scale[i]).astype(np.int64)
    return contour


def reduce_countour_dims(contour):
    contour = [[int(i[0][0]), int(i[0][1])] for i in contour]
    return contour
