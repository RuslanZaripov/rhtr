import onnxruntime
import albumentations as A
import numpy as np
import torchvision

from scipy import ndimage as ndi
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
import skimage
import cv2

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib


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


def contour2bbox(contour):
    """Get bbox from contour."""
    assert len(contour) > 0, "Contour is empty."
    x, y, w, h = cv2.boundingRect(contour.astype(np.float32))
    return x, y, x + w, y + h


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_energy(pred_output, mask_thresh=0.5, or_h=None, or_w=None):
    pred_masks = []

    for i in range(pred_output.shape[0]):
        pred_mask = pred_output[i, :, :]
        # pred_mask = cv2.resize(pred_mask, (or_h, or_w), interpolation=cv2.INTER_LINEAR)
        pred_masks.append(pred_mask)

    pred_energy = np.mean(pred_masks, axis=0)
    pred_mask = pred_masks[-1] > mask_thresh

    return pred_mask, pred_energy


rainbow = matplotlib.colormaps.get_cmap('rainbow')


def colorize(probability_map):
    return rainbow(probability_map)[:, :, :3]


def energy_baseline(msk, energy):
    # fig = plt.figure(figsize=(10, 10))

    # fig.add_subplot(1, 3, 1)
    # plt.imshow(colorize(energy))

    msk_ths = (np.copy(energy) > 0.1) * 1

    # fig.add_subplot(1, 3, 2)
    # plt.imshow(msk_ths)

    # import seaborn as sns
    # sns.heatmap(energy, cmap='rainbow', cbar=True, xticklabels=False, yticklabels=False)
    # plt.savefig(f'data/processed/energy.png')
    # plt.clf()

    energy_ths = (np.copy(energy) > 0.5) * 1

    # fig.add_subplot(1, 3, 3)
    # plt.imshow(energy_ths)

    # plt.show()

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
    # plt.savefig('/content/labels.png')

    return labels


def get_contours(probability_map, image):
    img_h, img_w = image.shape[:2]

    mask = probability_map > 0.5

    contours = get_contours_from_mask(mask, 50)

    contours = rescale_contours(
        contours=contours,
        pred_height=512,
        pred_width=512,
        image_height=img_h,
        image_width=img_w
    )

    bboxes = [contour2bbox(contour) for contour in contours]
    contours = reduce_contours_dims(contours)

    return zip(contours, bboxes)


def get_preds(images, preds, cuda_torch_input=True):
    pred_data = []
    for image, pred in zip(images, preds):
        img_h, img_w = image.shape[:2]
        pred_img = {'predictions': []}

        # print(f"{pred[0].shape=}")
        # print(f"{pred[1].shape=}")

        pred_mask, energy = calculate_energy(pred[0][0])

        labels = energy_baseline(pred[1][0], energy)

        contours = []
        for labeled_matrix in np.unique(labels):
            contour = skimage.measure.find_contours(labels == labeled_matrix, level=0.5)[0]
            contour = np.flip(contour, axis=1)
            contours.append(contour)

        areas = [cv2.contourArea(contour.astype(np.int32)) for contour in contours]
        mean = np.mean(areas)
        contours = [contour
                    for contour, area in zip(contours, areas)
                    if area > mean * 0.15]

        contours = [rescale_contour(contour,
                                    512, 512,
                                    img_h, img_w)
                    for contour in contours]

        boxes = [contour2bbox(contour)
                 for contour in contours]

        for contour, bbox in zip(contours, boxes):
            pred_img['predictions'].append(
                {
                    'polygon': contour,
                    'bbox': bbox,
                    'class_name': 'handwritten_text_shrinked_mask1'
                }
            )

        for contour, bbox in get_contours(pred[4][0][0], image):
            pred_img['predictions'].append(
                {
                    'polygon': contour,
                    'bbox': bbox,
                    'class_name': 'lines'
                }
            )

        pred_data.append(pred_img)
    return pred_data


class UNet:
    def __init__(self):
        self.ort_session = onnxruntime.InferenceSession(
            "models/segmentation/linknet-7.onnx")

        ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
        ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

        # note that you can include more fancy data augmentation methods here
        self.train_transform = A.Compose([
            A.Resize(width=512, height=512),
            A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ])

    def __call__(self, images):
        outputs = []
        for image in images:
            transformed_image = self.train_transform(image=image)['image']
            transformed_image = torchvision.transforms.ToTensor()(transformed_image)
            transformed_image = transformed_image[None, :, :, :]

            output = self.ort_session.run(None, {'input': transformed_image.numpy()})
            outputs.append(output)

        preds = get_preds(images, outputs)
        return preds
