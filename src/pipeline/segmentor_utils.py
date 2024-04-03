import onnxruntime
import albumentations as A
import numpy as np
import torchvision

from src.segmentation.predictor import get_contours_from_mask, rescale_contours, contour2bbox, reduce_contours_dims


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_energy(pred_output, or_h=None, or_w=None):
    pred_masks = []

    for i in range(pred_output.shape[0]):
        pred_mask = pred_output[i, :, :]
        # pred_mask = cv2.resize(pred_mask, (or_h, or_w), interpolation=cv2.INTER_LINEAR)
        pred_masks.append(pred_mask)

    pred_energy = (np.mean(pred_masks, axis=0) * 255).astype(np.uint8)
    pred_mask = pred_masks[-1] > 0.5

    return pred_mask, pred_energy


def get_preds(images, preds, cuda_torch_input=True):
    pred_data = []
    for image, pred in zip(images, preds):  # iterate through images
        img_h, img_w = image.shape[:2]
        pred_img = {'predictions': []}

        pred_binary = pred > 137

        contours = get_contours_from_mask(pred_binary)

        contours = rescale_contours(
            contours=contours,
            pred_height=512,
            pred_width=512,
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
                    'class_name': 'handwritten_text_shrinked_mask1'
                }
            )
        pred_data.append(pred_img)
    return pred_data


class UNet:
    def __init__(self):
        self.ort_session = onnxruntime.InferenceSession(
            "models/segmentation/unet_onnx/UNet-watershed-border-1.onnx")

        ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
        ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

        # note that you can include more fancy data augmentation methods here
        self.train_transform = A.Compose([
            A.Resize(width=512, height=512),
            A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ])

    def __call__(self, images):
        transformed_images = self.train_transform(image=images)['image']
        transformed_images = torchvision.transforms.ToTensor()(transformed_images)
        transformed_images = transformed_images[None, :, :, :]

        outputs = self.ort_session.run(None, {'input': transformed_images.numpy()})

        watershed_masks = sigmoid(outputs[0])
        print(f"{watershed_masks.shape=}")

        energies = []
        for batch_masks in watershed_masks:
            pred_mask, pred_energy = calculate_energy(batch_masks)
            energies.append(pred_energy)

        preds = get_preds(images, energies)
        return preds
