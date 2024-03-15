import torchvision
import torch
import numpy as np
import cv2


class ToTensor:
    def __call__(self, arr):
        return torch.from_numpy(arr)


class Normalize:
    def __call__(self, arr):
        return arr.astype(np.float32) / 255


class RescalePaddingImage:
    """
    Input image should have OpenCV format (H, W, C).
    """

    def __init__(self, out_height, out_width):
        self.out_height = out_height
        self.out_width = out_width

    def __call__(self, arr):
        in_height, in_width = arr.shape[:2]
        height_change_ratio = self.out_height / in_height
        new_width = min(int(in_width * height_change_ratio), self.out_width)

        arr = cv2.resize(arr,
                         (new_width, self.out_height),
                         interpolation=cv2.INTER_LINEAR)
        if len(arr.shape) < 3:
            arr = arr[:, :, np.newaxis]

        width_correction = self.out_width - new_width
        if width_correction > 0:
            arr = np.pad(arr, ((0, 0), (0, width_correction), (0, 0)), mode='constant', constant_values=0)
        return arr


class MoveChannels:
    """
    Move channels to the first dimension.
    OpenCV image shape: (H, W, C)
    PyTorch image shape: (C, H, W)
    """

    def __init__(self, move_channel_first=True):
        self.is_channel_first = move_channel_first

    def __call__(self, arr):
        if self.is_channel_first:
            return np.moveaxis(arr, source=-1, destination=0)
        else:
            return np.moveaxis(arr, source=0, destination=-1)


class RGBToGray:
    def __call__(self, arr):
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        arr = arr[:, :, np.newaxis]
        return arr


class DefaultBatchPreprocessor:
    def __init__(self, height, width):
        self.transforms = get_val_transform(height, width)

    def __call__(self, images):
        """
        Apply transforms to the batch of images.
        """
        # print(f"{len(images)=}")
        # print(f"{images[0].shape=}")
        return torch.stack([self.transforms(image) for image in images], 0)


def get_train_transform(height, width):
    return torchvision.transforms.Compose([
        RGBToGray(),
        RescalePaddingImage(height, width),
        MoveChannels(move_channel_first=True),
        Normalize(),
        ToTensor(),
    ])


def get_val_transform(height, width):
    return torchvision.transforms.Compose([
        RGBToGray(),
        RescalePaddingImage(height, width),
        MoveChannels(move_channel_first=True),
        Normalize(),
        ToTensor(),
    ])
