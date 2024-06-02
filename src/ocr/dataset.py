import os

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import src.ocr.tokenizer


def get_data_loader(
        dataset_root,
        csv_filename,
        h5_filename,
        batch_size,
        transform,
        shuffle,
        num_workers,
        drop_last,
):
    dataset = ImageDataset(root=dataset_root,
                           csv_filename=csv_filename,
                           h5_filename=h5_filename,
                           transform=transform)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn,
                             pin_memory=True,
                             drop_last=drop_last)
    return data_loader


class ImageDataset(Dataset):
    def __init__(self, root, csv_filename, h5_filename=None, transform=None):
        self.dataset_root = root
        self.subset_name = os.path.splitext(csv_filename)[0]
        self.img_labels = pd.read_csv(filepath_or_buffer=f'{self.dataset_root}/{csv_filename}',
                                      delimiter='\t',
                                      on_bad_lines='warn',
                                      engine='python')
        self.corrupted_image_count = 0
        self.h5_file = h5py.File(f'{self.dataset_root}/{h5_filename}', 'r') \
            if h5_filename is not None \
            else None
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        image_name = self.img_labels.iloc[index, 0]
        image_label = str(self.img_labels.iloc[index, 1])

        image_relative_path = f'{self.dataset_root}/{self.subset_name}/{image_name}'
        image = np.asarray(Image.open(image_relative_path))

        if self.transform:
            image = self.transform(image)

        encoded_image_label = src.ocr.tokenizer.Tokenizer().encode_word(image_label)
        target_length = [len(image_label)]

        encoded_image_label = torch.LongTensor(encoded_image_label)
        target_length = torch.LongTensor(target_length)

        return image, image_label, encoded_image_label, target_length


def collate_fn(batch):
    images, image_labels, encoded_image_label, target_lengths = zip(*batch)

    images = torch.stack(images, 0)
    encoded_image_label = torch.cat(encoded_image_label, 0)
    target_lengths = torch.cat(target_lengths, 0)

    return images, image_labels, encoded_image_label, target_lengths
