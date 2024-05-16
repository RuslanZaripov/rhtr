import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


class DatasetCreator:
    def __init__(self, save_path, dataset_path):
        self.save_path = save_path
        self.dataset_path = dataset_path

    def upload_set(self, dict):
        for set_name, csv in dict.items():
            with h5py.File(self.save_path, 'w') as f:
                data = pd.read_csv(csv)

                # use compression
                f.create_dataset(f'{set_name}_images', shape=(len(data),), dtype=h5py.vlen_dtype(np.uint8),
                                 chunks=True, compression="gzip", compression_opts=4)
                f.create_dataset(f'{set_name}_targets', shape=(len(data),), dtype=h5py.vlen_dtype(np.float32),
                                 chunks=True, compression="gzip", compression_opts=4)

                for i, row in tqdm(data.iterrows(), total=len(data)):
                    image_path = str(os.path.join(self.dataset_path, row['image']))
                    img = bytearray(open(image_path, 'rb').read())

                    target_path = os.path.join(self.dataset_path, row['target'])
                    target = bytearray(np.load(target_path).tobytes())

                    f[f'{set_name}_images'][i] = img
                    f[f'{set_name}_targets'][i] = target

    def read(self, set_name, idx):
        with h5py.File(self.save_path, 'r') as f:
            image = np.frombuffer(f[f'{set_name}_images'][idx], dtype=np.uint8)
            target = np.frombuffer(f[f'{set_name}_targets'][idx], dtype=np.float32)
        return image, target


def main():
    dataset = DatasetCreator(
        save_path='../data/data.h5',
        dataset_path='../data/raw/school_notebooks_RU',
    )
    dataset.upload_set({
        'train': '../data/train.csv',
        'val': '../data/val.csv',
        'test': '../data/test.csv',
    })
    dataset.read('train', 0)


if __name__ == '__main__':
    main()
