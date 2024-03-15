import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm


def main(args):
    images_dir = args.dir
    h5_filename = args.filename

    current_dir = os.path.basename(images_dir)
    parent_dir = os.path.dirname(images_dir)
    h5_filepath = os.path.join(parent_dir, h5_filename)

    with h5py.File(h5_filepath, 'w') as f:
        g = f.create_group(current_dir)
        for file in tqdm(os.listdir(images_dir)):
            if file.endswith('.png') or file.endswith('.jpg'):
                image_bytes = bytearray(open(os.path.join(images_dir, file), 'rb').read())
                image = np.asarray(image_bytes)
                g.create_dataset(file, shape=image.shape, data=image, compression="gzip")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--dir',
                             help='Path to directory containing images',
                             required=True)
    args_parser.add_argument('--filename',
                             help='Name of the h5 file',
                             default='images.h5')
    main(args_parser.parse_args())
