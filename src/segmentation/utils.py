import os
import time

import matplotlib
import matplotlib.pyplot as plt
import torch


def visualize(image, title='', cmap='viridis'):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.axis('off')
    plt.imshow(image, cmap=cmap)
    plt.show()


def background_ratio(tensor: torch.Tensor):
    assert torch.unique(tensor).size(0) == 2 or torch.unique(tensor).size(0) == 1, \
        "The tensor does not contain exactly 2 unique values."

    count_zeros = torch.sum(tensor == 0).item()
    total_elements = tensor.numel()
    ratio_zeros = count_zeros / total_elements
    return ratio_zeros


def get_filename(path):
    import os
    root_ext = os.path.splitext(path.split('/')[-1])[0]
    return root_ext


def delete_and_create_dir(path):
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result

    return wrapper


def validate_range(name: str, arr: torch.Tensor):
    print(f"{name}: {arr.min()}-{arr.max()}")


def dict_to(dictionary: dict, device: torch.device):
    for k, v in dictionary.items():
        # skip entries which are not tensors
        if 'polygons' not in k:
            dictionary[k] = v.to(device)
    return dictionary


rainbow = matplotlib.colormaps.get_cmap('rainbow')


def colorize(probability_map):
    return rainbow(probability_map)[:, :, :3]
