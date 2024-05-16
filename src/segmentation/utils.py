import os
import matplotlib.pyplot as plt
import torch


def visualize(image, title='', cmap='viridis'):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.axis('off')
    plt.imshow(image, cmap=cmap)
    plt.show()


def background_ratio(tensor):
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
