from functools import wraps

import cv2
import numpy as np
import inspect
import time
import matplotlib.pyplot as plt


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def get_constructor_params(cls):
    init_signature = inspect.signature(cls.__init__)
    return [param for param in init_signature.parameters][1:]


def collect_params_from_dict(params, args):
    return {
        param: args[param]
        if param in args.keys()
        else print(f"WARN: param {param} not found in args {args.keys()}")
        for param in params
    }


def visualize(image, data):
    image_copy = image.copy()

    for prediction in data['predictions']:
        if prediction["class_name"] != "handwritten_text_shrinked_mask1":
            continue

        class2color = {
            "handwritten_text_shrinked_mask1": (0, 255, 0),  # green # text
            "class_name_2": (0, 0, 255),  # red # comments
            "class_name_3": (255, 0, 0),  # blue # text_line
        }

        polygon = [tuple(point) for point in prediction["polygon"]]
        polygon_np = np.array(polygon, np.int32)
        polygon_np = polygon_np.reshape((-1, 1, 2))
        cv2.polylines(image_copy, [polygon_np],
                      isClosed=True,
                      color=class2color[prediction["class_name"]],
                      thickness=3)

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image_copy)
    plt.show()


class AverageMeter:
    """Computes and stores the average and current value"""
    avg: float
    sum: float
    count: int

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
