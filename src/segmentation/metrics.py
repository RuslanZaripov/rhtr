import numpy as np
import torch
from shapely.geometry import Polygon
from joblib import Parallel, delayed
from collections import defaultdict

from src.segmentation.predictor import upscale_contour
from src.segmentation.utils import time_it


class AverageMeter(object):
    """Computes and stores the average and current value."""
    name: str
    fmt: str
    val: float
    avg: float
    sum: float
    count: int

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MetricMonitor:
    metrics: defaultdict

    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0.0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def get(self, metric_name):
        return self.metrics[metric_name]["avg"]

    def __str__(self):
        return " | ".join(
            [
                f"{metric_name}: {metric['avg']:.{self.float_precision}f}"
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def batch_mean(preds, targets, metric, threshold=0.5):
    values = []
    for p, t in zip(preds, targets):
        p_threshold = (p > threshold).float()
        values.append(metric(p_threshold, t))
    return np.mean(values)


def accuracy(y_pred, y_true, eps=1e-7):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return acc.item()


def f1_score(y_pred, y_true, eps=1e-7):
    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    p = tp / (tp + fp + eps)

    tp = (y_true * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    r = tp / (tp + fn + eps)

    f1 = 2 * p * r / (p + r + eps)
    return f1.item()


def precision(y_pred, y_true, eps=1e-7):
    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    p = tp / (tp + fp + eps)
    return p.item()


def recall(y_pred, y_true, eps=1e-7):
    tp = (y_true * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    r = tp / (tp + fn + eps)
    return r.item()


def iou_pytorch(y_pred, y_true, eps=1e-7):
    intersection = (y_pred * y_true).sum()
    union = (y_pred + y_true).sum() - intersection
    return (intersection / (union + eps)).item()


def polygon_iou(poly1, poly2):
    """Calculate the intersection of two polygons using Shapely."""

    eps = 0.01

    if len(poly1) <= 2:
        new_point = np.array([poly1[-1][0] + eps, poly1[-1][1] + eps])
        poly1 = np.vstack([poly1, new_point])

    if len(poly2) <= 2:
        new_point = np.array([poly2[-1][0] + eps, poly2[-1][1] + eps])
        poly2 = np.vstack([poly2, new_point])

    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)

    if not polygon1.is_valid:
        polygon1 = polygon1.buffer(0)

    if not polygon2.is_valid:
        polygon2 = polygon2.buffer(0)

    if not polygon1.intersects(polygon2):
        return 0

    intersection = polygon1.intersection(polygon2)
    union = polygon1.union(polygon2)
    return intersection.area / union.area


def parallelized_iou_matrix(p, t):
    iou_matrix = np.zeros((len(p), len(t)))
    results = Parallel(n_jobs=-1)(delayed(polygon_iou)(pred_polygon, gt_polygon)
                                  for pred_polygon in p for gt_polygon in t)
    for idx, result in enumerate(results):
        pred_idx = idx // len(t)
        gt_idx = idx % len(t)
        iou_matrix[pred_idx, gt_idx] = result
    return iou_matrix


@time_it
def calc_mAPIoU(preds, targets, threshold=0.5, upscale_ratio=1.0, eps=1e-7):
    average_IoU = []
    average_P = []
    average_R = []
    average_f1 = []

    for p, t in zip(preds, targets):
        t = [ele for ele in t if len(ele) != 0]

        gt_boxes = np.zeros(len(t))
        tp = 0
        fp = 0
        total_true_gts = len(t)

        p = [upscale_contour(polygon, upscale_ratio) for polygon in p]

        iou_matrix = parallelized_iou_matrix(p, t)

        for pred_idx in range(len(p)):
            best_idx = np.argmax(iou_matrix[pred_idx])
            best_iou = iou_matrix[pred_idx, best_idx]

            if best_iou > threshold:
                if gt_boxes[best_idx] == 0:
                    tp += 1
                    gt_boxes[best_idx] = 1
                else:
                    fp += 1
            else:
                fp += 1

        fn = np.sum(gt_boxes == 0)
        iou = tp / (tp + fp + fn)
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r + eps)

        # print(f"{tp=} {fp=} {fn=} {iou=}")

        average_IoU.append(iou)
        average_P.append(p)
        average_R.append(r)
        average_f1.append(f1)

    meanIoU = np.mean(average_IoU)
    meanP = np.mean(average_P)
    meanR = np.mean(average_R)
    meanf1 = np.mean(average_f1)

    return meanIoU, meanP, meanR, meanf1
