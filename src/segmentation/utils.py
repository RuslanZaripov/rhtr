import torch
import os
import math
import logging
import time
import numpy as np

from tqdm import tqdm

from src.segmentation.metrics import get_iou, get_f1_score, AverageMeter, IOUMetric
from src.segmentation.predictor import predict


def configure_logging(log_path=None):
    logger = logging.getLogger(__name__)

    if len(logger.handlers) > 0:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )
    # Setup console logging
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    # Setup file logging as well
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def val_loop(data_loader, model, criterion, device, class_names, logger):
    loss_avg = AverageMeter('Loss', ':.4e')
    iou_avg = AverageMeter('IOU', ':6.2f')
    cls2iou = {cls_name: IOUMetric(cls_idx)
               for cls_idx, cls_name in enumerate(class_names)}
    f1_score_avg = AverageMeter('F1 score', ':6.2f')
    strat_time = time.time()
    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, targets in tqdm_data_loader:
        preds, targets = predict(images, model, device, targets)
        batch_size = len(images)

        loss = criterion(preds, targets)
        loss_avg.update(loss.item(), batch_size)

        iou_avg.update(get_iou(preds, targets), batch_size)
        f1_score_avg.update(get_f1_score(preds, targets), batch_size)
        for cls_name in class_names:
            cls2iou[cls_name](preds, targets)
    loop_time = sec2min(time.time() - strat_time)
    cls2iou_log = ''.join([f' IOU {cls_name}: {iou_fun.avg():.4f}'
                           for cls_name, iou_fun in cls2iou.items()])
    logger.info(f'Validation: {loss_avg}, {iou_avg}, {cls2iou_log}, {f1_score_avg}, loop_time: {loop_time}')
    return loss_avg.avg


def sec2min(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class FilesLimitControl:
    """Delete files from the disk if there are more files than the set limit.
    Args:
        max_weights_to_save (int, optional): The number of files that will be
            stored on the disk at the same time. Default is 3.
    """

    def __init__(self, logger=None, max_weights_to_save=2):
        self.saved_weights_paths = []
        self.max_weights_to_save = max_weights_to_save
        self.logger = logger if logger is not None else configure_logging()

    def __call__(self, save_path):
        self.saved_weights_paths.append(save_path)
        if len(self.saved_weights_paths) > self.max_weights_to_save:
            old_weights_path = self.saved_weights_paths.pop(0)
            if os.path.exists(old_weights_path):
                os.remove(old_weights_path)
                self.logger.info(f"Weigths removed '{old_weights_path}'")


def load_pretrain_model(weights_path, model, logger=None):
    """Load the entire pretrain model or as many layers as possible.
    """
    old_dict = torch.load(weights_path)
    new_dict = model.state_dict()
    if logger is None:
        logger = configure_logging()
    for key, weights in new_dict.items():
        if key in old_dict:
            if new_dict[key].shape == old_dict[key].shape:
                new_dict[key] = old_dict[key]
            else:
                logger.info('Weights {} were not loaded'.format(key))
        else:
            logger.info('Weights {} were not loaded'.format(key))
    return new_dict


"""
Source: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

Early stops the training if validation loss doesn't improve after a given patience.
Source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
Usage: https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
 
Another source: https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_03_4_early_stop.ipynb
where best model restoring happens as a feature
"""


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0,
                 logger=None,
                 load_best_weights=True,
                 control_files_limit=True):
        """
        Args:
            patience (int):
                How long to wait after last time validation loss improved.
                Default: 7
            min_delta (float):
                Minimum change in the monitored quantity to qualify as an improvement.
                Should be a positive number.
                Default: 0
            logger (logging.Logger):
                Logger object to use for logging. If None, no logging is done.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0

        self.load_best_weights = load_best_weights
        self.logger = logger if logger is not None else configure_logging()
        self.file_limit_control = FilesLimitControl(logger=self.logger) if control_files_limit else None

    def is_loss_decreased(self, val_loss):
        return self.best_loss - val_loss >= self.min_delta

    def __call__(self, val_loss, checkpoint_path, model):
        if self.is_loss_decreased(val_loss):
            self.log_info(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.best_loss = val_loss
            self.counter = 0

            self.save_checkpoint(model, checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.log_info(f'EarlyStopping triggered after {self.counter} epochs.')
                if self.load_best_weights:
                    model.load_state_dict(torch.load(checkpoint_path))
                return True
        return False

    def log_info(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def save_checkpoint(self, model, checkpoint_path):
        """Saves model when validation loss decrease."""
        torch.save(model.state_dict(), checkpoint_path)
        self.log_info(f'Model saved to {checkpoint_path}')
        if self.file_limit_control is not None:
            self.file_limit_control(checkpoint_path)
