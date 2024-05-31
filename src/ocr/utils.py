import matplotlib.pyplot as plt
import logging
import src.ocr as ocr
import time
from tqdm import tqdm
import os


def implt(im, cmap=None, title=None):
    plt.imshow(im, cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.show()


def configure_logging(log_path=None):
    logger = logging.getLogger(__name__)
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


def val_loop(data_loader, model, decoder, logger, device):
    acc_avg = ocr.AverageMeter()
    wer_avg = ocr.AverageMeter()
    cer_avg = ocr.AverageMeter()

    start_time = time.time()
    for images, labels, _, _ in tqdm(data_loader, total=len(data_loader)):
        batch_size = len(labels)

        label_predictions = ocr.predict(images, model, decoder, device)

        acc_avg.update(ocr.accuracy(labels, label_predictions), batch_size)
        wer_avg.update(ocr.wer(labels, label_predictions), batch_size)
        cer_avg.update(ocr.cer(labels, label_predictions), batch_size)

    elapsed_time = format_sec(time.time() - start_time)

    logger.info(f'Validation: '
                f'Acc: {acc_avg.avg:.4f}, '
                f'WER: {wer_avg.avg:.4f}, '
                f'CER: {cer_avg.avg:.4f}, '
                f'Elapsed Time: {elapsed_time}')

    return acc_avg.avg


def format_sec(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%02d' % (h, m, s)


class FilesLimitController:
    """
    Delete files from the disk if there are more files than the set limit.
    """

    def __init__(self, logger, max_weights_to_save=3):
        self.saved_weights_paths = []
        self.max_weights_to_save = max_weights_to_save
        self.logger = logger

    def __call__(self, save_path):
        self.saved_weights_paths.append(save_path)

        if len(self.saved_weights_paths) <= self.max_weights_to_save:
            return

        old_weights_path = self.saved_weights_paths.pop(0)
        if os.path.exists(old_weights_path):
            os.remove(old_weights_path)
            self.logger.info(f"Weights removed '{old_weights_path}' due to weight files count limit")
