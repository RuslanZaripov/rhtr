from tqdm import tqdm
import os
import time
import numpy as np
import torch
import argparse

from src.segmentation.utils import (
    val_loop, load_pretrain_model, FilesLimitControl, sec2min,
    configure_logging
)
from src.segmentation.dataset import get_data_loader
from src.segmentation.transforms import (
    get_train_transforms, get_image_transforms, get_mask_transforms
)
from src.segmentation.config import Config
from src.segmentation.metrics import get_iou, get_f1_score, AverageMeter, IOUMetric
from src.segmentation.losses import FbBceLoss
from src.segmentation.models import LinkResNet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(
    data_loader, model, criterion, optimizer, epoch, class_names, logger
):
    loss_avg = AverageMeter()
    iou_avg = AverageMeter()
    cls2iou = {cls_name: IOUMetric(cls_idx)
               for cls_idx, cls_name in enumerate(class_names)}
    f1_score_avg = AverageMeter()
    strat_time = time.time()
    model.train()
    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, targets in tqdm_data_loader:
        model.zero_grad()
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        batch_size = len(images)
        preds = model(images)

        loss = criterion(preds, targets)
        loss_avg.update(loss.item(), batch_size)

        iou_avg.update(get_iou(preds, targets), batch_size)
        f1_score_avg.update(get_f1_score(preds, targets), batch_size)
        for cls_name in class_names:
            cls2iou[cls_name](preds, targets)

        loss.backward()
        optimizer.step()
    loop_time = sec2min(time.time() - strat_time)
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    cls2iou_log = ''.join([f' IOU {cls_name}: {iou_fun.avg():.4f}'
                           for cls_name, iou_fun in cls2iou.items()])
    logger.info(f'Epoch {epoch}, '
                f'Loss: {loss_avg.avg:.5f}, '
                f'IOU avg: {iou_avg.avg:.4f}, '
                f'{cls2iou_log}, '
                f'F1 avg: {f1_score_avg.avg:.4f}, '
                f'LR: {lr:.7f}, '
                f'loop_time: {loop_time}')
    return loss_avg.avg


def get_loaders(config):
    mask_transforms = get_mask_transforms()
    image_transforms = get_image_transforms()
    train_transforms = get_train_transforms(config.get_image('height'),
                                            config.get_image('width'))
    train_loader = get_data_loader(
        train_transforms=train_transforms,
        image_transforms=image_transforms,
        mask_transforms=mask_transforms,
        csv_paths=config.get_train_datasets('processed_data_path'),
        dataset_probs=config.get_train_datasets('prob'),
        epoch_size=config.get_train('epoch_size'),
        batch_size=config.get_train('batch_size'),
        drop_last=True
    )
    val_loader = get_data_loader(
        train_transforms=None,
        image_transforms=image_transforms,
        mask_transforms=mask_transforms,
        csv_paths=config.get_val_datasets('processed_data_path'),
        dataset_probs=config.get_val_datasets('prob'),
        epoch_size=config.get_val('epoch_size'),
        batch_size=config.get_val('batch_size'),
        drop_last=False
    )
    return train_loader, val_loader


def main(args):
    config = Config(args.config_path)
    os.makedirs(config.get('save_dir'), exist_ok=True)
    log_path = os.path.join(config.get('save_dir'), "output.log")
    logger = configure_logging(log_path)

    train_loader, val_loader = get_loaders(config)

    class_names = config.get_classes().keys()
    model = LinkResNet(output_channels=len(class_names))
    if config.get('pretrain_path'):
        states = load_pretrain_model(
            config.get('pretrain_path'), model, logger)
        model.load_state_dict(states)
        logger.info(f"Load pretrained model {config.get('pretrain_path')}")
    model.to(DEVICE)

    criterion = FbBceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.6, patience=50)

    weight_limit_control = FilesLimitControl(logger=logger)
    best_loss = np.inf
    val_loss = val_loop(
        val_loader, model, criterion, DEVICE, class_names, logger)
    for epoch in range(config.get('num_epochs')):
        train_loss = train_loop(train_loader, model, criterion, optimizer,
                                epoch, class_names, logger)
        val_loss = val_loop(
            val_loader, model, criterion, DEVICE, class_names, logger)
        scheduler.step(train_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            model_save_path = os.path.join(
                config.get('save_dir'), f'model-{epoch}-{val_loss:.4f}.ckpt')
            torch.save(model.state_dict(), model_save_path)
            logger.info(f'Model weights saved {model_save_path}')
            weight_limit_control(model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='/workdir/scripts/segm_config.json',
                        help='Path to config.json.')
    args = parser.parse_args()

    main(args)
