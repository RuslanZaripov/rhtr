import argparse
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.segmentation.config import Config
from src.segmentation.dataset import get_data_loader
from src.segmentation.losses import FbBceLoss
from src.segmentation.metrics import get_iou, get_f1_score, AverageMeter, IOUMetric
from src.segmentation.models import LinkResNet
from src.segmentation.transforms import (
    get_train_transforms, get_image_transforms, get_mask_transforms
)
from src.segmentation.utils import (
    val_loop, load_pretrain_model, sec2min,
    configure_logging, EarlyStopping
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(
        data_loader, model, criterion, optimizer, epoch, class_names, logger, writer
):
    loss_avg = AverageMeter('Loss', ':.4e')
    iou_avg = AverageMeter('IOU', ':6.2f')
    cls2iou = {cls_name: IOUMetric(cls_idx)
               for cls_idx, cls_name in enumerate(class_names)}
    f1_score_avg = AverageMeter('F1 score', ':6.2f')
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
            cls2iou[cls_name].update(preds, targets)

        loss.backward()
        optimizer.step()

    loop_time = sec2min(time.time() - strat_time)
    lr = optimizer.param_groups[0]['lr']
    cls2iou_log = ''.join([f' IOU {cls_name}: {iou_fun.avg():.4f}'
                           for cls_name, iou_fun in cls2iou.items()])
    logger.info(f'Train: epoch {epoch}, {loss_avg}, {iou_avg}, {cls2iou_log}, {f1_score_avg}, '
                f'LR: {lr:.7f}, loop_time: {loop_time}')

    writer.add_scalar('Loss/train', loss_avg.avg, epoch)
    writer.add_scalar('IOU/train', iou_avg.avg, epoch)
    writer.add_scalar('F1_score/train', f1_score_avg.avg, epoch)
    writer.add_scalar('LR/train', lr, epoch)

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

    log_path = os.path.join(config.get('save_dir'), 'output.log')
    logger = configure_logging(log_path)

    writer = SummaryWriter(log_dir=config.get('tensorboard_log_dir'))

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

    def get_model_save_path(_epoch, _val_loss):
        return os.path.join(
            config.get('save_dir'),
            f'{model.__class__.__name__}-{_epoch}-{_val_loss:.4f}.ckpt'
        )

    early_stopping = EarlyStopping(logger=logger, load_best_weights=False)
    val_loss = val_loop(val_loader, model, criterion, DEVICE, class_names, logger)
    model_save_path = get_model_save_path(0, val_loss)
    early_stopping(val_loss, model_save_path, model)

    for epoch in range(config.get('num_epochs')):
        train_loss = train_loop(
            train_loader, model, criterion, optimizer, epoch, class_names, logger, writer)
        val_loss = val_loop(
            val_loader, model, criterion, DEVICE, class_names, logger, writer)

        scheduler.step(train_loss)

        model_save_path = get_model_save_path(epoch, val_loss)
        if early_stopping(val_loss, model_save_path, model):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='/workdir/scripts/segm_config.json',
                        help='Path to config.json.')
    main(parser.parse_args())
