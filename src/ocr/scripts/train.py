import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

import src.ocr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
        epoch,
        model,
        decoder,
        data_loader,
        criterion,
        optimizer,
        logger
):
    loss_avg = src.ocr.AverageMeter()
    acc_avg = src.ocr.AverageMeter()
    wer_avg = src.ocr.AverageMeter()
    cer_avg = src.ocr.AverageMeter()

    start_time = time.time()

    model.train()
    for images, labels, enc_labels, label_lengths in tqdm(data_loader, total=len(data_loader)):
        model.zero_grad()

        batch_size = len(labels)
        images = images.to(DEVICE)

        output = model(images)

        pred_labels = decoder.decode(output)

        acc_avg.update(src.ocr.accuracy(labels, pred_labels), batch_size)
        wer_avg.update(src.ocr.wer(labels, pred_labels), batch_size)
        cer_avg.update(src.ocr.cer(labels, pred_labels), batch_size)

        output_lenghts = torch.full(
            size=(output.size(1),),
            fill_value=output.size(0),
            dtype=torch.long
        )

        loss = criterion(output, enc_labels, output_lenghts, label_lengths)
        loss_avg.update(loss.item(), batch_size)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

    elapsed_time = src.ocr.format_sec(time.time() - start_time)
    lr = optimizer.param_groups[0]['lr']

    logger.info(f'Epoch: {epoch}, '
                f'Loss: {loss_avg.avg:.4f}, '
                f'Acc: {acc_avg.avg:.4f}, '
                f'WER: {wer_avg.avg:.4f}, '
                f'CER: {cer_avg.avg:.4f}, '
                f'LR: {lr:.4f}, '
                f'Elapsed Time: {elapsed_time}')

    return loss_avg.avg


def get_data_loaders(config):
    train_transform = src.ocr.get_train_transform(
        config.get_image('height'),
        config.get_image('width'),
    )
    val_transform = src.ocr.get_val_transform(
        config.get_image('height'),
        config.get_image('width'),
    )
    train_loader = src.ocr.get_data_loader(
        dataset_root=config.get('dataset_root'),
        csv_filename=config.get_train('csv_filename'),
        h5_filename=config.get('h5_filename'),
        batch_size=config.get_train('batch_size'),
        transform=train_transform,
        shuffle=True,
        num_workers=config.get_dataloader('num_workers'),
        drop_last=True,
    )
    val_loader = src.ocr.get_data_loader(
        dataset_root=config.get('dataset_root'),
        csv_filename=config.get_val('csv_filename'),
        h5_filename=config.get('h5_filename'),
        batch_size=config.get_val('batch_size'),
        transform=val_transform,
        shuffle=True,
        num_workers=config.get_dataloader('num_workers'),
        drop_last=False,
    )
    return train_loader, val_loader


def main(args):
    config = src.ocr.Config(args.config_path)

    os.makedirs(config.get('save_dir'), exist_ok=True)

    log_path = os.path.join(config.get('save_dir'), 'train.log')
    logger = src.ocr.configure_logging(log_path)

    img_height = config.get_image('height')
    img_width = config.get_image('width')

    tokenizer = src.ocr.Tokenizer(config.get('alphabet'))
    num_class = tokenizer.get_num_chars()

    train_loader, val_loader = get_data_loaders(config)

    crnn = src.ocr.CRNN(1, img_height, img_width, num_class,
                        map_to_seq_hidden=64,
                        rnn_hidden=255,
                        leaky_relu=False)
    """
    crnn = src.ocr.CRNN2(
        number_class_symbols=num_class,
    )
    """
    crnn.to(DEVICE)

    decoder = src.ocr.BestPathDecoder(tokenizer)

    optimizer = torch.optim.RMSprop(crnn.parameters(), lr=config.get_optimizer('lr'))
    criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)

    weight_files_controller = src.ocr.FilesLimitController(logger)

    best_acc = -np.inf
    for epoch in range(config.get_train('epochs')):
        acc_avg = src.ocr.val_loop(
            data_loader=val_loader,
            model=crnn,
            decoder=decoder,
            logger=logger,
            device=DEVICE,
        )
        loss_avg = train_one_epoch(
            epoch=epoch,
            model=crnn,
            decoder=decoder,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            logger=logger
        )
        if acc_avg > best_acc:
            best_acc = acc_avg

            model_name = crnn.__class__.__name__
            model_save_path = os.path.join(
                config.get('save_dir'),
                f'{model_name}-{epoch}-{acc_avg:.4f}.ckpt'
            )
            torch.save(crnn.state_dict(), model_save_path)
            logger.info(f'Model saved to {model_save_path}')
            weight_files_controller(model_save_path)


if __name__ == "__main__":
    print(os.environ.get('PYTHONPATH'))
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config_path',
                             help='Path to .yaml configuration file',
                             required=True,
                             type=str)
    main(args_parser.parse_args())
