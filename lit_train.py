import sys
import argparse

import torch
from compressai.zoo import models
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from lightning.pytorch.profilers import SimpleProfiler

from lit_model import CompressorLit
from lit_data import CLIC
from lit_config import Config


def main(args):
    parser = argparse.ArgumentParser(description="Lighting-based CompressAI training.")
    parser.add_argument('-c', '--comment', type=str, help='concise description of this experiment', required=True)
    args = parser.parse_args(args)

    model_name = f'{Config.Model.model}-lambda={Config.Model.lmbda}-beta={Config.Model.beta}'
    dir_name = 'ckpt' + '/' + model_name

    if Config.Trainer.seed is not None:
        seed_everything(Config.Trainer.seed, workers=True)

    if Config.Trainer.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(Config.Trainer.float32_matmul_precision)

    data_lit = CLIC(
        data_dir=Config.Data.dataset_directory,
        num_workers=Config.Data.num_workers,
        pin_memory=Config.Data.pin_memory,
        persistent_workers=Config.Data.persistent_workers,
        train_batch_size=Config.Data.train_batch_size,
        train_patch_size=Config.Data.train_patch_size,
        valid_batch_size=Config.Data.valid_batch_size,
        valid_patch_size=Config.Data.valid_patch_size,
    )
    net_lit = CompressorLit(
        net=models[Config.Model.model](Config.Model.quality),
        net_lr=Config.Trainer.net_lr,
        aux_lr=Config.Trainer.aux_lr,
        lmbda=Config.Model.lmbda,
        beta=Config.Model.beta,
        gradient_clip_norm=Config.Trainer.gradient_clip_norm,
        comment=args.comment
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='valid/loss',
        mode='min',
        filename='epoch={epoch}-val_loss={valid/loss:.4f}-best',
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
        save_last=True,
        verbose=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = 'epoch={epoch}-loss={train/loss:.4f}-last'

    profiler_to_file = SimpleProfiler(filename='simple_profile', extended=True)

    trainer = Trainer(
        default_root_dir=dir_name,
        accelerator=Config.Trainer.accelerator,
        devices=Config.Trainer.devices,
        max_epochs=Config.Trainer.num_epochs,
        check_val_every_n_epoch=Config.Trainer.validation_cadence,
        log_every_n_steps=Config.Trainer.log_cadence,
        callbacks=[LearningRateMonitor(logging_interval='epoch'), checkpoint_callback],
        strategy=Config.Trainer.strategy,  # due to .quantile parameters
        profiler=profiler_to_file,
    )
    trainer.logger._default_hp_metric = None

    trainer.fit(model=net_lit, datamodule=data_lit, ckpt_path=Config.Trainer.checkpoint_to_resume)


if __name__ == "__main__":
    main(sys.argv[1:])
