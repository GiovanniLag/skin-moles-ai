import argparse
import json
import os
import re

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from src.data.isic_datamodule import ISICDataModule
from src.models.dermanet_module import DermaNetLightning
from src.utils.configs import read_yaml, write_yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--img_dir', type=str, required=True)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--model_cfg', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default='outputs/dermanet')
    ap.add_argument('--experiment_name', type=str, default='training')
    ap.add_argument('--devices', type=int, default=1)
    ap.add_argument('--early_stop', type=int, default=8)
    ap.add_argument('--precision', type=str, default='32-true', choices=['16-mixed', '32-true']) 
    ap.add_argument('--gradient_clip_val', type=float, default=1.0, help='Set to 0 to disable gradient clipping')
    ap.add_argument('--resume_version', type=int, default=None, help='If specified, will resume from the latest checkpoint in this version directory.') 
    return ap.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    # Setup the datamodule
    dm = ISICDataModule(
        csv_path=args.csv,
        img_dir=args.img_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    dm.setup()

    num_classes = dm.num_classes
    class_weights = dm.class_weights

    if args.model_cfg:
        model_cfg = read_yaml(args.model_cfg)
        if 'model' in model_cfg:
            model_cfg = model_cfg['model']
    else:
        model_cfg = {}

    # drop num_classes if present in cfg
    if 'num_classes' in model_cfg:
        model_cfg.pop('num_classes')

    exp_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True) # ensure base experiment directory exists
    if not args.resume_version:
        # Find the next available version number
        versions = []
        for name in os.listdir(exp_dir):
            m = re.match(r'^version_(\d+)$', name)
            if m and os.path.isdir(os.path.join(exp_dir, name)):
                versions.append(int(m.group(1)))
        run_version = max(versions) + 1 if versions else 0  # Note: If versions is empty, start at 0 (logger's default).
    else:
        run_version = args.resume_version
        # Check that the specified version directory exists
        version_dir = os.path.join(exp_dir, f'version_{run_version}')
        if not os.path.isdir(version_dir):
            raise ValueError(f"Specified resume_version {run_version} does not exist in {exp_dir}.")
        
    # Create logger with explicit version
    logger = TensorBoardLogger(save_dir=args.output_dir, name=args.experiment_name, version=run_version)
    
    # Set run_dir to logger's log_dir (this creates the directory if needed)
    run_dir = logger.log_dir
    ckpt_dir = os.path.join(run_dir, 'ckpts')

    # Initialize the model
    model = DermaNetLightning(
        num_classes=num_classes,
        class_weights=class_weights,
        model_cfg=model_cfg,
        epochs=args.epochs,
        confusion_matrix_dir=os.path.join(run_dir, 'conf_matrix'),
        num_to_label=dm.num_to_label,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best',
        monitor='val/auroc',
        mode='max',
        save_top_k=1,
        save_last=True,
    )
    early_cb = EarlyStopping(monitor='val/acc1', mode='max', patience=args.early_stop)
    lr_cb = LearningRateMonitor(logging_interval='epoch')
    progress_cb = RichProgressBar()

    # Set the strategy for distributed training if multiple GPUs are used
    strategy = 'ddp' if args.devices and args.devices > 1 else 'auto'

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.devices,
        strategy=strategy,
        precision=args.precision,
        deterministic=True,
        benchmark=True,  # choose the best algorithm for CuDNN convolutions for the given input size
        callbacks=[ckpt_cb, early_cb, lr_cb, progress_cb],
        logger=logger,
        default_root_dir=args.output_dir,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else 0.0,
    )
    
    # Ensure checkpoint directory exists
    os.makedirs(ckpt_dir, exist_ok=True)

    # Find checkpoint to resume from if resuming a version
    resume_ckpt_path = None
    if args.resume_version:
        # Look for last.ckpt first (most recent state), then best.ckpt as fallback
        potential_ckpts = [
            os.path.join(ckpt_dir, 'last.ckpt'),
            os.path.join(ckpt_dir, 'best.ckpt')
        ]
        for ckpt_path in potential_ckpts:
            if os.path.isfile(ckpt_path):
                resume_ckpt_path = ckpt_path
                print(f"Resuming training from checkpoint: {ckpt_path}")
                break
        
        if resume_ckpt_path is None:
            print(f"Warning: No checkpoint found in {ckpt_dir}. Starting fresh training in version {run_version}.")

    # Train - validate - test (validation and test are done on best checkpoint)
    if resume_ckpt_path is None:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt_path)
    val_metrics = trainer.validate(model, datamodule=dm, ckpt_path='best')
    test_metrics = trainer.test(model, datamodule=dm, ckpt_path='best')

    # Save artifacts
    with open(os.path.join(run_dir, 'label_encoder.json'), 'w') as f:
        json.dump(dm.label_to_num, f, indent=2)
    with open(os.path.join(run_dir, 'class_weights.json'), 'w') as f:
        json.dump(class_weights.tolist(), f, indent=2)
    write_yaml(os.path.join(run_dir, 'config.yaml'), vars(args))
    metrics = {'val': val_metrics[0] if val_metrics else {},
               'test': test_metrics[0] if test_metrics else {}}
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
    # python -m scripts.train --csv data/isic2019/isic_2019_common.csv --img_dir data/isic2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input --img_size 224 --batch_size 64 --epochs 50 --experiment_name test1 --model_cfg cfgs/dermanet_default.yaml