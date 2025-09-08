import argparse
import json
import os
import re

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from src.data.isic_datamodule import ISICDataModule
from src.models.dermanet_module import DermaNetLightning
from src.utils.configs import read_yaml, write_yaml
from src.models.dermanet import DermResNetSE


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
    ap.add_argument('--pretrain_ckpt', type=str, default=None, help='Path to a pretrained checkpoint to initialize the model. If specified, training will start from this checkpoint.')
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
    if args.resume_version is None:
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
        
    print(f"Experiment directory: {exp_dir}, using version: {run_version}")
        
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

    # If a pretrained BYOL checkpoint is provided, try to load its encoder/backbone weights
    def load_byol_backbone(pretrain_path: str, target_model: DermResNetSE):
        """Load a BYOL-pretrained backbone into the target DermResNetSE instance.

        Supports two common formats:
        - A state_dict saved via `torch.save(model.encoder.state_dict(), path)` (usually .pt)
        - A Lightning checkpoint (.ckpt) containing a nested 'state_dict' with keys like 'encoder.backbone...'

        The loader will attempt to match relevant keys (backbone and encoder) and will load partially
        when shapes or keys mismatch, reporting missing and unexpected keys.
        """
        import torch as _torch

        if not pretrain_path or not os.path.isfile(pretrain_path):
            print(f"Pretrain checkpoint path does not exist: {pretrain_path}")
            return

        print(f"Loading pretrained BYOL checkpoint from: {pretrain_path}")

        # Load the file (allow map_location to cpu for safety)
        ckpt = _torch.load(pretrain_path, map_location='cpu')

        # Determine if this is a raw state_dict or a Lightning checkpoint
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        elif isinstance(ckpt, dict) and any(k.startswith('encoder') or k.startswith('target_encoder') for k in ckpt.keys()):
            sd = ckpt
        else:
            # assume it's a raw state_dict
            sd = ckpt

        # Normalize keys: remove leading 'module.' if present (DataParallel) and optional 'encoder.' or 'encoder.backbone.' prefixes
        new_sd = {}
        for k, v in sd.items():
            new_k = k
            if new_k.startswith('module.'):
                new_k = new_k[len('module.'):]
            # Many BYOL checkpoints store encoder as 'encoder.backbone.<...>' or 'encoder.<...>' depending on how it was saved
            if new_k.startswith('encoder.backbone.'):
                new_k = new_k[len('encoder.backbone.'):]
            elif new_k.startswith('encoder.'):
                # try to strip just 'encoder.' too
                new_k = new_k[len('encoder.'):]
            # handle raw encoder.state_dict() where keys start with 'backbone.'
            if new_k.startswith('backbone.'):
                new_k = new_k[len('backbone.'):]
            new_sd[new_k] = v

        # Now try to load into the target model's backbone (which lives under target_model.backbone)
        # We will attempt to find matching keys for either the whole model or the backbone only.
        target_sd = target_model.state_dict()

        # Build a filtered dict of keys that appear in both
        filtered = {k: v for k, v in new_sd.items() if k in target_sd and v.shape == target_sd[k].shape}

        missing_keys = [k for k in target_sd.keys() if k not in filtered]
        unexpected_keys = [k for k in new_sd.keys() if k not in filtered]

        # Load matched params
        if filtered:
            target_sd.update(filtered)
            target_model.load_state_dict(target_sd)
            print(f"Loaded {len(filtered)} param tensors into target backbone.")
        else:
            print("No matching parameter shapes/keys found to load into the target backbone. Skipping load.")

        if missing_keys:
            print(f"Warning: {len(missing_keys)} parameters in the target model were not found in the pretrained checkpoint (they will keep their initialized values).")
        if unexpected_keys:
            print(f"Note: {len(unexpected_keys)} parameters were present in the pretrained checkpoint but not used: (showing up to 20)\n" + str(unexpected_keys[:20]))

    # If user provided a pretraining checkpoint, attempt to load it into the underlying backbone
    if args.pretrain_ckpt is not None:
        try:
            # target: the DermResNetSE instance inside the lightning wrapper
            target_backbone = model.model
            load_byol_backbone(args.pretrain_ckpt, target_backbone)
        except Exception as e:
            print(f"Warning: failed to load pretrained checkpoint '{args.pretrain_ckpt}': {e}")

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
    if args.resume_version is not None:
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