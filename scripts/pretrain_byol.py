import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from src.models.byol import BYOL
from src.data.byol_datamodule import BYOLDataModule
from src.utils.configs import read_yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dirs', nargs='+', required=True, help='One or more directories with images (e.g., data/isic2019/ISIC_2019_Training_Input)')
    ap.add_argument('--img-size', type=int, default=448)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--max-epochs', type=int, default=100)
    ap.add_argument('--precision', type=str, default='16-mixed')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--base-momentum', type=float, default=0.996)
    ap.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'lars'])
    ap.add_argument('--devices', type=int, default=1)
    ap.add_argument('--accumulate-grad-batches', type=int, default=1)
    ap.add_argument('--log-dir', type=str, default='outputs/byol')
    ap.add_argument('--model-cfg', type=str, default=None, help='Path to model config YAML. It refers to the DermResNetSE kwargs.')
    ap.add_argument('--experiment-name', type=str, default='pretraining')
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()

    pl.seed_everything(args.seed)

    dm = BYOLDataModule(
        data_roots=args.data_dirs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.model_cfg:
        model_cfg = read_yaml(args.model_cfg)
        print("Using model config:", model_cfg)
        if 'model' in model_cfg:
            model_cfg = model_cfg['model']
    else:
        model_cfg = {}

    # drop nunm_classes if present in cfg
    if 'num_classes' in model_cfg:
        model_cfg.pop('num_classes')


    model = BYOL(
        backbone_kwargs=dict(  # tune to your DermResNetSE signature
            num_classes=0,  # ignored; backbone only
            **model_cfg
        ),
        proj_hidden_dim=4096,
        proj_out_dim=256,
        pred_hidden_dim=4096,
        base_momentum=args.base_momentum,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        use_sync_bn=(args.devices > 1),
    )

    os.makedirs(args.log_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.experiment_name)

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, 'ckpts'),
        filename='byol-{epoch:03d}-{train_loss:.4f}',
        save_top_k=3,
        monitor='train/loss_epoch',
        mode='min',
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval='epoch')

    strategy = 'ddp' if args.devices and args.devices > 1 else 'auto'

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy=strategy,
        default_root_dir=args.log_dir,
    )

    trainer.fit(model, datamodule=dm)

    # Save backbone weights for fine-tuning
    backbone_path = os.path.join(args.log_dir, 'dermanet_byol_backbone.pt')
    torch.save(model.encoder.state_dict(), backbone_path)
    print(f"Saved backbone to {backbone_path}")


if __name__ == '__main__':
    main()
    # Example usage:
    # python scripts/pretrain_byol.py --data-dirs data/isic2019/ISIC_2019_Training_Input data/isic2024/train-image/image --img-size 448 --batch-size 128 --max-epochs 100 --log-dir outputs/byol --model-cfg cfgs/dermanet_default.yaml