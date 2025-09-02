"""
BYOL (Bootstrap Your Own Latent) implementation for pretraining the backbone of Dermanet. 
"""
from __future__ import annotations
import math
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .dermanet import DermResNetSE


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, final_bn: bool = False) -> nn.Sequential:
    """Simple 3-layer MLP with optional final BatchNorm, used for projector and predictor in BYOL."""
    layers = [
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim, bias=False),
    ]
    if final_bn: # False by default. In the original paper they don't batchnorm neither projector nor predictor.
        layers.append(nn.BatchNorm1d(out_dim, affine=True)) # affine=True -> learns scale and bias parameters
    return nn.Sequential(*layers)


class Normalize(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1, p=2)


class DermNetBackbone(nn.Module):
    """
    Thiny wrapper around DermResNetSE to expose the ResNet backbone + GeM pooling.
    Uses forward_backbone(x) to return pooled features only.
    Exposes feat_dim attribute for feature size.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = DermResNetSE(**kwargs)
        self.feat_dim = int(self.backbone.feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_backbone(x)


class BYOL(pl.LightningModule):
    """Bootstrap Your Own Latent (BYOL) LightningModule for DermResNetSE.

    Args:
        backbone_kwargs: kwargs for DermResNetSE (e.g., depths, widths, activation, etc.)
        proj_hidden_dim: hidden dim of projector MLP
        proj_out_dim: output dim of projector (and predictor input). Common: 256
        pred_hidden_dim: hidden dim of predictor MLP
        base_momentum: initial EMA momentum for target network (e.g., 0.996)
        optimizer: 'adamw' (default) or 'lars'
        lr: learning rate for AdamW, or base lr for LARS (scaled by batch size)
        weight_decay: weight decay
        max_epochs: needed for cosine momentum schedule
    """

    def __init__(
        self,
        backbone_kwargs: Optional[dict] = None,
        proj_hidden_dim: int = 4096,
        proj_out_dim: int = 256,
        pred_hidden_dim: int = 4096,
        base_momentum: float = 0.996,
        optimizer: str = 'adamw',
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        use_sync_bn: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        backbone_kwargs = backbone_kwargs or {}
        self.encoder = DermNetBackbone(**backbone_kwargs)
        feat_dim = self.encoder.feat_dim

        # Projector and predictor
        self.projector = nn.Sequential(_mlp(feat_dim, proj_hidden_dim, proj_out_dim, final_bn=False), Normalize())
        self.predictor = nn.Sequential(_mlp(proj_out_dim, pred_hidden_dim, proj_out_dim, final_bn=False))

        # Target network (EMA copy of encoder+projector)
        self.target_encoder = deepcopy(self.encoder)
        self.target_projector = deepcopy(self.projector)
        # Stop gradients to the target network
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        if use_sync_bn: # optionally convert all BatchNorm layers to SyncBatchNorm for multi-GPU training
            self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            self.projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
            self.predictor = nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)
            self.target_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.target_encoder)
            self.target_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.target_projector)

        self.base_momentum = base_momentum
        self.current_momentum = base_momentum
        self.max_epochs_for_sched = max_epochs

    # ---------------------- utils ----------------------
    @staticmethod
    def _byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """2 - 2 * cos_sim(p, z)
        Both inputs are expected to be un-normalized; we'll L2 normalize inside.
        """
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return 2 - 2 * (p * z).sum(dim=1)

    @torch.no_grad()
    def _momentum_update(self):
        m = float(self.current_momentum)
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.data = target.data * m + online.data * (1.0 - m)
        for online, target in zip(self.projector.parameters(), self.target_projector.parameters()):
            target.data = target.data * m + online.data * (1.0 - m)

    def _update_momentum_schedule(self):
        # Cosine schedule to anneal (1 - m) -> 0 over training
        # m_t = 1 - (1 - m0) * (cos(pi * t/T) + 1)/2
        t = self.current_epoch + self.global_step / max(1, self.trainer.num_training_batches)
        T = max(1e-6, float(self.max_epochs_for_sched))
        m0 = float(self.base_momentum)
        self.current_momentum = 1.0 - (1.0 - m0) * (math.cos(math.pi * min(t / T, 1.0)) + 1.0) / 2.0

    # ---------------------- Lightning hooks ----------------------
    def training_step(self, batch, batch_idx):
        # batch: (view1, view2)
        x1, x2 = batch

        # Online network
        y1 = self.encoder(x1)  # [B, D]
        y2 = self.encoder(x2)
        z1 = self.projector(y1)  # [B, P]
        z2 = self.projector(y2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Target network (stop-grad)
        with torch.no_grad():
            self.target_encoder.eval()
            h1 = self.target_encoder(x1)
            h2 = self.target_encoder(x2)
            t1 = self.target_projector(h1)
            t2 = self.target_projector(h2)

        loss = self._byol_loss(p1, t2).mean() + self._byol_loss(p2, t1).mean() # Symmetric loss
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/momentum', float(self.current_momentum), on_step=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_momentum_schedule()
        with torch.no_grad():
            self._momentum_update()

    def configure_optimizers(self):
        opt_name = str(self.hparams.optimizer).lower()
        params = list(self.encoder.parameters()) + list(self.projector.parameters()) + list(self.predictor.parameters())

        if opt_name == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif opt_name == 'lars':
            try:
                from torch.optim import lr_scheduler
                from torch.optim.optimizer import Optimizer
                from torch.optim import SGD
            except Exception as e:
                raise ImportError('LARS selected but dependencies not available')
            # Lightweight LARS via torch.optim.SGD + manual LARS could be added; keep AdamW default for simplicity.
            optimizer = torch.optim.SGD(params, lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.weight_decay, nesterov=False)
        else:
            raise ValueError("optimizer must be 'adamw' or 'lars'")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr * 0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'train/loss',
            }
        }