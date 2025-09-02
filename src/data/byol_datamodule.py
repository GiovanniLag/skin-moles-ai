from __future__ import annotations
from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .augmentations import TwoAugmentTransform
from .datasets import UnlabeledImagePathsTwoViews


class BYOLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_roots: List[str],
        img_size: int = 448,
        batch_size: int = 128,
        num_workers: int = 8,
        val_split: float = 0.02,
        drop_last: bool = True,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_roots = data_roots
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.drop_last = drop_last
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        transform = TwoAugmentTransform(self.img_size)
        full = UnlabeledImagePathsTwoViews(self.data_roots, transform)
        if self.val_split and self.val_split > 0:
            n_val = max(1, int(len(full) * self.val_split))
            n_train = len(full) - n_val
            self.train_ds, self.val_ds = random_split(full, [n_train, n_val])
        else:
            self.train_ds, self.val_ds = full, None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else 0,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else 0,
        )