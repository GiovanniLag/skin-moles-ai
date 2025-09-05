from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

from .datasets import ISICDataset, collate_with_meta
from .augmentations import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
)


class ISICDataModule(pl.LightningDataModule):
    """LightningDataModule for ISIC classification datasets.

    Performs patient-level grouped splits to avoid leakage. If no grouping
    column is found, each sample is treated as its own group.
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        img_size: int = 224,
        batch_size: int = 64,
        num_workers: int = 8,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        drop_last: bool = False,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.drop_last = drop_last
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv(self.csv_path)
        df['label'] = df['label'].astype(str)

        # Determine grouping key
        if 'patient_id' in df.columns:
            group_key = 'patient_id'
        elif 'lesion_id' in df.columns:
            group_key = 'lesion_id'
        else: # If no grouping column, treat each sample as a different patient
            group_key = '_group_temp'
            df[group_key] = np.arange(len(df))
        self.group_key = group_key

        self.label_to_num = {lab: i for i, lab in enumerate(sorted(df['label'].unique()))}
        self.num_to_label = {i: lab for lab, i in self.label_to_num.items()}
        self.num_classes = len(self.label_to_num)

        # grouped, stratified split
        rng = np.random.default_rng(self.seed)
        group_df = df[[group_key, 'label']].drop_duplicates(subset=[group_key])
        train_groups, val_groups, test_groups = set(), set(), set()
        for label, g in group_df.groupby('label'):
            # Sorting by label keeps the original class distribution
            groups = g[group_key].tolist()
            rng.shuffle(groups)
            n = len(groups)
            n_train = int(n * self.train_frac)
            n_val = int(n * self.val_frac)
            train_groups.update(groups[:n_train])
            val_groups.update(groups[n_train:n_train + n_val])
            test_groups.update(groups[n_train + n_val:])

        # assign splits in the original dataframe
        df['split'] = 'train'
        df.loc[df[group_key].isin(val_groups), 'split'] = 'val'
        df.loc[df[group_key].isin(test_groups), 'split'] = 'test'

        # split dataframes
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        val_df = df[df['split'] == 'val'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)

        # compute class weights from train set
        counts = train_df['label'].map(self.label_to_num).value_counts().sort_index()
        counts = counts.reindex(range(self.num_classes), fill_value=0)
        weights = len(train_df) / (self.num_classes * counts) # inverse frequency
        self.class_weights = torch.tensor(weights.values, dtype=torch.float32)

        # log class distribution
        print('Train class distribution:', counts.to_dict())
        val_counts = val_df['label'].map(self.label_to_num).value_counts().sort_index()
        print('Val class distribution:', val_counts.to_dict())

        self.train_ds = ISICDataset(train_df, self.img_dir,
                                    transform=get_train_transforms(self.img_size),
                                    labels_map=self.label_to_num)
        self.val_ds = ISICDataset(val_df, self.img_dir,
                                  transform=get_val_transforms(self.img_size),
                                  labels_map=self.label_to_num)
        self.test_ds = ISICDataset(test_df, self.img_dir,
                                   transform=get_test_transforms(self.img_size),
                                   labels_map=self.label_to_num)

    # dataloaders
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=3 if self.num_workers > 0 else 0,
            collate_fn=collate_with_meta,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=3 if self.num_workers > 0 else 0,
            collate_fn=collate_with_meta,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=3 if self.num_workers > 0 else 0,
            collate_fn=collate_with_meta,
        )