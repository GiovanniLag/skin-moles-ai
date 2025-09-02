import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from glob import glob
from typing import List, Tuple, Optional

class ISICDataset(Dataset):
    """
    A PyTorch dataset for the ISIC datasets.
    It can be used for datasets that have been pre-processed into a common format.

    It can combine multiple datasets by providing a list of CSV files and corresponding root directories.

    The items returned by the dataset are dictionaries with the following structure:
    {
        'images': image_tensor,  # Transformed image tensor
        'labels': label_index,   # Numerical label index
        'meta': {                # Metadata dictionary (if available)
            'column1': value1,
            'column2': value2,
            ...
        }
    }
    The 'meta' dictionary contains all columns from the CSV file except 'image' and 'label'.
    Moreover, if 'return_image_id' is set to True, the 'meta' dictionary will also include the image ID.

    """
    def __init__(self, 
                 csv_path: str, 
                 root_dir: str, 
                 transform: Optional[callable] = None, 
                 return_image_id: bool = False, 
                 labels_map: Optional[dict] = None):
        """
        Parameters:
        ----------
        csv_path : str or list[str]
            Path to the CSV file containing image names and labels.
            The CSV should contain at least 'image' and 'label' columns. Other metadata columns are optional.
            You can provide a single CSV file or a list of CSV files. In this latter case, the datasets will be concatenated
            and you should provide a list of root directories of the same length and of matching order.
        root_dir : str or list[str]
            Directory with all the images.
            Can be a single directory or a list of directories if multiple CSV files are provided.
        transform : callable, optional
            Optional transform to be applied on a sample.
        return_image_id : bool, optional
            Whether to return the image ID along with the sample. Default is False.
            If True the image ID will be included in the meta dictionary.
        labels_map : dict, optional
            Optional mapping from string labels to numerical indices.
            If not provided, the mapping will be created from the dataset.
        """
        if isinstance(csv_path, list):
            dfs = []
            for i, path in enumerate(csv_path):
                df = pd.read_csv(path)
                df['dataset_idx'] = i
                dfs.append(df)
            self.data_frame = pd.concat(dfs, ignore_index=True)
        else:
            self.data_frame = pd.read_csv(csv_path)
            self.data_frame['dataset_idx'] = 0
        self.root_dirs = root_dir if isinstance(root_dir, list) else [root_dir]
        self.transform = transform
        self.return_image_id = return_image_id

        assert len(self.root_dirs) == self.data_frame['dataset_idx'].nunique(), \
            "Number of root directories must match number of datasets in CSV files."

        # Ensure labels are strings and create mappings
        self.data_frame['label'] = self.data_frame['label'].astype(str)
        if labels_map:
            self.label_to_num = labels_map
            self.num_to_label = {v: k for k, v in labels_map.items()}
        else:
            self._unique_labels = sorted(self.data_frame['label'].unique())
            self.label_to_num = {label: i for i, label in enumerate(self._unique_labels)}
            self.num_to_label = {i: label for i, label in enumerate(self._unique_labels)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dirs[self.data_frame.iloc[idx]['dataset_idx']], self.data_frame.iloc[idx]['image'] + ".jpg")
        labels = str(self.data_frame.iloc[idx]['label'])
        labels = self.label_to_num[labels]
        
        meta = self.data_frame.iloc[idx].to_dict()
        if not self.return_image_id:
            meta.pop('image', None)
        else:
            meta['image_name'] = self.data_frame.iloc[idx]['image']
        meta.pop('label', None)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sample = {'image': image, 'labels': labels, 'meta': meta}

        if self.transform:
            sample['image'] = self.transform(image=sample['image'])['image']

        return sample


class UnlabeledImagePathsTwoViews(Dataset):
    def __init__(self, roots: List[str], transform, exts=(".jpg", ".jpeg", ".png")):
        self.paths: List[str] = []
        for r in roots:
            for ext in exts:
                self.paths.extend(glob(os.path.join(r, f"**/*{ext}"), recursive=True))
        self.paths.sort()
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under: {roots}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.paths[idx]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        v1, v2 = self.transform(image=img)
        return v1, v2

