from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F

from .modules import BasicBlockSE, ConvBNAct, GeM


class DermResNetSE(nn.Module):
    """
    A ResNet-like architecture with Squeeze-and-Excitation blocks and GeM pooling.

    Parameters
    ----------
    layers : list[int] 
        number of blocks per stage (e.g. [2,2,2,2])
    widths : list[int] 
        number of channels per stage (e.g. [64,128,256,512])
    num_classes : int
        number of output classes, e.g. 8 for ISIC 2019
    act : str
        activation function, one of {'silu', 'relu', 'gelu'}
    se_ratio : float
        Squeeze-and-Excitation ratio. 0.25 means reduction r=4; set 0 to disable SE
    aux_binary : bool
        add a melanoma-vs-rest auxiliary head
    """
    def __init__(self,
                 layers=(2,2,2,2),
                 widths=(64,128,256,512),
                 num_classes=8,
                 act='silu',
                 se_ratio=0.25,
                 aux_binary=False):
        super().__init__()
        assert len(layers) == len(widths) == 4
        self.stem = nn.Sequential(
            ConvBNAct(3, 64, k=7, s=2, p=3, act=act),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        in_ch = 64
        self.stage1 = self._make_stage(in_ch, widths[0], layers[0], stride=1, act=act, se_ratio=se_ratio)
        in_ch = widths[0]
        self.stage2 = self._make_stage(in_ch, widths[1], layers[1], stride=2, act=act, se_ratio=se_ratio)
        in_ch = widths[1]
        self.stage3 = self._make_stage(in_ch, widths[2], layers[2], stride=2, act=act, se_ratio=se_ratio)
        in_ch = widths[2]
        self.stage4 = self._make_stage(in_ch, widths[3], layers[3], stride=2, act=act, se_ratio=se_ratio)
        self.out_ch = widths[3]
        self.pool = GeM(p=3.0, trainable=True) # This corresponds to y representation for the BYOL
        self.feat_dim = self.out_ch
        self.head = nn.Sequential(
            nn.LayerNorm(self.out_ch),
            nn.Dropout(0.2),
            nn.Linear(self.out_ch, num_classes)
        )
        self.aux_binary = aux_binary
        if aux_binary:
            self.bin_head = nn.Sequential(
                nn.LayerNorm(self.out_ch),
                nn.Dropout(0.2),
                nn.Linear(self.out_ch, 2)
            )
        self._init_weights()

    def _make_stage(self, in_ch, out_ch, n_blocks, stride, act, se_ratio):
        blocks = []
        blocks.append(BasicBlockSE(in_ch, out_ch, stride=stride, se_ratio=se_ratio, act=act))
        for _ in range(1, n_blocks):
            blocks.append(BasicBlockSE(out_ch, out_ch, stride=1, se_ratio=se_ratio, act=act))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)  # last conv feature map (for CAM)
        return x
    
    def forward_backbone(self, x):
        """Used for BYOL pretraining: returns pooled features only."""
        feats = self.forward_features(x)
        feats = self.pool(feats)
        return feats

    def forward(self, x):
        feat_map = self.forward_features(x)
        feats = self.pool(feat_map)
        logits = self.head(feats)
        if self.aux_binary:
            bin_logits = self.bin_head(feats)
            return logits, bin_logits, feats, feat_map
        return logits, feats, feat_map
    


if __name__ == "__main__":
    # Test the model
    import sys
    import os
    # Add: this_file parent / data to the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data.datasets import ISICDataset
    from data.augmentations import get_val_transforms
    from torch.utils.data import DataLoader

    model = DermResNetSE(layers=[2,2,2,2], widths=[64,128,256,512], num_classes=8, aux_binary=True)
    print(model)

    dataset = ISICDataset(csv_path='data/isic2019/isic_2019_common.csv',
                          root_dir='data/isic2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
                          transform=get_val_transforms(img_size=384),
                          labels_map={
                                'NV': 0,
                                'MEL': 1,
                                'BCC': 2,
                                'BKL': 3,
                                'AK': 4,
                                'SCC': 5,
                                'VASC': 6,
                                'DF': 7
                            }
                         )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Test the model
    batch = next(iter(dataloader))
    images, labels = batch['image'], batch['labels']
    outputs = model(images)
    print(f"output shapes: {[o.shape for o in outputs]}")

    back_bone_feats = model.forward_backbone(images)
    print(f"backbone pooled features shape: {back_bone_feats.shape}")
