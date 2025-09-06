from io import BytesIO
from typing import BinaryIO
from PIL import Image
import torch


def load_image(uploaded_file: BinaryIO) -> Image.Image:
    """Load an uploaded file into a PIL image."""
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        data = uploaded_file.read()
    return Image.open(BytesIO(data)).convert("RGB")


def denormalize_tensor(imgs, mean=None, std=None):
    """Denormalize a batch tensor that was normalized with albumentations A.Normalize.

    imgs: torch.Tensor of shape [B, C, H, W]
    mean, std: iterables of length 3 (RGB) in 0-1 scale
    Returns: tensor [B, C, H, W] with values clamped to [0,1]
    """
    if mean is None or std is None:
        mean = [0.696, 0.522, 0.426]
        std = [0.142, 0.135, 0.128]
    mean = torch.tensor(mean, device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std, device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
    imgs_denorm = imgs * std + mean
    imgs_denorm = imgs_denorm.clamp(0.0, 1.0)
    # convert to numpy array (put on CPU if needed)
    return imgs_denorm.permute(0, 2, 3, 1).contiguous().cpu().numpy()