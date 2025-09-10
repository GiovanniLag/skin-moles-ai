from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .inference import load_model, _preprocess_image, DEVICE  # reuse caching + preprocessing
from src.utils.image import denormalize_tensor, load_image

# NOTE: We purposely import the lightning module indirectly via load_model so that
# checkpoint loading & caching stay in a single place.

@dataclass
class GradCAMResult:
    heatmap: Image.Image          # Colored heatmap (jet)
    overlay: Image.Image          # Heatmap overlayed on original image
    raw_cam: np.ndarray           # Raw CAM (H, W) 0..1
    target_class: int             # Target class index used
    target_score: float           # Model prob/logit (prob) for that class


def _apply_colormap_on_image(img: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> Tuple[Image.Image, Image.Image]:
    """Apply jet colormap on cam and overlay on image.

    Parameters
    ----------
    img : np.ndarray
        Original image in [0,1], shape (H,W,3)
    cam : np.ndarray
        Grad-CAM heatmap already normalized to 0..1, shape (H,W)
    alpha : float
        Transparency of heatmap overlay.
    """
    import matplotlib.cm as cm

    colormap = cm.get_cmap("jet")
    colored = colormap(cam)[..., :3]  # (H,W,3) RGBA -> RGB
    heatmap_img = Image.fromarray((colored * 255).astype(np.uint8))

    overlay = (colored * alpha + img * (1 - alpha))
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    return heatmap_img, overlay_img


def generate_gradcam(single_image, model_path: str, target_class: Optional[int] = None, use_prob: bool = True, alpha: float = 0.45) -> GradCAMResult:
    """Generate Grad-CAM for a single image.

    Parameters
    ----------
    single_image : file-like | PIL.Image | np.ndarray
        Source image (RGB).
    model_path : str
        Path to checkpoint.
    target_class : Optional[int]
        If provided, use this class index; otherwise take argmax.
    use_prob : bool
        If True use softmax probability for target selection & reporting, else raw logit.
    alpha : float
        Overlay transparency.
    """
    model = load_model(model_path)
    model.eval()
    base_model = model.model  # DermResNetSE

    # Identify layer for Grad-CAM: last conv stage output (stage4 last block output)
    target_module = base_model.stage4[-1]

    # Prepare hooks
    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations['value'] = output.detach()

    def bwd_hook(_, grad_input, grad_output):  # grad_output is tuple
        gradients['value'] = grad_output[0].detach()

    handle_fwd = target_module.register_forward_hook(fwd_hook)
    handle_bwd = target_module.register_full_backward_hook(bwd_hook)  # PyTorch >=1.13

    try:
        pil_img = load_image(single_image)
        tensor = _preprocess_image(np.array(pil_img), img_size=224).to(DEVICE)  # shape [1,C,H,W]
        logits = model(tensor)  # [1,num_classes]

        if use_prob:
            probs = F.softmax(logits, dim=1)
        else:
            probs = logits

        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())

        target_score = probs[0, target_class]

        model.zero_grad(set_to_none=True)
        target_score.backward(retain_graph=False)

        act = activations['value']  # [B,C,h,w]
        grad = gradients['value']   # [B,C,h,w]

        # Global average pooling over spatial dims for weights
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (weights * act).sum(dim=1, keepdim=False)  # [B,h,w]
        cam = F.relu(cam)
        cam = cam[0].cpu().numpy()

        # Normalize CAM to 0..1
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # Resize CAM to input size (assume square 224)
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((tensor.shape[-1], tensor.shape[-2]), Image.BILINEAR)
        cam_resized = np.array(cam_img).astype(np.float32) / 255.0

        # Get original (denormalized) image for overlay (detach to avoid grad numpy error)
        denorm = denormalize_tensor(tensor.detach())
        img_np = denorm[0]  # (H,W,3) 0..1

        heatmap_img, overlay_img = _apply_colormap_on_image(img_np, cam_resized, alpha=alpha)

        return GradCAMResult(
            heatmap=heatmap_img,
            overlay=overlay_img,
            raw_cam=cam_resized,
            target_class=target_class,
            target_score=float(target_score.detach().cpu().item()),
        )
    finally:
        handle_fwd.remove()
        handle_bwd.remove()
