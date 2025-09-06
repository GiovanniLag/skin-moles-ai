from pathlib import Path
from typing import Dict

import streamlit as st
import torch
from PIL import Image
import numpy as np

# Make sure the repository root is on the path to import the model
import sys
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR / "src"))
STREAMLIT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(STREAMLIT_DIR))

# Import the existing model without copying its code
from models.dermanet_module import DermaNetLightning # noqa: E402
from data.augmentations import InferenceTransform # noqa: E402
from src.utils.image import load_image, denormalize_tensor # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model(model_path: str) -> DermaNetLightning:
    """Load the DermaNet model from a checkpoint."""
    ckpt = Path(model_path)
    model = DermaNetLightning.load_from_checkpoint(ckpt, map_location=DEVICE)
    model.eval()
    return model


@st.cache_data
def preprocess_image(image: Image.Image, img_size: int) -> torch.Tensor:
    """Preprocess a PIL image for model input."""
    transform = InferenceTransform(img_size)
    tensor = transform(image).unsqueeze(0)
    return tensor

def predict(image: torch.Tensor, model_path: str) -> Dict[str, float]:
    """Run inference on a PIL image and return class probabilities."""
    model = load_model(model_path)
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1).squeeze()
    labels = getattr(model, "num_to_label", {i: str(i) for i in range(len(probs))})
    return {labels[i]: float(probs[i]) for i in range(len(probs))}


def infer_images(images, model_path: str) -> Dict[str, Dict[str, float]]:
    """Run inference on one or multiple images and return a mapping from filename to class probabilities.

    Parameters
    ----------
    images : UploadedFile or list of UploadedFile
        One or more images uploaded via Streamlit's file uploader.

    Returns
    -------
    dict
        Mapping filename -> {label: probability}
    """
    results: Dict[str, Dict[str, float]] = {}
    # Process each uploaded file and run prediction. We return only the
    # `results` mapping because callers expect a mapping that can be passed
    # directly to `dict.update()` (see `home_page.on_file_upload`).
    files = images if isinstance(images, list) else [images]
    processed_images: Dict[str, np.ndarray] = {}
    for file in files:
        image = load_image(file)
        image = np.array(image)
        tensor = preprocess_image(image, img_size=224).to(DEVICE)
        results[file.name] = predict(tensor, model_path)
        # denormalize_tensor returns an array with shape [B, H, W, C]
        denorm = denormalize_tensor(tensor)
        # If batch dim is present, take first image
        if denorm.ndim == 4 and denorm.shape[0] == 1:
            denorm = denorm[0]
        # Convert float [0,1] -> uint8 [0,255] and to PIL Image for Streamlit
        denorm_img = (denorm * 255.0).round().astype(np.uint8)
        processed_images[file.name] = Image.fromarray(denorm_img)
    return results, processed_images