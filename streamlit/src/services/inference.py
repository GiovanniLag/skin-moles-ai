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
def _preprocess_image(image: np.ndarray, img_size: int) -> torch.Tensor:
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


def infer_images(
    images: list, model_path: str, filenames: list[str]
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Image.Image]]:
    """Run inference on one or multiple images and return a mapping from filename to class probabilities.

    Parameters
    ----------
    images : list
        A list of images to process. Can be file-like objects or PIL Images.
    model_path : str
        Path to the model checkpoint.
    filenames : list[str]
        A list of filenames corresponding to the images.

    Returns
    -------
    tuple
        - A dictionary mapping filename to {label: probability}.
        - A dictionary mapping filename to the processed PIL Image.
    """
    results: Dict[str, Dict[str, float]] = {}
    processed_images: Dict[str, Image.Image] = {}

    for i, file_or_img in enumerate(images):
        filename = filenames[i]
        image = load_image(file_or_img)  # Handles both file-like and PIL
        
        # Preprocess and predict
        tensor = _preprocess_image(np.array(image), img_size=224).to(DEVICE)
        results[filename] = predict(tensor, model_path)
        
        # Denormalize for display
        denorm = denormalize_tensor(tensor)
        if denorm.ndim == 4 and denorm.shape[0] == 1:
            denorm = denorm[0]
            
        denorm_img_arr = (denorm * 255.0).round().astype(np.uint8)
        processed_images[filename] = Image.fromarray(denorm_img_arr)
        
    return results, processed_images