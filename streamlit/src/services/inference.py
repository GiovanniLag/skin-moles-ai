from pathlib import Path
from typing import Dict

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Make sure the repository root is on the path to import the model
import sys
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR / "src"))

# Import the existing model without copying its code
from models.dermanet_module import DermaNetLightning


@st.cache_resource
def load_model(model_path: str) -> DermaNetLightning:
    """Load the DermaNet model from a checkpoint."""
    ckpt = Path(model_path)
    model = DermaNetLightning.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval()
    return model


def predict(image: Image.Image, model_path: str) -> Dict[str, float]:
    """Run inference on a PIL image and return class probabilities."""
    model = load_model(model_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
    labels = getattr(model, "num_to_label", {i: str(i) for i in range(len(probs))})
    return {labels[i]: float(probs[i]) for i in range(len(probs))}
