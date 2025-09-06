from pathlib import Path
from typing import BinaryIO
import uuid
from PIL import Image
import numpy as np


STORAGE_DIR = Path("streamlit/uploads")


def save_image(image: Image.Image | np.ndarray, filename: str | None = None) -> Path:
    """Save an image to the uploads directory (inside the streamlit app folder).
    
    Parameters
    ----------
    image: PIL Image or numpy array (H, W, C) in 0-255 range
        image to save
    filename: Optional str
        If None, a random UUID-based name is generated.

    Returns
    -------
    Path
        The path where the image was saved.
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"{uuid.uuid4().hex}.png"
    path = STORAGE_DIR / filename
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    image.save(path)
    return path


def save_uploaded_file(file: BinaryIO, filename: str | None = None) -> Path:
    image = Image.open(file).convert("RGB")
    return save_image(image, filename)


def load_images() -> dict[str, Image.Image]:
    """Load all images from the uploads directory."""
    images = {}
    if not STORAGE_DIR.exists():
        return images
    for path in STORAGE_DIR.iterdir():
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            try:
                img = Image.open(path).convert("RGB")
                images[path.name] = img
            except Exception:
                continue
    return images