from pathlib import Path
from typing import BinaryIO
import uuid
from PIL import Image


STORAGE_DIR = Path("uploads")


def save_image(image: Image.Image, filename: str | None = None) -> Path:
    """Save an image to the uploads directory."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"{uuid.uuid4().hex}.png"
    path = STORAGE_DIR / filename
    image.save(path)
    return path


def save_uploaded_file(file: BinaryIO, filename: str | None = None) -> Path:
    image = Image.open(file).convert("RGB")
    return save_image(image, filename)
