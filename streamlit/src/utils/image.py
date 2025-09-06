from io import BytesIO
from typing import BinaryIO
from PIL import Image


def load_image(uploaded_file: BinaryIO) -> Image.Image:
    """Load an uploaded file into a PIL image."""
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        data = uploaded_file.read()
    return Image.open(BytesIO(data)).convert("RGB")
