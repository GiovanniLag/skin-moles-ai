from typing import List
import streamlit as st
from PIL import Image

from ..components import prediction_bar
from ..services import inference, storage
from ..utils.image import load_image


def render() -> None:
    st.title("DermaNet")
    st.subheader("Upload Photos — drag & drop or browse")

    uploaded_files = st.file_uploader(
        "",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload dermatoscopic images to get predictions.")
        return

    images: List[Image.Image] = [load_image(f) for f in uploaded_files]
    names = [f.name for f in uploaded_files]
    for f, name in zip(uploaded_files, names):
        storage.save_uploaded_file(f, name)

    selected_name = st.selectbox("Select image", names)
    idx = names.index(selected_name)
    image = images[idx]

    st.image(image, caption=selected_name, width=300)

    model_path = st.secrets.get("MODEL_PATH", "model.ckpt")
    try:
        preds = inference.predict(image, model_path)
        st.markdown("### Prediction")
        prediction_bar(preds)
    except Exception as e:  # noqa: BLE001
        st.error(f"Prediction failed: {e}")

    st.markdown("### Your images")
    cols = st.columns(len(images))
    for col, img, name in zip(cols, images, names):
        col.image(img, caption=name, width=100)
