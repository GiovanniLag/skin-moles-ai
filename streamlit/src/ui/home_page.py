import io
import base64
from PIL import Image
import streamlit as st
from st_click_detector import click_detector

from ..components import prediction_bar
from ..services import inference, storage, statistics
from ..services.weighs_registry import get_active_weights, get_spec


def handle_uploads() -> None:
    """Save uploaded files to the storage directory."""
    files = st.session_state.get("uploaded_image", [])
    for file in files:
        img = Image.open(file).convert("RGB")
        # resize to 224x224 for consistency
        img = img.resize((224, 224))
        storage.save_image(img, file.name)


def _pil_to_base64(img: Image.Image) -> str:
    """Encode PIL image to base64 (PNG)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")



def render() -> None:
    """Render the home page: uploads, inference, predictions, and gallery."""
    # Upload
    st.file_uploader(
        label="Upload one or more images of skin lesions",
        type=["png", "jpg", "jpeg"],
        key="uploaded_image",
        help="Supported formats: PNG, JPG, JPEG",
        accept_multiple_files=True,
        on_change=handle_uploads,
    )

    # Load all images from storage (single source of truth)
    images = storage.load_images()
    if not images:
        st.info("Upload images to see predictions.")
        return

    # Weights & spec
    active_weights = get_active_weights()
    if not active_weights:
        st.error("No model weights selected. Please select model weights in the sidebar.")
        return
    spec = get_spec(active_weights)
    if not spec.get("exists", False):
        st.error(f"Model weights not found: {spec.get('ckpt', 'Unknown path')}")
        return
    model_path = spec["ckpt"]

    # Run inference
    try:
        image_list = list(images.values())
        filenames = list(images.keys())
        results, processed_images_pil = inference.infer_images(image_list, model_path, filenames)

        # ====== TOP ROW (two columns) ======
        col_pred, col_side = st.columns([0.5, 0.5], gap="large")

        with col_pred:
            st.subheader("Predictions")
            if results:
                for filename, preds in results.items():
                    st.markdown(f"**{filename}**")
                    prediction_bar(preds)
                    st.markdown("---")
            else:
                st.info("No predictions available.")

        with col_side:
            st.subheader("Risk indicator")
            # Legend (we're using orange for now; green/red shown for future)
            st.markdown(
                """
                <div class="dna-legend">
                  <span class="dna-legend-item"><span class="dna-legend-dot" style="background:#2ecc71;"></span>Low</span>
                  <span class="dna-legend-item"><span class="dna-legend-dot" style="background:#f6bb37;"></span>Medium (placeholder)</span>
                  <span class="dna-legend-item"><span class="dna-legend-dot" style="background:#e14e2d;"></span>High</span>
                </div>
                <p style="opacity:.8; font-size:.9rem; margin-top:.5rem;">
                  For now all images show <strong>orange (Medium)</strong>. We'll plug in the real risk score later.
                </p>
                """,
                unsafe_allow_html=True,
            )
            # Optional: show active weights/spec
            st.caption(f"Active weights: `{active_weights}`")
            if spec.get("ckpt"):
                st.caption(f"Checkpoint: `{spec['ckpt']}`")

        # ====== FULL-WIDTH GALLERY (under the columns) ======
        st.markdown("### Uploaded images")
        
        if processed_images_pil:
            cards_html = []
            LOW_RISK_COLOR = "#2ecc71"
            MEDIUM_RISK_COLOR = "#f6bb37"
            HIGH_RISK_COLOR = "#e14e2d"

            # keep order aligned to filenames
            for filename in filenames:
                img = processed_images_pil.get(filename)
                if img is None:
                    continue
                b64 = _pil_to_base64(img)
                risk_factor = statistics.risk_factor(results[filename])
                traffic_light_color = (
                    LOW_RISK_COLOR if risk_factor == "Low" else
                    MEDIUM_RISK_COLOR if risk_factor == "Medium" else
                    HIGH_RISK_COLOR if risk_factor == "High" else
                    "#cccccc"  # default/unknown
                )
                cards_html.append(
                    f"""
                    <div class="dna-card">
                        <span class="dna-dot" style="background:{traffic_light_color};"></span>
                        <a href='#' id=image-{filename}><img src="data:image/png;base64,{b64}" alt="{filename}"/></a>
                        <div class="dna-caption">{filename}</div>
                    </div>
                    """
                )

            gallery_html = f'<div class="dna-gallery">{"".join(cards_html)}</div>'
            st.html(gallery_html)

        else:
            st.info("No images to display in the gallery.")

    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
