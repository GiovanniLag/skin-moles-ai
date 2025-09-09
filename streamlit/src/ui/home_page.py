import io
import base64
from PIL import Image
import streamlit as st
from urllib.parse import quote_plus
from pathlib import Path
from st_click_detector import click_detector

from ..components import prediction_bar
from ..services import inference, storage, statistics
from ..services.weighs_registry import _REGISTRY, get_active_weights, get_spec, set_active_weights
from ..services.gradcam import generate_gradcam  # Import Grad-CAM service

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

_LAST_CLICKED = ""  # Keep track of the last clicked image to avoid repeated actions

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
        try: # try to set default weights if available
            default_ckpt_path = str(f"{ROOT_DIR}/streamlit/weights/DermaNet_default.ckpt")
            _REGISTRY["DermaNet-default"] = {"ckpt": default_ckpt_path, "img_size": 224}
            set_active_weights("DermaNet-default")
            st.session_state["weights"] = "DermaNet-default"
            active_weights = get_active_weights()
        except Exception:
            st.error("No model weights selected. Please select model weights in the sidebar.")
            return
    spec = get_spec(active_weights)
    if not spec.get("exists", False):
        st.error(f"Model weights not found: {spec.get('ckpt', 'Unknown path')}")
        return
    model_path = spec["ckpt"]

    # Sidebar controls for Grad-CAM
    with st.sidebar:
        # Add an empty spacer
        st.markdown("<br>", unsafe_allow_html=True)

        # Grad-CAM controls
        st.markdown("### 🔥 Grad-CAM")
        enable_gradcam = st.checkbox("Show Grad-CAM overlay", value=True)
        gradcam_alpha = st.slider("Overlay alpha", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
        target_mode = st.radio("Target class", ["Predicted", "Specific"], horizontal=True)
        target_class_input = None
        if target_mode == "Specific":
            target_class_input = st.number_input("Class index", min_value=0, step=1, value=0, help="Index of class for Grad-CAM (0-based)")

    try:
        image_list = list(images.values())
        filenames = list(images.keys())
        results, processed_images_pil = inference.infer_images(image_list, model_path, filenames)

        # ====== TOP ROW (two columns) ======
        col_side, col_pred = st.columns([0.4, 0.6], gap="large")

        # Set frist image as selected by default if none selected yet and images exist
        if "selected_image" not in st.session_state and filenames:
            st.session_state["selected_image"] = filenames[0]

        with col_side:
            selected_image_name = st.session_state.get("selected_image")
            if selected_image_name and selected_image_name in processed_images_pil:
                st.subheader("Selected image")
                orig_img = processed_images_pil[selected_image_name]

                # Display original + Grad-CAM if enabled
                if enable_gradcam:
                    with st.spinner("Computing Grad-CAM..."):
                        try:
                            # Determine target class if user specified one
                            target_class = None if target_mode == "Predicted" else int(target_class_input)
                            gc_res = generate_gradcam(images[selected_image_name], model_path=model_path, target_class=target_class, alpha=gradcam_alpha)
                            tab_orig, tab_overlay, tab_heat = st.tabs(["Original", "Overlay", "Heatmap"])
                            with tab_orig:
                                st.image(orig_img, caption=selected_image_name, width='stretch')
                            with tab_overlay:
                                st.image(gc_res.overlay, caption=f"Overlay (class {gc_res.target_class}, score {gc_res.target_score:.3f})", width='stretch')
                            with tab_heat:
                                st.image(gc_res.heatmap, caption="Grad-CAM heatmap", width='stretch')
                        except Exception as cam_e:
                            st.warning(f"Grad-CAM failed: {cam_e}")
                            st.image(orig_img, caption=selected_image_name, width='stretch')
                else:
                    st.image(orig_img, caption=selected_image_name, width='stretch')
                _LAST_CLICKED = f"image-{selected_image_name}"
            else:
                st.info("Click an image in the gallery below to see it here.")

        with col_pred:
            st.subheader("Predictions")
            selected_image_name = st.session_state.get("selected_image")
            if selected_image_name and selected_image_name in results:
                preds = results[selected_image_name]
                binary_scores = statistics.compute_malignancy_index(preds)
                malignant_score = binary_scores["malignant"]
                benign_score = binary_scores["benign"]
                simplified_preds = {"Benign": benign_score, "Malignant": malignant_score}
                prediction_bar(simplified_preds, use_custom_colors=True)
                with st.expander("🔍 Show detailed classification"):
                    st.markdown("**Detailed classification results:**")
                    prediction_bar(preds)
                    st.caption("*Individual class probabilities from the model*")
            else:
                st.info("Select an image from the gallery below to see its predictions.")

        # ====== FULL-WIDTH GALLERY (under the columns) ======
        st.markdown("### Uploaded images")
        st.markdown(
            """
                <div class="dna-legend">
                  <span class="dna-legend-item">Risk levels:</span>
                  <span class="dna-legend-item"><span class="dna-legend-dot" style="background:#2ecc71;"></span>Low</span>
                  <span class="dna-legend-item"><span class="dna-legend-dot" style="background:#f6bb37;"></span>Medium</span>
                  <span class="dna-legend-item"><span class="dna-legend-dot" style="background:#e14e2d;"></span>High</span>
                </div>
                """,
            unsafe_allow_html=True,
        )

        if processed_images_pil:
            cards_html = []
            LOW_RISK_COLOR = "#2ecc71"
            MEDIUM_RISK_COLOR = "#f6bb37"
            HIGH_RISK_COLOR = "#e14e2d"
            for idx, filename in enumerate(filenames):
                img = processed_images_pil.get(filename)
                if img is None:
                    continue
                b64 = _pil_to_base64(img)
                risk_factor = statistics.risk_factor(results[filename])
                traffic_light_color = (
                    LOW_RISK_COLOR if risk_factor == "Low" else
                    MEDIUM_RISK_COLOR if risk_factor == "Medium" else
                    HIGH_RISK_COLOR if risk_factor == "High" else
                    "#cccccc"
                )
                safe_name = quote_plus(filename)
                cards_html.append(
                    f"""
                    <div class="dna-card">
                        <span class="dna-dot" style="background:{traffic_light_color};"></span>
                        <a href="#" data-filename="{safe_name}" id="image-{filename}">
                            <img src="data:image/png;base64,{b64}" alt="{filename}" />
                        </a>
                        <div class="dna-caption">{filename}</div>
                    </div>
                    """
                )
            gallery_html = f'<div class="dna-gallery">{"".join(cards_html)}</div>'
            css_path = Path(__file__).resolve().parents[2] / "assets" / "styles.css"
            try:
                css_text = css_path.read_text(encoding="utf-8")
            except Exception:
                st.warning("Could not read styles.css for gallery styling.")
            css_inj_html = f"<style>{css_text}</style>"
            clicked = click_detector(css_inj_html + gallery_html)
            if clicked != _LAST_CLICKED and clicked.startswith("image-"):
                clicked_filename = clicked[len("image-") :]
                st.session_state["selected_image"] = clicked_filename
                st.rerun()
        else:
            st.info("No images to display in the gallery.")

    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
