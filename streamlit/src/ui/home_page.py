import streamlit as st

from ..components import prediction_bar
from ..services import inference
from ..services.weighs_registry import get_active_weights, get_spec


def on_file_upload() -> None:
    """Callback function for file uploader to run inference on uploaded images."""
    files = st.session_state.get("uploaded_image")
    
    # Get existing results and processed filenames
    existing_results = st.session_state.get("inference_results", {})
    processed_files = st.session_state.get("processed_files", set())
    processed_images = st.session_state.get("processed_images", {})
    
    if not files:
        # No files uploaded, clear everything
        st.session_state["inference_results"] = {}
        st.session_state["processed_files"] = set()
        st.session_state["processed_images"] = {}
        return
    
    # Check for removed files and clean up
    current_filenames = {file.name for file in files}
    removed_files = processed_files - current_filenames
    
    if removed_files:
        # Remove results for deleted files
        for filename in removed_files:
            existing_results.pop(filename, None)
        processed_files -= removed_files
        st.session_state["inference_results"] = existing_results
        st.session_state["processed_files"] = processed_files
        for filename in removed_files:
            processed_images.pop(filename, None)
        st.session_state["processed_images"] = processed_images
    
    # Get the active model weights
    active_weights = get_active_weights()
    if not active_weights:
        st.error("No model weights selected. Please select model weights in the sidebar.")
        return
    
    spec = get_spec(active_weights)
    if not spec.get("exists", False):
        st.error(f"Model weights not found: {spec.get('ckpt', 'Unknown path')}")
        return
    
    model_path = spec["ckpt"]
    
    # Filter out already processed files
    new_files = [file for file in files if file.name not in processed_files]
    
    if not new_files:
        # All files already processed
        return
    
    # Run inference only on new files
    try:
        new_results, new_processed_images = inference.infer_images(new_files, model_path)
        
        # Merge with existing results
        existing_results.update(new_results)
        st.session_state["inference_results"] = existing_results
        
        # Update processed files set and persist
        processed_files.update(file.name for file in new_files)
        st.session_state["processed_files"] = processed_files

        # Merge new processed images and persist to session state
        for filename, img in new_processed_images.items():
            processed_images[filename] = img
        st.session_state["processed_images"] = processed_images
        
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")


def render() -> None:
    # Upload section
    files = st.file_uploader(
        label="Upload one or more images of skin lesions",
        type=["png", "jpg", "jpeg"],
        key="uploaded_image",
        help="Supported formats: PNG, JPG, JPEG",
        accept_multiple_files=True,
        on_change=on_file_upload
    )

    # Prediction section - 2 column layout
    col1, col2 = st.columns([0.3, 0.7])
    
    with col1:
        st.subheader("Predictions")
        results = st.session_state.get("inference_results", {})
        if results:
            for filename, preds in results.items():
                st.markdown(f"**{filename}**")
                prediction_bar(preds)
                st.markdown("---")
        else:
            st.info("Upload images to see predictions")
    
    with col2:
        st.subheader("Uploaded Images")
        if files:
            for file in files:
                # Safely get the processed image (may not be present if processing failed)
                processed_images = st.session_state.get("processed_images", {})
                image = processed_images.get(file.name)
                if image is not None:
                    # Display at a fixed, smaller width to avoid huge images in the UI
                    st.image(image, caption=file.name, width=360)
        else:
            st.info("No images uploaded yet")

