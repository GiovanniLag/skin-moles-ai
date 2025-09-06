from pathlib import Path
import sys
import streamlit as st
from streamlit_option_menu import option_menu

# Ensure repository root is on the path so `src` can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
STREAMLIT_DIR = Path(__file__).resolve().parent

from src.ui import home_page, training_page, about_page # noqa: E402
from src.services.weighs_registry import set_active_weights, _REGISTRY # noqa: E402
from src.utils.theme import load_theme_config, transparency # noqa: E402
from src.utils.streamlit import load_css # noqa: E402
from src.components.popups import notify # noqa: E402

load_css(str(STREAMLIT_DIR / "assets" / "styles.css")) # Load custom CSS for styling
st.set_page_config(page_title="DermaNet", page_icon="🩺", layout="wide")

PAGES = {
    "Home": home_page.render,
    "Training": training_page.render,
    "About": about_page.render,
}

THEME = load_theme_config()
secbg = THEME.get("secondaryBackgroundColor", "#FBF0D8")
acc2 = THEME.get("accent2Color", "#f6bb37")
    

def _init_sidebar() -> None:
    """Initialise the sidebar with branding and weight selection.

    The selected weight is stored in ``st.session_state['weights']`` so that
    subsequent page reloads retain the choice. When the user selects a
    different weight the corresponding entry in the registry is updated.
    """
    st.sidebar.title("DermaNet")
    # --- Page navigation (placed above weights, like the screenshot) ---
    page_names = list(PAGES.keys())
    default_page = st.session_state.get("page", page_names[0])
    selected = default_page
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=page_names,
            icons=["house", "journal-bookmark", "info-circle"],
            menu_icon="cast",
            default_index=page_names.index(default_page),
            orientation="vertical",
            styles={
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": transparency(acc2, 0.2)},
                "nav-link-selected": {"background-color": acc2, "color": "black"},
                "container": {"background-color": secbg},
            },
        )

    # Persist selection
    if st.session_state.get("page") != selected:
        st.session_state["page"] = selected



    # Model weights selection at the bottom of the sidebar
    st.sidebar.markdown("<div id=weights-sel-sep-bar><hr></div>", unsafe_allow_html=True)

    # Use tkinter filedialog.askopenfilename directly to request a path.
    # We only import tkinter when the button is clicked to avoid import-time
    # issues on headless servers. If no file is selected, fall back to the
    # default checkpoint (below).
    if st.sidebar.button("Select model weights 📂", key="select-weights"):
        try:
            from tkinter import filedialog
            # filedialog may require a root on some platforms, but many TK
            # installations allow calling askopenfilename directly after import.
            file_path = filedialog.askopenfilename(title="Select model weights", filetypes=[("Checkpoint files", "*.ckpt *.pt"), ("All files", "*.*")])
            if file_path:
                name = f"{Path(file_path).name}"
                _REGISTRY[name] = {"ckpt": file_path, "img_size": 224}
                set_active_weights(name)
                st.session_state["weights"] = name
                notify(f"Selected weights: {Path(file_path).name}", "success", duration=4)
            else:
                notify("No file selected; using default weights.", "info", duration=4)
        except Exception as exc:
            # If tkinter isn't present or askopenfilename fails, notify and
            # continue to use default weights.
            notify(f"Could not open native file dialog: {exc}; using default weights.", "error", duration=4)

    # If no selection was made, ensure default checkpoint is used
    if st.session_state.get("weights") is None:
        default_name = "DermaNet-default"
        default_ckpt = str(f"{ROOT_DIR}/outputs/dermanet/training/version_0/ckpts/best.ckpt")
        _REGISTRY[default_name] = {"ckpt": default_ckpt, "img_size": 224}
        set_active_weights(default_name)
        st.session_state["weights"] = default_name
    
    # Display current selection
    active_weights = st.session_state.get("weights")
    st.sidebar.markdown(f"<div style='font-size:14px;'><strong>Active weights:</strong> {active_weights}</div>", unsafe_allow_html=True)

def main() -> None:
    _init_sidebar()

    # Dispatch to the currently selected page from session state
    current = st.session_state.get("page", list(PAGES.keys())[0])
    render_fn = PAGES.get(current, PAGES[list(PAGES.keys())[0]])
    render_fn()


if __name__ == "__main__":
    main()
