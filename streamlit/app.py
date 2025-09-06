from pathlib import Path
import sys
import streamlit as st

# Ensure repository root is on the path so `src` can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.ui import home_page, training_page, about_page  # noqa: E402

st.set_page_config(page_title="DermaNet", page_icon="🩺", layout="wide")

PAGES = {
    "Home": home_page.render,
    "Training": training_page.render,
    "About": about_page.render,
}


def main() -> None:
    with st.sidebar:
        st.title("DermaNet")
        choice = st.radio("Navigate", list(PAGES.keys()))
    PAGES[choice]()


if __name__ == "__main__":
    main()
