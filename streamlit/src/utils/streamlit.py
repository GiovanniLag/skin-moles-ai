import streamlit as st

def load_css(file_path: str) -> None:
    """Load a CSS file and inject it into the Streamlit app."""
    with open(file_path) as f:
        st.html(f"<style>{f.read()}</style>")