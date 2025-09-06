import streamlit as st


def render() -> None:
    st.title("About")
    st.markdown(
        """
        **DermaNet** is a skin-lesion classification demo built with Streamlit.\
        It uses a pretrained model from the `src.models` package to predict whether\
        an uploaded image is benign or suspicious.
        """
    )
