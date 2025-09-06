import streamlit as st

def notify(msg: str, level: str = "info", duration: float | str = "short") -> None:
    """Top-right fading toast with fallback for older Streamlit.

    Parameters
    ----------
    msg : str
        The message to display.
    level : str
        The level of the message (success, info, warning, error).
    duration : float | str
        The duration for which to display the message. Can be "short", "long", or a number of seconds.
    """
    icons = {"success": "✅", "info": "ℹ️", "warning": "⚠️", "error": "❌"}
    st.toast(msg, icon=icons.get(level, "ℹ️"), duration=duration)
