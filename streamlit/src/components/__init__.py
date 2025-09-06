from typing import Dict
import streamlit as st

PALETTE = ["#36b7d2", "#e76f51", "#f6bb37", "#f3b3b4", "#264653"]


def prediction_bar(preds: Dict[str, float]) -> None:
    """Render probabilities as color-coded bars."""
    for idx, (label, prob) in enumerate(preds.items()):
        color = PALETTE[idx % len(PALETTE)]
        st.markdown(
            f"""
            <div style='margin-bottom:0.5rem'>
              <div style='display:flex;justify-content:space-between;font-weight:600'>
                <span>{label}</span><span>{prob*100:.1f}%</span>
              </div>
              <div style='background-color:#e0e0e0;border-radius:4px;overflow:hidden'>
                <div style='height:0.5rem;width:{prob*100}%;background-color:{color}'></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
