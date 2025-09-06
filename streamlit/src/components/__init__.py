from typing import Dict
import streamlit as st

PALETTE = ["#36b7d2", "#e76f51", "#f6bb37", "#f3b3b4", "#264653"]


def prediction_bar(preds: Dict[str, float], use_custom_colors: bool = False) -> None:
    """Render probabilities as color-coded bars.
    
    Args:
        preds: Dictionary of label -> probability
        use_custom_colors: If True, uses custom colors for Benign/Malignant
    """
    for idx, (label, prob) in enumerate(preds.items()):
        if use_custom_colors and label.lower() in ["benign", "malignant"]:
            # Use custom styling for simplified view
            bar_class = "benign-bar" if label.lower() == "benign" else "malignant-bar"
            st.markdown(
                f"""
                <div class="prediction-bar-wrapper {bar_class}">
                  <div class="prediction-label">
                    <span>{label}</span>
                    <span class="prediction-percentage">{prob*100:.1f}%</span>
                  </div>
                  <div class="prediction-bar-bg">
                    <div class="prediction-bar-fill" style="width:{prob*100}%"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Use original styling for detailed view
            color = PALETTE[idx % len(PALETTE)]
            st.markdown(
                f"""
                <div class="prediction-bar-wrapper">
                  <div class="prediction-label">
                    <span>{label}</span>
                    <span class="prediction-percentage">{prob*100:.1f}%</span>
                  </div>
                  <div class="prediction-bar-bg">
                    <div class="prediction-bar-fill" style="width:{prob*100}%; --fill-color: {color}"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
