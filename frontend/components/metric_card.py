import streamlit as st
from typing import Optional, Union

def metric_card(
    label: str,
    value: Union[str, int, float],
    icon: str = "",
    delta: Optional[Union[str, int, float]] = None,
) -> None:
    """Render a metric card using st.metric.
    
    Args:
        label: The metric label/title.
        value: The main value to display.
        icon: Optional emoji or icon prefix for the label.
        delta: Optional delta value shown beneath the metric.
    """
    display_label = f"{icon} {label}".strip() if icon else label
    st.metric(label=display_label, value=value, delta=delta)
