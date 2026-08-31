import streamlit as st
import pandas as pd
from typing import Optional


def data_table(
    df: pd.DataFrame,
    key: Optional[str] = None,
    selectable: bool = False,
) -> Optional[pd.DataFrame]:
    """Render a styled dataframe using st.dataframe.

    Args:
        df: The DataFrame to display.
        key: Optional Streamlit key for the component (used for selection state).
        selectable: If True, enables row selection mode.

    Returns:
        Selected rows as a DataFrame if selectable=True, else None.
    """
    if df.empty:
        st.info("No records found.")
        return None

    # Format float columns to 2 decimal places for display
    format_dict = {}
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        format_dict[col] = "{:.2f}"

    if selectable:
        event = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key=key,
        )
        if event and hasattr(event, "selection") and event.selection.rows:
            return df.iloc[event.selection.rows]
        return None
    else:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            key=key,
        )
        return None
