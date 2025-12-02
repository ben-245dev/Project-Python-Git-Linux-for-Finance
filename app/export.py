import io
import pandas as pd
import streamlit as st

def render_export_button(df: pd.DataFrame, filename: str = "export.csv", label="📥 Télécharger l'historique CSV"):
    csv_buf = io.StringIO()
    df.to_csv(csv_buf)
    st.download_button(
        label=label,
        data=csv_buf.getvalue(),
        file_name=filename,
        mime="text/csv"
    )
