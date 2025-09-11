import streamlit as st
import pandas as pd

st.set_page_config(page_title="MVP Bancario", layout="wide")
st.title("📊 MVP Bancario – Arista 1 (Default/Impago)")

st.caption("Vista ejecutiva: comparación método actual vs. optimizado, KPIs y definiciones")
st.info("Placeholder de la app. En el siguiente paso conectaremos 'out/default_compare_*.csv'.")

st.subheader("Check rápido de archivos")
import os, glob
files = sorted(glob.glob("out/*.csv"))
if files:
    st.write("Archivos en out/:", files[:10])
else:
    st.warning("No hay archivos en out/. Sube outputs desde Colab (celdas 24.x) para visualizar.")
