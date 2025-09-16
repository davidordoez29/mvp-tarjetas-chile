# app_dashboard.py — Dashboard Bancario Optimización WDOF
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ==============================
# 🔧 Utilidades
# ==============================
def format_number(x, is_pct=False):
    """
    Formatea números en estilo CLP/Latam:
    - Miles con punto
    - Decimales con coma
    - Siempre dos decimales
    - Si is_pct=True → agrega %
    """
    try:
        if pd.isna(x): return "-"
        if is_pct:
            return f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return str(x)

def kpi_box(label, base, opt, unit=""):
    delta = (opt - base) / (abs(base) + 1e-9)
    col1, col2, col3 = st.columns(3)
    col1.metric(label, f"{format_number(base)} {unit}")
    col2.metric("Optimizado", f"{format_number(opt)} {unit}",
                delta=f"{delta*100:.2f}%")
    col3.metric("Variación", f"{format_number(opt-base)} {unit}")
    with st.expander(f"ℹ️ Definición de {label}"):
        if "Default" in label or "Pérdidas" in label:
            st.write("Mide las *pérdidas esperadas*: PD × LGD × EAD. "
                     "Nuestro modelo reduce estas pérdidas ajustando precios "
                     "y políticas, generando menos default para el mismo portafolio.")
        elif "Yield" in label or "Margen" in label:
            st.write("Mide el *margen financiero neto*. "
                     "Optimizado ajustando tasas y spreads para maximizar ingresos sin aumentar riesgos.")
        elif "Incentivos" in label:
            st.write("Compara el costo de incentivos frente al ingreso incremental. "
                     "Nuestro modelo asigna incentivos sólo donde generan ROI positivo.")
        elif "Capital" in label:
            st.write("Mide el capital regulatorio requerido. "
                     "La optimización libera capital manteniendo riesgo controlado, "
                     "permitiendo crecer más con el mismo balance.")

# ==============================
# 📂 Cargar bundle
# ==============================
def load_bundle():
    bundle_dir = Path(st.secrets.get("BUNDLE_DIR", "/content/mvp-tarjetas-chile/out/dashboard_bundle"))
    clients_p = bundle_dir / "dashboard_bundle_clients.csv"
    segs_p    = bundle_dir / "dashboard_bundle_segments.csv"
    if not clients_p.exists() or not segs_p.exists():
        st.error("❌ No se encontró el bundle (CSV). Ejecuta el pipeline previo.")
        return None, None
    return pd.read_csv(clients_p), pd.read_csv(segs_p)

clients, segs = load_bundle()
if clients is None: st.stop()

# ==============================
# ⚙️ Sidebar config
# ==============================
st.sidebar.header("⚙️ Configuración")
currency = st.sidebar.radio("Moneda", ["CLP", "USD"])
usd_to_clp = st.sidebar.number_input("Tipo de cambio USD → CLP", value=900, step=10)

def convert(x):
    if currency == "CLP":
        return x
    else:
        return x / usd_to_clp

# ==============================
# 📊 Dashboard en pestañas
# ==============================
st.title("📊 Dashboard Bancario – Optimización 4 Aristas")
st.caption("Simulación + Optimización de KPIs bancarios. Modelo matemático WDOF.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Arista 1 – Default",
    "Arista 2 – Yield",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital"
])

# ==============================
# 🔹 Arista 1: Default
# ==============================
with tab1:
    st.subheader("📉 Arista 1 — Default (Pérdidas Esperadas)")
    try:
        base = clients["EL_base"].sum()
        opt  = clients["EL_opt"].sum()
        kpi_box("Pérdidas Esperadas", convert(base), convert(opt), unit=currency)
    except Exception as e:
        st.error(f"Error en Arista 1: {e}")

# ==============================
# 🔹 Arista 2: Yield
# ==============================
with tab2:
    st.subheader("💰 Arista 2 — Yield (Margen Financiero)")
    try:
        base = clients["Yield_base"].sum()
        opt  = clients["Yield_opt"].sum()
        kpi_box("Margen Financiero", convert(base), convert(opt), unit=currency)
    except Exception as e:
        st.error(f"Error en Arista 2: {e}")

# ==============================
# 🔹 Arista 3: Incentivos
# ==============================
with tab3:
    st.subheader("🎁 Arista 3 — Incentivos")
    try:
        base = clients["Inc_cost_base"].sum()
        opt  = clients["Inc_cost_opt"].sum()
        kpi_box("Costo Incentivos", convert(base), convert(opt), unit=currency)

        base_b = clients["Inc_benefit_base"].sum()
        opt_b  = clients["Inc_benefit_opt"].sum()
        kpi_box("Beneficio Incentivos", convert(base_b), convert(opt_b), unit=currency)

        roi_base = (base_b - base) / (base+1e-9)
        roi_opt  = (opt_b - opt) / (opt+1e-9)
        kpi_box("ROI Incentivos", roi_base, roi_opt, unit="")
    except Exception as e:
        st.warning("No hay columnas inc_* en bundle, Incentivos no simulados aún.")

# ==============================
# 🔹 Arista 4: Capital
# ==============================
with tab4:
    st.subheader("🏦 Arista 4 — Capital")
    try:
        base = clients["Capital_req_base"].sum()
        opt  = clients["Capital_req_opt"].sum()
        kpi_box("Capital Requerido", convert(base), convert(opt), unit=currency)

        base_lib = clients["Capital_lib_base"].sum()
        opt_lib  = clients["Capital_lib_opt"].sum()
        kpi_box("Capital Liberado", convert(base_lib), convert(opt_lib), unit=currency)
    except Exception as e:
        st.error(f"Error en Arista 4: {e}")
