# app_dashboard.py
import os
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MVP Bancario", layout="wide")

# --- Config inicial desde variables de entorno (fallback si faltan) ---
BUNDLE_DIR = os.environ.get("BUNDLE_DIR", "").strip()
moneda_env = (os.environ.get("MONEDA") or "CLP").upper()
usdclp_env = float(os.environ.get("USDCLP") or 900.0)

st.sidebar.header("⚙️ Configuración")
colA, colB = st.sidebar.columns(2)
with colA:
    moneda = st.selectbox("Moneda", ["CLP","USD"], index=0 if moneda_env=="CLP" else 1)
with colB:
    usdclp = st.number_input("USDCLP", min_value=1.0, max_value=100000.0, value=usdclp_env, step=1.0)
bundle_input = st.sidebar.text_input("Ruta bundle", value=BUNDLE_DIR or "")
if bundle_input and os.path.isdir(bundle_input):
    BUNDLE_DIR = bundle_input

st.title("📊 MVP Bancario – 4 Aristas (Baseline vs Optimizado)")
st.caption("Carga automática desde dashboard_bundle; conversión CLP↔USD controlable.")

def to_currency(val, moneda="CLP", usdclp=900.0):
    try:
        v = float(val)
    except:
        return None
    return v if moneda == "CLP" else v / usdclp

def load_csv(name):
    path = os.path.join(BUNDLE_DIR, name)
    return pd.read_csv(path) if (BUNDLE_DIR and os.path.exists(path)) else None

if not BUNDLE_DIR:
    st.warning("No se ha definido BUNDLE_DIR. Genera el bundle (celdas 7–12) o coloca la ruta en la barra lateral.")
else:
    st.info(f"Bundle: {BUNDLE_DIR}")

# Carga de datasets
port_def   = load_csv("default_portfolio.csv")
seg_def    = load_csv("default_segment.csv")
det_def    = load_csv("default_detail.csv")

port_y     = load_csv("yield_portfolio.csv")
seg_y      = load_csv("yield_segment.csv")
curve_y    = load_csv("yield_curve_segment.csv")
det_y      = load_csv("yield_detail.csv")

inc_det    = load_csv("incentives_detail.csv")
inc_sum    = load_csv("incentives_diag_summary.csv")
inc_sens   = load_csv("incentives_sensitivity.csv")

cap_port   = load_csv("capital_compare_portfolio.csv")
cap_seg    = load_csv("capital_compare_segment.csv")
cap_det    = load_csv("capital_compare_detail.csv")

# Tabs para 4 aristas
tab1, tab2, tab3, tab4 = st.tabs(["Arista 1: Default", "Arista 2: Yield", "Arista 3: Incentivos", "Arista 4: Capital"])

# -----------------
# Arista 1 — Default
# -----------------
with tab1:
    st.subheader("Resumen de Portafolio (Default)")
    if port_def is not None:
        kpi_cols = [c for c in port_def.columns if any(k in c.lower() for k in
                   ["ead","el","ingreso","costo","utilidad","reduccion","delta"])]
        kpi = port_def.copy()
        for c in kpi_cols:
            if c in kpi.columns:
                kpi[c] = kpi[c].apply(lambda x: to_currency(x, moneda, usdclp))
        st.dataframe(kpi.T, use_container_width=True)
    else:
        st.warning("default_portfolio.csv no encontrado.")
    st.markdown("*Por segmento*")
    if seg_def is not None:
        show = seg_def.copy()
        money_cols = [c for c in show.columns if any(k in c.lower() for k in
                     ["ead","el","ingreso","costo","utilidad","reduccion"])]
        for c in money_cols: show[c] = show[c].apply(lambda x: to_currency(x, moneda, usdclp))
        st.dataframe(show, use_container_width=True)
    else:
        st.warning("default_segment.csv no encontrado.")
    st.markdown("*Detalle (muestra)*")
    if det_def is not None:
        show = det_def.head(200).copy()
        money_cols = [c for c in show.columns if any(k in c.lower() for k in
                     ["ead","el","ingreso","costo","utilidad"])]
        for c in money_cols: show[c] = show[c].apply(lambda x: to_currency(x, moneda, usdclp))
        st.dataframe(show, use_container_width=True)
    else:
        st.info("default_detail.csv no encontrado.")

# --------------
# Arista 2 — Yield
# --------------
with tab2:
    st.subheader("Resumen de Portafolio (Yield/Pricing)")
    if port_y is not None:
        kpi2 = port_y.copy()
        money_cols = [c for c in kpi2.columns if any(k in c.lower() for k in
                      ["ead","ingreso","utilidad","el","costo","delta","reduccion"])]
        for c in money_cols: kpi2[c] = kpi2[c].apply(lambda x: to_currency(x, moneda, usdclp))
        st.dataframe(kpi2.T, use_container_width=True)
    else:
        st.warning("yield_portfolio.csv no encontrado.")
    colL, colR = st.columns(2)
    with colL:
        st.markdown("*Por segmento*")
        if seg_y is not None:
            show = seg_y.copy()
            money_cols = [c for c in show.columns if any(k in c.lower() for k in
                         ["ead","el","ingreso","costo","utilidad","delta","reduccion"])]
            for c in money_cols: show[c] = show[c].apply(lambda x: to_currency(x, moneda, usdclp))
            st.dataframe(show, use_container_width=True)
        else:
            st.info("yield_segment.csv no encontrado.")
    with colR:
        st.markdown("*Curva de pricing*")
        if curve_y is not None and len(curve_y):
            segs = sorted(curve_y["segmento"].dropna().unique().tolist())
            sel = st.selectbox("Segmento", segs)
            sub = curve_y[curve_y["segmento"]==sel].copy()
            for m in ["EAD_model","Ingreso_model","EL_model","CostFin_model","CostOp_model","Utilidad_model"]:
                if m in sub.columns:
                    sub[m] = sub[m].apply(lambda x: to_currency(x, moneda, usdclp))
            st.line_chart(sub.set_index("r")[["Utilidad_model","Ingreso_model","EL_model"]])
        else:
            st.info("No hay yield_curve_segment.csv o está vacío.")

# --------------------
# Arista 3 — Incentivos
# --------------------
with tab3:
    st.subheader("Impacto de Incentivos")
    if inc_sum is not None:
        show = inc_sum.copy()
        money_cols = [c for c in show.columns if any(k in c.lower() for k in
                     ["ingreso","costo","utilidad","roi","monto"])]
        for c in money_cols: 
            if c in show.columns:
                show[c] = show[c].apply(lambda x: to_currency(x, moneda, usdclp))
        st.dataframe(show, use_container_width=True)
    else:
        st.info("incentives_diag_summary.csv no encontrado.")
    st.markdown("*Detalle por cliente (muestra)*")
    if inc_det is not None:
        show = inc_det.head(200).copy()
        money_cols = [c for c in show.columns if any(k in c.lower() for k in
                     ["ingreso","costo","utilidad","monto"])]
        for c in money_cols: show[c] = show[c].apply(lambda x: to_currency(x, moneda, usdclp))
        st.dataframe(show, use_container_width=True)
    else:
        st.info("incentives_detail.csv no encontrado.")
    st.markdown("*Sensibilidad (si existe)*")
    if inc_sens is not None:
        st.dataframe(inc_sens, use_container_width=True)
    else:
        st.info("incentives_sensitivity.csv no encontrado.")

# ----------------------------
# Arista 4 — Capital/Provisiones
# ----------------------------
with tab4:
    st.subheader("Capital y Provisiones (IFRS9/Basilea – proxy)")
    if cap_port is not None:
        show = cap_port.copy()
        money_cols = [c for c in show.columns if any(k in c.lower() for k in
                     ["capital","prov","liber","monto","utilidad"])]
        for c in money_cols: show[c] = show[c].apply(lambda x: to_currency(x, moneda, usdclp))
        st.dataframe(show.T, use_container_width=True)
    else:
        st.warning("capital_compare_portfolio.csv no encontrado.")
    st.markdown("*Por segmento*")
    if cap_seg is not None:
        show = cap_seg.copy()
        money_cols = [c for c in show.columns if any(k in c.lower() for k in
                     ["capital","prov","liber","monto","utilidad"])]
        for c in money_cols: show[c] = show[c].apply(lambda x: to_currency(x, moneda, usdclp))
        st.dataframe(show, use_container_width=True)
    else:
        st.info("capital_compare_segment.csv no encontrado.")
    st.markdown("*Detalle (muestra)*")
    if cap_det is not None:
        show = cap_det.head(200).copy()
        money_cols = [c for c in show.columns if any(k in c.lower() for k in
                     ["capital","prov","liber","monto","utilidad"])]
        for c in money_cols: show[c] = show[c].apply(lambda x: to_currency(x, moneda, usdclp))
        st.dataframe(show, use_container_width=True)
    else:
        st.info("capital_compare_detail.csv no encontrado.")

st.caption(f"Moneda: {moneda} | USDCLP: {usdclp}")
