# app/app_dashboard.py
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Config básica
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

DEFAULT_BUNDLE = "/content/mvp-tarjetas-chile/out/dashboard_bundle"

# Archivos esperados por arista
REQ_FILES = {
    # Arista 1 – Default/Impago
    "def_port": "default_portfolio.csv",
    "def_seg":  "default_segment.csv",
    "def_det":  "default_detail.csv",
    # Arista 2 – Yield
    "yld_port": "yield_portfolio.csv",
    "yld_seg":  "yield_segment.csv",
    "yld_det":  "yield_detail.csv",
    "yld_curve":"yield_curve_segment.csv",
    # Arista 3 – Incentivos
    "inc_det":  "incentives_detail.csv",
    "inc_sum":  "incentives_diag_summary.csv",
    "inc_sens": "incentives_sensitivity.csv",
    # Arista 4 – Capital/Provisiones
    "cap_port": "capital_portfolio.csv",   # nombre correcto
    "cap_seg":  "capital_segment.csv",
    "cap_det":  "capital_detail.csv",
    # Definiciones/meta (opcionales)
    "kpi_defs": "kpi_defs.json",
    "seg_defs": "segment_defs.json",
    "meta":     "bundle_meta.json",
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de I/O y formato
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame | None:
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict | None:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None

def fmt_money(x, moneda="CLP", usdclp=900.0):
    try:
        v = float(x)
    except:
        return "-"
    if moneda.upper() == "USD":
        v = v / float(usdclp)
        sym = "USD"
    else:
        sym = "CLP"
    # miles con punto, 2 decimales
    return f"{sym} {v:,.2f}".replace(",", "").replace(".", ",").replace("", ".")

def fmt_pct(x):
    try:
        v = float(x)
    except:
        return "-"
    return f"{v:,.2f}%".replace(",", "").replace(".", ",").replace("", ".")

def kpi_row(nombre, actual, optim, moneda, usdclp, ayuda=""):
    col1, col2, col3, col4 = st.columns([2, 2.2, 2.2, 1.6])
    with col1:
        st.markdown(f"*{nombre}*")
        if ayuda:
            st.caption(ayuda)
    with col2:
        st.metric("Actual", fmt_money(actual, moneda, usdclp))
    with col3:
        st.metric("Optimizado", fmt_money(optim, moneda, usdclp))
    with col4:
        var = (optim - actual) / actual * 100.0 if (actual is not None and actual != 0) else np.nan
        st.metric("VAR %", fmt_pct(var))

def kpi_row_pct(nombre, actual, optim, ayuda=""):
    col1, col2, col3, col4 = st.columns([2, 2.2, 2.2, 1.6])
    with col1:
        st.markdown(f"*{nombre}*")
        if ayuda:
            st.caption(ayuda)
    with col2:
        st.metric("Actual", fmt_pct(actual * 100 if pd.notna(actual) else np.nan))
    with col3:
        st.metric("Optimizado", fmt_pct(optim * 100 if pd.notna(optim) else np.nan))
    with col4:
        var = (optim - actual) * 100 if (actual is not None) else np.nan
        st.metric("Δ p.p.", fmt_pct(var))

def show_table_formatted(df_raw: pd.DataFrame, moneda: str, usdclp: float, title: str = ""):
    if df_raw is None or df_raw.empty:
        return
    df = df_raw.copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        name = c.lower()
        # Heurística: columnas de tasa → %
        if any(k in name for k in ["tasa","rate","apr","pd","lgd","pct","porc","r_opt","pdw","pd_pond"]):
            df[c] = pd.to_numeric(df[c], errors="coerce") * 100.0
            df[c] = df[c].apply(lambda v: fmt_pct(v))
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].apply(lambda v: fmt_money(v, moneda, usdclp))
    if title:
        st.markdown(f"*{title}*")
    st.dataframe(df, use_container_width=True, height=360)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar (moneda/paths)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")
bundle_dir = st.sidebar.text_input("Carpeta de bundle", value=DEFAULT_BUNDLE)
moneda = st.sidebar.radio("Moneda", options=["CLP","USD"], horizontal=True, index=0)
usdclp = st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0)
st.sidebar.caption("Actualiza el tipo de cambio para ver todo en CLP o USD.")

# Cargar todos los CSV/JSON
paths_abs = {k: os.path.join(bundle_dir, v) for k, v in REQ_FILES.items()}
dfs = {k: load_csv(p) for k, p in paths_abs.items()}
defs_kpi = load_json(paths_abs["kpi_defs"])
defs_seg = load_json(paths_abs["seg_defs"])
meta     = load_json(paths_abs["meta"])

# Info de carga
with st.expander("🔎 Archivos detectados"):
    miss = [k for k,v in dfs.items() if v is None and k not in ("kpi_defs","seg_defs","meta")]
    if miss:
        st.error("Faltan archivos: " + ", ".join(miss))
    else:
        st.success("Todos los CSV disponibles.")
    st.code(json.dumps(paths_abs, indent=2), language="json")

# ─────────────────────────────────────────────────────────────────────────────
# Encabezado
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 MVP Bancario – Motor de Optimización (4 Aristas)")
st.caption("Comparativo Actual vs Optimizado + curvas y detalle. Cambia moneda en el panel izquierdo.")
if meta:
    st.caption(f"Bundle meta: {meta}")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Arista 1 – Default/Impago",
                "Arista 2 – Yield/Pricing",
                "Arista 3 – Incentivos",
                "Arista 4 – Capital/Provisiones"])

# ================
# Arista 1 – Default / Impago
# ================
with tabs[0]:
    st.subheader("Arista 1 – Riesgo de Crédito (Default/Impago)")
    st.markdown("""
*Qué ves aquí:* Expected Loss (EL), EAD, Ingreso y Utilidad antes/después de optimizar límites y precio.<br>
*KPIs clave:* EAD, EL, Ingreso, Costos (fin+op), Utilidad, PD ponderado (EAD).
""", unsafe_allow_html=True)

    port = dfs["def_port"]
    seg  = dfs["def_seg"]
    det  = dfs["def_det"]

    if port is None:
        st.error("No se encontró default_portfolio.csv")
    else:
        p = port.iloc[0].copy()
        # Nombres esperados en Celda 7
        ead_a = p.get("EAD_actual", 0.0);            ead_o = p.get("EAD_optimizado", 0.0)
        el_a  = p.get("EL_actual", 0.0);             el_o  = p.get("EL_optimizado", 0.0)
        ing_a = p.get("Ingreso_actual", 0.0);        ing_o = p.get("Ingreso_optimizado", 0.0)
        cst_a = p.get("Costos_actual", 0.0);         cst_o = p.get("Costos_optimizado", 0.0)
        uti_a = p.get("Utilidad_actual", 0.0);       uti_o = p.get("Utilidad_optimizada", 0.0)
        pdw_a = p.get("PD_pond_actual", np.nan);     pdw_o = p.get("PD_pond_optimizado", np.nan)

        kpi_row("EAD (exposición)", ead_a, ead_o, moneda, usdclp, "Suma de exposiciones")
        kpi_row("Expected Loss (EL)", el_a, el_o, moneda, usdclp, "PD×LGD×EAD")
        kpi_row("Ingreso", ing_a, ing_o, moneda, usdclp, "APR×EAD")
        kpi_row("Costos (fin + op)", cst_a, cst_o, moneda, usdclp)
        kpi_row("Utilidad", uti_a, uti_o, moneda, usdclp)
        kpi_row_pct("PD ponderado (EAD)", pdw_a, pdw_o, "PD promedio ponderado por EAD")

    st.markdown("---")
    colL, colR = st.columns(2)
    with colL:
        if seg is not None:
            st.markdown("*Segmentos (Default/Impago)*")
            show_table_formatted(seg, moneda, usdclp)
    with colR:
        if det is not None:
            with st.expander("Detalle por cliente"):
                show_table_formatted(det.head(2000), moneda, usdclp, "Primeros 2.000")

# ================
# Arista 2 – Yield / Pricing
# ================
with tabs[1]:
    st.subheader("Arista 2 – Yield / Pricing")
    st.markdown("""
*Qué ves aquí:* Efecto del precio óptimo (r_opt) sobre ingreso/utilidad, separando el componente solo pricing.<br>
*KPIs:* Ingreso/Utilidad *base, **solo pricing* (EAD fijo) y *total optimizado* (precio+EAD).
""", unsafe_allow_html=True)

    port = dfs["yld_port"]
    seg  = dfs["yld_seg"]
    det  = dfs["yld_det"]
    curve= dfs["yld_curve"]

    if port is None:
        st.error("No se encontró yield_portfolio.csv")
    else:
        p = port.iloc[0].copy()
        ead_a  = p.get("EAD_actual", 0.0);           ead_o = p.get("EAD_optimizado", 0.0)
        ing_b  = p.get("ingreso_base", 0.0);         ing_iso = p.get("ingreso_iso", 0.0);     ing_o = p.get("ingreso_opt", 0.0)
        uti_b  = p.get("utilidad_base", 0.0);        uti_iso = p.get("utilidad_iso", 0.0);    uti_o = p.get("utilidad_opt", 0.0)
        el_b   = p.get("EL_baseline", 0.0);          el_iso = p.get("el_iso", 0.0);           el_o  = p.get("el_opt", 0.0)

        kpi_row("EAD (exposición)", ead_a, ead_o, moneda, usdclp, "Suma de exposiciones")
        kpi_row("Ingreso (Base → Optimizado)", ing_b, ing_o, moneda, usdclp)
        kpi_row("Ingreso (Solo pricing)", ing_b, ing_iso, moneda, usdclp, "Manteniendo EAD = baseline")
        kpi_row("Utilidad (Base → Optimizado)", uti_b, uti_o, moneda, usdclp)
        kpi_row("Utilidad (Solo pricing)", uti_b, uti_iso, moneda, usdclp)
        kpi_row("Expected Loss", el_b, el_o, moneda, usdclp)

    st.markdown("---")
    colL, colR = st.columns(2)
    with colL:
        if seg is not None:
            st.markdown("*Segmentos (Yield)*")
            show_table_formatted(seg, moneda, usdclp)
    with colR:
        if curve is not None:
            with st.expander("Curvas por segmento (r vs Utilidad/EAD)"):
                show_table_formatted(curve, moneda, usdclp, "Grid")

    if det is not None:
        with st.expander("Detalle por cliente"):
            show_table_formatted(det.head(2000), moneda, usdclp, "Primeros 2.000")

# ================
# Arista 3 – Incentivos (ACTUALIZADO)
# ================
with tabs[2]:
    st.subheader("Arista 3 – Incentivos")
    st.markdown("""
*Qué ves aquí:* costo de incentivos vs. incremento esperado (ingreso/utilidad). *ROI = Incremental / Costo*.<br>
*KPIs:* Costo total, Incremental total y ROI.
""", unsafe_allow_html=True)

    det = dfs["inc_det"]
    summ= dfs["inc_sum"]

    total_cost = np.nan
    uplift_val = np.nan
    roi        = np.nan

    if det is None or det.empty:
        st.error("No se encontró incentives_detail.csv")
    else:
        d = det.copy()
        # Normaliza numéricos donde aplique
        for col in d.columns:
            if pd.api.types.is_object_dtype(d[col]):
                try:
                    d[col] = pd.to_numeric(d[col], errors="ignore")
                except:
                    pass

        # 1) COSTO (monto): busca monto directo; si no, tasa * EAD (prioriza e_opt)
        cost_amount_col = next((c for c in [
            "inc_cost","costo_incentivo","costo_incentivo_total","costo_beneficios","costo_incentivos"
        ] if c in d.columns), None)
        if cost_amount_col is None:
            rate_col = next((c for c in [
                "costo_incentivo_tasa","inc_tasa","tasa_incentivo","incentivo_tasa"
            ] if c in d.columns), None)
            ead_col = "e_opt" if "e_opt" in d.columns else ("ead_baseline" if "ead_baseline" in d.columns else None)
            if rate_col and ead_col:
                d["_inc_cost_"] = pd.to_numeric(d[rate_col], errors="coerce").fillna(0.0) * \
                                    pd.to_numeric(d[ead_col], errors="coerce").fillna(0.0)
                cost_amount_col = "_inc_cost_"
        total_cost = pd.to_numeric(d[cost_amount_col], errors="coerce").fillna(0.0).sum() if cost_amount_col else 0.0

        # 2) INCREMENTAL (monto): prioriza utilidad; si no, ingreso; si no, deriva (opt-base)
        uplift_col = next((c for c in [
            "delta_util_total","utilidad_incremental","utilidad_uplift","uplift_utilidad"
        ] if c in d.columns), None)
        if uplift_col is None:
            uplift_col = next((c for c in [
                "uplift_ingreso","ingreso_incremental","delta_ingreso_total","delta_ingreso","ingreso_uplift"
            ] if c in d.columns), None)
        if uplift_col is None:
            if "utilidad_opt" in d.columns and "utilidad_base" in d.columns:
                d["_uplift_"] = pd.to_numeric(d["utilidad_opt"], errors="coerce").fillna(0.0) - \
                                  pd.to_numeric(d["utilidad_base"], errors="coerce").fillna(0.0)
                uplift_col = "_uplift_"
            elif "ingreso_opt" in d.columns and "ingreso_base" in d.columns:
                d["_uplift_"] = pd.to_numeric(d["ingreso_opt"], errors="coerce").fillna(0.0) - \
                                  pd.to_numeric(d["ingreso_base"], errors="coerce").fillna(0.0)
                uplift_col = "_uplift_"
        uplift_val = pd.to_numeric(d[uplift_col], errors="coerce").fillna(0.0).sum() if uplift_col else 0.0

        roi = (uplift_val / total_cost) if (total_cost and total_cost != 0) else np.nan

        # KPIs
        kpi_row("Costo de Incentivos", total_cost, total_cost, moneda, usdclp, "Suma de costos de beneficios")
        kpi_row("Incremental (Ingreso/Utilidad)", uplift_val, uplift_val, moneda, usdclp, "Suma de incrementos")
        st.metric("ROI (Incremental/Costo)", fmt_pct((roi*100) if pd.notna(roi) else np.nan))

        with st.expander("Detalle de incentivos (formateado)"):
            show_table_formatted(d, moneda, usdclp, "Registros")

    if summ is not None and not summ.empty:
        with st.expander("Diagnóstico (resumen)"):
            show_table_formatted(summ, moneda, usdclp, "Resumen")

# ================
# Arista 4 – Capital / Provisiones
# ================
with tabs[3]:
    st.subheader("Arista 4 – Capital / Provisiones")
    st.markdown("""
*Qué ves aquí:* consumo de capital y provisiones antes/después, y montos liberados por la optimización.<br>
*KPIs:* Capital requerido, Provisiones y montos *liberados*.
""", unsafe_allow_html=True)

    port = dfs["cap_port"]
    seg  = dfs["cap_seg"]
    det  = dfs["cap_det"]

    if port is None:
        st.error("No se encontró capital_portfolio.csv")
    else:
        p = port.iloc[0].copy()
        cap_a = p.get("capital_req_base", 0.0);   cap_o = p.get("capital_req_opt", 0.0)
        prv_a = p.get("prov_base", 0.0);          prv_o = p.get("prov_opt", 0.0)

        kpi_row("Capital Requerido", cap_a, cap_o, moneda, usdclp, "Proxy RW×K×EAD")
        kpi_row("Provisiones", prv_a, prv_o, moneda, usdclp, "≈ Expected Loss (IFRS 9)")

    st.markdown("---")
    colL, colR = st.columns(2)
    with colL:
        if seg is not None:
            st.markdown("*Segmentos (Capital/Provisiones)*")
            show_table_formatted(seg, moneda, usdclp)
    with colR:
        if det is not None:
            with st.expander("Detalle por cliente"):
                show_table_formatted(det.head(2000), moneda, usdclp, "Primeros 2.000")

# ─────────────────────────────────────────────────────────────────────────────
# Nota final
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
---
*¿Por qué nuestro modelo mejora al actual?* Integra *4 aristas* en un solo motor:
1) *Riesgo* (PD·LGD·EAD) para bajar EL,  
2) *Pricing* para subir margen sin disparar mora,  
3) *Incentivos* con ROI positivo (asignación eficiente),  
4) *Capital/Provisiones* para liberar consumo y subir ROE.  
Todo con *reglas configurables*, trazabilidad y métricas listas para supervisión.
""")
