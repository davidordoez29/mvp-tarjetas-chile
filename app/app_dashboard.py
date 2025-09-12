# app/app_dashboard.py
import os
import json
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG & CONSTANTS
# =========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

# Ubicación por defecto del bundle exportado desde notebooks
DEFAULT_BUNDLE_DIR = os.environ.get("BUNDLE_DIR", "").strip() or "out/dashboard_bundle"

# =========================
# HELPERS (formato & carga)
# =========================
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            # intento con ; por si Excel
            try:
                return pd.read_csv(path, sep=";")
            except Exception:
                return None
    return None

def fmt_money(x, moneda="CLP", usdclp=900.0):
    try:
        v = float(x)
    except:
        return "-"
    if moneda.upper() == "USD":
        v = v / float(usdclp if usdclp else 900.0)
        sym = "USD"
    else:
        sym = "CLP"
    # miles con punto, decimales con coma
    return f"{sym} {v:,.2f}".replace(",", "").replace(".", ",").replace("", ".")

def fmt_pct(x):
    try:
        v = float(x)
    except:
        return "-"
    # si viene 0.12 => 12.00%
    if v <= 1.0:
        v = v * 100.0
    return f"{v:,.2f}%".replace(",", "").replace(".", ",").replace("", ".")

def kpi_row(title, v_actual, v_opt, moneda, usdclp, help_txt=None):
    var_pct = None
    if v_actual is not None and v_opt is not None:
        try:
            base = float(v_actual)
            new  = float(v_opt)
            var_pct = ((new - base) / base * 100.0) if base != 0 else np.nan
        except:
            var_pct = np.nan

    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        st.metric(f"{title} (Actual)", fmt_money(v_actual, moneda, usdclp), help=help_txt)
    with c2:
        st.metric(f"{title} (Optimizado)", fmt_money(v_opt, moneda, usdclp))
    with c3:
        st.metric("Δ VAR (%)", fmt_pct(var_pct) if pd.notna(var_pct) else "-")

def show_table_formatted(df, moneda, usdclp, caption=""):
    if df is None or df.empty:
        st.info("Sin datos para mostrar.")
        return
    d = df.copy()

    # Detecta columnas de porcentaje vs. montos (heurística por nombre)
    pct_keys = ["tasa", "rate", "apr", "pd", "lgd", "pct", "porc", "r_opt", "pdw"]
    def is_pct_col(name):
        n = name.lower()
        return any(k in n for k in pct_keys)

    for col in d.columns:
        if pd.api.types.is_numeric_dtype(d[col]):
            if is_pct_col(col):
                d[col] = d[col].apply(fmt_pct)
            else:
                d[col] = d[col].apply(lambda x: fmt_money(x, moneda, usdclp))

    st.dataframe(d, use_container_width=True)
    if caption:
        st.caption(caption)

# =========================
# SIDEBAR – Control de moneda y bundle
# =========================
with st.sidebar:
    st.header("⚙️ Configuración")
    bundle_dir = st.text_input("Carpeta de datos (bundle)", DEFAULT_BUNDLE_DIR)
    moneda = st.radio("Moneda", ["CLP", "USD"], horizontal=True, index=0)
    usdclp = st.number_input("USD↔CLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0)
    st.caption("Todos los montos se convertirán al vuelo según esta tasa.")

# =========================
# Cargar CSVs (robusto a nombres)
# =========================
# Arista 1 (Default)
p_def_port = os.path.join(bundle_dir, "default_portfolio.csv")
p_def_seg  = os.path.join(bundle_dir, "default_segment.csv")
p_def_det  = os.path.join(bundle_dir, "default_detail.csv")

# Arista 2 (Yield)
p_y_port = os.path.join(bundle_dir, "yield_portfolio.csv")
p_y_seg  = os.path.join(bundle_dir, "yield_segment.csv")
p_y_det  = os.path.join(bundle_dir, "yield_detail.csv")
p_y_curv = os.path.join(bundle_dir, "yield_curve_segment.csv")

# Arista 3 (Incentivos)
p_inc_det  = os.path.join(bundle_dir, "incentives_detail.csv")
p_inc_sum  = os.path.join(bundle_dir, "incentives_diag_summary.csv")
p_inc_sens = os.path.join(bundle_dir, "incentives_sensitivity.csv")

# Arista 4 (Capital) – nombres corregidos
p_cap_port = os.path.join(bundle_dir, "capital_portfolio.csv")  # antes: capital_portafolio.csv (mal)
p_cap_seg  = os.path.join(bundle_dir, "capital_segment.csv")
p_cap_det  = os.path.join(bundle_dir, "capital_detail.csv")

# Metadatos/defs (opcionales)
p_kpi_defs = os.path.join(bundle_dir, "kpi_defs.json")
p_seg_defs = os.path.join(bundle_dir, "segment_defs.json")
p_meta     = os.path.join(bundle_dir, "bundle_meta.json")

def safe_read_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

dfs = dict(
    def_port = load_csv(p_def_port),
    def_seg  = load_csv(p_def_seg),
    def_det  = load_csv(p_def_det),
    y_port   = load_csv(p_y_port),
    y_seg    = load_csv(p_y_seg),
    y_det    = load_csv(p_y_det),
    y_curve  = load_csv(p_y_curv),
    inc_det  = load_csv(p_inc_det),
    inc_sum  = load_csv(p_inc_sum),
    inc_sens = load_csv(p_inc_sens),
    cap_port = load_csv(p_cap_port),
    cap_seg  = load_csv(p_cap_seg),
    cap_det  = load_csv(p_cap_det),
)

kpi_defs = safe_read_json(p_kpi_defs)
seg_defs = safe_read_json(p_seg_defs)
meta     = safe_read_json(p_meta)

# =========================
# HEADER
# =========================
st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Motor de decisión que integra riesgo (Default), Yield/Pricing, Incentivos y Capital/Provisiones. "
           "Esta demo compara tu baseline vs. la propuesta optimizada.")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "Arista 1 • Default/Impago",
    "Arista 2 • Yield/Pricing",
    "Arista 3 • Incentivos",
    "Arista 4 • Capital/Provisiones"
])

# -------------------------
# TAB 1: DEFAULT / IMPAGO
# -------------------------
with tab1:
    st.subheader("Arista 1 — Default / Impago")
    st.write("Modelos PD/LGD/EAD → Expected Loss (EL). Comparamos el portafolio actual vs. optimizado.")

    port = dfs["def_port"]
    seg  = dfs["def_seg"]
    det  = dfs["def_det"]

    if port is None:
        st.error("Falta default_portfolio.csv")
    else:
        # Esperados: EAD, EL, Ingreso, Costos, Utilidad (actual/opt)
        cols_need = [
            "EAD_actual","EAD_optimizado",
            "EL_actual","EL_optimizado",
            "Ingreso_actual","Ingreso_optimizado",
            "Costos_actual","Costos_optimizado",
            "Utilidad_actual","Utilidad_optimizada"
        ]
        for c in cols_need:
            if c not in port.columns:
                port[c] = 0.0

        c_kpi = st.container()
        with c_kpi:
            st.markdown("*KPIs del Portafolio (Actual vs. Optimizado)*")
            kpi_row("EAD total", port["EAD_actual"].sum(), port["EAD_optimizado"].sum(), moneda, usdclp, "Exposición")
            kpi_row("Expected Loss", port["EL_actual"].sum(), port["EL_optimizado"].sum(), moneda, usdclp, "Pérdida esperada")
            kpi_row("Ingreso Financiero", port["Ingreso_actual"].sum(), port["Ingreso_optimizado"].sum(), moneda, usdclp)
            kpi_row("Costos Totales", port["Costos_actual"].sum(), port["Costos_optimizado"].sum(), moneda, usdclp)
            kpi_row("Utilidad", port["Utilidad_actual"].sum(), port["Utilidad_optimizada"].sum(), moneda, usdclp)

        st.markdown("*Detalle por segmento*")
        show_table_formatted(seg, moneda, usdclp, "Resumen por segmento")
        with st.expander("Detalle por cliente (muestra)"):
            show_table_formatted(det.head(200) if det is not None else det, moneda, usdclp)

# -------------------------
# TAB 2: YIELD / PRICING
# -------------------------
with tab2:
    st.subheader("Arista 2 — Yield / Pricing")
    st.write("Comparamos ingreso/utilidad total y efecto aislado de pricing. Curvas por segmento con bandas de r.")

    y_port = dfs["y_port"]
    y_seg  = dfs["y_seg"]
    y_det  = dfs["y_det"]
    y_curv = dfs["y_curve"]

    if y_port is None:
        st.error("Falta yield_portfolio.csv")
    else:
        # Safeguard columnas típicas
        for c in ["ingreso_base","ingreso_iso","ingreso_opt","utilidad_base","utilidad_iso","utilidad_opt"]:
            if c not in y_port.columns:
                y_port[c] = 0.0

        st.markdown("*KPIs del Portafolio (Pricing)*")
        kpi_row("Ingreso (Total)",   y_port["ingreso_base"].sum(),  y_port["ingreso_opt"].sum(), moneda, usdclp)
        kpi_row("Ingreso (Solo Pricing)", y_port["ingreso_base"].sum(),  y_port["ingreso_iso"].sum(), moneda, usdclp)
        kpi_row("Utilidad (Total)",  y_port["utilidad_base"].sum(), y_port["utilidad_opt"].sum(), moneda, usdclp)
        kpi_row("Utilidad (Solo Pricing)", y_port["utilidad_base"].sum(), y_port["utilidad_iso"].sum(), moneda, usdclp)

        st.markdown("*Curvas por segmento*")
        show_table_formatted(y_curv, moneda, usdclp, "Útil para ver sensibilidad de r dentro de bandas")
        with st.expander("Detalle por cliente (muestra)"):
            show_table_formatted(y_det.head(200) if y_det is not None else y_det, moneda, usdclp)

# -------------------------
# TAB 3: INCENTIVOS
# -------------------------
with tab3:
    st.subheader("Arista 3 — Incentivos")
    st.write("Motor de asignación de beneficios (bonos, cuotas, cashback, etc.) buscando ROI positivo y control de EL.")

    det  = dfs["inc_det"]
    summ = dfs["inc_sum"]
    sens = dfs["inc_sens"]

    if det is None:
        st.error("Falta incentives_detail.csv")
    else:
        d = det.copy()
        # Asegura numéricos donde aplique (sin romper ids)
        for col in d.columns:
            if pd.api.types.is_object_dtype(d[col]):
                try:
                    d[col] = pd.to_numeric(d[col], errors="ignore")
                except:
                    pass

        # --- COSTO (monto): preferir columna de monto; si no existe, usar tasa * EAD (e_opt o ead_baseline)
        cost_amount_col = next((c for c in [
            "inc_cost","costo_incentivo","costo_incentivo_total","costo_beneficios","costo_incentivos"
        ] if c in d.columns), None)

        if cost_amount_col is None or pd.to_numeric(d.get(cost_amount_col, 0), errors="coerce").fillna(0).sum() == 0:
            rate_col = next((c for c in [
                "costo_incentivo_tasa","inc_tasa","tasa_incentivo","incentivo_tasa"
            ] if c in d.columns), None)
            ead_col = "e_opt" if "e_opt" in d.columns else ("ead_baseline" if "ead_baseline" in d.columns else None)
            if rate_col and ead_col:
                d["_inc_cost_"] = pd.to_numeric(d[rate_col], errors="coerce").fillna(0.0) * \
                                    pd.to_numeric(d[ead_col],  errors="coerce").fillna(0.0)
                cost_amount_col = "_inc_cost_"

        total_cost = pd.to_numeric(d.get(cost_amount_col, 0), errors="coerce").fillna(0.0).sum()

        # --- INCREMENTAL (monto): preferir utilidad; luego ingreso; si no, derivar (opt - base)
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

        uplift_val = pd.to_numeric(d.get(uplift_col, 0), errors="coerce").fillna(0.0).sum()
        roi = (uplift_val / total_cost) if (total_cost and total_cost != 0) else np.nan

        st.markdown("*KPIs del Programa*")
        kpi_row("Costo de Incentivos", total_cost, total_cost, moneda, usdclp, "Suma de costos")
        kpi_row("Incremental (Ingreso/Utilidad)", uplift_val, uplift_val, moneda, usdclp, "Suma de incrementos")
        st.metric("ROI (Incremental/Costo)", fmt_pct(roi*100) if pd.notna(roi) else "-")

        st.markdown("*Detalle / Diagnóstico*")
        show_table_formatted(d, moneda, usdclp, "Detalle por cliente/campaña")
        with st.expander("Resumen & Sensibilidades"):
            show_table_formatted(summ, moneda, usdclp, "Resumen de diagnóstico")
            show_table_formatted(sens, moneda, usdclp, "Sensibilidades")

# -------------------------
# TAB 4: CAPITAL / PROVISIONES
# -------------------------
with tab4:
    st.subheader("Arista 4 — Capital / Provisiones")
    st.write("Proxies IFRS9/Basilea para capital requerido y provisiones. "
             "Mostramos liberación/consumo al pasar de baseline a optimizado.")

    cap_port = dfs["cap_port"]
    cap_seg  = dfs["cap_seg"]
    cap_det  = dfs["cap_det"]

    if cap_port is None:
        st.error("Falta capital_portfolio.csv (ojo: nombre en inglés).")
    else:
        # Columnas típicas (tolerante a ausentes)
        for c in ["capital_req_base","capital_req_opt","prov_base","prov_opt"]:
            if c not in cap_port.columns:
                cap_port[c] = 0.0

        cap_lib_monto = (cap_port.get("capital_req_base", pd.Series([0])).sum() -
                         cap_port.get("capital_req_opt",  pd.Series([0])).sum())
        prov_lib_monto = (cap_port.get("prov_base", pd.Series([0])).sum() -
                          cap_port.get("prov_opt",  pd.Series([0])).sum())

        st.markdown("*KPIs de Capital/Provisiones*")
        kpi_row("Capital requerido", cap_port["capital_req_base"].sum(), cap_port["capital_req_opt"].sum(), moneda, usdclp)
        kpi_row("Provisiones",       cap_port["prov_base"].sum(),       cap_port["prov_opt"].sum(),       moneda, usdclp)
        kpi_row("Liberación de Capital (Δ)", cap_lib_monto, cap_lib_monto, moneda, usdclp)
        kpi_row("Liberación de Provisiones (Δ)", prov_lib_monto, prov_lib_monto, moneda, usdclp)

        st.markdown("*Detalle por segmento*")
        show_table_formatted(cap_seg, moneda, usdclp, "Resumen de capital/provisiones por segmento")

        with st.expander("Detalle por cliente (muestra)"):
            show_table_formatted(cap_det.head(200) if cap_det is not None else cap_det, moneda, usdclp)

# =========================
# FOOTER (ayuda ejecutiva)
# =========================
st.divider()
st.markdown("""
*¿Por qué nuestro modelo?* Integramos 4 aristas en un único motor:
- *Default/Impago*: control de pérdidas esperadas (PD·LGD·EAD) con bandas y restricciones de negocio.
- *Yield/Pricing*: precio por cliente ajustado a riesgo, con curvas y efecto aislado de pricing.
- *Incentivos*: asignación con ROI positivo (reconstruimos costos/incrementos si el input viene en tasa).
- *Capital/Provisiones*: proxies IFRS9/Basilea para liberar capital sin deteriorar EL ni ROE.

Con esto, el banco puede *subir ROE, **reducir EL* y *optimizar consumo de capital* con transparencia y reglas auditables.
""")
