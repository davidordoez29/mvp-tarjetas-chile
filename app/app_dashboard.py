# app/app_dashboard.py
import os, json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

# =========================
# AUTODETECCIÓN DEL BUNDLE
# =========================
ENV_BUNDLE = os.environ.get("BUNDLE_DIR", "").strip()
CANDIDATE_DIRS = [
    ENV_BUNDLE,
    "out/dashboard_bundle",
    "./out/dashboard_bundle",
    "../out/dashboard_bundle",
    "/content/mvp-tarjetas-chile/out/dashboard_bundle",
    "/content/out/dashboard_bundle",
    "/workspace/mvp-tarjetas-chile/out/dashboard_bundle",
    os.path.join(os.getcwd(), "out/dashboard_bundle"),
]

def first_existing_path(paths):
    for p in [x for x in paths if x]:
        if os.path.isdir(p):
            return p
    return None

bundle_dir = first_existing_path(CANDIDATE_DIRS) or "out/dashboard_bundle"  # fallback

# ====== ARCHIVOS EXACTOS DEL BUNDLE (según tu lista) ======
REQUIRED_FILES = [
    "bundle_health_report.csv",
    "bundle_health_report.json",
    "bundle_meta.json",
    "capital_detail.csv",
    "capital_portfolio.csv",
    "capital_segment.csv",
    "default_detail.csv",
    "default_portfolio.csv",
    "default_segment.csv",
    "incentives_detail.csv",
    "incentives_diag_summary.csv",
    "incentives_sensitivity.csv",
    "kpi_defs.json",
    "segment_defs.json",
    "yield_curve_segment.csv",
    "yield_detail.csv",
    "yield_portfolio.csv",
    "yield_segment.csv",
]

# =========================
# HELPERS
# =========================
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            try:
                return pd.read_csv(path, sep=";")
            except Exception:
                return None
    return None

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

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
    return f"{sym} {v:,.2f}".replace(",", "").replace(".", ",").replace("", ".")

def fmt_pct(x):
    try:
        v = float(x)
    except:
        return "-"
    if v <= 1.0:
        v = v * 100.0
    return f"{v:,.2f}%".replace(",", "").replace(".", ",").replace("", ".")

def kpi_row(title, v_actual, v_opt, moneda, usdclp, help_txt=None):
    var_pct = None
    if v_actual is not None and v_opt is not None:
        try:
            base = float(v_actual); new = float(v_opt)
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
    pct_keys = ["tasa", "rate", "apr", "pd", "lgd", "pct", "porc", "r_opt", "pdw", "roi"]
    def is_pct_col(name):
        n = str(name).lower()
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
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Configuración")
    st.write("*Rutas probadas (orden):*")
    st.code("\n".join([p or "<vacía>" for p in CANDIDATE_DIRS]), language="bash")
    bundle_dir = st.text_input("Carpeta de datos (bundle)", bundle_dir)
    moneda = st.radio("Moneda", ["CLP", "USD"], horizontal=True, index=0)
    usdclp = st.number_input("USD↔CLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0)
    # Explorador del bundle
    if os.path.isdir(bundle_dir):
        files_in_dir = sorted(os.listdir(bundle_dir))
        st.success(f"Bundle detectado: {bundle_dir}")
        st.caption("Archivos presentes:")
        st.code("\n".join(files_in_dir) if files_in_dir else "(vacío)")
    else:
        st.error(f"No existe la carpeta: {bundle_dir}")

    # Chequeo de faltantes por nombre exacto
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(bundle_dir, f))]
    if missing:
        st.warning("Faltantes (nombre exacto):\n- " + "\n- ".join(missing))

# =========================
# CARGA (usando NOMBRES EXACTOS)
# =========================
# Default
def_port = load_csv(os.path.join(bundle_dir, "default_portfolio.csv"))
def_seg  = load_csv(os.path.join(bundle_dir, "default_segment.csv"))
def_det  = load_csv(os.path.join(bundle_dir, "default_detail.csv"))

# Yield
y_port = load_csv(os.path.join(bundle_dir, "yield_portfolio.csv"))
y_seg  = load_csv(os.path.join(bundle_dir, "yield_segment.csv"))
y_det  = load_csv(os.path.join(bundle_dir, "yield_detail.csv"))
y_curv = load_csv(os.path.join(bundle_dir, "yield_curve_segment.csv"))

# Incentives
inc_det  = load_csv(os.path.join(bundle_dir, "incentives_detail.csv"))
inc_sum  = load_csv(os.path.join(bundle_dir, "incentives_diag_summary.csv"))
inc_sens = load_csv(os.path.join(bundle_dir, "incentives_sensitivity.csv"))

# Capital
cap_port = load_csv(os.path.join(bundle_dir, "capital_portfolio.csv"))
cap_seg  = load_csv(os.path.join(bundle_dir, "capital_segment.csv"))
cap_det  = load_csv(os.path.join(bundle_dir, "capital_detail.csv"))

# Meta y health
kpi_defs = load_json(os.path.join(bundle_dir, "kpi_defs.json"))
seg_defs = load_json(os.path.join(bundle_dir, "segment_defs.json"))
meta     = load_json(os.path.join(bundle_dir, "bundle_meta.json"))
health_csv  = load_csv(os.path.join(bundle_dir, "bundle_health_report.csv"))
health_json = load_json(os.path.join(bundle_dir, "bundle_health_report.json"))

# =========================
# HEADER
# =========================
st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Riesgo (Default), Yield/Pricing, Incentivos y Capital/Provisiones. Baseline vs optimizado, con reglas auditables.")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Arista 1 • Default/Impago",
    "Arista 2 • Yield/Pricing",
    "Arista 3 • Incentivos",
    "Arista 4 • Capital/Provisiones",
    "Diagnóstico Bundle"
])

# -------------------------
# TAB 1: DEFAULT / IMPAGO
# -------------------------
with tab1:
    st.subheader("Arista 1 — Default / Impago")
    st.write("PD·LGD·EAD → Expected Loss. Comparamos actual vs optimizado.")

    if def_port is None:
        st.error("Falta default_portfolio.csv")
    else:
        for c in [
            "EAD_actual","EAD_optimizado",
            "EL_actual","EL_optimizado",
            "Ingreso_actual","Ingreso_optimizado",
            "Costos_actual","Costos_optimizado",
            "Utilidad_actual","Utilidad_optimizada"
        ]:
            if c not in def_port.columns:
                def_port[c] = 0.0

        st.markdown("*KPIs del Portafolio*")
        kpi_row("EAD total", def_port["EAD_actual"].sum(), def_port["EAD_optimizado"].sum(), moneda, usdclp)
        kpi_row("Expected Loss", def_port["EL_actual"].sum(), def_port["EL_optimizado"].sum(), moneda, usdclp)
        kpi_row("Ingreso Financiero", def_port["Ingreso_actual"].sum(), def_port["Ingreso_optimizado"].sum(), moneda, usdclp)
        kpi_row("Costos Totales", def_port["Costos_actual"].sum(), def_port["Costos_optimizado"].sum(), moneda, usdclp)
        kpi_row("Utilidad", def_port["Utilidad_actual"].sum(), def_port["Utilidad_optimizada"].sum(), moneda, usdclp)

        st.markdown("*Detalle por segmento*")
        show_table_formatted(def_seg, moneda, usdclp, "Resumen por segmento")
        with st.expander("Detalle por cliente (muestra)"):
            show_table_formatted(def_det.head(200) if def_det is not None else def_det, moneda, usdclp)

# -------------------------
# TAB 2: YIELD / PRICING
# -------------------------
with tab2:
    st.subheader("Arista 2 — Yield / Pricing")
    st.write("Efecto total y efecto aislado de pricing. Curvas por segmento.")

    if y_port is None:
        st.error("Falta yield_portfolio.csv")
    else:
        for c in ["ingreso_base","ingreso_iso","ingreso_opt","utilidad_base","utilidad_iso","utilidad_opt"]:
            if c not in y_port.columns:
                y_port[c] = 0.0

        st.markdown("*KPIs del Portafolio (Pricing)*")
        kpi_row("Ingreso (Total)",   y_port["ingreso_base"].sum(),  y_port["ingreso_opt"].sum(), moneda, usdclp)
        kpi_row("Ingreso (Solo Pricing)", y_port["ingreso_base"].sum(),  y_port["ingreso_iso"].sum(), moneda, usdclp)
        kpi_row("Utilidad (Total)",  y_port["utilidad_base"].sum(), y_port["utilidad_opt"].sum(), moneda, usdclp)
        kpi_row("Utilidad (Solo Pricing)", y_port["utilidad_base"].sum(), y_port["utilidad_iso"].sum(), moneda, usdclp)

        st.markdown("*Curvas por segmento*")
        show_table_formatted(y_curv, moneda, usdclp, "Sensibilidad dentro de bandas")
        with st.expander("Detalle por cliente (muestra)"):
            show_table_formatted(y_det.head(200) if y_det is not None else y_det, moneda, usdclp)

# -------------------------
# TAB 3: INCENTIVOS
# -------------------------
with tab3:
    st.subheader("Arista 3 — Incentivos")
    st.write("Costo de incentivos vs incremental (ingreso/utilidad). ROI = incremental / costo.")

    if inc_det is None:
        st.error("Falta incentives_detail.csv")
    else:
        d = inc_det.copy()
        for col in d.columns:
            if pd.api.types.is_object_dtype(d[col]):
                try:
                    d[col] = pd.to_numeric(d[col], errors="ignore")
                except:
                    pass

        # Costo (monto)
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

        # Incremental
        uplift_col = next((c for c in [
            "delta_util_total","utilidad_incremental","utilidad_uplift","uplift_utilidad",
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
            show_table_formatted(inc_sum, moneda, usdclp, "Resumen de diagnóstico")
            show_table_formatted(inc_sens, moneda, usdclp, "Sensibilidades")

# -------------------------
# TAB 4: CAPITAL / PROVISIONES
# -------------------------
with tab4:
    st.subheader("Arista 4 — Capital / Provisiones")
    st.write("Proxies IFRS9/Basilea: capital requerido y provisiones; liberaciones por optimización.")

    if cap_port is None:
        st.error("Falta capital_portfolio.csv")
    else:
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
        show_table_formatted(cap_seg, moneda, usdclp, "Resumen por segmento")
        with st.expander("Detalle por cliente (muestra)"):
            show_table_formatted(cap_det.head(200) if cap_det is not None else cap_det, moneda, usdclp)

# -------------------------
# TAB 5: DIAGNÓSTICO BUNDLE
# -------------------------
with tab5:
    st.subheader("Diagnóstico del Bundle")
    st.write("Verifica integridad y consistencia del paquete de datos.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("*bundle_meta.json*")
        st.json(meta if meta else {"info": "bundle_meta.json no encontrado o vacío"})
        st.markdown("*bundle_health_report.json*")
        st.json(health_json if health_json else {"info": "bundle_health_report.json no encontrado o vacío"})

    with c2:
        st.markdown("*bundle_health_report.csv*")
        show_table_formatted(health_csv, moneda, usdclp, "Chequeos de sanidad (si existe)")

    st.markdown("*Archivos esperados vs presentes*")
    present, missing = [], []
    for f in REQUIRED_FILES:
        p = os.path.join(bundle_dir, f)
        (present if os.path.exists(p) else missing).append(f)
    st.success("Presentes:\n- " + "\n- ".join(present) if present else "No se detectó ninguno.")
    if missing:
        st.warning("Faltantes:\n- " + "\n- ".join(missing))
