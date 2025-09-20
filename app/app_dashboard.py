# app/app_dashboard.py — MVP Bancario (4 Aristas) v2.0
import os, json, math, re, glob
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ==========================
# Config de archivos
# ==========================
REQ_FILES_BASE = {
    # Arista 1
    "a1_port": "default_portfolio{S}.csv",
    "a1_seg":  "default_segment{S}.csv",
    "a1_det":  "default_detail{S}.csv",
    # Arista 2
    "a2_port": "yield_portfolio{S}.csv",
    "a2_seg":  "yield_segment{S}.csv",
    "a2_det":  "yield_detail{S}.csv",
    "a2_cur":  "yield_curve_segment{S}.csv",
    # Arista 3
    "a3_det":  "incentives_detail{S}.csv",
    "a3_sum":  "incentives_diag_summary{S}.csv",
    "a3_sens": "incentives_sensitivity{S}.csv",
    # Arista 4
    "a4_port": "capital_portfolio{S}.csv",
    "a4_seg":  "capital_segment{S}.csv",
    "a4_det":  "capital_detail{S}.csv",
    # Guardrails (catálogo export real)
    "gr_port": "guardrails_portfolio.csv",
    "gr_seg":  "guardrails_segment.csv",
    # Guardrails (evaluación celda 16)
    "gr_eval": "guardrails_eval_portfolio.csv",
}

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    "/content/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
    "/content/out",            # por si apuntan al OUT directo
    "./out",
]

# ==========================
# Formato de números
# ==========================
_num_like = re.compile(r"^-?\d+(\.\d+)?$")

def _to_float_or_nan(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return np.nan
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        s = v.strip().replace("%","").replace(",", ".")
        if _num_like.match(s):
            try: return float(s)
            except: return np.nan
        return np.nan
    return np.nan

def fmt_pct_val(val):
    # acepta números o strings; devuelve “—” si NaN
    if isinstance(val, str):
        s = val.strip()
        if s.endswith("%"):      # ya viene con porcentaje
            return s.replace(".", ",")
        x = _to_float_or_nan(s)
        if np.isnan(x): return "—"
        return f"{x:.2f}%".replace(".", ",")
    x = _to_float_or_nan(val)
    if np.isnan(x): return "—"
    return f"{x:.2f}%".replace(".", ",")

def _to_display_currency(val: float, target: str, usdclp: float) -> float:
    if pd.isna(val): return np.nan
    if target.upper() == "USD":
        return float(val) / float(usdclp) if usdclp else np.nan
    return float(val)

def fmt_money_val(val, moneda: str, usdclp: float):
    # int/float/None/np.nan; strings se devuelven como están
    if isinstance(val, str):
        v = val.strip()
        if v == "" or v.upper() == "N/A": return "—"
        return v
    if val is None or (isinstance(val, float) and math.isnan(val)): return "—"
    x = _to_display_currency(float(val), moneda, usdclp)
    if x is None or (isinstance(x, float) and math.isnan(x)): return "—"
    neg = x < 0; x = abs(x)
    ent = int(x); dec = int(round((x - ent) * 100))
    if dec == 100: ent += 1; dec = 0
    ent_str = f"{ent:,}".replace(",", ".")
    return f"-{ent_str},{dec:02d}" if neg else f"{ent_str},{dec:02d}"

def var_pct(actual, opt):
    a = _to_float_or_nan(actual); o = _to_float_or_nan(opt)
    if np.isnan(a) or a == 0: return None
    return (o - a) / a * 100.0

def kpi_row_money(label: str, actual, opt, moneda: str, usdclp: float, help_text: str = ""):
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        st.metric(label=f"{label} – Actual", value=fmt_money_val(actual, moneda, usdclp))
        if help_text: st.caption(help_text)
    with c2:
        st.metric(label=f"{label} – Optimizado", value=fmt_money_val(opt, moneda, usdclp))
    with c3:
        vp = var_pct(actual, opt)
        st.metric(label="VAR %", value=fmt_pct_val(vp) if vp is not None else "—")

def kpi_row_pct(label: str, actual_pct, opt_pct, help_text: str = ""):
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        st.metric(label=f"{label} – Actual", value=fmt_pct_val(actual_pct))
        if help_text: st.caption(help_text)
    with c2:
        st.metric(label=f"{label} – Optimizado", value=fmt_pct_val(opt_pct))
    with c3:
        vp = var_pct(actual_pct, opt_pct)
        st.metric(label="VAR %", value=fmt_pct_val(vp) if vp is not None else "—")

def format_df_currency(df: pd.DataFrame, cols: list[str], moneda: str, usdclp: float):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda v: fmt_money_val(v, moneda, usdclp))
    return df2

def format_df_pct(df: pd.DataFrame, cols: list[str]):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = df2[c].apply(fmt_pct_val)
    return df2

# ==========================
# Carga bundle
# ==========================
def _dir_ok(d: str, suf: str) -> bool:
    try:
        if not d or not os.path.isdir(d): return False
        hits = 0
        # sólo probamos algunos archivos clave
        for k in ["a1_port","a2_port","a4_port"]:
            fname = REQ_FILES_BASE[k].format(S=suf)
            if os.path.exists(os.path.join(d, fname)):
                hits += 1
        return hits >= 1
    except Exception:
        return False

def autodetect_bundle(suf: str) -> str | None:
    for d in CANDIDATE_DIRS:
        if _dir_ok(d, suf): return d
    return None

def load_csv(bundle_dir: str, base_name: str, suf: str) -> pd.DataFrame | None:
    path = os.path.join(bundle_dir, base_name.format(S=suf))
    if not os.path.exists(path): return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# ==========================
# UI
# ==========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

st.sidebar.title("⚙️ Configuración")

escenario = st.sidebar.radio("Escenario", ["Conservador", "Potenciado"], horizontal=True)
suf = "" if escenario == "Conservador" else "_agresivo"

# Moneda
moneda = st.sidebar.radio("Moneda", ["CLP", "USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))

# Bundle
default_dir = autodetect_bundle(suf)
bundle_dir = st.sidebar.text_input("📦 Ruta del bundle", value=(default_dir or ""), help="Ej: /content/out/dashboard_bundle").strip() or default_dir

st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Modelo matemático con cumplimiento (IFRS9/Basilea), comparando *Actual vs Optimizado* por arista y escenario.")

if not bundle_dir:
    st.error("No encuentro el bundle. Genera el paquete en el notebook y vuelve a cargar.")
    st.stop()

tabs = st.tabs([
    "Arista 1 – Default/Impago",
    "Arista 2 – Yield/Pricing",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital/Provisiones",
    "Guardrails (Resguardos)"
])

# ==============================
# Arista 1 – Default/Impago
# ==============================
with tabs[0]:
    st.header("Arista 1 – Default/Impago")
    st.markdown("*Objetivo:* Mantener EAD total estable, *recomponer mix* hacia segmentos con menor PD para *reducir EL* sin frenar crecimiento.")
    st.markdown("""
*KPIs clave*
- *EAD*: Exposición en riesgo (se mantiene; cambia la composición por segmento).  
- *Pérdida Esperada (EL): PD × LGD × EAD (12m IFRS9); buscamos *↓**.  
- *Interés Devengado Bruto*: APR × EAD (antes “Ingreso”); proxy de margen antes de pérdidas.  
- *Utilidad: Interés Devengado Bruto – EL – (costos si aplicaran). Buscamos *↑**.  
- *PD ponderado: PD promedio ponderado por EAD; buscamos *↓**.
    """)

    a1p = load_csv(bundle_dir, REQ_FILES_BASE["a1_port"], suf)
    if a1p is not None and not a1p.empty:
        def g0(df, c): return df[c].iloc[0] if c in df.columns else np.nan
        # Interés Devengado Bruto
        kpi_row_money("Interés Devengado Bruto",
                      g0(a1p,"Interes_devengado_bruto_actual"),
                      g0(a1p,"Interes_devengado_bruto_optimizado"),
                      moneda, usdclp,
                      "APR×EAD (proxy de margen bruto).")
        # Utilidad
        kpi_row_money("Utilidad",
                      g0(a1p,"Utilidad_actual"),
                      g0(a1p,"Utilidad_optimizada"),
                      moneda, usdclp)
        # EL
        kpi_row_money("Pérdida Esperada (EL)",
                      g0(a1p,"EL_actual"),
                      g0(a1p,"EL_optimizado"),
                      moneda, usdclp)
        # EAD
        kpi_row_money("EAD",
                      g0(a1p,"EAD_actual"),
                      g0(a1p,"EAD_optimizado"),
                      moneda, usdclp)
        # PD ponderado (viene 0–1; mostramos %)
        kpi_row_pct("PD Ponderado",
                    g0(a1p,"PD_pond_actual")*100 if "PD_pond_actual" in a1p.columns else np.nan,
                    g0(a1p,"PD_pond_optimizado")*100 if "PD_pond_optimizado" in a1p.columns else np.nan)

    a1d = load_csv(bundle_dir, REQ_FILES_BASE["a1_det"], suf)
    if a1d is not None and not a1d.empty:
        st.markdown("*Detalle por cliente*")
        money_cols = ["e_base","e_final","income_base","income_final","util_base","util_final","EL_base","EL_final"]
        pct_cols = ["pd_base","pd_final","lgd_base","lgd_final"]
        df_fmt = format_df_currency(a1d, money_cols, moneda, usdclp)
        df_fmt = format_df_pct(df_fmt, pct_cols)
        st.dataframe(df_fmt, use_container_width=True, height=360)

# ==============================
# Arista 2 – Yield/Pricing
# ==============================
with tabs[1]:
    st.header("Arista 2 – Yield/Pricing")
    st.markdown("*Objetivo:* Encontrar el *APR óptimo por segmento* que maximiza *Utilidad* equilibrando *precio vs volumen* (elasticidades).")
    st.markdown("""
*KPIs clave*
- *Utilidad Total: Interés Bruto – EL – COF. Objetivo *↑**.  
- *Interés Bruto Total*: APR × EAD_out.  
- *EAD_in / EAD_out*: Volumen antes/después de aplicar APR óptimo (impacto de elasticidad).  
- *APR óptimo (promedio)* por segmento: dentro de bandas y caps regulados.
    """)

    a2p = load_csv(bundle_dir, REQ_FILES_BASE["a2_port"], suf)
    a2s = load_csv(bundle_dir, REQ_FILES_BASE["a2_seg"], suf)
    if a2p is not None and not a2p.empty:
        def g0(df, c): return df[c].iloc[0] if c in df.columns else np.nan
        kpi_row_money("Utilidad Total",
                      g0(a2p,"utilidad_base"),
                      g0(a2p,"utilidad_opt"),
                      moneda, usdclp)
        kpi_row_money("Interés Bruto Total",
                      g0(a2p,"ingreso_base"),
                      g0(a2p,"ingreso_opt"),
                      moneda, usdclp)
        kpi_row_money("EAD (in → out)",
                      g0(a2p,"EAD_in"),
                      g0(a2p,"EAD_out"),
                      moneda, usdclp,
                      "Volumen afectado por elasticidad de demanda vs APR.")
    if a2s is not None and not a2s.empty:
        st.markdown("*Resumen por segmento (APR óptimo y resultados)*")
        df = a2s.copy()
        # si tenemos APR_opt, mostramos; formateo de EAD
        df1 = format_df_currency(df, ["EAD_in","EAD_out","ingreso_opt","utilidad_opt"], moneda, usdclp)
        st.dataframe(df1, use_container_width=True, height=360)

# ==============================
# Arista 3 – Incentivos
# ==============================
with tabs[2]:
    st.header("Arista 3 – Incentivos")
    st.markdown("*Objetivo:* Asignar incentivos *sólo* donde el *ROI* esperado es *> 0*, bajo presupuesto global.")
    st.markdown("""
*KPIs clave*
- *Costo de incentivos*: gasto ejecutado.  
- *Ingreso incremental*: aumento de ingresos atribuible a la acción.  
- *ROI: Ingreso incremental / Costo. Objetivo *> 1** y creciente.  
- *Sensibilidad*: cómo varían resultados al exigir mayor ROI mínimo.
    """)

    a3d = load_csv(bundle_dir, REQ_FILES_BASE["a3_det"], suf)
    a3s = load_csv(bundle_dir, REQ_FILES_BASE["a3_sum"], suf)
    a3x = load_csv(bundle_dir, REQ_FILES_BASE["a3_sens"], suf)

    if a3s is not None and not a3s.empty:
        r = a3s.iloc[0].to_dict()
        costo = r.get("budget_usado", 0.0)
        ingr  = r.get("ingreso_incremental", 0.0)
        roi   = (ingr / costo * 100.0) if costo and costo != 0 else np.nan
        kpi_row_money("Costo de incentivos", costo, costo, moneda, usdclp)
        kpi_row_money("Ingreso incremental", ingr, ingr, moneda, usdclp)
        st.metric("ROI", fmt_pct_val(roi))

    if a3d is not None and not a3d.empty:
        st.markdown("*Detalle seleccionado* (top por ROI dentro de presupuesto)")
        money_cols = ["costo_incentivo","ingreso_incremental"]
        pct_cols = ["roi"]
        df = a3d.copy()
        if "roi" not in df.columns and "ROI" in df.columns:
            df = df.rename(columns={"ROI":"roi"})
        df1 = format_df_currency(df, money_cols, moneda, usdclp)
        df1 = format_df_pct(df1, pct_cols)
        st.dataframe(df1, use_container_width=True, height=360)

    if a3x is not None and not a3x.empty:
        st.markdown("*Sensibilidades (ROI mínimo)*")
        df = a3x.copy()
        df1 = format_df_currency(df, ["budget","costo","ingreso_inc"], moneda, usdclp)
        df1 = format_df_pct(df1, ["ROI"])
        st.dataframe(df1, use_container_width=True, height=280)

# ==============================
# Arista 4 – Capital/Provisiones
# ==============================
with tabs[3]:
    st.header("Arista 4 – Capital/Provisiones")
    st.markdown("*Objetivo:* Hacer más eficiente el *consumo de capital* (RWA, K) y reducir *provisiones* (≈ EL) sin deteriorar la calidad.")
    st.markdown("""
*KPIs clave (portafolio)*
- *EAD (base vs optimizado)*.  
- *RWA (proxy Basel): RW × EAD. Buscamos *↓**.  
- *Capital (K): K_ratio × RWA. Buscamos *↓**.  
- *Provisiones (≈ EL): consistentes con IFRS9. Buscamos *↓**.
    """)

    a4p = load_csv(bundle_dir, REQ_FILES_BASE["a4_port"], suf)
    if a4p is not None and not a4p.empty:
        def g0(df, c): return df[c].iloc[0] if c in df.columns else np.nan
        # EAD
        kpi_row_money("EAD", g0(a4p,"EAD_base"), g0(a4p,"EAD_opt"), moneda, usdclp)
        # RWA
        kpi_row_money("RWA (proxy Basel)", g0(a4p,"RWA_base"), g0(a4p,"RWA_opt"), moneda, usdclp)
        # K
        kpi_row_money("Capital (K)", g0(a4p,"K_base"), g0(a4p,"K_opt"), moneda, usdclp)
        # Provisiones ~ EL
        kpi_row_money("Provisiones (≈ EL)", g0(a4p,"EL_base"), g0(a4p,"EL_opt"), moneda, usdclp)

    a4d = load_csv(bundle_dir, REQ_FILES_BASE["a4_det"], suf)
    if a4d is not None and not a4d.empty:
        st.markdown("*Detalle por cliente*")
        money_cols = ["EAD_base","EAD_opt","RWA_base","RWA_opt","K_base","K_opt","EL_base","EL_opt","Util_base","Util_opt"]
        df1 = format_df_currency(a4d, money_cols, moneda, usdclp)
        st.dataframe(df1, use_container_width=True, height=360)

# ==============================
# Guardrails
# ==============================
with tabs[4]:
    st.header("Guardrails (Resguardos)")
    st.markdown("Conjunto de *límites* regulatorios y de negocio + *evaluación automática* de consistencia (Celda 16).")

    # Catálogo exportado (opcional)
    gport = load_csv(bundle_dir, REQ_FILES_BASE["gr_port"], "")
    gseg  = load_csv(bundle_dir, REQ_FILES_BASE["gr_seg"], "")
    geval = load_csv(bundle_dir, REQ_FILES_BASE["gr_eval"], "")

    if gport is None and gseg is None:
        st.info("No se encontraron archivos de catálogo (guardrails_portfolio/segment).")
    else:
        if gport is not None and not gport.empty:
            st.subheader("Catálogo – Portafolio")
            # formateo respetando rangos como strings y % numéricos
            df = gport.copy()
            for c in ["umbral","observado_actual","observado_optimizado"]:
                if c in df.columns: df[c] = df[c].apply(fmt_pct_val)
            st.dataframe(df, use_container_width=True, height=280)
        if gseg is not None and not gseg.empty:
            st.subheader("Catálogo – Segmento")
            df = gseg.copy()
            if "observado" in df.columns:
                df["observado"] = df["observado"].apply(fmt_pct_val)
            st.dataframe(df, use_container_width=True, height=280)

    st.markdown("---")
    st.subheader("Evaluación automática (Celda 16)")
    if geval is None or geval.empty:
        st.info("No se encontró *guardrails_eval_portfolio.csv*. Ejecuta la Celda 16 en el Notebook.")
    else:
        df = geval.copy()
        st.dataframe(df, use_container_width=True, height=300)

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (IFRS9 + Basel proxy). Estructura lista para piloto IT.")
