# app/app_dashboard.py — MVP Bancario (4 Aristas) v2.4 (root fix + robust KPI)
import os, re, math
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
    # Guardrails
    "gr_port": "guardrails_portfolio.csv",
    "gr_seg":  "guardrails_segment.csv",
    "gr_eval": "guardrails_eval_portfolio.csv",
}

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    "/content/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
    "/content/out",
    "./out",
]

# ==========================
# Parsing / Formato numérico
# ==========================
def parse_num_any(v):
    """Convierte '1.234.567,89' | '1,234,567.89' | '1234,56' | '1234.56' | con/sin '%' | con/sin $ a float."""
    if v is None: return np.nan
    if isinstance(v, (int, float)): 
        try: return float(v)
        except: return np.nan
    s = str(v).strip().replace(" ", "").replace("−","-").replace("%","")
    s = s.replace("$","").replace("CLP","").replace("USD","")
    if s == "" or s.upper() in {"N/A","NA","NULL","NONE","—"}: return np.nan
    last_dot, last_com = s.rfind("."), s.rfind(",")
    try:
        if last_dot == -1 and last_com == -1:
            return float(s) if s.lstrip("-").replace(".","",1).isdigit() else np.nan
        if last_com > last_dot:
            s = s.replace(".","").replace(",",".")
        else:
            s = s.replace(",","")
        return float(s)
    except:
        return np.nan

def to_currency(val, moneda, usdclp):
    if pd.isna(val): return np.nan
    return float(val) / float(usdclp) if (moneda.upper()=="USD" and usdclp) else float(val)

def fmt_money(val, moneda, usdclp):
    x = parse_num_any(val)
    if np.isnan(x): return "—"
    x = to_currency(x, moneda, usdclp)
    neg = x < 0; x = abs(x)
    ent = int(x); dec = int(round((x - ent)*100))
    if dec == 100: ent += 1; dec = 0
    ent_str = f"{ent:,}".replace(",", ".")
    return f"-{ent_str},{dec:02d}" if neg else f"{ent_str},{dec:02d}"

def fmt_pct(val):
    x = parse_num_any(val)
    if np.isnan(x): return "—"
    return f"{x:.2f}%".replace(".", ",")

def var_pct(a, b):
    A = parse_num_any(a); B = parse_num_any(b)
    if np.isnan(A) or A == 0: return None
    return (B - A)/A*100.0

def kpi_row_money(label, a, b, moneda, usdclp, help_text=""):
    c1, c2, c3 = st.columns([1.2,1.2,0.8])
    with c1:
        st.metric(f"{label} – Actual", fmt_money(a, moneda, usdclp))
        if help_text: st.caption(help_text)
    with c2:
        st.metric(f"{label} – Optimizado", fmt_money(b, moneda, usdclp))
    with c3:
        vp = var_pct(a,b)
        st.metric("VAR %", fmt_pct(vp) if vp is not None else "—")

def kpi_row_pct(label, a, b, help_text=""):
    c1, c2, c3 = st.columns([1.2,1.2,0.8])
    with c1:
        st.metric(f"{label} – Actual", fmt_pct(a))
        if help_text: st.caption(help_text)
    with c2:
        st.metric(f"{label} – Optimizado", fmt_pct(b))
    with c3:
        vp = var_pct(a,b)
        st.metric("VAR %", fmt_pct(vp) if vp is not None else "—")

def df_fmt_money(df, cols, moneda, usdclp):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda v: fmt_money(v, moneda, usdclp))
    return df2

def df_fmt_pct(df, cols):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = df2[c].apply(fmt_pct)
    return df2

# ==========================
# Helpers de bundle y KPIs
# ==========================
def _exists(bundle_dir, key, suf):
    return os.path.exists(os.path.join(bundle_dir, REQ_FILES_BASE[key].format(S=suf)))

def scenario_autodetect(bundle_dir):
    has_cons = any(_exists(bundle_dir, k, "") for k in ["a1_port","a2_port","a4_port"])
    has_aggr = any(_exists(bundle_dir, k, "_agresivo") for k in ["a1_port","a2_port","a4_port"])
    if has_aggr and not has_cons: return "_agresivo"
    if has_cons and not has_aggr: return ""
    return ""  # por defecto conservador si hay ambos

def autodetect_bundle(suf):
    for d in CANDIDATE_DIRS:
        if d and os.path.isdir(d):
            if any(_exists(d,k,suf) for k in ["a1_port","a2_port","a4_port"]):
                return d
            s2 = scenario_autodetect(d)
            if any(_exists(d,k,s2) for k in ["a1_port","a2_port","a4_port"]):
                return d
    return None

def load_csv(bundle_dir, pattern, suf):
    p = os.path.join(bundle_dir, pattern.format(S=suf))
    if not os.path.exists(p): return None
    try:
        return pd.read_csv(p)
    except:
        return None

def first_value(df, candidates):
    """Primer valor de la primera columna encontrada (fila 0), parseado."""
    if df is None or df.empty: return None
    for c in candidates:
        if c in df.columns:
            try:
                return parse_num_any(df[c].iloc[0])
            except:
                continue
    return None

def sum_col(df, candidates):
    """Suma columna (parseando cada celda)."""
    if df is None or df.empty: return None
    for c in candidates:
        if c in df.columns:
            return df[c].map(parse_num_any).sum()
    return None

# ==========================
# App
# ==========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")
st.sidebar.title("⚙️ Configuración")

escenario_choice = st.sidebar.radio("Escenario", ["Conservador", "Potenciado"], horizontal=True)
suf_ui = "" if escenario_choice=="Conservador" else "_agresivo"

moneda = st.sidebar.radio("Moneda", ["CLP","USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))

default_dir = autodetect_bundle(suf_ui)
bundle_dir = st.sidebar.text_input("📦 Ruta del bundle", value=(default_dir or ""), help="Ej: /content/out/dashboard_bundle").strip() or default_dir

st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Modelo matemático con cumplimiento (IFRS9 + Basel proxy). Comparación *Actual vs Optimizado* por arista y escenario.")

if not bundle_dir:
    st.error("No encuentro el bundle. Revisa la ruta o genera el paquete en el Notebook."); st.stop()

# Si el escenario seleccionado no existe, forzar el disponible
suf_real = suf_ui
if not any(_exists(bundle_dir,k,suf_real) for k in ["a1_port","a2_port","a4_port"]):
    suf_real = scenario_autodetect(bundle_dir)
    if suf_real == "_agresivo":
        st.info("⚠️ Sólo hay archivos del escenario *Potenciado*. Se ajustó automáticamente.")
    else:
        st.info("⚠️ Sólo hay archivos del escenario *Conservador*. Se ajustó automáticamente.")

with st.expander("🔎 Diagnóstico del bundle", expanded=False):
    found = []
    for key, pat in REQ_FILES_BASE.items():
        p = os.path.join(bundle_dir, pat.format(S=suf_real))
        if os.path.exists(p): found.append(f"✅ {os.path.basename(p)}")
    st.write("\n".join(found) if found else "No se encontraron archivos para el escenario.")

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
    st.markdown("*Objetivo:* Mantener *EAD* total, recomponer mix a menor *PD* para *reducir EL* sin frenar negocio.")
    st.markdown("""
*KPIs*
- *EAD* total (estable).
- *EL* = PD × LGD × EAD (↓).
- *Interés Devengado Bruto* = APR × EAD.
- *Utilidad* = Interés – EL – costos.
- *PD ponderado* (↓).
    """)

    a1p = load_csv(bundle_dir, REQ_FILES_BASE["a1_port"], suf_real)
    a1d = load_csv(bundle_dir, REQ_FILES_BASE["a1_det"], suf_real)

    # KPI desde portfolio; si falta, recomponer desde detail
    inc_a = first_value(a1p, ["Interes_devengado_bruto_actual","Ingreso_actual","ingreso_base"])
    inc_b = first_value(a1p, ["Interes_devengado_bruto_optimizado","Ingreso_optimizado","ingreso_final","ingreso_opt"])
    util_a = first_value(a1p, ["Utilidad_actual","utilidad_base","Util_actual"])
    util_b = first_value(a1p, ["Utilidad_optimizada","utilidad_final","utilidad_opt","Util_opt"])
    el_a  = first_value(a1p, ["EL_actual","EL_base"])
    el_b  = first_value(a1p, ["EL_optimizado","EL_final","EL_opt"])
    ead_a = first_value(a1p, ["EAD_actual","EAD_base","EAD"])
    ead_b = first_value(a1p, ["EAD_optimizado","EAD_final","EAD_opt","EAD"])

    if a1p is None or a1p.empty or all(v is None for v in [inc_a,inc_b,util_a,util_b,el_a,el_b,ead_a,ead_b]):
        # Recalcular desde detail
        inc_a = inc_a or sum_col(a1d, ["income_base"])
        inc_b = inc_b or sum_col(a1d, ["income_final","income_opt"])
        util_a= util_a or sum_col(a1d, ["util_base"])
        util_b= util_b or sum_col(a1d, ["util_final","util_opt"])
        el_a  = el_a  or sum_col(a1d, ["EL_base"])
        el_b  = el_b  or sum_col(a1d, ["EL_final","EL_opt"])
        ead_a = ead_a or sum_col(a1d, ["e_base","EAD_base"])
        ead_b = ead_b or sum_col(a1d, ["e_final","EAD_opt"])

    kpi_row_money("Interés Devengado Bruto", inc_a, inc_b, moneda, usdclp, "APR×EAD (proxy margen).")
    kpi_row_money("Utilidad",                 util_a, util_b, moneda, usdclp)
    kpi_row_money("Pérdida Esperada (EL)",    el_a,   el_b,   moneda, usdclp)
    kpi_row_money("EAD",                      ead_a,  ead_b,  moneda, usdclp)

    # PD ponderado si viene en portfolio
    pd_a = first_value(a1p, ["PD_pond_actual","PDpond_actual"])
    pd_b = first_value(a1p, ["PD_pond_optimizado","PDpond_optimizado"])
    if (pd_a is not None) or (pd_b is not None):
        kpi_row_pct("PD Ponderado", (pd_a*100 if pd_a is not None else None), (pd_b*100 if pd_b is not None else None))

    if a1d is not None and not a1d.empty:
        st.markdown("*Detalle por cliente*")
        money_cols = ["e_base","e_final","income_base","income_final","util_base","util_final","EL_base","EL_final","EAD_base","EAD_opt"]
        pct_cols   = ["pd_base","pd_final","lgd_base","lgd_final","PD_12m","LGD_adj"]
        st.dataframe(df_fmt_pct(df_fmt_money(a1d, money_cols, moneda, usdclp), pct_cols), use_container_width=True, height=360)

# ==============================
# Arista 2 – Yield/Pricing
# ==============================
with tabs[1]:
    st.header("Arista 2 – Yield/Pricing")
    st.markdown("*Objetivo:* Hallar *APR óptimo* por segmento maximizando *Utilidad* (trade-off precio/volumen).")
    st.markdown("""
*KPIs*
- *Utilidad Total* (↑).
- *Interés Bruto Total* = APR × EAD_out.
- *EAD_in / EAD_out* (impacto de elasticidad).
    """)

    a2p = load_csv(bundle_dir, REQ_FILES_BASE["a2_port"], suf_real)
    a2s = load_csv(bundle_dir, REQ_FILES_BASE["a2_seg"],  suf_real)
    a2d = load_csv(bundle_dir, REQ_FILES_BASE["a2_det"],  suf_real)

    util_a = first_value(a2p, ["utilidad_base","Utilidad_base"])
    util_b = first_value(a2p, ["utilidad_opt","Utilidad_opt"])
    inc_a  = first_value(a2p, ["ingreso_base","Ingreso_base"])
    inc_b  = first_value(a2p, ["ingreso_opt","Ingreso_opt"])
    e_in   = first_value(a2p, ["EAD_in","EAD_base","EAD"])
    e_out  = first_value(a2p, ["EAD_out","EAD_opt","EAD"])

    # fallback desde detail si falta
    if a2p is None or a2p.empty or all(v is None for v in [util_a,util_b,inc_a,inc_b,e_in,e_out]):
        util_a = util_a or sum_col(a2d, ["util_base"])
        util_b = util_b or sum_col(a2d, ["util_opt"])
        inc_a  = inc_a  or sum_col(a2d, ["income_base"])
        inc_b  = inc_b  or sum_col(a2d, ["income_opt"])
        e_in   = e_in   or sum_col(a2d, ["ead_in","EAD_in"])
        e_out  = e_out  or sum_col(a2d, ["e_out","EAD_out"])

    kpi_row_money("Utilidad Total",     util_a, util_b, moneda, usdclp)
    kpi_row_money("Interés Bruto Total",inc_a,  inc_b,  moneda, usdclp)
    kpi_row_money("EAD (in → out)",     e_in,   e_out, moneda, usdclp, "Volumen afectado por elasticidad vs APR.")

    if a2s is not None and not a2s.empty:
        st.markdown("*Resumen por segmento*")
        st.dataframe(df_fmt_money(a2s, ["EAD_in","EAD_out","ingreso_opt","utilidad_opt"], moneda, usdclp),
                     use_container_width=True, height=340)
    if a2d is not None and not a2d.empty:
        st.markdown("*Detalle por cliente (pricing)*")
        st.dataframe(df_fmt_money(a2d, ["ead_in","e_out","income_opt","EL_opt","COF_opt","util_opt"], moneda, usdclp),
                     use_container_width=True, height=340)

# ==============================
# Arista 3 – Incentivos
# ==============================
with tabs[2]:
    st.header("Arista 3 – Incentivos")
    st.markdown("*Objetivo:* Asignar incentivos donde *ROI>0*, respetando presupuesto.")
    st.markdown("""
*KPIs*
- *Costo de incentivos*.
- *Ingreso incremental*.
- *ROI* = ingreso_inc / costo (↑).
    """)

    a3d = load_csv(bundle_dir, REQ_FILES_BASE["a3_det"],  suf_real)
    a3s = load_csv(bundle_dir, REQ_FILES_BASE["a3_sum"],  suf_real)
    a3x = load_csv(bundle_dir, REQ_FILES_BASE["a3_sens"], suf_real)

    # resumen principal (si no hay summary, intentar desde detail)
    if a3s is not None and not a3s.empty:
        r = a3s.iloc[0].to_dict()
        costo = parse_num_any(r.get("budget_usado", r.get("costo", 0.0)))
        ingr  = parse_num_any(r.get("ingreso_incremental", r.get("ingreso_inc", 0.0)))
    else:
        costo = sum_col(a3d, ["costo_incentivo","budget_usado"])
        ingr  = sum_col(a3d, ["ingreso_incremental","ingreso_inc"])

    roi = (ingr / costo * 100.0) if (costo not in (None, 0, np.nan)) else np.nan
    kpi_row_money("Costo de incentivos", costo, costo, moneda, usdclp)
    kpi_row_money("Ingreso incremental", ingr,  ingr,  moneda, usdclp)
    st.metric("ROI", fmt_pct(roi))

    if a3d is not None and not a3d.empty:
        st.markdown("*Detalle seleccionado*")
        df = a3d.copy()
        if "ROI" in df.columns and "roi" not in df.columns: df = df.rename(columns={"ROI":"roi"})
        st.dataframe(df_fmt_pct(df_fmt_money(df, ["costo_incentivo","ingreso_incremental","budget_usado"], moneda, usdclp), ["roi"]),
                     use_container_width=True, height=360)
    if a3x is not None and not a3x.empty:
        st.markdown("*Sensibilidades (ROI mínimo)*")
        st.dataframe(df_fmt_pct(df_fmt_money(a3x, ["budget","costo","ingreso_inc"], moneda, usdclp), ["ROI"]),
                     use_container_width=True, height=280)

# ==============================
# Arista 4 – Capital/Provisiones
# ==============================
with tabs[3]:
    st.header("Arista 4 – Capital/Provisiones")
    st.markdown("*Objetivo:* Reducir *RWA, **K* y *Provisiones (≈EL)* con calidad conservada.")
    st.markdown("""
*KPIs (portafolio)*
- *EAD* (base vs opt)
- *RWA* = RW × EAD (↓)
- *Capital (K)* = K_ratio × RWA (↓)
- *Provisiones (≈ EL)* (↓)
    """)

    a4p = load_csv(bundle_dir, REQ_FILES_BASE["a4_port"], suf_real)
    a4d = load_csv(bundle_dir, REQ_FILES_BASE["a4_det"],  suf_real)
    a4s = load_csv(bundle_dir, REQ_FILES_BASE["a4_seg"],  suf_real)

    ead_a = first_value(a4p, ["EAD_base","EAD"])
    ead_b = first_value(a4p, ["EAD_opt","EAD"])
    rwa_a = first_value(a4p, ["RWA_base","RWA"])
    rwa_b = first_value(a4p, ["RWA_opt","RWA"])
    k_a   = first_value(a4p, ["K_base","K"])
    k_b   = first_value(a4p, ["K_opt","K"])
    el_a  = first_value(a4p, ["EL_base","EL"])
    el_b  = first_value(a4p, ["EL_opt","EL"])

    if a4p is None or a4p.empty or all(v is None for v in [ead_a,ead_b,rwa_a,rwa_b,k_a,k_b,el_a,el_b]):
        ead_a = ead_a or sum_col(a4d, ["EAD_base"])
        ead_b = ead_b or sum_col(a4d, ["EAD_opt"])
        rwa_a = rwa_a or sum_col(a4d, ["RWA_base"])
        rwa_b = rwa_b or sum_col(a4d, ["RWA_opt"])
        k_a   = k_a   or sum_col(a4d, ["K_base"])
        k_b   = k_b   or sum_col(a4d, ["K_opt"])
        el_a  = el_a  or sum_col(a4d, ["EL_base"])
        el_b  = el_b  or sum_col(a4d, ["EL_opt"])

    kpi_row_money("EAD",                ead_a, ead_b, moneda, usdclp)
    kpi_row_money("RWA (proxy Basel)",  rwa_a, rwa_b, moneda, usdclp)
    kpi_row_money("Capital (K)",        k_a,   k_b,   moneda, usdclp)
    kpi_row_money("Provisiones (≈ EL)", el_a,  el_b,  moneda, usdclp)

    if a4d is not None and not a4d.empty:
        st.markdown("*Detalle por cliente*")
        cols = ["EAD_base","EAD_opt","RWA_base","RWA_opt","K_base","K_opt","EL_base","EL_opt","Util_base","Util_opt"]
        st.dataframe(df_fmt_money(a4d, cols, moneda, usdclp), use_container_width=True, height=360)
    if a4s is not None and not a4s.empty:
        st.markdown("*Resumen por segmento*")
        cols = ["EAD_base","EAD_opt","RWA_base","RWA_opt","K_base","K_opt","EL_base","EL_opt","Util_base","Util_opt"]
        st.dataframe(df_fmt_money(a4s, cols, moneda, usdclp), use_container_width=True, height=320)

# ==============================
# Guardrails
# ==============================
with tabs[4]:
    st.header("Guardrails (Resguardos)")
    st.markdown("Catálogo y *evaluación automática* (Celda 16).")

    gport = load_csv(bundle_dir, REQ_FILES_BASE["gr_port"], "")
    gseg  = load_csv(bundle_dir, REQ_FILES_BASE["gr_seg"], "")
    geval = load_csv(bundle_dir, REQ_FILES_BASE["gr_eval"], "")

    if gport is None and gseg is None and (geval is None or geval.empty):
        st.info("No se encontraron archivos de guardrails. Ejecuta Celdas 15–16 y exporta al bundle.")
    else:
        if gport is not None and not gport.empty:
            st.subheader("Catálogo – Portafolio")
            df = gport.copy()
            for c in ["umbral","observado_actual","observado_optimizado"]:
                if c in df.columns: df[c] = df[c].apply(lambda x: fmt_pct(x))
            st.dataframe(df, use_container_width=True, height=280)
        if gseg is not None and not gseg.empty:
            st.subheader("Catálogo – Segmento")
            df = gseg.copy()
            if "observado" in df.columns:
                df["observado"] = df["observado"].apply(lambda x: fmt_pct(x))
            st.dataframe(df, use_container_width=True, height=280)
        st.markdown("---")
        st.subheader("Evaluación automática (Celda 16)")
        if geval is None or geval.empty:
            st.info("No se encontró *guardrails_eval_portfolio.csv*.")
        else:
            st.dataframe(geval, use_container_width=True, height=320)

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (IFRS9 + Basel proxy). Estructura lista para piloto IT.")
