# app/app_dashboard.py — MVP Bancario (Fix inmediato: Potenciado-only, KPIs desde detail)
import os, re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ==========================
# Config: nombres esperados (escenario Potenciado)
# ==========================
FILES = {
    # Arista 1
    "a1_det":  "default_detail_agresivo.csv",
    "a1_seg":  "default_segment_agresivo.csv",
    "a1_port": "default_portfolio_agresivo.csv",
    # Arista 2
    "a2_det":  "yield_detail_agresivo.csv",
    "a2_seg":  "yield_segment_agresivo.csv",
    "a2_port": "yield_portfolio_agresivo.csv",
    "a2_curv": "yield_curve_segment_agresivo.csv",
    # Arista 3
    "a3_det":  "incentives_detail_agresivo.csv",
    "a3_sum":  "incentives_diag_summary_agresivo.csv",
    "a3_sens": "incentives_sensitivity_agresivo.csv",
    # Arista 4
    "a4_det":  "capital_detail_agresivo.csv",
    "a4_seg":  "capital_segment_agresivo.csv",
    "a4_port": "capital_portfolio_agresivo.csv",
    # Guardrails (sin sufijo)
    "gr_port": "guardrails_portfolio.csv",
    "gr_seg":  "guardrails_segment.csv",
    "gr_eval": "guardrails_eval_portfolio.csv",
}

CANDIDATES = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    "/content/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
    "/content/out",
    "./out",
]

# ==========================
# Parseo y formato
# ==========================
def parse_num_any(v):
    if v is None: return np.nan
    if isinstance(v, (int, float)):
        try: return float(v)
        except: return np.nan
    s = str(v).strip().replace(" ", "").replace("−","-").replace("%","")
    s = s.replace("$","").replace("CLP","").replace("USD","")
    if s == "" or s.upper() in {"N/A","NA","NULL","NONE","—"}: return np.nan
    d, c = s.rfind("."), s.rfind(",")
    try:
        if d == -1 and c == -1:
            return float(s) if s.lstrip("-").replace(".","",1).isdigit() else np.nan
        if c > d:
            s = s.replace(".","").replace(",",".")
        else:
            s = s.replace(",","")
        return float(s)
    except:
        return np.nan

def to_currency(val, moneda, usdclp):
    if pd.isna(val): return np.nan
    return float(val)/float(usdclp) if (moneda=="USD" and usdclp) else float(val)

def fmt_money(v, moneda, usdclp):
    x = parse_num_any(v)
    if np.isnan(x): return "—"
    x = to_currency(x, moneda, usdclp)
    neg = x < 0; x = abs(x)
    ent = int(x); dec = int(round((x - ent)*100))
    if dec == 100: ent += 1; dec = 0
    ent_str = f"{ent:,}".replace(",", ".")
    return f"-{ent_str},{dec:02d}" if neg else f"{ent_str},{dec:02d}"

def fmt_pct(v):
    x = parse_num_any(v)
    if np.isnan(x): return "—"
    return f"{x:.2f}%".replace(".", ",")

def var_pct(a,b):
    A,B = parse_num_any(a), parse_num_any(b)
    if np.isnan(A) or A==0: return None
    return (B-A)/A*100.0

def kpi_row_money(label, a, b, moneda, usdclp, help_text=""):
    c1,c2,c3 = st.columns([1.2,1.2,0.8])
    with c1:
        st.metric(f"{label} – Actual", fmt_money(a,moneda,usdclp))
        if help_text: st.caption(help_text)
    with c2:
        st.metric(f"{label} – Optimizado", fmt_money(b,moneda,usdclp))
    with c3:
        vp = var_pct(a,b)
        st.metric("VAR %", fmt_pct(vp) if vp is not None else "—")

def kpi_row_pct(label, a, b, help_text=""):
    c1,c2,c3 = st.columns([1.2,1.2,0.8])
    with c1:
        st.metric(f"{label} – Actual", fmt_pct(a)); 
        if help_text: st.caption(help_text)
    with c2:
        st.metric(f"{label} – Optimizado", fmt_pct(b))
    with c3:
        vp = var_pct(a,b)
        st.metric("VAR %", fmt_pct(vp) if vp is not None else "—")

def df_fmt_money(df, cols, moneda, usdclp):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns: df2[c] = df2[c].apply(lambda v: fmt_money(v,moneda,usdclp))
    return df2

def df_fmt_pct(df, cols):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns: df2[c] = df2[c].apply(fmt_pct)
    return df2

# ==========================
# Helpers de carga/aggregación
# ==========================
def autodetect_bundle():
    for d in CANDIDATES:
        if d and os.path.isdir(d):
            # verificamos que al menos existan los archivos potenciado clave
            ok = all(os.path.exists(os.path.join(d, FILES[k])) for k in [
                "a1_det","a2_det","a3_det","a4_det"
            ])
            if ok: return d
    # si no están todos, con que haya varios ya sirve (devolvemos el primero válido)
    for d in CANDIDATES:
        if d and os.path.isdir(d):
            return d
    return None

def load(bundle_dir, key):
    p = os.path.join(bundle_dir, FILES[key])
    if not os.path.exists(p): return None
    try: return pd.read_csv(p)
    except: return None

def sum_col(df, candidates):
    if df is None or df.empty: return None
    for c in candidates:
        if c in df.columns:
            return df[c].map(parse_num_any).sum()
    return None

def first_val(df, candidates):
    if df is None or df.empty: return None
    for c in candidates:
        if c in df.columns:
            try: return parse_num_any(df[c].iloc[0])
            except: continue
    return None

# ==========================
# App
# ==========================
st.set_page_config(page_title="MVP Bancario – Potenciado", layout="wide")
st.sidebar.title("⚙️ Configuración (Potenciado)")

moneda = st.sidebar.radio("Moneda", ["CLP","USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))

default_dir = autodetect_bundle()
bundle_dir = st.sidebar.text_input("📦 Ruta del bundle", value=(default_dir or ""), help="Ej: /content/out/dashboard_bundle").strip() or default_dir

st.title("📊 MVP Bancario — Escenario Potenciado (Agresivo regulado)")
st.caption("KPIs calculadas directamente desde los detail para evitar desalineaciones. IFRS9 + Basel proxy.")

if not bundle_dir:
    st.error("No encuentro el bundle. Revisa la ruta o genera el paquete en el Notebook."); st.stop()

# Diagnóstico
with st.expander("🔎 Diagnóstico del bundle (Potenciado)", expanded=True):
    rows = []
    for k,f in FILES.items():
        p = os.path.join(bundle_dir, f)
        rows.append({"archivo": f, "existe": os.path.exists(p)})
    df_diag = pd.DataFrame(rows)
    st.dataframe(df_diag, use_container_width=True, height=220)
    st.caption("Si un archivo clave no existe, la pestaña correspondiente puede mostrar advertencias.")

tabs = st.tabs([
    "Arista 1 – Default/Impago",
    "Arista 2 – Yield/Pricing",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital/Provisiones",
    "Guardrails"
])

# ==============================
# Arista 1 (desde detail)
# ==============================
with tabs[0]:
    st.header("Arista 1 – Default/Impago (Potenciado)")
    A1D = load(bundle_dir, "a1_det")
    if A1D is None or A1D.empty:
        st.error("Falta default_detail_agresivo.csv"); 
    else:
        # Sumatorias robustas
        ead_a = sum_col(A1D, ["e_base","EAD_base"])
        ead_b = sum_col(A1D, ["e_final","EAD_opt"])
        el_a  = sum_col(A1D, ["EL_base"])
        el_b  = sum_col(A1D, ["EL_final","EL_opt"])
        inc_a = sum_col(A1D, ["income_base"])
        inc_b = sum_col(A1D, ["income_final","income_opt"])
        util_a= sum_col(A1D, ["util_base"])
        util_b= sum_col(A1D, ["util_final","util_opt"])

        kpi_row_money("Interés Devengado Bruto", inc_a, inc_b, moneda, usdclp, "APR×EAD (proxy margen).")
        kpi_row_money("Pérdida Esperada (EL)", el_a, el_b, moneda, usdclp)
        kpi_row_money("Utilidad", util_a, util_b, moneda, usdclp)
        kpi_row_money("EAD", ead_a, ead_b, moneda, usdclp)

        st.markdown("*Detalle por cliente*")
        money = ["e_base","e_final","income_base","income_final","util_base","util_final","EL_base","EL_final","EAD_base","EAD_opt"]
        pct   = ["pd_base","pd_final","lgd_base","lgd_final","PD_12m","LGD_adj"]
        st.dataframe(df_fmt_pct(df_fmt_money(A1D, money, moneda, usdclp), pct), use_container_width=True, height=360)

# ==============================
# Arista 2 (desde detail)
# ==============================
with tabs[1]:
    st.header("Arista 2 – Yield/Pricing (Potenciado)")
    A2D = load(bundle_dir, "a2_det")
    if A2D is None or A2D.empty:
        st.error("Falta yield_detail_agresivo.csv");
    else:
        util_a = sum_col(A2D, ["util_base"])         # si no existe, quedará NaN (se muestra —)
        util_b = sum_col(A2D, ["util_opt"])
        inc_a  = sum_col(A2D, ["income_base"])
        inc_b  = sum_col(A2D, ["income_opt"])
        e_in   = sum_col(A2D, ["ead_in","EAD_in"])
        e_out  = sum_col(A2D, ["e_out","EAD_out"])

        kpi_row_money("Utilidad Total",     util_a, util_b, moneda, usdclp)
        kpi_row_money("Interés Bruto Total",inc_a,  inc_b,  moneda, usdclp)
        kpi_row_money("EAD (in → out)",     e_in,   e_out,  moneda, usdclp, "Volumen afectado por elasticidad vs APR.")

        st.markdown("*Detalle por cliente (pricing)*")
        money = ["ead_in","e_out","income_opt","EL_opt","COF_opt","util_opt","income_base","util_base"]
        st.dataframe(df_fmt_money(A2D, money, moneda, usdclp), use_container_width=True, height=360)

# ==============================
# Arista 3 (desde summary o detail)
# ==============================
with tabs[2]:
    st.header("Arista 3 – Incentivos (Potenciado)")
    A3S = load(bundle_dir, "a3_sum")
    A3D = load(bundle_dir, "a3_det")
    A3X = load(bundle_dir, "a3_sens")

    if (A3S is None or A3S.empty) and (A3D is None or A3D.empty):
        st.error("Faltan incentives_diag_summary_agresivo.csv e incentives_detail_agresivo.csv")
    else:
        if A3S is not None and not A3S.empty:
            r = A3S.iloc[0].to_dict()
            costo = parse_num_any(r.get("budget_usado", r.get("costo", 0.0)))
            ingr  = parse_num_any(r.get("ingreso_incremental", r.get("ingreso_inc", 0.0)))
        else:
            costo = sum_col(A3D, ["costo_incentivo","budget_usado"])
            ingr  = sum_col(A3D, ["ingreso_incremental","ingreso_inc"])
        roi = (ingr/costo*100.0) if (costo not in (None, 0, np.nan)) else np.nan

        kpi_row_money("Costo de incentivos", costo, costo, moneda, usdclp)
        kpi_row_money("Ingreso incremental", ingr,  ingr,  moneda, usdclp)
        st.metric("ROI", fmt_pct(roi))

        if A3D is not None and not A3D.empty:
            st.markdown("*Detalle seleccionado*")
            df = A3D.copy()
            if "ROI" in df.columns and "roi" not in df.columns: df = df.rename(columns={"ROI":"roi"})
            money = ["costo_incentivo","ingreso_incremental","budget_usado"]
            st.dataframe(df_fmt_pct(df_fmt_money(df, money, moneda, usdclp), ["roi"]),
                         use_container_width=True, height=360)

        if A3X is not None and not A3X.empty:
            st.markdown("*Sensibilidades (ROI mínimo)*")
            st.dataframe(df_fmt_pct(df_fmt_money(A3X, ["budget","costo","ingreso_inc"], moneda, usdclp), ["ROI"]),
                         use_container_width=True, height=260)

# ==============================
# Arista 4 (desde detail)
# ==============================
with tabs[3]:
    st.header("Arista 4 – Capital/Provisiones (Potenciado)")
    A4D = load(bundle_dir, "a4_det")
    if A4D is None or A4D.empty:
        st.error("Falta capital_detail_agresivo.csv")
    else:
        ead_a = sum_col(A4D, ["EAD_base"])
        ead_b = sum_col(A4D, ["EAD_opt"])
        rwa_a = sum_col(A4D, ["RWA_base"])
        rwa_b = sum_col(A4D, ["RWA_opt"])
        k_a   = sum_col(A4D, ["K_base"])
        k_b   = sum_col(A4D, ["K_opt"])
        el_a  = sum_col(A4D, ["EL_base"])
        el_b  = sum_col(A4D, ["EL_opt"])

        kpi_row_money("EAD",                ead_a, ead_b, moneda, usdclp)
        kpi_row_money("RWA (proxy Basel)",  rwa_a, rwa_b, moneda, usdclp)
        kpi_row_money("Capital (K)",        k_a,   k_b,   moneda, usdclp)
        kpi_row_money("Provisiones (≈ EL)", el_a,  el_b,  moneda, usdclp)

        st.markdown("*Detalle por cliente*")
        money = ["EAD_base","EAD_opt","RWA_base","RWA_opt","K_base","K_opt","EL_base","EL_opt","Util_base","Util_opt"]
        st.dataframe(df_fmt_money(A4D, money, moneda, usdclp), use_container_width=True, height=360)

# ==============================
# Guardrails (opcionales)
# ==============================
with tabs[4]:
    st.header("Guardrails")
    GP = load(bundle_dir, "gr_port")
    GS = load(bundle_dir, "gr_seg")
    GE = load(bundle_dir, "gr_eval")

    if GP is None and GS is None and (GE is None or GE.empty):
        st.info("No se encontraron archivos de guardrails. Ejecuta Celdas 15–16 y exporta al bundle.")
    else:
        if GP is not None and not GP.empty:
            st.subheader("Catálogo – Portafolio")
            df = GP.copy()
            for c in ["umbral","observado_actual","observado_optimizado"]:
                if c in df.columns: df[c] = df[c].apply(fmt_pct)
            st.dataframe(df, use_container_width=True, height=260)
        if GS is not None and not GS.empty:
            st.subheader("Catálogo – Segmento")
            df = GS.copy()
            if "observado" in df.columns:
                df["observado"] = df["observado"].apply(fmt_pct)
            st.dataframe(df, use_container_width=True, height=260)
        st.markdown("---")
        st.subheader("Evaluación automática (Celda 16)")
        if GE is None or GE.empty:
            st.info("No se encontró *guardrails_eval_portfolio.csv*.")
        else:
            st.dataframe(GE, use_container_width=True, height=320)

st.markdown("---")
st.caption("© MVP Bancario — Potenciado (KPIs desde detail).")
