# app/app_dashboard.py — MVP Bancario (4 Aristas) v3.4
# Escenarios: Conservador / Potenciado | Moneda: CLP / USD
# Correcciones clave:
#  - Lectura estricta por escenario con sufijo
#  - Guardrails sin sufijo
#  - Diagnóstico por pestaña (muestra archivo exacto leído)
#  - Mensajes de error útiles si falta archivo/columna

import os, re, math, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ==========================
# Config de archivos
# ==========================
REQ = {
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
    # Guardrails (sin sufijo)
    "gr_port": "guardrails_portfolio.csv",
    "gr_seg":  "guardrails_segment.csv",
    "gr_eval": "guardrails_eval_portfolio.csv",
}

SCENARIOS = {"Conservador": "", "Potenciado": "_agresivo"}

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR","").strip(),
    "/content/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
    "/content/out",
    "./out",
]

# ==========================
# Utilidades numéricas/formatos
# ==========================
def parse_num_any(v):
    if v is None: return np.nan
    if isinstance(v, (int, float)):
        try: return float(v)
        except: return np.nan
    s = str(v).strip().replace(" ", "").replace("−","-")
    s = s.replace("%","").replace("$","").replace("CLP","").replace("USD","")
    if s == "" or s.upper() in {"N/A","NA","NULL","NONE","—"}: return np.nan
    d, c = s.rfind("."), s.rfind(",")
    try:
        if d == -1 and c == -1:
            return float(s) if re.fullmatch(r"-?\d+(\.\d+)?", s) else np.nan
        if c > d: s = s.replace(".","").replace(",",".")
        else:     s = s.replace(",","")
        return float(s)
    except:
        return np.nan

def to_currency(val, moneda, usdclp):
    if pd.isna(val): return np.nan
    return float(val)/float(usdclp) if (moneda.upper()=="USD" and usdclp) else float(val)

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
        if c in df2.columns: df2[c] = df2[c].apply(lambda v: fmt_money(v,moneda,usdclp))
    return df2

def df_fmt_pct(df, cols):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns: df2[c] = df2[c].apply(fmt_pct)
    return df2

# ==========================
# I/O Bundle + diagnóstico por archivo
# ==========================
def autodetect_bundle():
    for d in CANDIDATE_DIRS:
        if d and os.path.isdir(d):
            return d
    return None

def file_path(bundle_dir, key, suf):
    return os.path.join(bundle_dir, REQ[key].format(S=suf))

def load_csv_strict(bundle_dir, key, suf):
    """Lee EXACTAMENTE el archivo del escenario elegido.
       No hace fallback silencioso. Devuelve (df, path, exists_bool)."""
    p = file_path(bundle_dir, key, suf)
    if not os.path.exists(p): return None, p, False
    try:
        return pd.read_csv(p), p, True
    except Exception as e:
        st.error(f"Error leyendo {os.path.basename(p)}: {e}")
        return None, p, True

def sum_col(df, candidates):
    if df is None or df.empty: return None
    for c in candidates:
        if c in df.columns:
            return df[c].map(parse_num_any).sum()
    return None

def first_value(df, candidates):
    if df is None or df.empty: return None
    for c in candidates:
        if c in df.columns:
            try: return parse_num_any(df[c].iloc[0])
            except: continue
    return None

def wavg_from_detail(df, val_col_candidates, weight_col_candidates):
    if df is None or df.empty: return None
    vcol = next((c for c in val_col_candidates if c in df.columns), None)
    wcol = next((c for c in weight_col_candidates if c in df.columns), None)
    if not vcol or not wcol: return None
    v = df[vcol].map(parse_num_any)
    w = df[wcol].map(parse_num_any)
    tot_w = w.sum()
    if pd.isna(tot_w) or tot_w == 0: return None
    return float((v*w).sum() / tot_w)

def show_missing_file(path_expected):
    st.warning(f"Falta el archivo para este escenario:\n`{os.path.basename(path_expected)}\n\nRuta esperada:\n{path_expected}`")
    st.stop()

def show_columns(df, title="Columnas detectadas"):
    if df is None: return
    cols = list(df.columns)
    st.caption(f"*{title}:* {', '.join(cols) if cols else '(sin columnas)'}")

# ==========================
# App
# ==========================
st.set_page_config(page_title="MVP Bancario — 4 Aristas", layout="wide")
st.sidebar.title("⚙️ Configuración")

escenario = st.sidebar.radio("Escenario", list(SCENARIOS.keys()), horizontal=True, index=1)  # por defecto Potenciado
suf = SCENARIOS[escenario]

moneda = st.sidebar.radio("Moneda", ["CLP","USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))

default_dir = autodetect_bundle() or "/content/out/dashboard_bundle"
bundle_dir = st.sidebar.text_input("📦 Ruta del bundle", value=(default_dir or ""), help="Ej: /content/out/dashboard_bundle").strip() or default_dir

st.title("📊 MVP Bancario — Optimización en 4 Aristas")
st.caption("Modelo matemático (IFRS9 + Basel proxy). *Actual vs Optimizado* por arista, escenario y moneda.")

if not os.path.isdir(bundle_dir):
    st.error(f"No encuentro la carpeta del bundle: {bundle_dir}"); st.stop()

# Diagnóstico de ambos escenarios (tabla comparativa)
with st.expander("🔎 Diagnóstico del bundle (Conservador vs Potenciado)", expanded=False):
    def diag_for(label, sufval):
        rows = []
        for key, pat in REQ.items():
            p = os.path.join(bundle_dir, pat.format(S=sufval))
            rows.append({"escenario": label, "archivo": os.path.basename(p), "existe": os.path.exists(p)})
        return pd.DataFrame(rows)
    st.dataframe(pd.concat([diag_for("Conservador",""), diag_for("Potenciado","_agresivo")], ignore_index=True),
                 use_container_width=True, height=340)

tabs = st.tabs([
    "Arista 1 – Default/Impago",
    "Arista 2 – Yield/Pricing",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital/Provisiones",
    "Guardrails (Resguardos)"
])

# ==============================
# A1 – Default/Impago
# ==============================
with tabs[0]:
    st.header("Arista 1 – Default/Impago")
    st.markdown("*Objetivo:* Mantener *EAD* total estable y recomponer mix hacia menor *PD* para *reducir EL* sin frenar el negocio.")
    st.info(f"Escenario activo: *{escenario}*")

    try:
        A1P, p_port, ok_port = load_csv_strict(bundle_dir, "a1_port", suf)
        A1D, p_det,  ok_det  = load_csv_strict(bundle_dir, "a1_det",  suf)

        st.caption(f"Fuente portfolio: {os.path.basename(p_port)} (existe={ok_port})")
        st.caption(f"Fuente detail:    {os.path.basename(p_det)} (existe={ok_det})")

        if not ok_port and not ok_det:
            show_missing_file(p_port)  # muestra el esperado del escenario

        # KPIs
        inc_a = first_value(A1P, ["Interes_devengado_bruto_actual","Ingreso_actual","ingreso_base"])
        inc_b = first_value(A1P, ["Interes_devengado_bruto_optimizado","Ingreso_optimizado","ingreso_final","ingreso_opt"])
        util_a= first_value(A1P, ["Utilidad_actual","utilidad_base","Util_actual"])
        util_b= first_value(A1P, ["Utilidad_optimizada","utilidad_final","utilidad_opt","Util_opt"])
        el_a  = first_value(A1P, ["EL_actual","EL_base"])
        el_b  = first_value(A1P, ["EL_optimizado","EL_final","EL_opt"])
        ead_a = first_value(A1P, ["EAD_actual","EAD_base","EAD"])
        ead_b = first_value(A1P, ["EAD_optimizado","EAD_final","EAD_opt","EAD"])

        if (A1P is None or A1P.empty) or all(v is None for v in [inc_a,inc_b,util_a,util_b,el_a,el_b,ead_a,ead_b]):
            # fallback desde detail
            inc_a = inc_a or sum_col(A1D, ["income_base"])
            inc_b = inc_b or sum_col(A1D, ["income_final","income_opt"])
            util_a= util_a or sum_col(A1D, ["util_base"])
            util_b= util_b or sum_col(A1D, ["util_final","util_opt"])
            el_a  = el_a  or sum_col(A1D, ["EL_base"])
            el_b  = el_b  or sum_col(A1D, ["EL_final","EL_opt"])
            ead_a = ead_a or sum_col(A1D, ["e_base","EAD_base"])
            ead_b = ead_b or sum_col(A1D, ["e_final","EAD_opt"])

        kpi_row_money("Interés Devengado Bruto", inc_a, inc_b, moneda, usdclp, "APR×EAD (proxy margen).")
        kpi_row_money("Pérdida Esperada (EL)", el_a, el_b, moneda, usdclp)
        kpi_row_money("Utilidad", util_a, util_b, moneda, usdclp)
        kpi_row_money("EAD", ead_a, ead_b, moneda, usdclp)

        # PD ponderado
        pd_a = first_value(A1P, ["PD_pond_actual","PDpond_actual"])
        pd_b = first_value(A1P, ["PD_pond_optimizado","PDpond_optimizado"])
        if (pd_a is None) and (A1D is not None and not A1D.empty):
            pd_a = wavg_from_detail(A1D, ["pd_base","PD_12m"], ["e_base","EAD_base"])
        if (pd_b is None) and (A1D is not None and not A1D.empty):
            pd_b = wavg_from_detail(A1D, ["pd_final","PD_12m","pd_base"], ["e_final","EAD_opt","e_base"])
        if (pd_a is not None) or (pd_b is not None):
            kpi_row_pct("PD Ponderado", (pd_a*100 if pd_a is not None else None), (pd_b*100 if pd_b is not None else None))

        if A1D is not None and not A1D.empty:
            st.markdown("*Detalle por cliente*")
            money = ["e_base","e_final","income_base","income_final","util_base","util_final","EL_base","EL_final","EAD_base","EAD_opt"]
            pct   = ["pd_base","pd_final","lgd_base","lgd_final","PD_12m","LGD_adj"]
            show_columns(A1D)
            st.dataframe(df_fmt_pct(df_fmt_money(A1D, money, moneda, usdclp), pct), use_container_width=True, height=360)

    except Exception:
        st.error("Error en Arista 1:\n\n" + traceback.format_exc())

# ==============================
# A2 – Yield/Pricing
# ==============================
with tabs[1]:
    st.header("Arista 2 – Yield/Pricing")
    st.info(f"Escenario activo: *{escenario}*")

    try:
        A2P, p_port, ok_port = load_csv_strict(bundle_dir, "a2_port", suf)
        A2S, p_seg,  ok_seg  = load_csv_strict(bundle_dir, "a2_seg",  suf)
        A2D, p_det,  ok_det  = load_csv_strict(bundle_dir, "a2_det",  suf)

        st.caption(f"Fuente portfolio: {os.path.basename(p_port)} (existe={ok_port})")
        st.caption(f"Fuente segment:   {os.path.basename(p_seg)} (existe={ok_seg})")
        st.caption(f"Fuente detail:    {os.path.basename(p_det)} (existe={ok_det})")

        if not ok_port and not ok_det:
            show_missing_file(p_port)

        util_a = first_value(A2P, ["utilidad_base","Utilidad_base"])
        util_b = first_value(A2P, ["utilidad_opt","Utilidad_opt"])
        inc_a  = first_value(A2P, ["ingreso_base","Ingreso_base"])
        inc_b  = first_value(A2P, ["ingreso_opt","Ingreso_opt"])
        e_in   = first_value(A2P, ["EAD_in","EAD_base","EAD"])
        e_out  = first_value(A2P, ["EAD_out","EAD_opt","EAD"])

        if (A2P is None or A2P.empty) or all(v is None for v in [util_a,util_b,inc_a,inc_b,e_in,e_out]):
            util_a = util_a or sum_col(A2D, ["util_base"])
            util_b = util_b or sum_col(A2D, ["util_opt"])
            inc_a  = inc_a  or sum_col(A2D, ["income_base"])
            inc_b  = inc_b  or sum_col(A2D, ["income_opt"])
            e_in   = e_in   or sum_col(A2D, ["ead_in","EAD_in"])
            e_out  = e_out  or sum_col(A2D, ["e_out","EAD_out"])

        kpi_row_money("Utilidad Total", util_a, util_b, moneda, usdclp)
        kpi_row_money("Interés Bruto Total", inc_a, inc_b, moneda, usdclp)
        kpi_row_money("EAD (in → out)", e_in, e_out, moneda, usdclp, "Volumen afectado por elasticidad vs APR.")

        if A2S is not None and not A2S.empty:
            st.markdown("*Resumen por segmento*")
            show_columns(A2S)
            st.dataframe(df_fmt_money(A2S, ["EAD_in","EAD_out","ingreso_opt","utilidad_opt"], moneda, usdclp),
                         use_container_width=True, height=340)
        if A2D is not None and not A2D.empty:
            st.markdown("*Detalle por cliente (pricing)*")
            show_columns(A2D)
            st.dataframe(df_fmt_money(A2D, ["ead_in","e_out","income_opt","EL_opt","COF_opt","util_opt","income_base","util_base"], moneda, usdclp),
                         use_container_width=True, height=360)

    except Exception:
        st.error("Error en Arista 2:\n\n" + traceback.format_exc())

# ==============================
# A3 – Incentivos
# ==============================
with tabs[2]:
    st.header("Arista 3 – Incentivos")
    st.info(f"Escenario activo: *{escenario}*")

    try:
        A3S, p_sum,  ok_sum  = load_csv_strict(bundle_dir, "a3_sum",  suf)
        A3D, p_det,  ok_det  = load_csv_strict(bundle_dir, "a3_det",  suf)
        A3X, p_sens, ok_sens = load_csv_strict(bundle_dir, "a3_sens", suf)

        st.caption(f"Fuente summary: {os.path.basename(p_sum)} (existe={ok_sum})")
        st.caption(f"Fuente detail:  {os.path.basename(p_det)} (existe={ok_det})")
        st.caption(f"Fuente sens.:   {os.path.basename(p_sens)} (existe={ok_sens})")

        if not ok_sum and not ok_det:
            show_missing_file(p_sum)

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
            if "ROI" in df.columns and "roi" not in df.columns:
                df = df.rename(columns={"ROI":"roi"})
            show_columns(df)
            st.dataframe(df_fmt_pct(df_fmt_money(df, ["costo_incentivo","ingreso_incremental","budget_usado"], moneda, usdclp), ["roi"]),
                         use_container_width=True, height=360)

        if A3X is not None and not A3X.empty:
            st.markdown("*Sensibilidades (ROI mínimo)*")
            show_columns(A3X)
            st.dataframe(df_fmt_pct(df_fmt_money(A3X, ["budget","costo","ingreso_inc"], moneda, usdclp), ["ROI"]),
                         use_container_width=True, height=300)

    except Exception:
        st.error("Error en Arista 3:\n\n" + traceback.format_exc())

# ==============================
# A4 – Capital/Provisiones
# ==============================
with tabs[3]:
    st.header("Arista 4 – Capital/Provisiones")
    st.info(f"Escenario activo: *{escenario}*")

    try:
        A4P, p_port, ok_port = load_csv_strict(bundle_dir, "a4_port", suf)
        A4S, p_seg,  ok_seg  = load_csv_strict(bundle_dir, "a4_seg",  suf)
        A4D, p_det,  ok_det  = load_csv_strict(bundle_dir, "a4_det",  suf)

        st.caption(f"Fuente portfolio: {os.path.basename(p_port)} (existe={ok_port})")
        st.caption(f"Fuente segment:   {os.path.basename(p_seg)} (existe={ok_seg})")
        st.caption(f"Fuente detail:    {os.path.basename(p_det)} (existe={ok_det})")

        if not ok_port and not ok_det:
            show_missing_file(p_port)

        ead_a = first_value(A4P, ["EAD_base","EAD"])
        ead_b = first_value(A4P, ["EAD_opt","EAD"])
        rwa_a = first_value(A4P, ["RWA_base","RWA"])
        rwa_b = first_value(A4P, ["RWA_opt","RWA"])
        k_a   = first_value(A4P, ["K_base","K"])
        k_b   = first_value(A4P, ["K_opt","K"])
        el_a  = first_value(A4P, ["EL_base","EL"])
        el_b  = first_value(A4P, ["EL_opt","EL"])

        if (A4P is None or A4P.empty) or all(v is None for v in [ead_a,ead_b,rwa_a,rwa_b,k_a,k_b,el_a,el_b]):
            ead_a = ead_a or sum_col(A4D, ["EAD_base"])
            ead_b = ead_b or sum_col(A4D, ["EAD_opt"])
            rwa_a = rwa_a or sum_col(A4D, ["RWA_base"])
            rwa_b = rwa_b or sum_col(A4D, ["RWA_opt"])
            k_a   = k_a   or sum_col(A4D, ["K_base"])
            k_b   = k_b   or sum_col(A4D, ["K_opt"])
            el_a  = el_a  or sum_col(A4D, ["EL_base"])
            el_b  = el_b  or sum_col(A4D, ["EL_opt"])

        kpi_row_money("EAD",                ead_a, ead_b, moneda, usdclp)
        kpi_row_money("RWA (proxy Basel)",  rwa_a, rwa_b, moneda, usdclp)
        kpi_row_money("Capital (K)",        k_a,   k_b,   moneda, usdclp)
        kpi_row_money("Provisiones (≈ EL)", el_a,  el_b,  moneda, usdclp)

        if A4D is not None and not A4D.empty:
            st.markdown("*Detalle por cliente*")
            cols = ["EAD_base","EAD_opt","RWA_base","RWA_opt","K_base","K_opt","EL_base","EL_opt","Util_base","Util_opt"]
            show_columns(A4D)
            st.dataframe(df_fmt_money(A4D, cols, moneda, usdclp), use_container_width=True, height=360)
        if A4S is not None and not A4S.empty:
            st.markdown("*Resumen por segmento*")
            cols = ["EAD_base","EAD_opt","RWA_base","RWA_opt","K_base","K_opt","EL_base","EL_opt","Util_base","Util_opt"]
            show_columns(A4S)
            st.dataframe(df_fmt_money(A4S, cols, moneda, usdclp), use_container_width=True, height=320)

    except Exception:
        st.error("Error en Arista 4:\n\n" + traceback.format_exc())

# ==============================
# Guardrails (SIN sufijo siempre)
# ==============================
with tabs[4]:
    st.header("Guardrails (Resguardos)")
    st.caption("Los guardrails se exportan *sin sufijo* desde el Notebook (Celdas 15–16).")
    try:
        GP, p_gp, ok_gp = load_csv_strict(bundle_dir, "gr_port", "")
        GS, p_gs, ok_gs = load_csv_strict(bundle_dir, "gr_seg",  "")
        GE, p_ge, ok_ge = load_csv_strict(bundle_dir, "gr_eval", "")

        st.caption(f"Guardrails portfolio: {os.path.basename(p_gp)} (existe={ok_gp})")
        st.caption(f"Guardrails segmento:  {os.path.basename(p_gs)} (existe={ok_gs})")
        st.caption(f"Guardrails evaluación: {os.path.basename(p_ge)} (existe={ok_ge})")

        if not ok_gp and not ok_gs and not ok_ge:
            st.warning("No se encontraron archivos de guardrails en el bundle. Ejecuta Celdas 15–16 y exporta al bundle.")
        else:
            if GP is not None and not GP.empty:
                st.subheader("Catálogo – Portafolio")
                df = GP.copy()
                for c in ["umbral","observado_actual","observado_optimizado"]:
                    if c in df.columns: df[c] = df[c].apply(fmt_pct)
                show_columns(df)
                st.dataframe(df, use_container_width=True, height=280)
            if GS is not None and not GS.empty:
                st.subheader("Catálogo – Segmento")
                df = GS.copy()
                if "observado" in df.columns:
                    df["observado"] = df["observado"].apply(fmt_pct)
                show_columns(df)
                st.dataframe(df, use_container_width=True, height=280)

            st.markdown("---")
            st.subheader("Evaluación automática (Celda 16)")
            if GE is None or GE.empty:
                st.info("No se encontró *guardrails_eval_portfolio.csv*.")
            else:
                show_columns(GE)
                st.dataframe(GE, use_container_width=True, height=320)

    except Exception:
        st.error("Error en Guardrails:\n\n" + traceback.format_exc())

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (IFRS9 + Basel proxy). Listo para piloto IT.")
