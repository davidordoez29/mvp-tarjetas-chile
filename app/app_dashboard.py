# app/app_dashboard.py — MVP Bancario (4 Aristas) v3.0
# Escenarios: Conservador / Potenciado  |  Moneda: CLP / USD
# IFRS9 + Basel (proxy) | KPIs robustos con fallback desde detail | Pitches ejecutivos

import os, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ================
# Config archivos
# ================
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

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR","").strip(),
    "/content/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
    "/content/out",
    "./out",
]

SCENARIOS = {"Conservador": "", "Potenciado": "_agresivo"}

# ================
# Parseo/formatos
# ================
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

# ===================
# Bundle I/O helpers
# ===================
def _exists(bundle_dir, key, suf):
    return os.path.exists(os.path.join(bundle_dir, REQ[key].format(S=suf)))

def autodetect_bundle(suf):
    for d in CANDIDATE_DIRS:
        if d and os.path.isdir(d):
            if any(_exists(d,k,suf) for k in ["a1_port","a2_port","a4_port"]):
                return d
            # si no había de ese sufijo, aceptar de todos modos si hay archivos
            if any(os.path.exists(os.path.join(d, REQ[k].format(S=s))) for k in ["a1_port","a2_port","a4_port"] for s in ["","_agresivo"]):
                return d
    return None

def scenario_available(bundle_dir):
    has_cons = any(_exists(bundle_dir,k,"") for k in ["a1_port","a2_port","a4_port"])
    has_aggr = any(_exists(bundle_dir,k,"_agresivo") for k in ["a1_port","a2_port","a4_port"])
    return has_cons, has_aggr

def load_csv(bundle_dir, key, suf):
    p = os.path.join(bundle_dir, REQ[key].format(S=suf))
    if not os.path.exists(p): return None
    try:
        return pd.read_csv(p)
    except:
        return None

def first_value(df, candidates):
    if df is None or df.empty: return None
    for c in candidates:
        if c in df.columns:
            try: return parse_num_any(df[c].iloc[0])
            except: continue
    return None

def sum_col(df, candidates):
    if df is None or df.empty: return None
    for c in candidates:
        if c in df.columns:
            return df[c].map(parse_num_any).sum()
    return None

# ===================
# App layout
# ===================
st.set_page_config(page_title="MVP Bancario — 4 Aristas", layout="wide")

st.sidebar.title("⚙️ Configuración")
escenario_ui = st.sidebar.radio("Escenario", list(SCENARIOS.keys()), horizontal=True, index=1)  # default Potenciado
suf_ui = SCENARIOS[escenario_ui]

moneda = st.sidebar.radio("Moneda", ["CLP","USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))

default_dir = autodetect_bundle(suf_ui) or "/content/out/dashboard_bundle"
bundle_dir = st.sidebar.text_input("📦 Ruta del bundle", value=(default_dir or ""), help="Ej: /content/out/dashboard_bundle").strip() or default_dir

st.title("📊 MVP Bancario — Optimización en 4 Aristas")
st.caption("Modelo matemático con cumplimiento (IFRS9 + Basel proxy). *Actual vs Optimizado* por arista, escenario y moneda.")

if not bundle_dir or not os.path.isdir(bundle_dir):
    st.error("No encuentro el bundle. Revisa la ruta o genera el paquete en el Notebook.")
    st.stop()

has_cons, has_aggr = scenario_available(bundle_dir)
if suf_ui == "" and not has_cons and has_aggr:
    st.info("⚠️ No hay archivos Conservador; usando *Potenciado* disponibles.")
    suf_ui = "_agresivo"
elif suf_ui == "_agresivo" and not has_aggr and has_cons:
    st.info("⚠️ No hay archivos Potenciado; usando *Conservador* disponibles.")
    suf_ui = ""

# Diagnóstico rápido
with st.expander("🔎 Diagnóstico del bundle", expanded=False):
    rows = []
    for key, pat in REQ.items():
        p = os.path.join(bundle_dir, pat.format(S=suf_ui))
        rows.append({"archivo": os.path.basename(p), "existe": os.path.exists(p)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=240)

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
    st.markdown("*Objetivo:* Mantener *EAD* total estable, recomponiendo el mix hacia menor *PD* para *reducir EL* sin frenar el negocio.")
    st.markdown("""
*KPIs Clave*
- *EAD*: Exposición en riesgo (total estable; cambia la composición por segmento).  
- *Pérdida Esperada (EL): PD × LGD × EAD (IFRS9 12m). Buscamos *↓**.  
- *Interés Devengado Bruto*: APR × EAD (proxy del margen antes de pérdidas).  
- *Utilidad: Interés – EL – (costos aplicables). Buscamos *↑**.  
- *PD ponderado: Promedio ponderado por EAD; buscaremos *↓**.
    """)
    st.success("*Pitch:* Reasignamos exposición desde clientes/segmentos más riesgosos a más sanos manteniendo producción, reduciendo EL y mejorando utilidad sin comprometer el crecimiento.")

    A1P = load_csv(bundle_dir, "a1_port", suf_ui)
    A1D = load_csv(bundle_dir, "a1_det",  suf_ui)

    inc_a = first_value(A1P, ["Interes_devengado_bruto_actual","Ingreso_actual","ingreso_base"])
    inc_b = first_value(A1P, ["Interes_devengado_bruto_optimizado","Ingreso_optimizado","ingreso_final","ingreso_opt"])
    util_a = first_value(A1P, ["Utilidad_actual","utilidad_base","Util_actual"])
    util_b = first_value(A1P, ["Utilidad_optimizada","utilidad_final","utilidad_opt","Util_opt"])
    el_a  = first_value(A1P, ["EL_actual","EL_base"])
    el_b  = first_value(A1P, ["EL_optimizado","EL_final","EL_opt"])
    ead_a = first_value(A1P, ["EAD_actual","EAD_base","EAD"])
    ead_b = first_value(A1P, ["EAD_optimizado","EAD_final","EAD_opt","EAD"])

    if A1P is None or A1P.empty or all(v is None for v in [inc_a,inc_b,util_a,util_b,el_a,el_b,ead_a,ead_b]):
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

    # PD ponderado si viene en portfolio
    pd_a = first_value(A1P, ["PD_pond_actual","PDpond_actual"])
    pd_b = first_value(A1P, ["PD_pond_optimizado","PDpond_optimizado"])
    if (pd_a is not None) or (pd_b is not None):
        kpi_row_pct("PD Ponderado", (pd_a*100 if pd_a is not None else None), (pd_b*100 if pd_b is not None else None))

    if A1D is not None and not A1D.empty:
        st.markdown("*Detalle por cliente*")
        money = ["e_base","e_final","income_base","income_final","util_base","util_final","EL_base","EL_final","EAD_base","EAD_opt"]
        pct   = ["pd_base","pd_final","lgd_base","lgd_final","PD_12m","LGD_adj"]
        st.dataframe(df_fmt_pct(df_fmt_money(A1D, money, moneda, usdclp), pct), use_container_width=True, height=360)

# ==============================
# Arista 2 – Yield/Pricing
# ==============================
with tabs[1]:
    st.header("Arista 2 – Yield/Pricing")
    st.markdown("*Objetivo:* Encontrar el *APR óptimo* por segmento maximizando *Utilidad* balanceando precio y volumen (elasticidad).")
    st.markdown("""
*KPIs*
- *Utilidad Total* (↑).  
- *Interés Bruto Total* = APR × EAD_out.  
- *EAD_in / EAD_out*: Volumen antes/después por efecto de precio.  
- *APR óptimo por segmento* respetando bandas/caps.
    """)
    st.success("*Pitch:* Movemos la perilla del precio para capturar el punto de máxima utilidad; si sube mucho, cae el volumen; si baja de más, no cubre riesgo/costo. El equilibrio correcto maximiza el resultado.")

    A2P = load_csv(bundle_dir, "a2_port", suf_ui)
    A2S = load_csv(bundle_dir, "a2_seg",  suf_ui)
    A2D = load_csv(bundle_dir, "a2_det",  suf_ui)

    util_a = first_value(A2P, ["utilidad_base","Utilidad_base"])
    util_b = first_value(A2P, ["utilidad_opt","Utilidad_opt"])
    inc_a  = first_value(A2P, ["ingreso_base","Ingreso_base"])
    inc_b  = first_value(A2P, ["ingreso_opt","Ingreso_opt"])
    e_in   = first_value(A2P, ["EAD_in","EAD_base","EAD"])
    e_out  = first_value(A2P, ["EAD_out","EAD_opt","EAD"])

    if A2P is None or A2P.empty or all(v is None for v in [util_a,util_b,inc_a,inc_b,e_in,e_out]):
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
        st.dataframe(df_fmt_money(A2S, ["EAD_in","EAD_out","ingreso_opt","utilidad_opt"], moneda, usdclp),
                     use_container_width=True, height=340)
    if A2D is not None and not A2D.empty:
        st.markdown("*Detalle por cliente (pricing)*")
        st.dataframe(df_fmt_money(A2D, ["ead_in","e_out","income_opt","EL_opt","COF_opt","util_opt","income_base","util_base"], moneda, usdclp),
                     use_container_width=True, height=360)

# ==============================
# Arista 3 – Incentivos
# ==============================
with tabs[2]:
    st.header("Arista 3 – Incentivos")
    st.markdown("*Objetivo:* Asignar incentivos sólo donde *ROI>0, bajo un **presupuesto* global. Aseguramos impacto positivo neto.")
    st.markdown("""
*KPIs*
- *Costo de incentivos*.  
- *Ingreso incremental (uplift atribuible)*.  
- *ROI* = ingreso_inc / costo (↑).  
- *Sensibilidades* por umbral de ROI y presupuesto.
    """)
    st.success("*Pitch:* Fertilizamos solo donde responde: cada peso invertido genera retorno multiplicado. Sin desperdicios generalizados.")

    A3S = load_csv(bundle_dir, "a3_sum",  suf_ui)
    A3D = load_csv(bundle_dir, "a3_det",  suf_ui)
    A3X = load_csv(bundle_dir, "a3_sens", suf_ui)

    if (A3S is None or A3S.empty) and (A3D is None or A3D.empty):
        st.warning("No hay datos de incentivos para este escenario.")
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
            st.dataframe(df_fmt_pct(df_fmt_money(df, ["costo_incentivo","ingreso_incremental","budget_usado"], moneda, usdclp), ["roi"]),
                         use_container_width=True, height=360)

        if A3X is not None and not A3X.empty:
            st.markdown("*Sensibilidades (ROI mínimo)*")
            st.dataframe(df_fmt_pct(df_fmt_money(A3X, ["budget","costo","ingreso_inc"], moneda, usdclp), ["ROI"]),
                         use_container_width=True, height=300)

# ==============================
# Arista 4 – Capital/Provisiones
# ==============================
with tabs[3]:
    st.header("Arista 4 – Capital/Provisiones")
    st.markdown("*Objetivo:* Hacer más eficiente el *consumo de capital* (RWA, K) y reducir *Provisiones (≈ EL)* manteniendo la calidad.")
    st.markdown("""
*KPIs (Portafolio)*
- *EAD* (base vs opt).  
- *RWA* = RW × EAD (proxy Basel).  
- *Capital (K)* = K_ratio × RWA.  
- *Provisiones (≈ EL)*.  
    """)
    st.success("*Pitch:* Reorganizamos el capital protegido: seguimos cubiertos sin exceso inmovilizado. Se libera capacidad para crecer con control de riesgo.")

    A4P = load_csv(bundle_dir, "a4_port", suf_ui)
    A4D = load_csv(bundle_dir, "a4_det",  suf_ui)
    A4S = load_csv(bundle_dir, "a4_seg",  suf_ui)

    ead_a = first_value(A4P, ["EAD_base","EAD"])
    ead_b = first_value(A4P, ["EAD_opt","EAD"])
    rwa_a = first_value(A4P, ["RWA_base","RWA"])
    rwa_b = first_value(A4P, ["RWA_opt","RWA"])
    k_a   = first_value(A4P, ["K_base","K"])
    k_b   = first_value(A4P, ["K_opt","K"])
    el_a  = first_value(A4P, ["EL_base","EL"])
    el_b  = first_value(A4P, ["EL_opt","EL"])

    if A4P is None or A4P.empty or all(v is None for v in [ead_a,ead_b,rwa_a,rwa_b,k_a,k_b,el_a,el_b]):
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
        st.dataframe(df_fmt_money(A4D, cols, moneda, usdclp), use_container_width=True, height=360)
    if A4S is not None and not A4S.empty:
        st.markdown("*Resumen por segmento*")
        cols = ["EAD_base","EAD_opt","RWA_base","RWA_opt","K_base","K_opt","EL_base","EL_opt","Util_base","Util_opt"]
        st.dataframe(df_fmt_money(A4S, cols, moneda, usdclp), use_container_width=True, height=320)

# ==============================
# Guardrails
# ==============================
with tabs[4]:
    st.header("Guardrails (Resguardos)")
    st.markdown("Catálogo y *evaluación automática* (Notebook Celdas 15–16).")
    GP = load_csv(bundle_dir, "gr_port", "")
    GS = load_csv(bundle_dir, "gr_seg",  "")
    GE = load_csv(bundle_dir, "gr_eval", "")

    if GP is None and GS is None and (GE is None or GE.empty):
        st.info("No se encontraron archivos de guardrails. Ejecuta Celdas 15–16 y exporta al bundle.")
    else:
        if GP is not None and not GP.empty:
            st.subheader("Catálogo – Portafolio")
            df = GP.copy()
            for c in ["umbral","observado_actual","observado_optimizado"]:
                if c in df.columns: df[c] = df[c].apply(fmt_pct)
            st.dataframe(df, use_container_width=True, height=280)
        if GS is not None and not GS.empty:
            st.subheader("Catálogo – Segmento")
            df = GS.copy()
            if "observado" in df.columns:
                df["observado"] = df["observado"].apply(fmt_pct)
            st.dataframe(df, use_container_width=True, height=280)

        st.markdown("---")
        st.subheader("Evaluación automática (Celda 16)")
        if GE is None or GE.empty:
            st.info("No se encontró *guardrails_eval_portfolio.csv*.")
        else:
            st.dataframe(GE, use_container_width=True, height=320)

# ================
# Footer
# ================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (IFRS9 + Basel proxy). Listo para piloto IT.")
