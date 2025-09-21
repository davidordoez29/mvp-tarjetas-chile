# app/app_dashboard.py — MVP Bancario (Conservador/Potenciado + CLP/USD)
import os, re, math
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ==========================
# Config inicial
# ==========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

REQ_FILES = {
    # A1
    "def_port": "default_portfolio{S}.csv",
    "def_seg":  "default_segment{S}.csv",
    "def_det":  "default_detail{S}.csv",
    # A2
    "yld_port": "yield_portfolio{S}.csv",
    "yld_seg":  "yield_segment{S}.csv",
    "yld_det":  "yield_detail{S}.csv",
    "yld_curv": "yield_curve_segment{S}.csv",
    # A3
    "inc_det":  "incentives_detail{S}.csv",
    "inc_sum":  "incentives_diag_summary{S}.csv",
    "inc_sens": "incentives_sensitivity{S}.csv",
    # A4
    "cap_port": "capital_portfolio{S}.csv",
    "cap_seg":  "capital_segment{S}.csv",
    "cap_det":  "capital_detail{S}.csv",
    # Guardrails (sin sufijo)
    "guard_port": "guardrails_portfolio.csv",
    "guard_seg":  "guardrails_segment.csv",
    "guard_eval":"guardrails_eval_portfolio.csv",
}

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    "/content/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
]

_SCEN = {"Conservador": "", "Potenciado": "_agresivo"}

# ==========================
# Utils formato
# ==========================
_num_like = re.compile(r"^-?\d+(\.\d+)?$")

def _to_float_or_nan(v):
    if v is None: return np.nan
    if isinstance(v, (int,float)): return float(v)
    if isinstance(v, str):
        s = v.strip().replace("%","").replace(",","." )
        try: return float(s)
        except: return np.nan
    return np.nan

def fmt_money(val, moneda, usdclp):
    if isinstance(val, str):
        s = val.strip()
        if not s: return "—"
        return s
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    x = float(val)
    if moneda=="USD":
        x = x/float(usdclp) if usdclp else np.nan
    if np.isnan(x): return "—"
    neg = x<0; x=abs(x)
    ent=int(x); dec=int(round((x-ent)*100))
    if dec==100: ent+=1; dec=0
    ent_str = f"{ent:,}".replace(",", ".")
    return f"-{ent_str},{dec:02d}" if neg else f"{ent_str},{dec:02d}"

def fmt_pct(val):
    x = _to_float_or_nan(val)
    if np.isnan(x): return "—"
    return f"{x:.2f}%".replace(".", ",")

def kpi_row_money(label, actual, opt, moneda, usdclp, help_text=""):
    c1,c2,c3 = st.columns([1.0,1.0,0.6])
    with c1:
        st.metric(f"{label} – Actual", fmt_money(actual, moneda, usdclp))
        if help_text: st.caption(help_text)
    with c2:
        st.metric(f"{label} – Optimizado", fmt_money(opt, moneda, usdclp))
    with c3:
        if actual not in [None,0,np.nan]:
            try:
                var = (float(opt)-float(actual))/float(actual)*100.0
                st.metric("VAR %", fmt_pct(var))
            except:
                st.metric("VAR %", "—")
        else:
            st.metric("VAR %", "—")

def kpi_row_pct(label, actual, opt, help_text=""):
    c1,c2,c3 = st.columns([1.0,1.0,0.6])
    with c1:
        st.metric(f"{label} – Actual", fmt_pct(actual))
        if help_text: st.caption(help_text)
    with c2:
        st.metric(f"{label} – Optimizado", fmt_pct(opt))
    with c3:
        try:
            a=_to_float_or_nan(actual); o=_to_float_or_nan(opt)
            var=(o-a) if not (a==0) else np.nan
            st.metric("Δ p.p.", fmt_pct(var))
        except:
            st.metric("Δ p.p.", "—")

def autodetect_bundle():
    for d in CANDIDATE_DIRS:
        if d and os.path.isdir(d):
            return d
    return ""

def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Error leyendo {os.path.basename(path)}: {e}")
        return None

# ==========================
# Sidebar
# ==========================
st.sidebar.title("⚙️ Configuración")
bundle_dir = st.sidebar.text_input("📦 Ruta del bundle", value=autodetect_bundle()).strip()
if not bundle_dir:
    st.error("No encuentro el bundle. Genera el paquete en el notebook (Celda 22) y vuelve a cargar.")
    st.stop()

escenario = st.sidebar.radio("Escenario", list(_SCEN.keys()), index=0, horizontal=True)
suf = _SCEN[escenario]

moneda = st.sidebar.radio("Moneda", ["CLP","USD"], horizontal=True, index=0)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))

st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Modelo matemático aplicado sobre un portafolio simulado. Comparación Actual vs Optimizado por arista.")

tabs = st.tabs([
    "Arista 1 – Default/Impago",
    "Arista 2 – Yield/Pricing",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital/Provisiones",
    "Guardrails (Resguardos)"
])

# Helpers lectura por clave
def _p(fname):
    return Path(bundle_dir) / fname

def get_df(key, mandatory=True):
    fname = REQ_FILES[key].format(S=suf) if "{S}" in REQ_FILES[key] else REQ_FILES[key]
    path = _p(fname)
    if not path.exists():
        if mandatory:
            st.error(f"Falta {fname} en el bundle.")
        return None
    return read_csv_safe(path)

# ================
# Arista 1 — Default/Impago
# ================
with tabs[0]:
    st.header("Arista 1 – Default/Impago")
    st.markdown("*¿Qué resolvemos?* Disminuimos la pérdida esperada (EL) reasignando exposición hacia segmentos más sanos, *sin* frenar el crecimiento.")
    st.markdown("*KPIs:* EAD (exposición), *EL = PD × LGD × EAD, **Ingreso financiero* (interés devengado) y *Utilidad* (Ingreso − EL − costos). El *PD ponderado* se computa con peso EAD.")
    port = get_df("def_port")
    det  = get_df("def_det")
    if port is not None and not port.empty:
        def g(c): return float(port[c].iloc[0]) if c in port.columns else np.nan
        kpi_row_money("EAD", g("EAD_actual"), g("EAD_optimizado"), moneda, usdclp, "Exposición total (EAD). Ideal: estable o ↑ levemente.")
        kpi_row_money("EL (Pérdida Esperada)", g("EL_actual"), g("EL_optimizado"), moneda, usdclp, "Menor EL es deseable a igual crecimiento.")
        kpi_row_money("Utilidad", g("Utilidad_actual"), g("Utilidad_optimizada"), moneda, usdclp, "Utilidad = Ingreso − EL − costos.")
        if "PD_pond_actual" in port.columns and "PD_pond_optimizado" in port.columns:
            kpi_row_pct("PD ponderado (p.p.)", float(port["PD_pond_actual"].iloc[0])*100.0, float(port["PD_pond_optimizado"].iloc[0])*100.0,
                        "Promedio de PD ponderado por EAD. Debe bajar.")
    if det is not None and not det.empty:
        # formateo de columnas monetarias y %
        dfv = det.copy()
        for c in ["r_base","r_final","income_base","income_final","EL_base","EL_final","e_base","e_out"]:
            if c in dfv.columns:
                dfv[c] = dfv[c].apply(lambda x: fmt_money(x, moneda, usdclp))
        st.markdown("*Detalle por cliente (formateado)*")
        st.dataframe(dfv.head(500), use_container_width=True)

# ================
# Arista 2 — Yield/Pricing
# ================
with tabs[1]:
    st.header("Arista 2 – Yield/Pricing")
    st.markdown("*¿Qué resolvemos?* Encontramos la tasa (APR) que maximiza utilidad, equilibrando precio y volumen (elasticidad).")
    st.markdown("*KPIs:* *Ingreso total* (interés tras elasticidad), *Utilidad total* y *EAD_in → EAD_out* (volumen resp).")
    port = get_df("yld_port")
    det  = get_df("yld_det")
    if port is not None and not port.empty:
        def g(c): return float(port[c].iloc[0]) if c in port.columns else np.nan
        kpi_row_money("Ingreso total", g("ingreso_base"), g("ingreso_opt"), moneda, usdclp, "Ingreso tras respuesta de volumen.")
        kpi_row_money("Utilidad total", g("utilidad_base"), g("utilidad_opt"), moneda, usdclp, "Tiene en cuenta EL y costos.")
        if "EAD_in" in port.columns and "EAD_out" in port.columns:
            c1,c2,c3 = st.columns([1.0,1.0,0.6])
            with c1: st.metric("EAD in", fmt_money(port["EAD_in"].iloc[0], moneda, usdclp))
            with c2: st.metric("EAD out", fmt_money(port["EAD_out"].iloc[0], moneda, usdclp))
            with c3:
                try:
                    var = (float(port["EAD_out"].iloc[0])-float(port["EAD_in"].iloc[0]))/float(port["EAD_in"].iloc[0])*100
                    st.metric("VAR %", fmt_pct(var))
                except:
                    st.metric("VAR %", "—")
    if det is not None and not det.empty:
        dfv = det.copy()
        for c in ["r_base","r_opt","income_base","income_opt","ead_in","e_out"]:
            if c in dfv.columns:
                dfv[c] = dfv[c].apply(lambda x: fmt_money(x, moneda, usdclp))
        st.dataframe(dfv.head(500), use_container_width=True)

# ================
# Arista 3 — Incentivos
# ================
with tabs[2]:
    st.header("Arista 3 – Incentivos")
    st.markdown("*¿Qué resolvemos?* Invertimos incentivos *sólo* donde el *ROI* es positivo, con presupuesto y caps por segmento.")
    st.markdown("*KPIs:* *Costo incentivos, **Ingreso incremental* y *ROI* (Ingreso/Costo).")

    det = get_df("inc_det", mandatory=False)
    summ = get_df("inc_sum", mandatory=False)
    sens = get_df("inc_sens", mandatory=False)

    if det is None and summ is None:
        st.info("No hay archivos de incentivos en el bundle para este escenario.")
    else:
        costo = float(pd.to_numeric(det["costo_incentivo"], errors="coerce").fillna(0).sum()) if det is not None and "costo_incentivo" in det.columns else 0.0
        ingr  = float(pd.to_numeric(det["ingreso_incremental"], errors="coerce").fillna(0).sum()) if det is not None and "ingreso_incremental" in det.columns else 0.0
        roi   = (ingr/costo*100.0) if costo>0 else np.nan
        c1,c2,c3 = st.columns(3)
        c1.metric("Costo incentivos", fmt_money(costo, moneda, usdclp))
        c2.metric("Ingreso incremental", fmt_money(ingr, moneda, usdclp))
        c3.metric("ROI", fmt_pct(roi))

        if det is not None and not det.empty:
            dfv = det.copy()
            for c in ["e_out","r_opt","costo_incentivo","ingreso_incremental","roi"]:
                if c in dfv.columns:
                    if c=="roi":
                        dfv[c] = dfv[c].apply(lambda x: fmt_pct(float(x)*100.0 if pd.notna(x) else np.nan))
                    else:
                        dfv[c] = dfv[c].apply(lambda x: fmt_money(x, moneda, usdclp))
            st.dataframe(dfv.head(500), use_container_width=True)

        if sens is not None and not sens.empty:
            st.markdown("*Sensibilidad ROI vs umbral*")
            st.dataframe(sens, use_container_width=True)

# ================
# Arista 4 — Capital/Provisiones
# ================
with tabs[3]:
    st.header("Arista 4 – Capital/Provisiones")
    st.markdown("*¿Qué resolvemos?* Hacemos más eficiente el capital requerido (RWA, K) y provisiones (≈EL).")
    st.markdown("*KPIs:* *EAD, **RWA, **K (ratio)* y *Provisiones (EL)*.")

    port = get_df("cap_port")
    if port is not None and not port.empty:
        def g(c): return float(port[c].iloc[0]) if c in port.columns else np.nan
        kpi_row_money("EAD", g("EAD_base"), g("EAD_opt"), moneda, usdclp)
        kpi_row_money("RWA", g("RWA_base"), g("RWA_opt"), moneda, usdclp)
        kpi_row_money("K (capital)", g("K_base"), g("K_opt"), moneda, usdclp)
        kpi_row_money("Provisiones (EL)", g("EL_base"), g("EL_opt"), moneda, usdclp)

# ================
# Guardrails
# ================
with tabs[4]:
    st.header("Guardrails (Resguardos)")
    gp = _p(REQ_FILES["guard_port"])
    gs = _p(REQ_FILES["guard_seg"])
    ge = _p(REQ_FILES["guard_eval"])
    if not gp.exists() and not gs.exists() and not ge.exists():
        st.info("No encuentro guardrails en el bundle. Asegura Celda 15, 16 y 22.")
    else:
        if gp.exists():
            st.subheader("Catálogo — Portafolio")
            df = read_csv_safe(gp)
            if df is not None:
                dfv = df.copy()
                for c in ["umbral","observado_actual","observado_optimizado"]:
                    if c in dfv.columns:
                        dfv[c] = dfv[c].apply(lambda x: fmt_pct(x))
                st.dataframe(dfv, use_container_width=True)
        if gs.exists():
            st.subheader("Catálogo — Segmento")
            df = read_csv_safe(gs)
            if df is not None:
                st.dataframe(df, use_container_width=True)
        if ge.exists():
            st.subheader("Evaluación (checks)")
            df = read_csv_safe(ge)
            if df is not None:
                st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas integradas).")
