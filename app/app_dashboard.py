# app/app_dashboard.py — MVP Bancario (4 Aristas) con escenarios y carga robusta

import os, json, math, re
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# ==========================
# Utilidades de formato
# ==========================
_num_like = re.compile(r"^-?\d+(\.\d+)?$")

def _to_float_or_nan(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return np.nan
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        s = v.strip().replace("%","").replace(",", ".")
        if _num_like.match(s):
            try:
                return float(s)
            except Exception:
                return np.nan
        return np.nan
    return np.nan

def _to_display_currency(val: float, target: str, usdclp: float) -> float:
    if pd.isna(val): return np.nan
    if target.upper() == "USD":
        return float(val) / float(usdclp) if usdclp else np.nan
    return float(val)

def fmt_money_val(val, target: str, usdclp: float) -> str:
    if isinstance(val, str):
        v = val.strip()
        if v == "" or v.upper() == "N/A": return "—"
        return v
    if val is None or (isinstance(val, float) and math.isnan(val)): return "—"
    x = _to_display_currency(float(val), target, usdclp)
    if x is None or (isinstance(x, float) and math.isnan(x)): return "—"
    neg = x < 0; x = abs(x)
    ent = int(x); dec = int(round((x - ent) * 100))
    if dec == 100: ent += 1; dec = 0
    ent_str = f"{ent:,}".replace(",", ".")
    return f"-{ent_str},{dec:02d}" if neg else f"{ent_str},{dec:02d}"

def fmt_pct_val(val):
    if isinstance(val, str):
        s = val.strip()
        if s.endswith("%"):
            return s.replace(".", ",")
        if not _num_like.match(s.replace(",", ".")):
            return s
    x = _to_float_or_nan(val)
    if np.isnan(x):
        return "—" if (val is None or (isinstance(val, float) and math.isnan(val))) else str(val)
    return f"{x:.2f}%".replace(".", ",")

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

# ==========================
# Archivos requeridos (por escenario)
# ==========================
REQ_FILES_BASE = {
    "def_port": "default_portfolio.csv",
    "def_seg":  "default_segment.csv",
    "def_det":  "default_detail.csv",
    "yld_port": "yield_portfolio.csv",
    "yld_seg":  "yield_segment.csv",
    "yld_det":  "yield_detail.csv",
    "yld_curv": "yield_curve_segment.csv",
    "inc_det":  "incentives_detail.csv",
    "inc_sum":  "incentives_diag_summary.csv",
    "inc_sens": "incentives_sensitivity.csv",
    "cap_port": "capital_portfolio.csv",
    "cap_seg":  "capital_segment.csv",
    "cap_det":  "capital_detail.csv",
    # Comunes a ambos escenarios (sin sufijo)
    "guard_port": "guardrails_portfolio.csv",
    "guard_seg":  "guardrails_segment.csv",
    "kpis":       "kpis_unificados.csv",
}

SUFFIX = {"Conservador": "", "Potenciado": "_agresivo"}

def files_for_scenario(scenario: str) -> dict:
    suf = SUFFIX.get(scenario, "")
    files = {}
    for k, base in REQ_FILES_BASE.items():
        if k in {"guard_port", "guard_seg", "kpis"}:
            files[k] = base
        else:
            if suf:
                stem, ext = base.rsplit(".", 1)
                files[k] = f"{stem}{suf}.{ext}"
            else:
                files[k] = base
    return files

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    "/content/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
]

# ==========================
# Carga de bundle (escenario-aware)
# ==========================
def _dir_ok(d: str) -> bool:
    try:
        if not d or not os.path.isdir(d): 
            return False
        req_cons = files_for_scenario("Conservador")
        hits_cons = sum(os.path.exists(os.path.join(d, v)) for v in req_cons.values())
        req_pot = files_for_scenario("Potenciado")
        hits_pot = sum(os.path.exists(os.path.join(d, v)) for v in req_pot.values())
        return max(hits_cons, hits_pot) >= 6
    except Exception:
        return False

def autodetect_bundle() -> str | None:
    for d in CANDIDATE_DIRS:
        if _dir_ok(d):
            return d
    return None

def load_bundle(bundle_dir: str, scenario: str):
    req = files_for_scenario(scenario)
    dfs, missing = {}, []
    for key, fname in req.items():
        path = os.path.join(bundle_dir, fname)
        if not os.path.exists(path):
            missing.append(fname); dfs[key] = None; continue
        try:
            dfs[key] = pd.read_csv(path)
        except Exception as e:
            missing.append(f"{fname} (error: {e})")
            dfs[key] = None
    return dfs, missing

# ==========================
# App
# ==========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

st.sidebar.title("⚙️ Configuración")
scenario = st.sidebar.radio("Escenario", ["Conservador", "Potenciado"], horizontal=True)

default_dir = autodetect_bundle()
bundle_dir = st.sidebar.text_input(
    "📦 Ruta del bundle",
    value=(default_dir or ""),
    help="Ej: /content/out/dashboard_bundle"
).strip() or default_dir

if not bundle_dir:
    st.error("No encuentro el bundle. Genera el paquete en el notebook y vuelve a cargar.")
    st.stop()

dfs, missing = load_bundle(bundle_dir, scenario)
if missing:
    st.warning(f"Faltan archivos en el bundle (escenario *{scenario}*):\n- " + "\n- ".join(missing))

moneda = st.sidebar.radio("Moneda a visualizar", ["CLP", "USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))

st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Modelo matemático aplicado sobre un portafolio simulado. Comparación Actual vs Optimizado. Selecciona el *escenario* en la barra lateral.")

tabs = st.tabs([
    "Arista 1 – Default/Impago",
    "Arista 2 – Yield/Pricing",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital/Provisiones",
    "Guardrails (Resguardos)"
])

# ================
# Arista 1 – Default / Impago
# ================
with tabs[0]:
    st.header("Arista 1 – Default/Impago")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Reducimos la pérdida esperada (EL) reasignando la exposición a segmentos menos riesgosos, sin frenar el negocio.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - EAD: Exposición en riesgo.  
    - EL: Pérdida Esperada = PD × LGD × EAD.  
    - Ingreso: APR × EAD.  
    - Utilidad: Ingreso – EL – Costos.  
    - PD ponderado: Probabilidad de default promedio, ponderada por EAD.
    """)

    st.markdown("### Análisis Ejecutivo")
    st.success("La optimización disminuye la pérdida esperada y aumenta utilidad redirigiendo exposición a clientes más sanos, sin afectar producción.")

    port = dfs.get("def_port")
    if port is not None and not port.empty:
        def g0(df, col): return df[col].iloc[0] if col in df.columns else np.nan
        kpi_row_money("EAD", g0(port,"EAD_actual"), g0(port,"EAD_optimizado"), moneda, usdclp)
        kpi_row_money("EL (Pérdida Esperada)", g0(port,"EL_actual"), g0(port,"EL_optimizado"), moneda, usdclp)
        kpi_row_money("Ingreso", g0(port,"Ingreso_actual"), g0(port,"Ingreso_optimizado"), moneda, usdclp)
        kpi_row_money("Utilidad", g0(port,"Utilidad_actual"), g0(port,"Utilidad_optimizada"), moneda, usdclp)
        if "PD_pond_actual" in port.columns and "PD_pond_optimizado" in port.columns:
            # mostramos en %
            kpi_row_pct("PD Ponderado", _to_float_or_nan(g0(port,"PD_pond_actual"))*100, _to_float_or_nan(g0(port,"PD_pond_optimizado"))*100)

    det = dfs.get("def_det")
    if det is not None and not det.empty:
        with st.expander("Detalle por cliente (muestra)"):
            st.dataframe(det.head(100), use_container_width=True)

# ================
# Arista 2 – Yield / Pricing
# ================
with tabs[1]:
    st.header("Arista 2 – Yield/Pricing")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Encontramos la tasa (APR) óptima que maximiza utilidad equilibrando precio y volumen (elasticidades).")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - Ingreso total: Flujo de intereses ajustado por volumen.  
    - Utilidad total: Ingreso – EL – Costos.  
    - EAD in/out: Exposición antes/después del ajuste de precio.
    """)

    st.markdown("### Análisis Ejecutivo")
    st.success("Ajuste de precio al punto de máximo retorno: si cobras demasiado, cae el volumen; si cobras poco, se sacrifica rentabilidad. El balance correcto maximiza utilidad.")

    port = dfs.get("yld_port")
    if port is not None and not port.empty:
        def g0(df, name): return df[name].iloc[0] if name in df.columns else np.nan
        kpi_row_money("Ingreso Total", g0(port,"ingreso_base"), g0(port,"ingreso_opt"), moneda, usdclp)
        kpi_row_money("Utilidad Total", g0(port,"utilidad_base"), g0(port,"utilidad_opt"), moneda, usdclp)
        kpi_row_money("EAD In", g0(port,"EAD_in"), g0(port,"EAD_in"), moneda, usdclp)
        kpi_row_money("EAD Out", g0(port,"EAD_out"), g0(port,"EAD_out"), moneda, usdclp)

    curv = dfs.get("yld_curv")
    if curv is not None and not curv.empty:
        with st.expander("Curva APR–Volumen (por segmento)"):
            st.dataframe(curv, use_container_width=True)

# ================
# Arista 3 – Incentivos
# ================
with tabs[2]:
    st.header("Arista 3 – Incentivos")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Invertimos en incentivos solo donde el ROI es positivo: más ingresos por cada peso invertido, bajo un presupuesto.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - Costo incentivos: gasto total en beneficios.  
    - Ingreso incremental: ingresos adicionales generados.  
    - ROI: Retorno de la inversión = Ingreso / Costo.
    """)

    st.markdown("### Análisis Ejecutivo")
    st.success("Asignación selectiva: fertilizamos solo las plantas que responden. Cada peso invertido retorna múltiplos cuando ROI>0.")

    det = dfs.get("inc_det"); summ = dfs.get("inc_sum")
    if det is not None and not det.empty:
        cost_cols = [c for c in det.columns if "cost" in c.lower()]
        up_cols   = [c for c in det.columns if "uplift" in c.lower() or "ingreso_inc" in c.lower() or "delta_ingreso" in c.lower()]
        costo = pd.to_numeric(det[cost_cols].sum(axis=1), errors="coerce").fillna(0).sum() if cost_cols else np.nan
        uplift = pd.to_numeric(det[up_cols].sum(axis=1), errors="coerce").fillna(0).sum() if up_cols else np.nan
        roi = uplift/costo if (isinstance(costo,(int,float)) and costo>0) else np.nan
        kpi_row_money("Costo de Incentivos", costo, costo, moneda, usdclp)
        kpi_row_money("Ingreso Incremental", uplift, uplift, moneda, usdclp)
        st.metric("ROI", fmt_pct_val(roi*100 if not pd.isna(roi) else np.nan))
    if summ is not None and not summ.empty:
        with st.expander("Resumen de diagnóstico"):
            st.dataframe(summ, use_container_width=True)

# ================
# Arista 4 – Capital / Provisiones
# ================
with tabs[3]:
    st.header("Arista 4 – Capital / Provisiones")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Hacemos más eficiente el capital requerido y reducimos provisiones, liberando recursos para crecer (bajo proxies Basel III e IFRS9).")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - Capital requerido (K): K_ratio × RWA (proxy).  
    - RWA: Activos ponderados por riesgo (proxy ~ RW% × EAD).  
    - Provisiones: reservas por riesgo crediticio ≈ EL.
    """)

    st.markdown("### Análisis Ejecutivo")
    st.success("Reorganizamos el capital inmovilizado sin perder protección: menos RWA/K por la misma producción, liberando espacio para crecer.")

    cap_port = dfs.get("cap_port")
    if cap_port is not None and not cap_port.empty:
        def g0(df, name): return df[name].iloc[0] if name in df.columns else np.nan
        kpi_row_money("Capital Requerido (K)", g0(cap_port,"K"), g0(cap_port,"K"), moneda, usdclp)
        kpi_row_money("RWA", g0(cap_port,"RWA"), g0(cap_port,"RWA"), moneda, usdclp)
        # Si el portfolio trae EL o provisiones, mostrar:
        if "EL" in cap_port.columns:
            kpi_row_money("Provisiones (≈EL)", g0(cap_port,"EL"), g0(cap_port,"EL"), moneda, usdclp)

# ================
# Guardrails
# ================
with tabs[4]:
    st.header("Guardrails (Resguardos)")
    st.markdown("Límites regulatorios y de negocio para asegurar robustez y cumplimiento.")

    gport = dfs.get("guard_port"); gseg = dfs.get("guard_seg")

    if gport is None or gport.empty:
        st.info("No hay tablas de guardrails en el bundle. Genera con la Celda 16/21 del notebook.")
    else:
        gport_fmt = gport.copy()
        for c in ["umbral","observado_actual","observado_optimizado","observado"]:
            if c in gport_fmt.columns:
                gport_fmt[c] = gport_fmt[c].apply(fmt_pct_val)
        st.subheader("Portafolio")
        st.dataframe(gport_fmt, use_container_width=True)

    if gseg is not None and not gseg.empty:
        gseg_fmt = gseg.copy()
        for c in ["observado","umbral"]:
            if c in gseg_fmt.columns:
                gseg_fmt[c] = gseg_fmt[c].apply(fmt_pct_val)
        st.subheader("Segmento")
        st.dataframe(gseg_fmt, use_container_width=True)

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas integradas).")
