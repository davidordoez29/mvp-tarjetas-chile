# app/app_dashboard.py — MVP Bancario (4 Aristas) con escenarios y formatos

import os, json, math, re
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# =========================================
# Config / archivos requeridos por arista
# =========================================
REQ = {
    "A1": {
        "portfolio": "default_portfolio{SFX}.csv",
        "segment":   "default_segment{SFX}.csv",
        "detail":    "default_detail{SFX}.csv",
    },
    "A2": {
        "portfolio": "yield_portfolio{SFX}.csv",
        "segment":   "yield_segment{SFX}.csv",
        "detail":    "yield_detail{SFX}.csv",
        "curve":     "yield_curve_segment{SFX}.csv",
    },
    "A3": {
        # portfolio puede no existir; se reconstruye con detail si falta
        "portfolio": "incentives_portfolio{SFX}.csv",
        "detail":    "incentives_detail{SFX}.csv",
        "diag":      "incentives_diag_summary{SFX}.csv",
        "sens":      "incentives_sensitivity{SFX}.csv",
    },
    "A4": {
        "portfolio": "capital_portfolio{SFX}.csv",
        "segment":   "capital_segment{SFX}.csv",
        "detail":    "capital_detail{SFX}.csv",
    },
    "GR": {
        "portfolio": "guardrails_portfolio.csv",   # sin sufijo de escenario
        "segment":   "guardrails_segment.csv",     # sin sufijo
        "eval":      "guardrails_eval_portfolio.csv",  # opcional
    }
}

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    "/content/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
]

# =========================================
# Utilidades de formato
# =========================================
_NUM_LIKE = re.compile(r"^-?\d+(\.\d+)?$")
PCT_HINTS = ["pct", "%", "porc", "pd_pond", "pd%", "var%", "pd", "ratio"]

def _to_float_or_nan(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return np.nan
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        s = v.strip().replace("%","").replace(",", ".")
        if _NUM_LIKE.match(s):
            try: return float(s)
            except: return np.nan
        return np.nan
    return np.nan

def _to_display_currency(val: float, target: str, usdclp: float) -> float:
    if pd.isna(val): return np.nan
    if target.upper() == "USD":
        return float(val) / float(usdclp) if usdclp else np.nan
    return float(val)

def fmt_money_val(val, target: str, usdclp: float) -> str:
    # acepta int/float/None/np.nan; strings se devuelven como están
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
    # Acepta número o string. Si no es numérico, devuelve original limpio.
    if isinstance(val, str):
        s = val.strip()
        if s.endswith("%"):
            return s.replace(".", ",")
        if not _NUM_LIKE.match(s.replace(",", ".")):
            return s
    x = _to_float_or_nan(val)
    if np.isnan(x):
        return "—" if (val is None or (isinstance(val, float) and math.isnan(val))) else str(val)
    return f"{x:.2f}%".replace(".", ",")

def var_pct(actual, opt):
    a = _to_float_or_nan(actual); o = _to_float_or_nan(opt)
    if np.isnan(a) or a == 0 or np.isnan(o): return None
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

def _apply_series_fmt(series: pd.Series, fn):
    return series.apply(lambda v: fn(v))

def format_df_currency(df: pd.DataFrame, cols: list[str], moneda: str, usdclp: float):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = _apply_series_fmt(df2[c], lambda v: fmt_money_val(v, moneda, usdclp))
    return df2

def format_df_pct(df: pd.DataFrame, cols: list[str]):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = _apply_series_fmt(df2[c], fmt_pct_val)
    return df2

def format_df_auto(df: pd.DataFrame, moneda: str, usdclp: float):
    """Heurística: columnas con 'EAD', 'EL', 'Ingreso', 'Util' → dinero; con hints de % → pct."""
    if df is None or df.empty: return df
    money_hints = ["ead", "el", "ingreso", "util", "capital", "rwa", "k", "prov", "cof", "income"]
    df2 = df.copy()
    for c in df2.columns:
        lc = c.lower()
        if any(h in lc for h in PCT_HINTS):
            df2[c] = df2[c].apply(fmt_pct_val)
        elif any(h in lc for h in money_hints):
            df2[c] = df2[c].apply(lambda v: fmt_money_val(v, moneda, usdclp))
        # r_base / r_opt (tasa): lo formateamos como %
        elif lc in ("r_base","r_opt","apr","apr_opt","tasa","tasa_opt"):
            df2[c] = df2[c].apply(lambda v: fmt_pct_val(_to_float_or_nan(v)*100.0))
    return df2

# =========================================
# Carga bundle
# =========================================
def _dir_ok(d: str) -> bool:
    try:
        if not d or not os.path.isdir(d): return False
        # basta con que existan algunos archivos clave
        hits = 0
        for patt in [
            REQ["A1"]["portfolio"].format(SFX=""),
            REQ["A2"]["portfolio"].format(SFX=""),
            REQ["A4"]["portfolio"].format(SFX=""),
        ]:
            if os.path.exists(os.path.join(d, patt)):
                hits += 1
        return hits >= 1
    except Exception:
        return False

def autodetect_bundle() -> str | None:
    for d in CANDIDATE_DIRS:
        if _dir_ok(d): return d
    return None

def _read_csv(path: Path):
    if not path.exists(): return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"⚠️ Error leyendo {path.name}: {e}")
        return None

def load_by_arista(bundle_dir: Path, arista: str, scenario_suffix: str):
    """Devuelve dict con dataframes de la arista dada, respetando sufijo de escenario."""
    dfs = {}
    spec = REQ[arista]
    for key, patt in spec.items():
        # guardrails no llevan sufijo
        if arista == "GR":
            fname = patt
        else:
            fname = patt.format(SFX=scenario_suffix)
        p = bundle_dir / fname
        dfs[key] = _read_csv(p)
    return dfs

# =========================================
# App — Sidebar
# =========================================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

st.sidebar.title("⚙️ Configuración")
default_dir = autodetect_bundle()
bundle_dir_in = st.sidebar.text_input(
    "📦 Ruta del bundle",
    value=(default_dir or ""),
    help="Ej: /content/out/dashboard_bundle",
).strip()
bundle_dir = Path(bundle_dir_in) if bundle_dir_in else (Path(default_dir) if default_dir else None)

if not bundle_dir or not bundle_dir.exists():
    st.error("No encuentro el bundle. Genera el paquete en el notebook (Celda 22) y vuelve a cargar.")
    st.stop()

scenario = st.sidebar.radio("Escenario", ["Conservador", "Potenciado"], horizontal=True)
SFX = "" if scenario == "Conservador" else "_agresivo"

moneda = st.sidebar.radio("Moneda a visualizar", ["CLP", "USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))

st.title("📊 MVP Bancario — Optimización en 4 Aristas")
st.caption("Modelo matemático aplicado sobre un portafolio. Comparación Actual vs Optimizado con guardrails IFRS9/Basilea/Negocio.")

tabs = st.tabs([
    "Arista 1 – Default/Impago",
    "Arista 2 – Yield/Pricing",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital/Provisiones",
    "Guardrails (Resguardos)"
])

# =========================================
# 📍 Arista 1 – Default / Impago
# =========================================
with tabs[0]:
    st.header("Arista 1 – Default/Impago")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.success("Reducimos la *Pérdida Esperada (EL)* reasignando EAD hacia segmentos/clientes con menor riesgo, *sin frenar el negocio* ni violar límites regulatorios (concentración, bandas de precio, tracking vs cartera base).")

    st.markdown("### KPIs y definiciones")
    st.markdown("""
    - *EAD (Exposición)*: saldo expuesto al riesgo de crédito.  
    - *EL (Pérdida Esperada)*: PD × LGD × EAD.  
    - *Utilidad*: ingresos financieros menos pérdidas esperadas y costos (simplificado).  
    - *PD ponderado*: promedio de PD ponderado por EAD (indicador de riesgo promedio del libro).
    """)

    st.markdown("### ¿Qué cálculos hace el motor aquí?")
    st.markdown("""
    1) Estima PD y LGD cumpliendo IFRS9 (staging, lifetime, overlays).  
    2) Calcula EL por cliente/segmento y portafolio.  
    3) Rebalancea *EAD* dentro de límites (guardrails) para reducir EL y bajar el *PD ponderado*, preservando el volumen comercial.
    """)

    A1 = load_by_arista(bundle_dir, "A1", SFX)
    port, seg, det = A1["portfolio"], A1["segment"], A1["detail"]

    if port is not None and not port.empty:
        # columnas esperadas
        def g0(df, name): return df[name].iloc[0] if name in df.columns else np.nan
        kpi_row_money("EAD", g0(port,"EAD_actual"), g0(port,"EAD_optimizado"), moneda, usdclp)
        kpi_row_money("EL (Pérdida Esperada)", g0(port,"EL_actual"), g0(port,"EL_optimizado"), moneda, usdclp)
        kpi_row_money("Utilidad", g0(port,"Utilidad_actual"), g0(port,"Utilidad_optimizada"), moneda, usdclp)
        # PD ponderado puede venir 0–1; mostramos en %
        if "PD_pond_actual" in port.columns or "PD_pond_optimizado" in port.columns:
            a = g0(port,"PD_pond_actual"); b = g0(port,"PD_pond_optimizado")
            kpi_row_pct("PD ponderado", a*100 if pd.notna(a) else np.nan, b*100 if pd.notna(b) else np.nan)

    st.subheader("Detalle por segmento")
    if seg is not None and not seg.empty:
        st.dataframe(format_df_auto(seg, moneda, usdclp), use_container_width=True, height=360)

    st.subheader("Detalle por cliente (muestra)")
    if det is not None and not det.empty:
        sample = det.head(3000)  # evitar render pesado
        st.dataframe(format_df_auto(sample, moneda, usdclp), use_container_width=True, height=360)

# =========================================
# 📍 Arista 2 – Yield / Pricing
# =========================================
with tabs[1]:
    st.header("Arista 2 – Yield/Pricing")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.success("Ajustamos el *precio (APR)* por segmento/cliente para *maximizar utilidad, equilibrando precio y volumen con **elasticidades* y bandas reguladas.")

    st.markdown("### KPIs y definiciones")
    st.markdown("""
    - *Ingreso total*: interés esperado sobre el EAD efectivamente servido tras la decisión de precio.  
    - *Utilidad total*: ingreso menos EL y costos (COF/operativos) bajo el nuevo precio.  
    - *EAD_in / EAD_out*: exposición potencial vs exposición servida después del precio (impacto volumen).
    """)

    st.markdown("### ¿Qué cálculos hace el motor aquí?")
    st.markdown("""
    1) Aplica *curvas precio–demanda* por segmento (elasticidades) respetando bandas y caps.  
    2) Determina r_opt y el EAD_out resultante.  
    3) Recalcula *ingresos* y *utilidad* esperada con el nuevo mix de precio/volumen.
    """)

    A2 = load_by_arista(bundle_dir, "A2", SFX)
    port, seg, det, curve = A2["portfolio"], A2["segment"], A2["detail"], A2["curve"]

    if port is not None and not port.empty:
        def g0(df, name): return df[name].iloc[0] if name in df.columns else np.nan
        kpi_row_money("Ingreso total", g0(port,"ingreso_base"), g0(port,"ingreso_opt"), moneda, usdclp)
        kpi_row_money("Utilidad total", g0(port,"utilidad_base"), g0(port,"utilidad_opt"), moneda, usdclp)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Segmento")
        if seg is not None and not seg.empty:
            st.dataframe(format_df_auto(seg, moneda, usdclp), use_container_width=True, height=360)
    with c2:
        st.subheader("Curva precio–segmento")
        if curve is not None and not curve.empty:
            st.dataframe(format_df_auto(curve, moneda, usdclp), use_container_width=True, height=360)

    st.subheader("Detalle por cliente (muestra)")
    if det is not None and not det.empty:
        sample = det.head(3000)
        st.dataframe(format_df_auto(sample, moneda, usdclp), use_container_width=True, height=360)

# =========================================
# 📍 Arista 3 – Incentivos
# =========================================
with tabs[2]:
    st.header("Arista 3 – Incentivos")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.success("Invertimos en *incentivos* solo donde el *ROI* es positivo: más ingresos por cada peso invertido, con *caps presupuestarios* y reglas de selección.")

    st.markdown("### KPIs y definiciones")
    st.markdown("""
    - *Costo incentivos*: gasto total en beneficios/apoyos.  
    - *Ingreso incremental*: ingresos adicionales atribuibles al incentivo.  
    - *ROI*: Ingreso incremental / Costo. Debe ser > 0 y creciente tras la optimización.
    """)

    st.markdown("### ¿Qué cálculos hace el motor aquí?")
    st.markdown("""
    1) Estima *uplift* de ingreso por cliente ante incentivo (propensión / sensibilidad).  
    2) Selecciona el subconjunto con ROI > 0 bajo un *presupuesto* y caps (guardrails).  
    3) Calcula *ingreso incremental, **costo* y *ROI* agregados por segmento/portafolio.
    """)

    A3 = load_by_arista(bundle_dir, "A3", SFX)
    det, diag, sens = A3["detail"], A3["diag"], A3["sens"]

    # Reconstrucción de KPIs portafolio si no existe archivo portfolio
    total_cost = total_uplift = np.nan
    if det is not None and not det.empty:
        # buscar columnas costo/uplift/roi con tolerancia de nombres
        cost_cols = [c for c in det.columns if "costo" in c.lower() or c.lower() in ("cost","costo_est","costo_real")]
        up_cols   = [c for c in det.columns if "ingreso_inc" in c.lower() or "uplift" in c.lower()]
        if cost_cols: 
            total_cost = pd.to_numeric(det[cost_cols].sum(axis=1), errors="coerce").fillna(0).sum()
        if up_cols:
            total_uplift = pd.to_numeric(det[up_cols].sum(axis=1), errors="coerce").fillna(0).sum()

    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        st.metric("Costo incentivos", fmt_money_val(total_cost, moneda, usdclp) if pd.notna(total_cost) else "—")
    with c2:
        st.metric("Ingreso incremental", fmt_money_val(total_uplift, moneda, usdclp) if pd.notna(total_uplift) else "—")
    with c3:
        roi = (total_uplift / total_cost * 100.0) if (pd.notna(total_cost) and total_cost > 0 and pd.notna(total_uplift)) else np.nan
        st.metric("ROI", fmt_pct_val(roi))

    st.subheader("Diagnóstico / Selección")
    if diag is not None and not diag.empty:
        st.dataframe(format_df_auto(diag, moneda, usdclp), use_container_width=True, height=300)

    st.subheader("Sensibilidad (presupuesto / reglas)")
    if sens is not None and not sens.empty:
        st.dataframe(format_df_auto(sens, moneda, usdclp), use_container_width=True, height=300)

    st.subheader("Detalle por cliente (muestra)")
    if det is not None and not det.empty:
        sample = det.head(3000)
        st.dataframe(format_df_auto(sample, moneda, usdclp), use_container_width=True, height=360)

# =========================================
# 📍 Arista 4 – Capital / Provisiones
# =========================================
with tabs[3]:
    st.header("Arista 4 – Capital / Provisiones")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.success("Hacemos *más eficiente* el uso de capital (RWA, K) y *provisiones* (≈EL), cumpliendo Basilea/IFRS, liberando recursos para crecer.")

    st.markdown("### KPIs y definiciones")
    st.markdown("""
    - *Capital requerido (K)*: RWA × K_ratio.  
    - *RWA*: activos ponderados por riesgo (proxy por clase/segmento).  
    - *Provisiones*: ≈ EL bajo IFRS9.  
    - *Liberación*: diferencia pre/post optimización.
    """)

    st.markdown("### ¿Qué cálculos hace el motor aquí?")
    st.markdown("""
    1) Calcula *EL* y *RWA* por segmento/portafolio.  
    2) Aplica *K_ratio* y buffers para obtener capital requerido.  
    3) Compara *base vs optimizado* y verifica guardrails.
    """)

    A4 = load_by_arista(bundle_dir, "A4", SFX)
    port, seg, det = A4["portfolio"], A4["segment"], A4["detail"]

    if port is not None and not port.empty:
        # Detectar pares base/opt con tolerancia
        def pick(df, base, opt):
            a = df[base].iloc[0] if base in df.columns else np.nan
            b = df[opt].iloc[0]  if opt  in df.columns else np.nan
            return a, b

        EAD_a, EAD_b = pick(port, "EAD_base", "EAD_opt")
        EL_a,  EL_b  = pick(port, "EL_base",  "EL_opt")
        RWA_a, RWA_b = pick(port, "RWA_base", "RWA_opt")
        K_a,   K_b   = pick(port, "K_base",   "K_opt")

        kpi_row_money("EAD", EAD_a, EAD_b, moneda, usdclp)
        kpi_row_money("EL (Provisiones≈)", EL_a, EL_b, moneda, usdclp)
        kpi_row_money("RWA", RWA_a, RWA_b, moneda, usdclp)
        kpi_row_money("Capital (K)", K_a, K_b, moneda, usdclp)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Segmento")
        if seg is not None and not seg.empty:
            st.dataframe(format_df_auto(seg, moneda, usdclp), use_container_width=True, height=360)
    with c2:
        st.subheader("Detalle por cliente (muestra)")
        if det is not None and not det.empty:
            sample = det.head(3000)
            st.dataframe(format_df_auto(sample, moneda, usdclp), use_container_width=True, height=360)

# =========================================
# 📍 Guardrails (Resguardos)
# =========================================
with tabs[4]:
    st.header("Guardrails (Resguardos)")
    st.markdown("Verificación automática de *límites regulatorios y de negocio* sobre los resultados.")

    GR = load_by_arista(bundle_dir, "GR", SFX)  # sin sufijo
    gport, gseg, geval = GR["portfolio"], GR["segment"], GR["eval"]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Portafolio")
        if gport is None or gport.empty:
            st.info("No hay guardrails_portfolio.csv en el bundle.")
        else:
            st.dataframe(format_df_auto(gport, moneda, usdclp), use_container_width=True, height=360)
    with c2:
        st.subheader("Segmento")
        if gseg is None or gseg.empty:
            st.info("No hay guardrails_segment.csv en el bundle.")
        else:
            st.dataframe(format_df_auto(gseg, moneda, usdclp), use_container_width=True, height=360)

    if geval is not None and not geval.empty:
        st.subheader("Evaluación (checks resumidos)")
        st.dataframe(format_df_auto(geval, moneda, usdclp), use_container_width=True, height=300)

# =========================================
# Footer
# =========================================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas). IFRS9/Basilea/Negocio.")
