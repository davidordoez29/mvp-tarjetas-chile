# app/app_dashboard.py — MVP Bancario (4 Aristas) CONS/POT con CLP/USD
# Ajustes:
# - Pitch restaurado en “¿Qué resolvemos aquí?” (por arista).
# - KPIs y Definiciones ampliadas (lenguaje profesional y claro).
# - Análisis Ejecutivo: describe cálculos del motor sin expresiones coloquiales.
# - Formato numérico robusto (CLP/USD y %), aplicado a todas las tablas.
# - Fix definitivo: _PCT_HINTS (y helpers) definidos antes de usarse.

import os, json, math, re
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# ==========================
# Config carga de bundle
# ==========================
REQ_FILES_BASE = {
    # Arista 1
    "def_port": "default_portfolio.csv",
    "def_seg":  "default_segment.csv",
    "def_det":  "default_detail.csv",
    # Arista 2
    "yld_port": "yield_portfolio.csv",
    "yld_seg":  "yield_segment.csv",
    "yld_det":  "yield_detail.csv",
    "yld_curv": "yield_curve_segment.csv",
    # Arista 3
    "inc_det":  "incentives_detail.csv",
    "inc_sum":  "incentives_diag_summary.csv",
    "inc_sens": "incentives_sensitivity.csv",
    # Arista 4
    "cap_port": "capital_portfolio.csv",
    "cap_seg":  "capital_segment.csv",
    "cap_det":  "capital_detail.csv",
    # Guardrails (no por escenario)
    "gr_port":  "guardrails_portfolio.csv",
    "gr_seg":   "guardrails_segment.csv",
    "gr_eval":  "guardrails_eval_portfolio.csv",
    # KPI unificados (opcional)
    "kpis_all": "kpis_unificados.csv",
    # Meta opcional
    "manifest": "manifest.json",
    "run_meta": "run_meta.json",
}
SUF = {"Conservador": "", "Potenciado": "_agresivo"}

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    "/content/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
]

# ==========================
# Hints y formateo numérico (definidos ANTES de usarse)
# ==========================
_num_like = re.compile(r"^-?\d+(\.\d+)?$")

# Columnas que trataremos como porcentajes (insensible a may/min)
# Incluimos pd, roi y tasas (apr, r_base, r_opt) para mostrarlas en %.
PCT_HINTS   = ("pd_pond", "pd", "pd12", "pd_12", "roi", "apr", "r_", "%")

# Columnas que NUNCA se formatean como dinero/porcentaje (llaves u objetos categóricos)
_MONEY_AVOID = ("id", "id_cliente", "segmento")

def _to_display_currency(val: float, target: str, usdclp: float) -> float | None:
    if pd.isna(val): return None
    if target.upper() == "USD":
        return float(val) / float(usdclp) if usdclp else None
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

def _to_float_or_nan(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return np.nan
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        s = v.strip().replace("%","").replace(",", ".")
        if _num_like.match(s):
            try: return float(s)
            except Exception: return np.nan
        return np.nan
    return np.nan

def fmt_pct_val(val):
    if isinstance(val, str):
        s = val.strip()
        if s.endswith("%"): return s.replace(".", ",")
        if not _num_like.match(s.replace(",", ".")): return s
    x = _to_float_or_nan(val)
    if np.isnan(x): return "—"
    return f"{x:.2f}%".replace(".", ",")

def var_pct(actual, opt):
    a = _to_float_or_nan(actual); o = _to_float_or_nan(opt)
    if np.isnan(a) or a == 0: return None
    return (o - a) / a * 100.0

def kpi_row(label: str, actual, opt, moneda: str, usdclp: float, help_text: str = ""):
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

def format_df_auto(df: pd.DataFrame, moneda: str, usdclp: float) -> pd.DataFrame:
    """Formatea todas las columnas como % o moneda según hints, excluye id/segmento."""
    if df is None or df.empty: return df
    out = df.copy()
    for c in out.columns:
        lc = c.lower()
        if lc in _MONEY_AVOID:
            continue
        if any(h in lc for h in _PCT_HINTS):
            out[c] = out[c].apply(fmt_pct_val)
        else:
            out[c] = out[c].apply(lambda v: fmt_money_val(v, moneda, usdclp))
    return out

def _first(df: pd.DataFrame, col: str):
    try:
        return df[col].iloc[0] if df is not None and (not df.empty) and (col in df.columns) else np.nan
    except Exception:
        return np.nan

# ==========================
# Utilidades de carga bundle
# ==========================
def _dir_ok(d: str) -> bool:
    try:
        if not d or not os.path.isdir(d): return False
        hits = 0
        for k, v in REQ_FILES_BASE.items():
            p = os.path.join(d, v)
            if os.path.exists(p):
                hits += 1
        return hits >= 8
    except Exception:
        return False

def autodetect_bundle() -> str | None:
    for d in CANDIDATE_DIRS:
        if _dir_ok(d): return d
    return None

def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Error leyendo {path.name}: {e}")
    return None

def _load_for_scenario(bundle: Path, escenario: str) -> dict[str, pd.DataFrame | None]:
    """Carga todos los archivos necesarios, aplicando sufijo por escenario cuando corresponda.
       Guardrails se leen sin sufijo (comparten para ambos escenarios)."""
    suf = SUF.get(escenario, "")
    dfs = {}

    def pick(name_base: str, scenario_dependent=True):
        fname = REQ_FILES_BASE[name_base]
        if scenario_dependent and suf:
            p = bundle / (fname.replace(".csv", f"{suf}.csv"))
            if p.exists():
                return _read_csv(p)
        return _read_csv(bundle / fname)

    # A1
    dfs["def_port"] = pick("def_port")
    dfs["def_seg"]  = pick("def_seg")
    dfs["def_det"]  = pick("def_det")
    # A2
    dfs["yld_port"] = pick("yld_port")
    dfs["yld_seg"]  = pick("yld_seg")
    dfs["yld_det"]  = pick("yld_det")
    dfs["yld_curv"] = pick("yld_curv")
    # A3
    dfs["inc_det"]  = pick("inc_det")
    dfs["inc_sum"]  = pick("inc_sum")
    dfs["inc_sens"] = pick("inc_sens")
    # A4
    dfs["cap_port"] = pick("cap_port")
    dfs["cap_seg"]  = pick("cap_seg")
    dfs["cap_det"]  = pick("cap_det")
    # Guardrails (sin sufijo)
    dfs["gr_port"]  = pick("gr_port", scenario_dependent=False)
    dfs["gr_seg"]   = pick("gr_seg",  scenario_dependent=False)
    dfs["gr_eval"]  = pick("gr_eval", scenario_dependent=False)
    # KPIs unificados globales (opcional)
    dfs["kpis_all"] = pick("kpis_all", scenario_dependent=False)
    return dfs

# ==========================
# App Layout
# ==========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

st.sidebar.title("⚙️ Configuración")
default_dir = autodetect_bundle()
bundle_dir_input = st.sidebar.text_input("📦 Ruta del bundle", value=(default_dir or ""), help="Ej: /content/out/dashboard_bundle")
bundle_dir = (bundle_dir_input or default_dir or "").strip()

escenario = st.sidebar.radio("Escenario", ["Conservador", "Potenciado"], horizontal=True)
moneda = st.sidebar.radio("Moneda", ["CLP", "USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))

st.title("📊 MVP Bancario — Optimización en 4 Aristas")
st.caption("Comparación *Actual vs Optimizado*, por escenario y arista. Moneda y formato homogéneos.")

if not bundle_dir:
    st.error("No encuentro el bundle. Genera el paquete con la Celda 22 del notebook y vuelve a cargar.")
    st.stop()

bundle = Path(bundle_dir)
dfs = _load_for_scenario(bundle, escenario)

# Diagnóstico rápido
missing = []
for k, base in REQ_FILES_BASE.items():
    p1 = bundle / base
    p2 = bundle / base.replace(".csv", f"{SUF[escenario]}.csv")
    if k.startswith("gr_") or k in ["kpis_all","manifest","run_meta"]:
        if not p1.exists(): missing.append(base)
    else:
        if not (p1.exists() or p2.exists()): missing.append(base + " (con/sin sufijo)")
if missing:
    st.warning("Faltan archivos en el bundle:\n- " + "\n- ".join(missing))

tabs = st.tabs([
    "Arista 1 — Default/Impago",
    "Arista 2 — Yield/Pricing",
    "Arista 3 — Incentivos",
    "Arista 4 — Capital/Provisiones",
    "Guardrails (Resguardos)"
])

# ================
# Arista 1 — Default/Impago
# ================
with tabs[0]:
    st.header("Arista 1 — Default/Impago")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Reducimos la *pérdida esperada (EL)* reasignando exposición hacia perfiles de menor riesgo, conservando volumen de negocio y respetando guardrails.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
- *EAD (Exposición): monto expuesto al riesgo. El objetivo es **mantener el total* para no frenar el negocio.  
- *EL (Expected Loss): pérdida esperada promedio: **EL = PD × LGD × EAD*.  
- *Interés: ingresos financieros del portafolio: **APR × EAD*.  
- *Utilidad: resultado económico luego de riesgo y fondeo: **Interés − EL − COF*.  
- *PD ponderado: probabilidad media de incumplimiento ponderada por exposición: **Σ(PD × EAD)/Σ(EAD)*.
    """)

    st.markdown("### Análisis Ejecutivo")
    st.markdown("""
Se evalúa rentabilidad-riesgo por cliente/segmento, se aplican *límites de traslado de EAD* y se redistribuye exposición desde perfiles *alto PD/baja utilidad* hacia perfiles *menor PD/mayor utilidad*, siempre bajo los resguardos definidos.  
Luego se recalculan *EL, **PD ponderado, **interés* y *utilidad* para el portafolio optimizado.
    """)

    port = dfs.get("def_port")
    if port is not None and not port.empty:
        kpi_row("EAD", _first(port,"EAD_actual"), _first(port,"EAD_optimizado"), moneda, usdclp, "Exposición total.")
        kpi_row("EL (Pérdida Esperada)", _first(port,"EL_actual"), _first(port,"EL_optimizado"), moneda, usdclp, "Disminuye al mejorar el mix de riesgo.")
        kpi_row("Utilidad", _first(port,"Utilidad_actual"), _first(port,"Utilidad_optimizada"), moneda, usdclp, "Interés neto de riesgo y costo de fondos.")
        if "PD_pond_actual" in port.columns and "PD_pond_optimizado" in port.columns:
            kpi_row_pct("PD Ponderado", _first(port,"PD_pond_actual")*100, _first(port,"PD_pond_optimizado")*100, "Probabilidad de incumplimiento media ponderada.")

    seg = dfs.get("def_seg")
    if seg is not None and not seg.empty:
        st.markdown("#### Resultados por segmento")
        st.dataframe(format_df_auto(seg, moneda, usdclp), use_container_width=True, height=360)

    det = dfs.get("def_det")
    if det is not None and not det.empty:
        st.markdown("#### Detalle por cliente")
        st.dataframe(format_df_auto(det, moneda, usdclp).head(500), use_container_width=True, height=360)

# ================
# Arista 2 — Yield / Pricing
# ================
with tabs[1]:
    st.header("Arista 2 — Yield/Pricing")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Determinamos la *tasa óptima (APR)* por segmento para maximizar *utilidad*, equilibrando precio y volumen retenido/adquirido.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
- *Interés total: flujo de intereses resultante (APR_opt × EAD_out*).  
- *Utilidad total: **Interés − EL − COF* con la tasa óptima.  
- *EAD_in / EAD_out*: exposición antes/después del ajuste de precio (elasticidad de demanda por tasa).
    """)

    st.markdown("### Análisis Ejecutivo")
    st.markdown("""
Se estiman elasticidades por segmento, se prueban *candidatos de APR, se proyecta **EAD_out* y se recomputan *interés, **EL* y *utilidad*.  
Se elige la APR que maximiza *utilidad* respetando límites (bandas de APR, calidad crediticia y guardrails).
    """)

    port = dfs.get("yld_port")
    if port is not None and not port.empty:
        kpi_row("Interés total", _first(port,"ingreso_base"), _first(port,"ingreso_opt"), moneda, usdclp)
        kpi_row("Utilidad total", _first(port,"utilidad_base"), _first(port,"utilidad_opt"), moneda, usdclp)

    seg = dfs.get("yld_seg")
    if seg is not None and not seg.empty:
        st.markdown("#### Resultados por segmento")
        st.dataframe(format_df_auto(seg, moneda, usdclp), use_container_width=True, height=360)

    det = dfs.get("yld_det")
    if det is not None and not det.empty:
        st.markdown("#### Detalle por cliente (pricing)")
        st.dataframe(format_df_auto(det, moneda, usdclp).head(500), use_container_width=True, height=360)

# ================
# Arista 3 — Incentivos
# ================
with tabs[2]:
    st.header("Arista 3 — Incentivos")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Asignar *incentivos* únicamente donde el *ROI* sea positivo, maximizando el ingreso incremental por peso invertido.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
- *Costo de incentivos*: gasto total en beneficios.  
- *Ingreso incremental*: ingresos adicionales atribuibles al incentivo (uplift estimado/medido).  
- *ROI: **Ingreso incremental / Costo* (objetivo: *> 0* y por encima del umbral definido).
    """)

    st.markdown("### Análisis Ejecutivo")
    st.markdown("""
A partir del detalle de pricing se delimita *universo elegible, se estima **uplift* por cliente/segmento, se *asignan* incentivos bajo presupuesto y umbrales de ROI, y se consolida el impacto neto.  
Las asignaciones priorizan *eficiencia* y cumplen límites de negocio y regulación.
    """)

    det = dfs.get("inc_det")
    if det is not None and not det.empty:
        cost_cols = [c for c in det.columns if any(k in c.lower() for k in ["cost","costo"])]
        up_cols   = [c for c in det.columns if any(k in c.lower() for k in ["uplift","ingreso_inc","ingreso_incremental","delta_ingreso"])]
        costo = float(pd.to_numeric(det[cost_cols].sum(axis=1), errors="coerce").sum()) if cost_cols else 0.0
        uplift = float(pd.to_numeric(det[up_cols].sum(axis=1), errors="coerce").sum()) if up_cols else 0.0
        roi = (uplift / costo) if costo > 0 else np.nan
        kpi_row("Costo de incentivos", costo, costo, moneda, usdclp)
        kpi_row("Ingreso incremental", uplift, uplift, moneda, usdclp)
        st.metric("ROI", fmt_pct_val(roi*100 if pd.notna(roi) else np.nan))
        st.markdown("#### Detalle de asignación")
        st.dataframe(format_df_auto(det, moneda, usdclp).head(500), use_container_width=True, height=360)
    else:
        st.info("No hay universo elegible (o no se generaron incentivos). Revisa la Celda 13 del notebook.")

# ================
# Arista 4 — Capital / Provisiones
# ================
with tabs[3]:
    st.header("Arista 4 — Capital/Provisiones")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Optimizar *RWA* y *capital requerido (K)* y, cuando aplique, *provisiones (≈EL)*, liberando capacidad para crecer.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
- *EAD*: exposición en riesgo (base vs optimizada).  
- *RWA*: activos ponderados por riesgo (Basilea).  
- *K (capital requerido): **RWA × K_ratio*.  
- *Provisiones (≈EL)*: reservas por riesgo crediticio.
    """)

    st.markdown("### Análisis Ejecutivo")
    st.markdown("""
Con el portafolio más sano (Default) y el precio eficiente (Pricing), disminuye el *riesgo efectivo, se recalculan **RWA* y *K, y se evalúan **provisiones (≈EL)*.  
La reducción sostenida implica *liberación de capital* manteniendo el cumplimiento regulatorio.
    """)

    cap = dfs.get("cap_port")
    if cap is not None and not cap.empty:
        kpi_row("EAD", _first(cap,"EAD_base"), _first(cap,"EAD_opt"), moneda, usdclp)
        kpi_row("RWA", _first(cap,"RWA_base"), _first(cap,"RWA_opt"), moneda, usdclp)
        kpi_row("K (Capital requerido)", _first(cap,"K_base"), _first(cap,"K_opt"), moneda, usdclp)
        el_a = _first(cap, "EL_base") if "EL_base" in cap.columns else _first(cap, "prov_base")
        el_b = _first(cap, "EL_opt")  if "EL_opt"  in cap.columns else _first(cap, "prov_opt")
        if not (np.isnan(el_a) and np.isnan(el_b)):
            kpi_row("Provisiones (≈EL)", el_a, el_b, moneda, usdclp)

    seg = dfs.get("cap_seg")
    if seg is not None and not seg.empty:
        st.markdown("#### Resultados por segmento")
        st.dataframe(format_df_auto(seg, moneda, usdclp), use_container_width=True, height=360)

    det = dfs.get("cap_det")
    if det is not None and not det.empty:
        st.markdown("#### Detalle por cliente")
        st.dataframe(format_df_auto(det, moneda, usdclp).head(500), use_container_width=True, height=360)

# ================
# Guardrails
# ================
with tabs[4]:
    st.header("Guardrails (Resguardos)")

    st.markdown("""
Consolidación de límites regulatorios (Basilea III, IFRS 9) y de negocio.  
Se muestran umbrales, observados y la evaluación automática de cumplimiento por portafolio y segmento.
    """)

    gport = dfs.get("gr_port")
    gseg  = dfs.get("gr_seg")
    geval = dfs.get("gr_eval")

    def _fmt_cols(df, cols):
        if df is None or df.empty: return df
        out = df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = out[c].apply(fmt_pct_val)
        return out

    if gport is not None and not gport.empty:
        st.subheader("Portafolio")
        st.dataframe(_fmt_cols(gport, ["umbral","observado_actual","observado_optimizado"]), use_container_width=True)
    else:
        st.info("No se encontró guardrails_portfolio.csv en el bundle.")

    if gseg is not None and not gseg.empty:
        st.subheader("Segmento")
        if "observado" in gseg.columns:
            gseg = _fmt_cols(gseg, ["observado"])
        st.dataframe(gseg, use_container_width=True)

    if geval is not None and not geval.empty:
        st.subheader("Evaluación automática")
        geval = _fmt_cols(geval, ["obs","exp"])
        st.dataframe(geval, use_container_width=True)

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption(f"© MVP Bancario — Motor clásico consolidado. Escenario: *{escenario}* · Moneda: *{moneda}*")
