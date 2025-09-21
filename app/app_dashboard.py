# app/app_dashboard.py — MVP Bancario (4 Aristas) con escenarios CONS/POT y CLP/USD
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
    # KPI unificados (para resumen opcional)
    "kpis_all": "kpis_unificados.csv",
    # Meta opcional
    "manifest": "manifest.json",
    "run_meta": "run_meta.json",
}

# sufijos por escenario
SUF = {"Conservador": "", "Potenciado": "_agresivo"}

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    "/content/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
]

# ==========================
# Utilidades varias
# ==========================
def _dir_ok(d: str) -> bool:
    try:
        if not d or not os.path.isdir(d): return False
        # checamos existencia de al menos 1 archivo por arista para evitar falsos positivos
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
       Para guardrails usa SIEMPRE archivos sin sufijo (si existen)."""
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
    # Guardrails (no dependen de escenario)
    dfs["gr_port"]  = pick("gr_port", scenario_dependent=False)
    dfs["gr_seg"]   = pick("gr_seg",  scenario_dependent=False)
    dfs["gr_eval"]  = pick("gr_eval", scenario_dependent=False)
    # KPIs unificados globales (opcional)
    dfs["kpis_all"] = pick("kpis_all", scenario_dependent=False)

    return dfs

# ==========================
# Formato de números
# ==========================
_num_like = re.compile(r"^-?\d+(\.\d+)?$")

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

def _first(df: pd.DataFrame, col: str):
    try:
        return df[col].iloc[0] if df is not None and (not df.empty) and (col in df.columns) else np.nan
    except Exception:
        return np.nan

def _sum(df: pd.DataFrame, col: str):
    try:
        return float(pd.to_numeric(df[col], errors="coerce").sum()) if df is not None and col in df.columns else np.nan
    except Exception:
        return np.nan

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

st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Modelo matemático aplicado sobre un portafolio simulado. Comparación *Actual vs Optimizado* por escenario y arista.")

if not bundle_dir:
    st.error("No encuentro el bundle. Genera el paquete con la Celda 22 del notebook y vuelve a cargar.")
    st.stop()

bundle = Path(bundle_dir)
dfs = _load_for_scenario(bundle, escenario)

# Diagnóstico rápido
missing = []
for k, base in REQ_FILES_BASE.items():
    needed = (k.startswith("gr_") or k in ["kpis_all","manifest","run_meta"]) and (SUF.get(escenario,"")=="")
    p1 = bundle / base
    p2 = bundle / base.replace(".csv", f"{SUF[escenario]}.csv")
    if k.startswith("gr_") or k in ["kpis_all","manifest","run_meta"]:
        if not p1.exists(): missing.append(base)
    else:
        if not (p1.exists() or p2.exists()): missing.append(base + " (c/ sufijo opcional)")
if missing:
    st.warning("Faltan archivos en el bundle:\n- " + "\n- ".join(missing))

tabs = st.tabs([
    "Arista 1 – Default/Impago",
    "Arista 2 – Yield/Pricing",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital/Provisiones",
    "Guardrails (Resguardos)"
])

# ================
# Arista 1 – Default
# ================
with tabs[0]:
    st.header("Arista 1 – Default/Impago")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("""
    *Objetivo:* Reducir la *Pérdida Esperada (EL)* redistribuyendo la exposición (*EAD) desde clientes/segmentos de mayor riesgo hacia perfiles más sanos, **sin frenar el negocio*.
    """)

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - *EAD (Exposición)*: Saldo sujeto a riesgo.  
    - *EL (Expected Loss)*: PD × LGD × EAD — pérdida promedio esperada por riesgo de crédito.  
    - *Interés*: Ingreso financiero por tasa aplicada al EAD (APR × EAD).  
    - *Utilidad*: Interés − EL − COF (COF=cost of funds).  
    - *PD ponderado: Σ(PD × EAD) / Σ(EAD) — *mix de riesgo del portafolio.
    """)

    st.markdown("### Análisis Ejecutivo (Pitch)")
    st.success("El motor *redistribuye EAD* hacia clientes con *PD menor* sin reducir el total de exposición. "
               "El resultado: *EL disminuye, **PD ponderado baja, y **la utilidad sube* al mejorar el mix de riesgo.")

    port = dfs.get("def_port")
    if port is not None and not port.empty:
        kpi_row("EAD", _first(port,"EAD_actual"), _first(port,"EAD_optimizado"), moneda, usdclp,
                "Saldo total expuesto al riesgo (debe mantenerse estable).")
        kpi_row("EL (Pérdida Esperada)", _first(port,"EL_actual"), _first(port,"EL_optimizado"), moneda, usdclp,
                "PD × LGD × EAD; bajar EL indica menor pérdida promedio.")
        kpi_row("Utilidad", _first(port,"Utilidad_actual"), _first(port,"Utilidad_optimizada"), moneda, usdclp,
                "Interés − EL − COF; debe aumentar.")
        if "PD_pond_actual" in port.columns and "PD_pond_optimizado" in port.columns:
            kpi_row_pct("PD Ponderado", _first(port,"PD_pond_actual")*100, _first(port,"PD_pond_optimizado")*100,
                        "PD promedio ponderada por EAD.")

    det = dfs.get("def_det")
    if det is not None and not det.empty:
        st.markdown("#### Detalle por cliente")
        det_fmt = format_df_currency(det, 
                                     ["e_base","e_out","income_base","income_final","EL_base","EL_final","COF_base","COF_final","util_base","util_final"], 
                                     moneda, usdclp)
        st.dataframe(det_fmt.head(500), use_container_width=True, height=380)

# ================
# Arista 2 – Yield / Pricing
# ================
with tabs[1]:
    st.header("Arista 2 – Yield/Pricing")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("""
    *Objetivo:* Encontrar la *tasa (APR) óptima* que maximiza utilidad equilibrando *precio y volumen* (elasticidad).
    """)

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - *Ingreso total*: Flujo de intereses ajustado por el volumen resultante (APR_opt × EAD_out).  
    - *Utilidad total*: Ingreso − EL − COF.  
    - *EAD_in / EAD_out*: Exposición antes/después del ajuste de precio por demanda.
    """)
    st.info("Nota: ‘Ingreso’ aquí es *interés total* (flujo), no confundir con utilidad. Lo nombramos explícitamente para evitar ambigüedad.")

    st.markdown("### Análisis Ejecutivo (Pitch)")
    st.success("El precio óptimo mejora el *interés total* sin deteriorar la *utilidad*: si la tasa sube demasiado baja el volumen; "
               "si es muy baja, sube el volumen pero *no* la utilidad. Encontramos el *punto balance* por segmento.")

    port = dfs.get("yld_port")
    if port is not None and not port.empty:
        kpi_row("Interés Total", _first(port,"ingreso_base"), _first(port,"ingreso_opt"), moneda, usdclp)
        kpi_row("Utilidad Total", _first(port,"utilidad_base"), _first(port,"utilidad_opt"), moneda, usdclp)

    seg = dfs.get("yld_seg")
    if seg is not None and not seg.empty:
        st.markdown("#### Resultados por segmento")
        seg_fmt = format_df_currency(seg, ["ingreso_opt","utilidad_opt","EAD_in","EAD_out"], moneda, usdclp)
        st.dataframe(seg_fmt, use_container_width=True, height=380)

    det = dfs.get("yld_det")
    if det is not None and not det.empty:
        st.markdown("#### Detalle por cliente (pricing)")
        det_fmt = format_df_currency(det, ["ead_in","e_out","income_opt","EL_opt","COF_opt","util_opt"], moneda, usdclp)
        st.dataframe(det_fmt.head(500), use_container_width=True, height=380)

# ================
# Arista 3 – Incentivos
# ================
with tabs[2]:
    st.header("Arista 3 – Incentivos")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("""
    *Objetivo:* Invertir en *incentivos selectivos* donde el *ROI sea positivo, maximizando el **ingreso incremental* por peso invertido.
    """)

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - *Costo incentivos*: Gasto total en beneficios aplicados.  
    - *Ingreso incremental*: Ingreso adicional atribuible al incentivo (uplift).  
    - *ROI: Ingreso incremental / Costo — debe ser *> 0**.
    """)

    st.markdown("### Análisis Ejecutivo (Pitch)")
    st.success("Aplicamos incentivos *solo* donde el retorno es comprobable: más ingreso por cada peso. "
               "Si el ROI no supera el umbral, *no* se asignan incentivos.")

    det = dfs.get("inc_det")
    summ = dfs.get("inc_sum")
    if det is not None and not det.empty:
        # Intento robusto de columnas costo/uplift en diferentes nombres
        cost_cols = [c for c in det.columns if "cost" in c.lower() or "costo" in c.lower()]
        up_cols   = [c for c in det.columns if any(k in c.lower() for k in ["uplift","ingreso_inc","ingreso_incremental","delta_ingreso"])]
        costo = pd.to_numeric(det[cost_cols].sum(axis=1), errors="coerce").sum() if cost_cols else 0.0
        uplift = pd.to_numeric(det[up_cols].sum(axis=1), errors="coerce").sum() if up_cols else 0.0
        roi = uplift/costo if costo>0 else np.nan
        kpi_row("Costo de Incentivos", costo, costo, moneda, usdclp)
        kpi_row("Ingreso Incremental", uplift, uplift, moneda, usdclp)
        st.metric("ROI", fmt_pct_val(roi*100 if pd.notna(roi) else np.nan))
        det_fmt = format_df_currency(det, cost_cols+up_cols, moneda, usdclp)
        st.dataframe(det_fmt.head(500), use_container_width=True, height=380)
    else:
        st.info("No hay universo elegible (o no se generaron incentivos). Revisa la Celda 13 del notebook.")

# ================
# Arista 4 – Capital / Provisiones
# ================
with tabs[3]:
    st.header("Arista 4 – Capital / Provisiones")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("""
    *Objetivo:* Mejorar el consumo de *capital regulatorio* y optimizar *provisiones*, liberando recursos para crecer sin aumentar el riesgo.
    """)

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - *EAD*: Exposición en riesgo (base vs optimizada).  
    - *RWA*: Activos ponderados por riesgo (proxy de Basilea).  
    - *K*: Capital requerido (RWA × K_ratio).  
    - *Provisiones (≈ EL)*: Reservas por riesgo crediticio.
    """)

    st.markdown("### Análisis Ejecutivo (Pitch)")
    st.success("La reasignación de riesgo y el pricing mejorado *reducen RWA* y *provisiones* manteniendo el negocio, "
               "lo que *libera capital* y mejora la *rentabilidad ajustada por riesgo*.")

    cap = dfs.get("cap_port")
    if cap is not None and not cap.empty:
        def g0(df, name): return _first(df, name)
        kpi_row("EAD", g0(cap,"EAD_base"), g0(cap,"EAD_opt"), moneda, usdclp)
        kpi_row("RWA", g0(cap,"RWA_base"), g0(cap,"RWA_opt"), moneda, usdclp)
        kpi_row("K (Capital requerido)", g0(cap,"K_base"), g0(cap,"K_opt"), moneda, usdclp)
        # Algunas versiones exportan EL/provisiones
        if "EL_base" in cap.columns or "prov_base" in cap.columns:
            el_a = g0(cap, "EL_base") if "EL_base" in cap.columns else g0(cap, "prov_base")
            el_b = g0(cap, "EL_opt")  if "EL_opt"  in cap.columns else g0(cap, "prov_opt")
            kpi_row("Provisiones (≈EL)", el_a, el_b, moneda, usdclp)

    seg = dfs.get("cap_seg")
    if seg is not None and not seg.empty:
        st.markdown("#### Resultados por segmento")
        seg_fmt = format_df_currency(seg, ["EAD_base","EAD_opt","RWA_base","RWA_opt","K_base","K_opt","EL_base","EL_opt","prov_base","prov_opt"], moneda, usdclp)
        st.dataframe(seg_fmt, use_container_width=True, height=380)

    det = dfs.get("cap_det")
    if det is not None and not det.empty:
        st.markdown("#### Detalle por cliente")
        det_fmt = format_df_currency(det, ["EAD_base","EAD_opt","R_base","R_opt","EL_base","EL_opt","K_base","K_opt","RWA_base","RWA_opt","prov_base","prov_opt"], moneda, usdclp)
        st.dataframe(det_fmt.head(500), use_container_width=True, height=380)

# ================
# Guardrails
# ================
with tabs[4]:
    st.header("Guardrails (Resguardos)")
    st.markdown("Límites regulatorios (Basilea III, IFRS 9) y de negocio. *Nunca* deben violarse.")

    gport = dfs.get("gr_port")
    gseg  = dfs.get("gr_seg")
    geval = dfs.get("gr_eval")

    cols_pct_like = ["umbral","observado_actual","observado_optimizado","observado","exp"]
    if gport is not None and not gport.empty:
        st.subheader("Portafolio")
        gport_fmt = gport.copy()
        for c in cols_pct_like:
            if c in gport_fmt.columns:
                gport_fmt[c] = gport_fmt[c].apply(fmt_pct_val)
        st.dataframe(gport_fmt, use_container_width=True)
    else:
        st.info("No se encontró guardrails_portfolio.csv en el bundle.")

    if gseg is not None and not gseg.empty:
        st.subheader("Segmento")
        gseg_fmt = gseg.copy()
        if "observado" in gseg_fmt.columns:
            gseg_fmt["observado"] = gseg_fmt["observado"].apply(fmt_pct_val)
        st.dataframe(gseg_fmt, use_container_width=True)

    if geval is not None and not geval.empty:
        st.subheader("Evaluación automática")
        geval_fmt = geval.copy()
        for c in ["obs","exp"]:
            if c in geval_fmt.columns:
                geval_fmt[c] = geval_fmt[c].apply(fmt_pct_val)
        st.dataframe(geval_fmt, use_container_width=True)

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas integradas). Escenario: *{}. Moneda: *{}**.".format(escenario, moneda))
