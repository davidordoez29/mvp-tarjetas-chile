# app/app_dashboard.py — storytelling por arista (compat v2.0 + contrato + fixes KPIs)
import os, json, math, re
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ==========================
# Archivos requeridos (por escenario)
# ==========================
# Claves lógicas -> base filename (sin sufijo)
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
    # Estos son comunes (sin sufijo por escenario)
    "guard_port": "guardrails_portfolio.csv",
    "guard_seg":  "guardrails_segment.csv",
    "kpis":       "kpis_unificados.csv",
}

SUFFIX = {"Conservador": "", "Potenciado": "_agresivo"}

def files_for_scenario(scenario: str) -> dict[str, str]:
    """Devuelve {key: filename} aplicando sufijo por escenario donde corresponda."""
    suf = SUFFIX.get(scenario, "")
    files = {}
    for k, base in REQ_FILES_BASE.items():
        if k in {"guard_port", "guard_seg", "kpis"}:
            files[k] = base  # sin sufijo
        else:
            # inserta sufijo antes de .csv si aplica
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
# Utilidades de carga (escenario-aware)
# ==========================
def _dir_ok(d: str) -> bool:
    """Carpeta válida si contiene ~6 archivos del escenario conservador o potenciado."""
    try:
        if not d or not os.path.isdir(d): 
            return False
        # prueba conservador
        req_cons = files_for_scenario("Conservador")
        hits_cons = sum(os.path.exists(os.path.join(d, v)) for v in req_cons.values())
        # prueba potenciado
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
# Normalizadores (mapear columnas a lo que la UI espera)
# ==========================
def _num(x):
    return pd.to_numeric(x, errors="coerce")

def norm_default_port(df: pd.DataFrame) -> pd.DataFrame:
    """Soporta default_portfolio.csv y arista1_portfolio.csv; completa Ingreso/Utilidad si faltan."""
    if df is None or df.empty:
        return df
    df2 = df.copy()
    cols = {c.lower(): c for c in df2.columns}
    def pick(name): return cols.get(name.lower())

    EAD_ACT = pick("EAD_actual") or pick("ead_base")
    EAD_OPT = pick("EAD_optimizado") or pick("ead_final")
    EL_ACT  = pick("EL_actual") or pick("EL_base")
    EL_OPT  = pick("EL_optimizado") or pick("EL_final")
    APR_ACT = pick("apr_base_w")
    APR_OPT = pick("apr_final_w")

    if "EAD_actual" not in df2.columns and EAD_ACT: df2["EAD_actual"] = _num(df2[EAD_ACT])
    if "EAD_optimizado" not in df2.columns and EAD_OPT: df2["EAD_optimizado"] = _num(df2[EAD_OPT])
    if "EL_actual" not in df2.columns and EL_ACT: df2["EL_actual"] = _num(df2[EL_ACT])
    if "EL_optimizado" not in df2.columns and EL_OPT: df2["EL_optimizado"] = _num(df2[EL_OPT])

    if APR_ACT and "EAD_actual" in df2.columns and "ingreso_actual" not in df2.columns:
        df2["ingreso_actual"] = _num(df2[APR_ACT]) * _num(df2["EAD_actual"])
    if APR_OPT and "EAD_optimizado" in df2.columns and "ingreso_optimizado" not in df2.columns:
        df2["ingreso_optimizado"] = _num(df2[APR_OPT]) * _num(df2["EAD_optimizado"])

    if "Utilidad_actual" not in df2.columns and "ingreso_actual" in df2.columns:
        df2["Utilidad_actual"] = df2["ingreso_actual"] - (df2["EL_actual"] if "EL_actual" in df2.columns else 0.0)
    if "Utilidad_optimizada" not in df2.columns and "ingreso_optimizado" in df2.columns:
        df2["Utilidad_optimizada"] = df2["ingreso_optimizado"] - (df2["EL_optimizado"] if "EL_optimizado" in df2.columns else 0.0)

    return df2

def norm_yield_port(df: pd.DataFrame) -> pd.DataFrame:
    """Soporta yield_portfolio.csv y arista2_portfolio.csv."""
    if df is None or df.empty:
        return df
    df2 = df.copy()
    cols = {c.lower(): c for c in df2.columns}
    def pick(name): return cols.get(name.lower())

    if "ingreso_base" not in df2.columns and pick("income_base"):
        df2["ingreso_base"] = _num(df2[pick("income_base")])
    if "ingreso_opt" not in df2.columns and pick("income_final"):
        df2["ingreso_opt"] = _num(df2[pick("income_final")])

    for base, alt in [("utilidad_base","profit_base"), ("utilidad_opt","profit_final")]:
        if base not in df2.columns and pick(alt):
            df2[base] = _num(df2[pick(alt)])
    return df2

def norm_cap_port(df: pd.DataFrame) -> pd.DataFrame:
    """Soporta capital_portfolio.csv y arista4_portfolio.csv."""
    if df is None or df.empty:
        return df
    df2 = df.copy()
    cols = {c.lower(): c for c in df2.columns}
    def pick(name): return cols.get(name.lower())

    if "capital_req_base" not in df2.columns and pick("K_base"):
        df2["capital_req_base"] = _num(df2[pick("K_base")])
    if "capital_req_opt" not in df2.columns and pick("K_final"):
        df2["capital_req_opt"] = _num(df2[pick("K_final")])

    if "prov_base" not in df2.columns and pick("prov_base"):
        df2["prov_base"] = _num(df2[pick("prov_base")])
    if "prov_opt" not in df2.columns and pick("prov_final"):
        df2["prov_opt"] = _num(df2[pick("prov_final")])
    return df2

# ==========================
# Formato de números (robusto)
# ==========================
_num_like = re.compile(r"^-?\d+(\.\d+)?$")

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

def fmt_pct_val(val):
    if isinstance(val, str):
        s = val.strip()
        if s.endswith("%"):  # ya viene formateado
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

# ==========================
# App
# ==========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

st.sidebar.title("⚙️ Configuración")

# Selección de escenario (Conservador vs Potenciado)
scenario = st.sidebar.radio("Escenario", ["Conservador", "Potenciado"], horizontal=True)

# Path del bundle
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
st.caption("Modelo matemático aplicado sobre un portafolio. Comparación Actual vs Optimizado.")

tabs = st.tabs([
    "Arista 1 – Default/Impago",
    "Arista 2 – Yield/Pricing",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital/Provisiones",
    "Guardrails (Resguardos)"
])

# ================
# Arista 1 – Default/Impago (dos escenarios en una sola pestaña)
# ================
with tabs[0]:
    st.header("Arista 1 – Default/Impago")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Reducimos la pérdida esperada (EL) reasignando la exposición a segmentos menos riesgosos, sin frenar el negocio. *Esta arista NO toca tasas (APR)*.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - EAD: Exposición en riesgo (monto sobre el que se calcula pérdida e ingreso).  
    - EL: Pérdida Esperada = PD × LGD × EAD.  
    - Ingreso: r_base × EAD (no cambiamos r_base aquí).  
    - Utilidad: Ingreso – EL (sin costos operativos en esta vista).  
    - PD ponderado: Probabilidad de default promedio, ponderada por EAD.
    """)

    # ----- utilidades locales
    def g0(df, name):
        return df[name].iloc[0] if (df is not None and not df.empty and name in df.columns) else np.nan

    def render_kpis_block(title, port_df):
        st.subheader(title)
        if port_df is None or port_df.empty:
            st.info("No se encontraron datos para este escenario en el bundle.")
            return
        c_ead_a  = g0(port_df,"EAD_actual");       c_ead_o  = g0(port_df,"EAD_optimizado")
        c_el_a   = g0(port_df,"EL_actual");        c_el_o   = g0(port_df,"EL_optimizado")
        c_inc_a  = g0(port_df,"Ingreso_actual");   c_inc_o  = g0(port_df,"Ingreso_optimizado")
        c_ut_a   = g0(port_df,"Utilidad_actual");  c_ut_o   = g0(port_df,"Utilidad_optimizada")
        c_pd_a   = g0(port_df,"PD_pond_actual");   c_pd_o   = g0(port_df,"PD_pond_optimizado")

        kpi_row("EAD", c_ead_a, c_ead_o, moneda, usdclp, help_text="Exposición total (debe mantenerse ≈ constante).")
        kpi_row("EL (Pérdida Esperada)", c_el_a, c_el_o, moneda, usdclp, help_text="Objetivo: EL optimizado menor (−5% a −20%).")
        kpi_row("Ingreso", c_inc_a, c_inc_o, moneda, usdclp)
        kpi_row("Utilidad", c_ut_a, c_ut_o, moneda, usdclp, help_text="Ingreso – EL (sin costos).")
        if pd.notna(c_pd_a) and pd.notna(c_pd_o):
            # PD viene en [0,1]; lo mostramos en %
            kpi_row_pct("PD Ponderado", c_pd_a*100.0, c_pd_o*100.0)

    # ----- Análisis ejecutivo
    st.markdown("### Análisis Ejecutivo")
    st.success("Primero mostramos un escenario *Conservador* (movimientos acotados de EAD) y debajo un escenario *Potenciado* (mayor eficiencia, siempre respetando guardrails). Ambos cumplen IFRS9 y Basilea III.")

    # ----- KPIs: Conservador (default_portfolio.csv)
    port_cons = dfs.get("def_port")
    render_kpis_block("Escenario Conservador", port_cons)

    st.divider()

    # ----- KPIs: Potenciado (default_portfolio_agresivo.csv)
    port_poten = dfs.get("def_port_aggr")
    render_kpis_block("Escenario Potenciado", port_poten)

    # ----- Tablas opcionales (expandibles)
    with st.expander("Ver tablas de detalle (Conservador)"):
        seg = dfs.get("def_seg"); det = dfs.get("def_det")
        if seg is not None and not seg.empty:
            st.markdown("*Segmento (Conservador)*")
            # Formateos suaves: moneda para montos y % para PD
            seg_show = seg.copy()
            for c in ["EAD_actual","EAD_optimizado","EL_actual","EL_optimizado","Ingreso_actual","Ingreso_optimizado","Utilidad_actual","Utilidad_optimizada"]:
                if c in seg_show.columns: seg_show[c] = seg_show[c].apply(lambda v: fmt_money_val(v, moneda, usdclp))
            if "PD_pond_actual" in seg_show.columns: seg_show["PD_pond_actual"] = seg_show["PD_pond_actual"].apply(lambda v: fmt_pct_val((v if pd.isna(v) else v*100)))
            if "PD_pond_optimizado" in seg_show.columns: seg_show["PD_pond_optimizado"] = seg_show["PD_pond_optimizado"].apply(lambda v: fmt_pct_val((v if pd.isna(v) else v*100)))
            st.dataframe(seg_show, use_container_width=True)
        if det is not None and not det.empty:
            st.markdown("*Detalle clientes (Conservador)*")
            st.dataframe(det.head(2000), use_container_width=True)

    with st.expander("Ver tablas de detalle (Potenciado)"):
        seg_a = dfs.get("def_seg_aggr"); det_a = dfs.get("def_det_aggr")
        if seg_a is not None and not seg_a.empty:
            st.markdown("*Segmento (Potenciado)*")
            seg_show = seg_a.copy()
            for c in ["EAD_actual","EAD_optimizado","EL_actual","EL_optimizado","Ingreso_actual","Ingreso_optimizado","Utilidad_actual","Utilidad_optimizada"]:
                if c in seg_show.columns: seg_show[c] = seg_show[c].apply(lambda v: fmt_money_val(v, moneda, usdclp))
            if "PD_pond_actual" in seg_show.columns: seg_show["PD_pond_actual"] = seg_show["PD_pond_actual"].apply(lambda v: fmt_pct_val((v if pd.isna(v) else v*100)))
            if "PD_pond_optimizado" in seg_show.columns: seg_show["PD_pond_optimizado"] = seg_show["PD_pond_optimizado"].apply(lambda v: fmt_pct_val((v if pd.isna(v) else v*100)))
            st.dataframe(seg_show, use_container_width=True)
        if det_a is not None and not det_a.empty:
            st.markdown("*Detalle clientes (Potenciado)*")
            st.dataframe(det_a.head(2000), use_container_width=True)

# ================
# Arista 2 – Yield / Pricing (Conservador + Potenciado en una pestaña)
# ================
with tabs[1]:
    st.header("Arista 2 – Yield/Pricing")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Encontramos la tasa (APR) óptima que maximiza utilidad equilibrando precio y volumen, *respetando bandas/caps* y elasticidades razonables.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - Ingreso total: r × EAD.  
    - Utilidad total: Ingreso – EL.  
    - EAD in/out: EAD de entrada (post Arista 1) vs EAD tras pricing (elasticidad).  
    """)

    def g0(df, name):
        return df[name].iloc[0] if (df is not None and not df.empty and name in df.columns) else np.nan

    def render_yield_block(title, port_df):
        st.subheader(title)
        if port_df is None or port_df.empty:
            st.info("No se encontraron datos para este escenario en el bundle.")
            return
        inc_b = g0(port_df,"ingreso_base"); inc_o = g0(port_df,"ingreso_opt")
        utl_b = g0(port_df,"utilidad_base"); utl_o = g0(port_df,"utilidad_opt")
        ead_in = g0(port_df,"EAD_in"); ead_out = g0(port_df,"EAD_out")

        kpi_row("Ingreso Total", inc_b, inc_o, moneda, usdclp)
        kpi_row("Utilidad Total", utl_b, utl_o, moneda, usdclp)
        kpi_row("EAD (in → out)", ead_in, ead_out, moneda, usdclp, help_text="EAD_in: post Arista 1 | EAD_out: tras pricing (elasticidad)")

    st.markdown("### Análisis Ejecutivo")
    st.success("Arriba mostramos el escenario *Conservador* (ajustes moderados de APR y baja elasticidad). Abajo el escenario *Potenciado* (ajustes más amplios pero *siempre dentro de caps*).")

    # Conservador
    render_yield_block("Escenario Conservador", dfs.get("yld_port"))
    st.divider()
    # Potenciado
    render_yield_block("Escenario Potenciado", dfs.get("yld_port_aggr"))

    # Tablas expandibles
    with st.expander("Ver tablas de detalle (Conservador)"):
        seg = dfs.get("yld_seg"); det = dfs.get("yld_det"); cur = dfs.get("yld_curv")
        if seg is not None and not seg.empty:
            st.markdown("*Segmento (Conservador)*")
            seg_show = seg.copy()
            for c in ["ingreso_base","ingreso_opt","utilidad_base","utilidad_opt","EAD_in","EAD_out"]:
                if c in seg_show.columns: seg_show[c] = seg_show[c].apply(lambda v: fmt_money_val(v, moneda, usdclp))
            st.dataframe(seg_show, use_container_width=True)
        if cur is not None and not cur.empty:
            st.markdown("*Curva por segmento (APR ponderado & EAD)*")
            st.dataframe(cur, use_container_width=True)
        if det is not None and not det.empty:
            st.markdown("*Detalle clientes (Conservador)*")
            st.dataframe(det.head(2000), use_container_width=True)

    with st.expander("Ver tablas de detalle (Potenciado)"):
        seg = dfs.get("yld_seg_aggr"); det = dfs.get("yld_det_aggr"); cur = dfs.get("yld_curv_aggr")
        if seg is not None and not seg.empty:
            st.markdown("*Segmento (Potenciado)*")
            seg_show = seg.copy()
            for c in ["ingreso_base","ingreso_opt","utilidad_base","utilidad_opt","EAD_in","EAD_out"]:
                if c in seg_show.columns: seg_show[c] = seg_show[c].apply(lambda v: fmt_money_val(v, moneda, usdclp))
            st.dataframe(seg_show, use_container_width=True)
        if cur is not None and not cur.empty:
            st.markdown("*Curva por segmento (APR ponderado & EAD)*")
            st.dataframe(cur, use_container_width=True)
        if det is not None and not det.empty:
            st.markdown("*Detalle clientes (Potenciado)*")
            st.dataframe(det.head(2000), use_container_width=True)

# ================
# Arista 3 – Incentivos (Conservador + Potenciado en una pestaña)
# ================
with tabs[2]:
    st.header("Arista 3 – Incentivos")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Invertimos en incentivos *solo donde el ROI es positivo, con un **presupuesto* que puede ser conservador o potenciado.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - Costo incentivos: gasto total en beneficios asignados.  
    - Ingreso incremental: ingresos adicionales generados por incentivos.  
    - ROI: Ingreso incremental / Costo (debe ser > 0).  
    """)

    def render_incent_block(title, det_df, sum_df):
        st.subheader(title)
        if (det_df is None or det_df.empty) and (sum_df is None or sum_df.empty):
            st.info("No se encontraron datos para este escenario en el bundle.")
            return
        # Cálculo robusto por si el summary no está
        if sum_df is not None and not sum_df.empty:
            costo = float(sum_df.get("budget_usado", pd.Series([np.nan])).iloc[0])
            ingreso_inc = float(sum_df.get("ingreso_inc_total", pd.Series([np.nan])).iloc[0])
        else:
            # fallback al detalle
            costo = pd.to_numeric(det_df.get("incentivo", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() if det_df is not None else np.nan
            ingreso_inc = pd.to_numeric(det_df.get("ingreso_incremental", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() if det_df is not None else np.nan
        roi = (ingreso_inc / costo) if (costo and costo>0) else np.nan

        kpi_row("Costo de Incentivos", costo, costo, moneda, usdclp)
        kpi_row("Ingreso Incremental", ingreso_inc, ingreso_inc, moneda, usdclp)
        st.metric("ROI", fmt_pct_val((roi*100) if pd.notna(roi) else np.nan))

        # Tabla detalle
        if det_df is not None and not det_df.empty:
            with st.expander("Ver detalle de clientes"):
                st.dataframe(det_df.head(2000), use_container_width=True)
        if sum_df is not None and not sum_df.empty:
            with st.expander("Ver resumen de diagnóstico"):
                st.dataframe(sum_df, use_container_width=True)

    st.markdown("### Análisis Ejecutivo")
    st.success("El escenario *Conservador* usa un presupuesto pequeño (cobertura acotada). El *Potenciado* amplía cobertura manteniendo *ROI>0* y límites definidos.")

    # Conservador
    render_incent_block("Escenario Conservador", dfs.get("inc_det"), dfs.get("inc_sum"))
    st.divider()
    # Potenciado
    render_incent_block("Escenario Potenciado", dfs.get("inc_det_aggr"), dfs.get("inc_sum_aggr"))

    # Sensibilidad de presupuesto (si existe)
    with st.expander("Sensibilidad de presupuesto (si está disponible)"):
        sens_c = dfs.get("inc_sens"); sens_a = dfs.get("inc_sens_aggr")
        if sens_c is not None and not sens_c.empty:
            st.markdown("*Conservador*")
            st.dataframe(sens_c, use_container_width=True)
        if sens_a is not None and not sens_a.empty:
            st.markdown("*Potenciado*")
            st.dataframe(sens_a, use_container_width=True)

# ================
# Arista 4 – Capital / Provisiones (Conservador + Potenciado en una pestaña)
# ================
with tabs[3]:
    st.header("Arista 4 – Capital / Provisiones")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Hacemos más eficiente el *capital requerido* (RWA, K) y reducimos *provisiones* (≈EL IFRS9) manteniendo el mismo negocio.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - Capital requerido (K): proporción regulatoria aplicada a RWA.  
    - RWA: activos ponderados por riesgo (proxy estándar sobre EAD).  
    - Provisiones (≈EL): PD × LGD × EAD (IFRS9).  
    """)

    def g0(df, name):
        return df[name].iloc[0] if (df is not None and not df.empty and name in df.columns) else np.nan

    def render_capital_block(title, port_df):
        st.subheader(title)
        if port_df is None or port_df.empty:
            st.info("No se encontraron datos para este escenario en el bundle.")
            return
        ead = g0(port_df,"EAD"); el = g0(port_df,"EL")
        rwa = g0(port_df,"RWA"); k  = g0(port_df,"K")

        kpi_row("EAD", ead, ead, moneda, usdclp)
        kpi_row("Provisiones (≈EL)", el, el, moneda, usdclp)
        kpi_row("RWA", rwa, rwa, moneda, usdclp)
        kpi_row("Capital Requerido (K)", k, k, moneda, usdclp)

    st.markdown("### Análisis Ejecutivo")
    st.success("El escenario *Conservador* muestra el consumo de capital y provisiones con ajustes moderados. El *Potenciado* refleja mayor eficiencia en consumo de capital, siempre dentro del marco Basel/IFRS.")

    # Conservador
    render_capital_block("Escenario Conservador", dfs.get("cap_port"))
    st.divider()
    # Potenciado
    render_capital_block("Escenario Potenciado", dfs.get("cap_port_aggr"))

    # Tablas expandibles
    with st.expander("Ver tablas de detalle"):
        seg_c = dfs.get("cap_seg"); det_c = dfs.get("cap_det")
        seg_a = dfs.get("cap_seg_aggr"); det_a = dfs.get("cap_det_aggr")

        if seg_c is not None and not seg_c.empty:
            st.markdown("*Segmento (Conservador)*")
            seg_show = seg_c.copy()
            for c in ["EAD","EL","RWA","K"]:
                if c in seg_show.columns: seg_show[c] = seg_show[c].apply(lambda v: fmt_money_val(v, moneda, usdclp))
            st.dataframe(seg_show, use_container_width=True)

        if seg_a is not None and not seg_a.empty:
            st.markdown("*Segmento (Potenciado)*")
            seg_show = seg_a.copy()
            for c in ["EAD","EL","RWA","K"]:
                if c in seg_show.columns: seg_show[c] = seg_show[c].apply(lambda v: fmt_money_val(v, moneda, usdclp))
            st.dataframe(seg_show, use_container_width=True)

        if det_c is not None and not det_c.empty:
            st.markdown("*Detalle clientes (Conservador)*")
            st.dataframe(det_c.head(2000), use_container_width=True)
        if det_a is not None and not det_a.empty:
            st.markdown("*Detalle clientes (Potenciado)*")
            st.dataframe(det_a.head(2000), use_container_width=True)

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas integradas).")
