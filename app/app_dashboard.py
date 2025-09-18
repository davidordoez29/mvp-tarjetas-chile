# app/app_dashboard.py — storytelling por arista (compat v2.0 + contrato)
import os, glob, json, math, re
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ==========================
# Archivos requeridos (nombres "clásicos" que usa la UI)
# ==========================
REQ_FILES = {
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
    "guard_port": "guardrails_portfolio.csv",
    "guard_seg":  "guardrails_segment.csv",
}

# Fallbacks compatibles con el Notebook v2.0 (aristaX_*.csv)
FALLBACK_FILES = {
    "def_port": ["arista1_portfolio.csv"],
    "def_seg":  ["arista1_segment.csv"],
    "def_det":  ["arista1_detail.csv"],
    "yld_port": ["arista2_portfolio.csv"],
    "yld_seg":  ["arista2_segment.csv"],
    "yld_det":  ["arista2_detail.csv"],
    "inc_det":  ["arista3_detail.csv"],
    "cap_port": ["arista4_portfolio.csv"],
    "cap_seg":  ["arista4_segment.csv"],
    "cap_det":  ["arista4_detail.csv"],
    # guardrails ya calzan nombres
}

# Detección de bundle: añadimos rutas del notebook v2.0
CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    os.environ.get("WDOF_BUNDLE_DIR", "").strip(),   # opcional: symlink/var de entorno
    "/content/out",                                   # outputs sueltos
    "/content/out/dashboard_bundle",                  # si empaquetas aquí
    "/content/bundle",                                # raíz bundles
    "./out/dashboard_bundle",
    "./dashboard_bundle",
]

# ==========================
# Utilidades de carga
# ==========================
def _dir_ok(d: str) -> bool:
    try:
        if not d or not os.path.isdir(d): 
            return False
        # con 6 archivos ya damos por válido
        hits = 0
        for v in REQ_FILES.values():
            if os.path.exists(os.path.join(d, v)):
                hits += 1
        # también aceptamos si hay guardrails + kpis + aristas nuevas
        hits += sum(os.path.exists(os.path.join(d, f)) for lst in FALLBACK_FILES.values() for f in lst)
        return hits >= 6
    except Exception:
        return False

def autodetect_bundle() -> str | None:
    # Preferencia: si hay contrato, priorizar su folder raíz
    contract_env = os.environ.get("DASHBOARD_CONTRACT", "").strip()
    if contract_env and Path(contract_env).exists():
        try:
            with open(contract_env, "r", encoding="utf-8") as f:
                contract = json.load(f)
            # si paths del contrato existen, tomamos su carpeta base
            any_path = next(iter(contract.get("paths", {}).values()), None)
            if any_path and Path(any_path).exists():
                base = str(Path(any_path).parent)
                if _dir_ok(base):
                    return base
        except Exception:
            pass

    # Si no, intentamos dirs candidatos
    for d in CANDIDATE_DIRS:
        if _dir_ok(d): 
            return d
    return None

def _read_csv(path: str):
    try:
        return pd.read_csv(path)
    except Exception as e:
        return None

def _resolve_path(bundle_dir: str, fname: str, fallbacks: list[str]) -> str | None:
    """Devuelve el primer path existente entre fname y sus fallbacks."""
    cand = [fname] + (fallbacks or [])
    for c in cand:
        p = os.path.join(bundle_dir, c)
        if os.path.exists(p):
            return p
    return None

def load_bundle(bundle_dir: str):
    """Carga dataframes según REQ_FILES, con fallbacks v2.0. Devuelve dfs, missing."""
    dfs, missing = {}, []
    for key, fname in REQ_FILES.items():
        fpath = _resolve_path(bundle_dir, fname, FALLBACK_FILES.get(key, []))
        if not fpath:
            missing.append(fname + " (no encontrado)")
            dfs[key] = None
            continue
        df = _read_csv(fpath)
        if df is None:
            missing.append(fname + f" (error al leer: {fpath})")
        dfs[key] = df
    return dfs, missing

# ==========================
# Normalizadores (mapear columnas a lo que UI espera)
# ==========================
def _num(x):
    return pd.to_numeric(x, errors="coerce")

def norm_default_port(df: pd.DataFrame) -> pd.DataFrame:
    """Soporta 'default_portfolio.csv' y 'arista1_portfolio.csv'."""
    if df is None or df.empty:
        return df
    df2 = df.copy()
    cols = {c.lower(): c for c in df2.columns}

    # Arista1 v2.0: ead_base, ead_final, EL_base, EL_final, apr_base_w, apr_final_w
    def pick(name):
        return cols.get(name.lower())

    if pick("EAD_actual") not in cols and pick("ead_base"):
        df2["EAD_actual"] = _num(df2[pick("ead_base")])
    if pick("EAD_optimizado") not in cols and pick("ead_final"):
        df2["EAD_optimizado"] = _num(df2[pick("ead_final")])
    if pick("EL_actual") not in cols and pick("EL_base"):
        df2["EL_actual"] = _num(df2[pick("EL_base")])
    if pick("EL_optimizado") not in cols and pick("EL_final"):
        df2["EL_optimizado"] = _num(df2[pick("EL_final")])

    # Utilidad: si no existe, aproximamos como Ingreso - EL, usando APR ponderado
    for side in [("actual","base"), ("optimizado","final")]:
        ui_name = f"Utilidad_{side[0]}"
        if ui_name not in df2.columns:
            r = pick(f"apr_{side[1]}_w")
            e = pick(f"ead_{side[1]}")
            el= pick(f"EL_{side[1]}")
            if r and e:
                ingreso = _num(df2[r]) * _num(df2[e])
                util = ingreso - (_num(df2[el]) if el in df2.columns else 0.0)
                df2[ui_name] = ingreso if el not in df2.columns else util

    # PD ponderado: si no existe, lo dejamos NaN (la UI es robusta)
    return df2

def norm_yield_port(df: pd.DataFrame) -> pd.DataFrame:
    """Soporta 'yield_portfolio.csv' y 'arista2_portfolio.csv'."""
    if df is None or df.empty:
        return df
    df2 = df.copy()
    cols = {c.lower(): c for c in df2.columns}
    def pick(name): return cols.get(name.lower())

    # v2.0: income_base, income_final
    if pick("ingreso_base") not in cols and pick("income_base"):
        df2["ingreso_base"] = _num(df2[pick("income_base")])
    if pick("ingreso_opt") not in cols and pick("income_final"):
        df2["ingreso_opt"] = _num(df2[pick("income_final")])

    # utilidad_* si existen en alguna variante
    for base, alt in [("utilidad_base","profit_base"), ("utilidad_opt","profit_final")]:
        if base not in df2.columns and pick(alt):
            df2[base] = _num(df2[pick(alt)])
    return df2

def norm_cap_port(df: pd.DataFrame) -> pd.DataFrame:
    """Soporta 'capital_portfolio.csv' y 'arista4_portfolio.csv'."""
    if df is None or df.empty:
        return df
    df2 = df.copy()
    cols = {c.lower(): c for c in df2.columns}
    def pick(name): return cols.get(name.lower())

    # Mapas comunes:
    # capital_req_base/opt  ~ K_base/K_final  ó campos equivalentes
    if "capital_req_base" not in df2.columns:
        c = pick("K_base")
        if c: df2["capital_req_base"] = _num(df2[c])
    if "capital_req_opt" not in df2.columns:
        c = pick("K_final")
        if c: df2["capital_req_opt"] = _num(df2[c])

    if "prov_base" not in df2.columns:
        c = pick("prov_base")
        if c: df2["prov_base"] = _num(df2[c])
    if "prov_opt" not in df2.columns:
        c = pick("prov_final")
        if c: df2["prov_opt"] = _num(df2[c])
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
default_dir = autodetect_bundle()
bundle_dir = st.sidebar.text_input(
    "📦 Ruta del bundle",
    value=(default_dir or ""),
    help="Ej: /content/out o /content/bundle/bundle_<RUN_ID>"
).strip() or default_dir

contract_hint = st.sidebar.text_input(
    "📄 (Opcional) dashboard_contract.json",
    value=os.environ.get("DASHBOARD_CONTRACT","").strip()
).strip()

if not bundle_dir:
    st.error("No encuentro el bundle. Genera el paquete en el notebook y vuelve a cargar.")
    st.stop()

# Cargar contrato si se indicó, para hints (no obligatorio)
if contract_hint and Path(contract_hint).exists():
    try:
        with open(contract_hint, "r", encoding="utf-8") as f:
            _contract = json.load(f)
        st.sidebar.caption("Contrato cargado ✓")
    except Exception as _:
        st.sidebar.caption("No pude leer el contrato (opcional).")

dfs, missing = load_bundle(bundle_dir)
if missing:
    st.warning("Faltan archivos en el bundle:\n- " + "\n- ".join(missing))

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
# Arista 1 – Default
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
    st.success("La optimización disminuye la pérdida esperada y aumenta utilidad redirigiendo exposición a clientes más sanos.")

    port = norm_default_port(dfs.get("def_port"))
    if port is not None and not port.empty:
        def g0(df, col): return df[col].iloc[0] if col in df.columns else np.nan
        kpi_row("EAD", g0(port,"EAD_actual"), g0(port,"EAD_optimizado"), moneda, usdclp)
        kpi_row("EL (Pérdida Esperada)", g0(port,"EL_actual"), g0(port,"EL_optimizado"), moneda, usdclp)
        kpi_row("Utilidad", g0(port,"Utilidad_actual"), g0(port,"Utilidad_optimizada"), moneda, usdclp)
        # PD ponderado si existe
        if "PD_pond_actual" in port.columns and "PD_pond_optimizado" in port.columns:
            kpi_row_pct("PD Ponderado", port["PD_pond_actual"].iloc[0]*100, port["PD_pond_optimizado"].iloc[0]*100)

# ================
# Arista 2 – Yield / Pricing
# ================
with tabs[1]:
    st.header("Arista 2 – Yield/Pricing")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Encontramos la tasa (APR) óptima que maximiza utilidad equilibrando precio y volumen.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - Ingreso total: Flujo de intereses ajustado por volumen.  
    - Utilidad total: Ingreso – EL – Costos.  
    - Ingreso/Utilidad aislado: Solo efecto precio, manteniendo EAD fijo.
    """)

    st.markdown("### Análisis Ejecutivo")
    st.success("Ajustamos el precio como un tendero que encuentra el punto ideal: si cobra demasiado, pierde clientes; si cobra poco, gana volumen pero no rentabilidad.")

    port = norm_yield_port(dfs.get("yld_port"))
    if port is not None and not port.empty:
        def g0(df, name): return df[name].iloc[0] if name in df.columns else np.nan
        kpi_row("Ingreso Total", g0(port,"ingreso_base"), g0(port,"ingreso_opt"), moneda, usdclp)
        # Utilidad total si está disponible
        ub = g0(port, "utilidad_base"); uo = g0(port, "utilidad_opt")
        if not (np.isnan(ub) and np.isnan(uo)):
            kpi_row("Utilidad Total", ub, uo, moneda, usdclp)

# ================
# Arista 3 – Incentivos
# ================
with tabs[2]:
    st.header("Arista 3 – Incentivos")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Invertimos en incentivos solo donde el ROI es positivo: más ingresos por cada peso invertido.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - Costo incentivos: gasto total en beneficios.  
    - Ingreso incremental: ingresos adicionales generados.  
    - ROI: Retorno de la inversión = Ingreso / Costo.
    """)

    st.markdown("### Análisis Ejecutivo")
    st.success("Es como fertilizar solo las plantas que realmente responden: cada peso en incentivos genera retorno multiplicado.")

    det = dfs.get("inc_det"); summ = dfs.get("inc_sum")
    if det is not None and not det.empty:
        cost_cols = [c for c in det.columns if "cost" in c.lower()]
        up_cols   = [c for c in det.columns if "uplift" in c.lower() or "ingreso_incremental" in c.lower() or "delta_ingreso" in c.lower()]
        cost = pd.to_numeric(det[cost_cols].sum(axis=1), errors="coerce").fillna(0).sum() if cost_cols else 0.0
        uplift = pd.to_numeric(det[up_cols].sum(axis=1), errors="coerce").fillna(0).sum() if up_cols else 0.0
        roi = uplift/cost if cost>0 else np.nan
        kpi_row("Costo de Incentivos", cost, cost, moneda, usdclp)
        kpi_row("Ingreso Incremental", uplift, uplift, moneda, usdclp)
        st.metric("ROI", fmt_pct_val(roi*100 if pd.notna(roi) else np.nan))

# ================
# Arista 4 – Capital / Provisiones
# ================
with tabs[3]:
    st.header("Arista 4 – Capital / Provisiones")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Hacemos más eficiente el capital requerido y reducimos provisiones, liberando recursos para crecer.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - Capital requerido: Consumo de capital regulatorio (proxy RW×K×EAD).  
    - Provisiones: reservas por riesgo crediticio ≈ EL.  
    - Liberación: diferencia de capital/provisiones antes y después.
    """)

    st.markdown("### Análisis Ejecutivo")
    st.success("Reorganizamos el capital inmovilizado: seguimos protegidos, pero con menos exceso, liberando recursos para oportunidades rentables.")

    cap_port = norm_cap_port(dfs.get("cap_port"))
    if cap_port is not None and not cap_port.empty:
        def g0(df, name): return df[name].iloc[0] if name in df.columns else np.nan
        kpi_row("Capital Requerido", g0(cap_port,"capital_req_base"), g0(cap_port,"capital_req_opt"), moneda, usdclp)
        kpi_row("Provisiones", g0(cap_port,"prov_base"), g0(cap_port,"prov_opt"), moneda, usdclp)

# ================
# Guardrails
# ================
with tabs[4]:
    st.header("Guardrails (Resguardos)")
    st.markdown("Límites regulatorios y de negocio para asegurar robustez y cumplimiento.")

    gport = dfs.get("guard_port"); gseg = dfs.get("guard_seg")

    if gport is None or gport.empty:
        st.info("No hay tablas de guardrails en el bundle. Genera con las celdas 15–16 del notebook.")
    else:
        gport_fmt = gport.copy()
        pct_like_cols = [c for c in gport_fmt.columns if "share" in c.lower() or "ratio" in c.lower() or "pct" in c.lower()]
        for c in pct_like_cols:
            gport_fmt[c] = gport_fmt[c].apply(fmt_pct_val)
        st.subheader("Portafolio")
        st.dataframe(gport_fmt, use_container_width=True)

    if gseg is not None and not gseg.empty:
        gseg_fmt = gseg.copy()
        # Si hay 'concentration_share' u otras proporciones
        for c in gseg_fmt.columns:
            if "share" in c.lower() or "pct" in c.lower():
                gseg_fmt[c] = gseg_fmt[c].apply(fmt_pct_val)
        st.subheader("Segmento")
        st.dataframe(gseg_fmt, use_container_width=True)

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas integradas).")
