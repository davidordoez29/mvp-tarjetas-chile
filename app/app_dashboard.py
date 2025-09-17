# app/app_dashboard.py — storytelling por arista (con fixes)

import os, glob, json, math, re
import pandas as pd
import numpy as np
import streamlit as st

# ==========================
# Archivos requeridos
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

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    "/content/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
]

# ==========================
# Utilidades de carga
# ==========================
def _dir_ok(d: str) -> bool:
    try:
        if not d or not os.path.isdir(d): return False
        hits = sum(os.path.exists(os.path.join(d, v)) for v in REQ_FILES.values())
        return hits >= 6
    except Exception:
        return False

def autodetect_bundle() -> str | None:
    for d in CANDIDATE_DIRS:
        if _dir_ok(d): return d
    return None

def load_bundle(bundle_dir: str):
    dfs, missing = {}, []
    for key, fname in REQ_FILES.items():
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
# Formato de números (robusto)
# ==========================
_num_like = re.compile(r"^-?\d+(\.\d+)?$")

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
        # si viene con separadores raros, mejor devuélvelo tal cual
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
    # Acepta número o string. Si no es numérico, devuelve el original limpio.
    if isinstance(val, str):
        s = val.strip()
        # si ya trae %, devuélvelo (normalizando coma/punto mínimamente)
        if s.endswith("%"):
            return s.replace(".", ",")  # preferimos coma para decimales
        # si es rango u otro texto, déjalo intacto
        if not _num_like.match(s.replace(",", ".")):
            return s
        # si es numérico en string, seguimos abajo
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
bundle_dir = st.sidebar.text_input("📦 Ruta del bundle", value=(default_dir or ""), help="Ej: /content/out/dashboard_bundle").strip() or default_dir

if not bundle_dir:
    st.error("No encuentro el bundle. Genera el paquete en el notebook y vuelve a cargar.")
    st.stop()

dfs, missing = load_bundle(bundle_dir)
if missing:
    st.warning("Faltan archivos en el bundle:\n- " + "\n- ".join(missing))

moneda = st.sidebar.radio("Moneda a visualizar", ["CLP", "USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))

st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Modelo matemático aplicado sobre un portafolio simulado. Comparación *Actual vs Optimizado*.")

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
    st.markdown("Reducimos la *pérdida esperada (EL)* reasignando la exposición a segmentos menos riesgosos, sin frenar el negocio.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - *EAD*: Exposición en riesgo.  
    - *EL*: Pérdida Esperada = PD × LGD × EAD.  
    - *Ingreso*: APR × EAD.  
    - *Utilidad*: Ingreso – EL – Costos.  
    - *PD ponderado*: Probabilidad de default promedio, ponderada por EAD.
    """)

    st.markdown("### Análisis Ejecutivo")
    st.success("La optimización disminuye la pérdida esperada y aumenta utilidad redirigiendo exposición a clientes más sanos. Es como mover inversión de un terreno riesgoso a uno más estable: menos pérdidas y más retorno.")

    port = dfs.get("def_port")
    if port is not None and not port.empty:
        def g0(df, col): return df[col].iloc[0] if col in df.columns else np.nan
        kpi_row("EAD", g0(port,"EAD_actual"), g0(port,"EAD_optimizado"), moneda, usdclp)
        kpi_row("EL (Pérdida Esperada)", g0(port,"EL_actual"), g0(port,"EL_optimizado"), moneda, usdclp)
        kpi_row("Utilidad", g0(port,"Utilidad_actual"), g0(port,"Utilidad_optimizada"), moneda, usdclp)
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
    - *Ingreso total*: Flujo de intereses ajustado por volumen.  
    - *Utilidad total*: Ingreso – EL – Costos.  
    - *Ingreso/Utilidad aislado*: Solo efecto precio, manteniendo EAD fijo.
    """)

    st.markdown("### Análisis Ejecutivo")
    st.success("Ajustamos el precio como un tendero que encuentra el punto ideal: si cobra demasiado, pierde clientes; si cobra poco, gana volumen pero no rentabilidad. El balance correcto aumenta utilidad.")

    port = dfs.get("yld_port")
    if port is not None and not port.empty:
        def g0(df, name): return df[name].iloc[0] if name in df.columns else np.nan
        kpi_row("Ingreso Total", g0(port,"ingreso_base"), g0(port,"ingreso_opt"), moneda, usdclp)
        kpi_row("Utilidad Total", g0(port,"utilidad_base"), g0(port,"utilidad_opt"), moneda, usdclp)

# ================
# Arista 3 – Incentivos
# ================
with tabs[2]:
    st.header("Arista 3 – Incentivos")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown("Invertimos en incentivos solo donde el *ROI* es positivo: más ingresos por cada peso invertido.")

    st.markdown("### KPIs y Definiciones")
    st.markdown("""
    - *Costo incentivos*: gasto total en beneficios.  
    - *Ingreso incremental*: ingresos adicionales generados.  
    - *ROI*: Retorno de la inversión = Ingreso / Costo.
    """)

    st.markdown("### Análisis Ejecutivo")
    st.success("Es como fertilizar solo las plantas que realmente responden: cada peso en incentivos genera retorno multiplicado, en vez de regar parejo y perder recursos.")

    det = dfs.get("inc_det"); summ = dfs.get("inc_sum")
    if det is not None and not det.empty:
        # Robustez: detectar columnas costo/uplift si cambian nombres
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
    - *Capital requerido*: Consumo de capital regulatorio (proxy RW×K×EAD).  
    - *Provisiones*: reservas por riesgo crediticio ≈ EL.  
    - *Liberación*: diferencia de capital/provisiones antes y después.
    """)

    st.markdown("### Análisis Ejecutivo")
    st.success("Es como reorganizar el dinero guardado para emergencias: seguimos protegidos, pero sin exceso inmovilizado. Esto libera capital para invertir en oportunidades más rentables.")

    cap_port = dfs.get("cap_port")
    if cap_port is not None and not cap_port.empty:
        def g0(df, name): return df[name].iloc[0] if name in df.columns else np.nan
        kpi_row("Capital Requerido", g0(cap_port,"capital_req_base"), g0(cap_port,"capital_req_opt"), moneda, usdclp)
        kpi_row("Provisiones", g0(cap_port,"prov_base"), g0(cap_port,"prov_opt"), moneda, usdclp)

# ================
# Guardrails
# ================
with tabs[4]:
    st.header("Guardrails (Resguardos)")

    st.markdown("Estos son los límites regulatorios y de negocio que verificamos para asegurar robustez y cumplimiento.")

    gport = dfs.get("guard_port"); gseg = dfs.get("guard_seg")

    if gport is None or gport.empty:
        st.info("No hay tablas de guardrails en el bundle. Genera con la Celda 13.1 del notebook.")
    else:
        # Intento de formateo: si columnas parecen % numéricos, se formatean; si son cadenas (rangos), se respetan.
        gport_fmt = gport.copy()
        # columnas candidatas con % (si son numéricas, formatear)
        pct_like_cols = ["umbral","observado_actual","observado_optimizado"]
        for c in pct_like_cols:
            if c in gport_fmt.columns:
                gport_fmt[c] = gport_fmt[c].apply(fmt_pct_val)
        st.subheader("Portafolio")
        st.dataframe(gport_fmt, use_container_width=True)

    if gseg is not None and not gseg.empty:
        gseg_fmt = gseg.copy()
        if "observado" in gseg_fmt.columns:
            gseg_fmt["observado"] = gseg_fmt["observado"].apply(fmt_pct_val)
        st.subheader("Segmento")
        st.dataframe(gseg_fmt, use_container_width=True)

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas integradas).")
