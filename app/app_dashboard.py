# app/app_dashboard.py — Dashboard MVP Bancario (Conservador/Potenciado, CLP/USD)
# ------------------------------------------------------------------------------
# Características:
# - Autodetección segura del bundle (sin recorrer /proc)
# - Selector de escenario (Conservador / Potenciado) y moneda (CLP / USD)
# - KPIs explicados y tablas con formateo automático (porcentaje / dinero / tasas)
# - Compatibilidad con nombres de columnas alternativos (robusto)
# - Guardrails y verificación de archivos presentes
# ------------------------------------------------------------------------------

import os, math, re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# =====================================
# Config inicial
# =====================================
st.set_page_config(page_title="MVP Bancario — 4 Aristas", layout="wide")

# =====================================
# Utilidades de Paths / Bundle
# =====================================
def autodetect_bundle_safe() -> Path | None:
    """
    Busca el bundle en ubicaciones seguras sin recorrer /proc (evita OSError).
    Prioriza: env BUNDLE_DIR > /content/out/dashboard_bundle > ./out/dashboard_bundle > ./dashboard_bundle
    """
    env_dir = (os.environ.get("BUNDLE_DIR") or "").strip()
    candidates = [env_dir] if env_dir else []
    candidates += [
        "/content/out/dashboard_bundle",
        "./out/dashboard_bundle",
        "./dashboard_bundle",
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c).resolve()
        if p.is_dir() and len(list(p.glob("*.csv"))) >= 3:
            return p
    return None

def sidebar_bundle_picker(default_path: Path | None):
    st.sidebar.title("⚙️ Configuración")

    default_txt = str(default_path) if default_path else ""
    bundle_txt = st.sidebar.text_input(
        "📦 Ruta del bundle",
        value=default_txt,
        key="bundle_path_input",
        help="Ej: /content/out/dashboard_bundle",
    ).strip()
    bundle_dir = Path(bundle_txt).resolve() if bundle_txt else None
    if not bundle_dir or not bundle_dir.exists():
        st.sidebar.warning("No encuentro el bundle en la ruta indicada. Prueba /content/out/dashboard_bundle.")
        bundle_dir = default_path

    scenario = st.sidebar.radio(
        "Escenario",
        options=["Conservador", "Potenciado"],
        horizontal=True,
        key="scenario_radio",
    )
    currency = st.sidebar.radio(
        "Moneda",
        options=["CLP", "USD"],
        horizontal=True,
        key="currency_radio",
    )
    usdclp = float(
        st.sidebar.number_input(
            "USDCLP (1 USD = ? CLP)",
            min_value=1.0,
            value=900.0,
            step=1.0,
            key="usdclp_input",
        )
    )
    return bundle_dir, scenario, currency, usdclp

def suffix_from_scenario(scenario: str) -> str:
    return "" if scenario.strip().lower().startswith("conserv") else "_agresivo"

def quick_bundle_check(bundle_dir: Path):
    samples = [p.name for p in (bundle_dir.glob("*.csv"))][:8]
    st.caption(f"Bundle OK → {bundle_dir}")
    if samples:
        st.caption(f"Muestras de archivos: {', '.join(samples)}")

# =====================================
# Carga por arista
# =====================================
def load_by_arista(bundle_dir: Path, arista: str, suffix: str = "") -> dict[str, pd.DataFrame]:
    """
    Carga los CSV de una arista (A1/A2/A3/A4/GR) según convención de nombres.
    Retorna dict {nombre_base: DataFrame}. Omite los que no existan.
    """
    mapping = {
        "A1": ["default_portfolio", "default_segment", "default_detail"],
        "A2": ["yield_portfolio", "yield_segment", "yield_detail", "yield_curve_segment"],
        "A3": ["incentives_portfolio", "incentives_detail", "incentives_diag_summary", "incentives_sensitivity"],
        "A4": ["capital_portfolio", "capital_segment", "capital_detail"],
        "GR": ["guardrails_portfolio", "guardrails_segment", "guardrails_eval_portfolio"],
    }
    out = {}
    names = mapping.get(arista, [])
    for base in names:
        use_suffix = "" if arista == "GR" else suffix
        fname = f"{base}{use_suffix}.csv"
        fpath = (bundle_dir / fname).resolve()
        if fpath.exists():
            try:
                out[base] = pd.read_csv(fpath)
            except Exception as e:
                st.warning(f"[{arista}] Error leyendo {fname}: {e}")
    return out

# =====================================
# Formateo numérico y helpers
# =====================================
# ---- Constantes de formateo (DEBEN estar ANTES de format_df_auto) ----
PCT_HINTS   = ["%", "pct", "porc", "pd", "lgd", "roi", "var_%", "pd_pond", "k_ratio"]
MONEY_HINTS = ["monto","ead","el","ingreso","util","capital","k_","income","cost","exposure","saldo","rwa"]
_RATE_HINTS  = ["apr","tasa","rate","cof","r_base","r_opt"]

def _to_display_currency(val: float, target: str, usdclp: float) -> float:
    if pd.isna(val): 
        return np.nan
    return (float(val) / float(usdclp)) if target.upper() == "USD" else float(val)

def fmt_money_val(val, target: str, usdclp: float) -> str:
    if isinstance(val, str):
        v = val.strip()
        if v == "" or v.upper() == "N/A": 
            return "—"
        return v
    if val is None or (isinstance(val, float) and math.isnan(val)): 
        return "—"
    x = _to_display_currency(float(val), target, usdclp)
    if x is None or (isinstance(x, float) and math.isnan(x)): 
        return "—"
    neg = x < 0
    x = abs(x)
    ent = int(x)
    dec = int(round((x - ent) * 100))
    if dec == 100:
        ent += 1
        dec = 0
    ent_str = f"{ent:,}".replace(",", ".")
    return f"-{ent_str},{dec:02d}" if neg else f"{ent_str},{dec:02d}"

_num_like = re.compile(r"^-?\d+(\.\d+)?$")

def _to_float_or_nan(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): 
        return np.nan
    if isinstance(v, (int, float)): 
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace("%", "").replace(",", ".")
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

def var_pct(a, b):
    a = _to_float_or_nan(a)
    b = _to_float_or_nan(b)
    if np.isnan(a) or a == 0 or np.isnan(b): 
        return np.nan
    return (b - a) / a * 100.0

def kpi_row_money(label: str, a, b, moneda: str, usdclp: float, help_text: str = ""):
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        st.metric(label=f"{label} – Actual", value=fmt_money_val(a, moneda, usdclp))
        if help_text: 
            st.caption(help_text)
    with c2:
        st.metric(label=f"{label} – Optimizado", value=fmt_money_val(b, moneda, usdclp))
    with c3:
        vp = var_pct(a, b)
        st.metric(label="VAR %", value=fmt_pct_val(vp) if pd.notna(vp) else "—")

def kpi_row_pct(label: str, a_pct, b_pct, help_text: str = ""):
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        st.metric(label=f"{label} – Actual", value=fmt_pct_val(a_pct))
        if help_text: 
            st.caption(help_text)
    with c2:
        st.metric(label=f"{label} – Optimizado", value=fmt_pct_val(b_pct))
    with c3:
        vp = var_pct(a_pct, b_pct)
        st.metric(label="VAR %", value=fmt_pct_val(vp) if pd.notna(vp) else "—")

def format_df_auto(df: pd.DataFrame, moneda: str, usdclp: float):
    """Formatea heurísticamente por nombre de columna."""
    if df is None or df.empty: 
        return df
    out = df.copy()
    lower = {c: c.lower() for c in out.columns}
    for c in out.columns:
        lc = lower[c]
        # porcentaje
        if any(h in lc for h in _PCT_HINTS):
            out[c] = out[c].apply(fmt_pct_val)
            continue
        # moneda / montos
        if any(h in lc for h in _MONEY_HINTS):
            out[c] = out[c].apply(lambda v: fmt_money_val(v, moneda, usdclp))
            continue
        # tasa (apr/r_base/r_opt/cof)
        if any(h in lc for h in RATE_HINTS) or lc.startswith("r"):
            out[c] = out[c].apply(
                lambda v: fmt_pct_val(_to_float_or_nan(v) * 100.0) 
                if pd.notna(_to_float_or_nan(v)) else v
            )
    return out

def pick(df: pd.DataFrame, names: list[str], default=None):
    for n in names:
        if n and n in df.columns:
            return n
    return default

# =====================================
# Tabs
# =====================================
def build_tabs():
    return st.tabs([
        "Arista 1 – Default/Impago",
        "Arista 2 – Yield/Pricing",
        "Arista 3 – Incentivos",
        "Arista 4 – Capital/Provisiones",
        "Guardrails (Resguardos)"
    ])

# =====================================
# Inicio App
# =====================================
_auto = autodetect_bundle_safe()
bundle_dir, scenario, moneda, usdclp = sidebar_bundle_picker(_auto)
if not bundle_dir:
    st.error("No encuentro el bundle. Genera el paquete en el notebook y vuelve a cargar.")
    st.stop()

SFX = suffix_from_scenario(scenario)
quick_bundle_check(bundle_dir)
tabs = build_tabs()

# =====================================
# A1 — Default / Impago
# =====================================
with tabs[0]:
    st.header("Arista 1 – Default/Impago")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown(
        "Reducimos la *pérdida esperada (EL)* reubicando exposición (EAD) desde segmentos/clientes de mayor riesgo a otros más sanos, "
        "manteniendo el tamaño del negocio."
    )

    st.markdown("### KPIs y Definiciones")
    st.markdown(
        "- *EAD*: Exposición en riesgo (saldo sujeto a crédito).\n"
        "- *EL (Expected Loss): *PD × LGD × EAD.\n"
        "- *Utilidad*: Ingreso financiero – EL – costos financieros (COF aprox)."
    )

    st.markdown("### ¿Qué cálculos hace el motor aquí?")
    st.markdown(
        "Calcula PD/LGD (IFRS9), estima EL por cliente, detecta espacios para mover EAD con límites/reglas y recompone el portafolio manteniendo el total."
    )

    A1 = load_by_arista(bundle_dir, "A1", SFX)
    port = A1.get("default_portfolio")
    seg  = A1.get("default_segment")
    det  = A1.get("default_detail")

    if port is not None and not port.empty:
        # columnas robustas
        ca = pick(port, ["EAD_actual","EAD_base","EAD_a","EAD"])
        cb = pick(port, ["EAD_optimizado","EAD_opt","EAD_b"])
        ela = pick(port, ["EL_actual","EL_base","EL_a"])
        elb = pick(port, ["EL_optimizado","EL_opt","EL_b"])
        ua  = pick(port, ["Utilidad_actual","utilidad_base","Util_a"])
        ub  = pick(port, ["Utilidad_optimizada","utilidad_opt","Util_b"])

        def g(df, col):
            return (df[col].iloc[0] if col and col in df.columns else np.nan)

        kpi_row_money("EAD", g(port, ca), g(port, cb), moneda, usdclp, "Exposición total.")
        kpi_row_money("EL (Pérdida Esperada)", g(port, ela), g(port, elb), moneda, usdclp, "PD×LGD×EAD agregado.")
        kpi_row_money("Utilidad", g(port, ua), g(port, ub), moneda, usdclp, "Ingreso menos pérdidas y costos.")

    if seg is not None and not seg.empty:
        st.subheader("Segmento")
        st.dataframe(format_df_auto(seg, moneda, usdclp), use_container_width=True, height=360)
    if det is not None and not det.empty:
        st.subheader("Detalle por cliente")
        st.dataframe(format_df_auto(det, moneda, usdclp), use_container_width=True, height=360)

# =====================================
# A2 — Yield / Pricing
# =====================================
with tabs[1]:
    st.header("Arista 2 – Yield/Pricing")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown(
        "Encontramos la *tasa óptima (APR)* que maximiza utilidad equilibrando precio y volumen, respetando bandas y guardrails."
    )

    st.markdown("### KPIs y Definiciones")
    st.markdown(
        "- *Ingreso total*: Intereses estimados tras ajuste de tasa y volumen.\n"
        "- *Utilidad total*: Ingreso – EL – costo financiero (COF).\n"
        "- *EAD_in / EAD_out*: EAD de entrada al pricing vs EAD resultante tras elasticidad."
    )

    st.markdown("### ¿Qué cálculos hace el motor aquí?")
    st.markdown(
        "Aplica una función de demanda (elasticidad) por segmento para simular cómo cambia el volumen al mover la tasa, "
        "y evalúa ingreso, EL y utilidad con la nueva tasa."
    )

    A2 = load_by_arista(bundle_dir, "A2", SFX)
    port = A2.get("yield_portfolio")
    seg  = A2.get("yield_segment")
    det  = A2.get("yield_detail")
    curve= A2.get("yield_curve_segment")

    if port is not None and not port.empty:
        ia = pick(port, ["ingreso_base","Ingreso_base","Interes_a","Ingreso_a"])
        ib = pick(port, ["ingreso_opt","Ingreso_opt","Interes_b","Ingreso_b"])
        ua = pick(port, ["utilidad_base","Utilidad_base","Util_a"])
        ub = pick(port, ["utilidad_opt","Utilidad_opt","Util_b"])
        ein = pick(port, ["EAD_in","EAD_a","EAD_base"])
        eout= pick(port, ["EAD_out","EAD_b","EAD_opt"])

        def g(df, col): 
            return (df[col].iloc[0] if col and col in df.columns else np.nan)

        kpi_row_money("Ingreso total", g(port, ia), g(port, ib), moneda, usdclp, "Intereses tras pricing.")
        kpi_row_money("Utilidad total", g(port, ua), g(port, ub), moneda, usdclp, "Ingreso – EL – COF.")
        if ein or eout:
            kpi_row_money("EAD (in→out)", g(port, ein), g(port, eout), moneda, usdclp, "Cambio de volumen por tasa.")

    if seg is not None and not seg.empty:
        st.subheader("Segmento")
        st.dataframe(format_df_auto(seg, moneda, usdclp), use_container_width=True, height=360)
    if curve is not None and not curve.empty:
        st.subheader("Curva de referencia por segmento")
        st.dataframe(format_df_auto(curve, moneda, usdclp), use_container_width=True, height=360)
    if det is not None and not det.empty:
        st.subheader("Detalle por cliente")
        st.dataframe(format_df_auto(det, moneda, usdclp), use_container_width=True, height=360)

# =====================================
# A3 — Incentivos
# =====================================
with tabs[2]:
    st.header("Arista 3 – Incentivos")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown(
        "Inversiones en incentivos solo donde el *ROI* sea positivo: más ingreso por cada peso invertido, con límites presupuestarios."
    )

    st.markdown("### KPIs y Definiciones")
    st.markdown(
        "- *Costo de incentivos*: gasto total en beneficios.\n"
        "- *Ingreso incremental*: ingreso adicional estimado atribuible al incentivo.\n"
        "- *ROI*: Ingreso incremental / Costo."
    )

    st.markdown("### ¿Qué cálculos hace el motor aquí?")
    st.markdown(
        "Filtra candidatos que responden a incentivos (según elasticidad/propensión), calcula costo y uplift esperado, y selecciona "
        "hasta agotar presupuesto con ROI>0."
    )

    A3 = load_by_arista(bundle_dir, "A3", SFX)
    port = A3.get("incentives_portfolio")
    det  = A3.get("incentives_detail")
    diag = A3.get("incentives_diag_summary")

    # KPI desde detalle (si no hay portfolio)
    cost_total, inc_total, roi_val = np.nan, np.nan, np.nan
    if det is not None and not det.empty:
        cost_cols = [c for c in det.columns if "cost" in c.lower()]
        inc_cols  = [c for c in det.columns if any(k in c.lower() for k in ["uplift","ingreso_inc","delta_ingreso"])]
        if cost_cols:
            cost_total = pd.to_numeric(det[cost_cols].sum(axis=1), errors="coerce").fillna(0).sum()
        if inc_cols:
            inc_total = pd.to_numeric(det[inc_cols].sum(axis=1), errors="coerce").fillna(0).sum()
        roi_val = (inc_total / cost_total * 100.0) if (isinstance(cost_total, (int,float)) and cost_total>0) else np.nan

    if port is not None and not port.empty:
        ca = pick(port, ["costo","cost_total","costo_total"])
        ia = pick(port, ["ingreso_incremental","uplift_total","ingreso_inc_total"])
        ra = pick(port, ["roi_pct","roi","roi%"])
        def g(df, col): 
            return (df[col].iloc[0] if col and col in df.columns else np.nan)
        kpi_row_money("Costo incentivos", g(port, ca) if ca else cost_total, g(port, ca) if ca else cost_total, moneda, usdclp)
        kpi_row_money("Ingreso incremental", g(port, ia) if ia else inc_total, g(port, ia) if ia else inc_total, moneda, usdclp)
        st.metric("ROI", fmt_pct_val(g(port, ra) if ra else roi_val))
    else:
        kpi_row_money("Costo incentivos", cost_total, cost_total, moneda, usdclp)
        kpi_row_money("Ingreso incremental", inc_total, inc_total, moneda, usdclp)
        st.metric("ROI", fmt_pct_val(roi_val))

    if diag is not None and not diag.empty:
        st.subheader("Diagnóstico/Sensibilidad")
        st.dataframe(format_df_auto(diag, moneda, usdclp), use_container_width=True, height=360)
    if det is not None and not det.empty:
        st.subheader("Detalle por cliente")
        st.dataframe(format_df_auto(det, moneda, usdclp), use_container_width=True, height=360)

# =====================================
# A4 — Capital / Provisiones
# =====================================
with tabs[3]:
    st.header("Arista 4 – Capital / Provisiones")

    st.markdown("### ¿Qué resolvemos aquí?")
    st.markdown(
        "Hacemos más eficiente el *capital requerido* (RWA, K) y *provisiones* (EL), liberando recursos sin relajar el riesgo."
    )

    st.markdown("### KPIs y Definiciones")
    st.markdown(
        "- *EL*: pérdida esperada (PD×LGD×EAD).\n"
        "- *RWA*: activos ponderados por riesgo (aprox. RW×EAD).\n"
        "- *K*: capital regulatorio = K_ratio × RWA."
    )

    st.markdown("### ¿Qué cálculos hace el motor aquí?")
    st.markdown(
        "Con PD/LGD y EAD (pre/post), calculamos EL; con RW y K_ratio, derivamos RWA y capital pre/post; "
        "presentamos variaciones agregadas y por segmento."
    )

    A4 = load_by_arista(bundle_dir, "A4", SFX)
    port = A4.get("capital_portfolio")
    seg  = A4.get("capital_segment")
    det  = A4.get("capital_detail")

    if port is not None and not port.empty:
        ead_a = pick(port, ["EAD_base","EAD_actual","EAD_a"])
        ead_b = pick(port, ["EAD_opt","EAD_optimizado","EAD_b"])
        el_a  = pick(port, ["EL_base","EL_actual","EL_a"])
        el_b  = pick(port, ["EL_opt","EL_optimizado","EL_b"])
        rwa_a = pick(port, ["RWA_base","RWA_a"])
        rwa_b = pick(port, ["RWA_opt","RWA_b"])
        k_a   = pick(port, ["K_base","K_a"])
        k_b   = pick(port, ["K_opt","K_b"])

        def g(df, col): 
            return (df[col].iloc[0] if col and col in df.columns else np.nan)

        kpi_row_money("EL", g(port, el_a), g(port, el_b), moneda, usdclp, "Pérdida esperada total.")
        kpi_row_money("RWA", g(port, rwa_a), g(port, rwa_b), moneda, usdclp, "RW×EAD agregado.")
        kpi_row_money("Capital (K)", g(port, k_a), g(port, k_b), moneda, usdclp, "K_ratio×RWA.")

    if seg is not None and not seg.empty:
        st.subheader("Segmento")
        st.dataframe(format_df_auto(seg, moneda, usdclp), use_container_width=True, height=360)
    if det is not None and not det.empty:
        st.subheader("Detalle por cliente")
        st.dataframe(format_df_auto(det, moneda, usdclp), use_container_width=True, height=360)

# =====================================
# Guardrails
# =====================================
with tabs[4]:
    st.header("Guardrails (Resguardos)")
    st.markdown(
        "Límites regulatorios y de negocio verificados para asegurar cumplimiento y robustez."
    )
    GR = load_by_arista(bundle_dir, "GR", "")
    gport = GR.get("guardrails_portfolio")
    gseg  = GR.get("guardrails_segment")
    geval = GR.get("guardrails_eval_portfolio")

    if gport is not None and not gport.empty:
        st.subheader("Portafolio")
        st.dataframe(format_df_auto(gport, moneda, usdclp), use_container_width=True, height=360)
    else:
        st.info("No se encontró guardrails_portfolio.csv en el bundle.")

    if gseg is not None and not gseg.empty:
        st.subheader("Segmento")
        st.dataframe(format_df_auto(gseg, moneda, usdclp), use_container_width=True, height=360)

    if geval is not None and not geval.empty:
        st.subheader("Evaluación de checks")
        st.dataframe(format_df_auto(geval, moneda, usdclp), use_container_width=True, height=360)

# =====================================
# Footer
# =====================================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas). IFRS9/Basilea/Negocio.")
