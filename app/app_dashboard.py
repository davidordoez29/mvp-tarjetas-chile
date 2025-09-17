# app_dashboard.py — Dashboard Bancario Optimización WDOF (sin st.secrets)
import os
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ==============================
# 🔧 Utilidades
# ==============================
def format_number(x, is_pct=False, currency=None):
    """
    - Miles con punto, decimales con coma, 2 decimales.
    - Si is_pct=True → agrega % (x viene como proporción 0-1).
    - currency: "CLP" o "USD" (solo para mostrar etiqueta; formato es el mismo).
    """
    try:
        if pd.isna(x):
            return "-"
        if is_pct:
            return f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            return f"{s}" if currency is None else f"{s} {currency}"
    except Exception:
        return str(x)

def to_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def safe_sum(s):
    try:
        return float(pd.to_numeric(s, errors="coerce").fillna(0).sum())
    except Exception:
        return 0.0

def map_or_default(df, target, candidates, default_value=np.nan):
    """
    Si target no existe, intenta mapear desde 'candidates' (lista de nombres alternativos).
    Si ninguno existe, crea 'target' con default_value.
    """
    if target in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            df = df.rename(columns={c: target})
            return df
    df[target] = default_value
    return df

def ensure_columns(clients, funding_cost=0.03, rw=0.75, cap_ratio=0.105):
    """
    Asegura presencia de columnas base/opt para cada arista.
    Si faltan, intenta derivarlas o crea con ceros para que la UI no rompa.
    """
    # --- mapeos básicos que suele traer el bundle Celda14 ---
    clients = map_or_default(clients, "EL_base", ["EL_base"])
    clients = map_or_default(clients, "EL_opt",  ["EL_opt"])

    # Yield: preferir columnas ingreso_* y mapear a Yield_*
    if ("Yield_base" not in clients.columns) and ("ingreso_base" in clients.columns):
        clients = clients.rename(columns={"ingreso_base": "Yield_base"})
    else:
        clients = map_or_default(clients, "Yield_base", ["Yield_base"], default_value=0.0)

    if ("Yield_opt" not in clients.columns) and ("ingreso_opt" in clients.columns):
        clients = clients.rename(columns={"ingreso_opt": "Yield_opt"})
    else:
        clients = map_or_default(clients, "Yield_opt", ["Yield_opt"], default_value=0.0)

    # Incentivos: costos y beneficios
    # En Celda 14: inc_spend (costo optimizado), inc_benefit_base=0, inc_benefit_opt=(m_unit*dEAD - inc_spend)
    if "Inc_cost_base" not in clients.columns:
        clients["Inc_cost_base"] = 0.0
    if "Inc_cost_opt" not in clients.columns:
        if "inc_spend" in clients.columns:
            clients["Inc_cost_opt"] = pd.to_numeric(clients["inc_spend"], errors="coerce").fillna(0.0)
        else:
            clients["Inc_cost_opt"] = 0.0

    if "Inc_benefit_base" not in clients.columns:
        if "inc_benefit_base" in clients.columns:
            clients = clients.rename(columns={"inc_benefit_base": "Inc_benefit_base"})
        else:
            clients["Inc_benefit_base"] = 0.0

    if "Inc_benefit_opt" not in clients.columns:
        if "inc_benefit_opt" in clients.columns:
            clients = clients.rename(columns={"inc_benefit_opt": "Inc_benefit_opt"})
        else:
            clients["Inc_benefit_opt"] = 0.0

    # Capital requerido: si no viene, derivar de EAD y parámetros
    # Necesitamos ead_baseline (EAD_base) y e_opt (EAD_opt)
    if "ead_baseline" not in clients.columns:
        # intentar mapear desde alternativas
        for alt in ["EAD_base", "ead_base", "exposure_base", "saldo_base"]:
            if alt in clients.columns:
                clients = clients.rename(columns={alt: "ead_baseline"})
                break
        if "ead_baseline" not in clients.columns:
            clients["ead_baseline"] = 0.0

    if "e_opt" not in clients.columns:
        for alt in ["EAD_opt", "ead_opt", "exposure_opt", "saldo_opt"]:
            if alt in clients.columns:
                clients = clients.rename(columns={alt: "e_opt"})
                break
        if "e_opt" not in clients.columns:
            clients["e_opt"] = pd.to_numeric(clients["ead_baseline"], errors="coerce").fillna(0.0)

    if "Capital_req_base" not in clients.columns:
        clients["Capital_req_base"] = pd.to_numeric(clients["ead_baseline"], errors="coerce").fillna(0.0) * rw * cap_ratio
    if "Capital_req_opt" not in clients.columns:
        clients["Capital_req_opt"]  = pd.to_numeric(clients["e_opt"], errors="coerce").fillna(0.0) * rw * cap_ratio

    if "Capital_lib_base" not in clients.columns:
        clients["Capital_lib_base"] = 0.0
    if "Capital_lib_opt" not in clients.columns:
        # Capital liberado = max(0, Capital_base - Capital_opt) a nivel cliente (acotado)
        cap_base = pd.to_numeric(clients["Capital_req_base"], errors="coerce").fillna(0.0)
        cap_opt  = pd.to_numeric(clients["Capital_req_opt"],  errors="coerce").fillna(0.0)
        clients["Capital_lib_opt"] = (cap_base - cap_opt).clip(lower=0.0)

    # PD/LGD/r y r_opt (para drilldown/consistencia)
    clients = map_or_default(clients, "pd_score", ["pd_score","PD_base","pd"])
    clients = map_or_default(clients, "lgd_pred", ["lgd_pred","LGD_base","lgd"])
    clients = map_or_default(clients, "apr_efectiva", ["apr_efectiva","r_base","tasa","apr"])
    clients = map_or_default(clients, "r_opt", ["r_opt","apr_opt"])

    # Coerción a num
    for c in ["EL_base","EL_opt","Yield_base","Yield_opt",
              "Inc_cost_base","Inc_cost_opt","Inc_benefit_base","Inc_benefit_opt",
              "Capital_req_base","Capital_req_opt","Capital_lib_base","Capital_lib_opt",
              "ead_baseline","e_opt","pd_score","lgd_pred","apr_efectiva","r_opt"]:
        if c in clients.columns:
            clients[c] = pd.to_numeric(clients[c], errors="coerce").fillna(0.0)

    return clients

def aggregate_totals(clients):
    """Devuelve totales (base/opt/delta) por arista."""
    out = {}
    out["Default"] = {
        "base": safe_sum(clients["EL_base"]) if "EL_base" in clients.columns else 0.0,
        "opt":  safe_sum(clients["EL_opt"])  if "EL_opt"  in clients.columns else 0.0
    }
    out["Yield"] = {
        "base": safe_sum(clients["Yield_base"]) if "Yield_base" in clients.columns else 0.0,
        "opt":  safe_sum(clients["Yield_opt"])  if "Yield_opt"  in clients.columns else 0.0
    }
    out["Incentivos"] = {
        "base_cost": safe_sum(clients["Inc_cost_base"]) if "Inc_cost_base" in clients.columns else 0.0,
        "opt_cost":  safe_sum(clients["Inc_cost_opt"])  if "Inc_cost_opt"  in clients.columns else 0.0,
        "base_ben":  safe_sum(clients["Inc_benefit_base"]) if "Inc_benefit_base" in clients.columns else 0.0,
        "opt_ben":   safe_sum(clients["Inc_benefit_opt"])  if "Inc_benefit_opt"  in clients.columns else 0.0,
    }
    out["Capital"] = {
        "base": safe_sum(clients["Capital_req_base"]) if "Capital_req_base" in clients.columns else 0.0,
        "opt":  safe_sum(clients["Capital_req_opt"])  if "Capital_req_opt"  in clients.columns else 0.0,
        "lib_base": safe_sum(clients["Capital_lib_base"]) if "Capital_lib_base" in clients.columns else 0.0,
        "lib_opt":  safe_sum(clients["Capital_lib_opt"])  if "Capital_lib_opt"  in clients.columns else 0.0,
    }
    return out

# ==============================
# 📂 Cargar bundle (sin st.secrets)
# ==============================
def load_bundle():
    # 1) Variable de entorno (es la que seteamos en deploy)
    env_bundle = os.environ.get("BUNDLE_DIR", "").strip()
    if env_bundle:
        bundle_dir = Path(env_bundle)
    else:
        # 2) ruta por defecto dentro del repo
        bundle_dir = Path("/content/mvp-tarjetas-chile/out/dashboard_bundle")

    clients_p = bundle_dir / "dashboard_bundle_clients.csv"
    segs_p    = bundle_dir / "dashboard_bundle_segments.csv"

    if not clients_p.exists() or not segs_p.exists():
        st.error("❌ No se encontró el bundle (CSV). "
                 "Asegúrate de ejecutar el pipeline y de sincronizar al repo (Celda 16).")
        st.write("Ruta buscada:", str(bundle_dir))
        return None, None

    try:
        clients = pd.read_csv(clients_p)
        segs = pd.read_csv(segs_p)
    except Exception as e:
        st.error(f"❌ Error leyendo CSV del bundle: {e}")
        return None, None

    # Asegurar columnas mínimas y derivadas
    clients = ensure_columns(clients)
    return clients, segs

# ==============================
# ⚙️ Sidebar config
# ==============================
st.set_page_config(page_title="Dashboard Bancario WDOF", layout="wide")
st.sidebar.header("⚙️ Configuración")
currency = st.sidebar.radio("Moneda", ["CLP", "USD"], index=0)
usd_to_clp = st.sidebar.number_input("Tipo de cambio USD → CLP", value=900, step=10, min_value=1)

def convert_amount(x):
    x = to_float(x)
    if currency == "USD":
        return x / max(usd_to_clp, 1)
    return x

# ==============================
# 📊 Cargar datos
# ==============================
clients, segs = load_bundle()
if clients is None:
    st.stop()

# ==============================
# 🧭 Layout principal
# ==============================
st.title("📊 Dashboard Bancario – Optimización en 4 Aristas (WDOF)")
st.caption("Simulación + Optimización de KPIs bancarios • Modelo matemático con guardrails de riesgo y capital.")

with st.sidebar:
    st.markdown("### Diagnóstico Bundle")
    bundle_dir = os.environ.get("BUNDLE_DIR", "/content/mvp-tarjetas-chile/out/dashboard_bundle")
    st.write("BUNDLE_DIR:", bundle_dir)
    st.write("Clientes:", len(clients))
    cols_needed = ["EL_base","EL_opt","Yield_base","Yield_opt",
                   "Inc_cost_base","Inc_cost_opt","Inc_benefit_base","Inc_benefit_opt",
                   "Capital_req_base","Capital_req_opt","Capital_lib_base","Capital_lib_opt"]
    missing = [c for c in cols_needed if c not in clients.columns]
    if missing:
        st.warning(f"⚠️ Faltan columnas en bundle: {missing}")

# ==============================
# 📑 Pestañas por Arista
# ==============================
tab1, tab2, tab3, tab4 = st.tabs([
    "Arista 1 – Default",
    "Arista 2 – Yield",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital"
])

totals = aggregate_totals(clients)

# ---------- Arista 1: Default ----------
with tab1:
    st.subheader("📉 Arista 1 — Default (Pérdidas Esperadas)")
    base = totals["Default"]["base"]
    opt  = totals["Default"]["opt"]
    delta = opt - base
    var = (delta / (abs(base) + 1e-9)) if base != 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Pérdidas (Base)", format_number(convert_amount(base), currency=currency))
    col2.metric("Pérdidas (Opt)", format_number(convert_amount(opt), currency=currency),
                delta=f"{var*100:.2f}%")
    col3.metric("Variación", format_number(convert_amount(delta), currency=currency))

    with st.expander("ℹ️ Definición y método"):
        st.markdown(
            "- *KPI*: EL = PD × LGD × EAD.\n"
            "- *Optimización*: Ajuste de tasa por riesgo (β) + guardrails para que EL no crezca.\n"
            "- *Ventaja: Reducimos pérdidas esperadas **sin destruir* el margen ni disparar capital."
        )

# ---------- Arista 2: Yield ----------
with tab2:
    st.subheader("💰 Arista 2 — Yield (Margen Financiero)")
    base = totals["Yield"]["base"]
    opt  = totals["Yield"]["opt"]
    delta = opt - base
    var = (delta / (abs(base) + 1e-9)) if base != 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Margen (Base)", format_number(convert_amount(base), currency=currency))
    col2.metric("Margen (Opt)", format_number(convert_amount(opt), currency=currency),
                delta=f"{var*100:.2f}%")
    col3.metric("Variación", format_number(convert_amount(delta), currency=currency))

    with st.expander("ℹ️ Definición y método"):
        st.markdown(
            "- *KPI*: (tasa efectiva – costo fondeo) × EAD.\n"
            "- *Optimización*: Subimos precio en bajo riesgo, bajamos en alto riesgo.\n"
            "- *Ventaja*: Aumenta el margen total manteniendo riesgo controlado."
        )

# ---------- Arista 3: Incentivos ----------
with tab3:
    st.subheader("🎁 Arista 3 — Incentivos (ROI)")
    base_cost = totals["Incentivos"]["base_cost"]
    opt_cost  = totals["Incentivos"]["opt_cost"]
    base_ben  = totals["Incentivos"]["base_ben"]
    opt_ben   = totals["Incentivos"]["opt_ben"]

    base_roi = (base_ben - base_cost) / (base_cost + 1e-9) if base_cost > 0 else 0.0
    opt_roi  = (opt_ben  - opt_cost)  / (opt_cost  + 1e-9) if opt_cost  > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Costo (Base)", format_number(convert_amount(base_cost), currency=currency))
    col2.metric("Costo (Opt)",  format_number(convert_amount(opt_cost),  currency=currency),
                delta=f"{((opt_cost-base_cost)/(abs(base_cost)+1e-9))*100:.2f}%")
    col3.metric("Δ Costo",      format_number(convert_amount(opt_cost - base_cost), currency=currency))

    col4, col5, col6 = st.columns(3)
    col4.metric("Beneficio (Base)", format_number(convert_amount(base_ben), currency=currency))
    col5.metric("Beneficio (Opt)",  format_number(convert_amount(opt_ben),  currency=currency),
                delta=f"{((opt_ben-base_ben)/(abs(base_ben)+1e-9))*100:.2f}%")
    col6.metric("Δ Beneficio",      format_number(convert_amount(opt_ben - base_ben), currency=currency))

    col7, col8 = st.columns(2)
    col7.metric("ROI (Base)", format_number(base_roi, is_pct=True))
    col8.metric("ROI (Opt)",  format_number(opt_roi,  is_pct=True),
                delta=f"{(opt_roi-base_roi)*100:.2f}%")

    with st.expander("ℹ️ Definición y método"):
        st.markdown(
            "- *KPI*: Beneficio neto incremental de campañas vs su costo.\n"
            "- *Optimización: Asignación **greedy* con elasticidad (ε) y costo (k) bajo un presupuesto B.\n"
            "- *Ventaja: Invertimos **solo* donde el CLP rinde más en margen neto, penalizando consumo de capital si aplica."
        )

# ---------- Arista 4: Capital ----------
with tab4:
    st.subheader("🏦 Arista 4 — Capital (uso eficiente)")
    base = totals["Capital"]["base"]
    opt  = totals["Capital"]["opt"]
    delta = opt - base
    var = (delta / (abs(base) + 1e-9)) if base != 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Requerido (Base)", format_number(convert_amount(base), currency=currency))
    col2.metric("Requerido (Opt)",  format_number(convert_amount(opt), currency=currency),
                delta=f"{var*100:.2f}%")
    col3.metric("Δ Requerido",      format_number(convert_amount(delta), currency=currency))

    base_lib = totals["Capital"]["lib_base"]
    opt_lib  = totals["Capital"]["lib_opt"]
    dlib = opt_lib - base_lib
    varlib = (dlib / (abs(base_lib) + 1e-9)) if base_lib != 0 else 0.0

    col4, col5, col6 = st.columns(3)
    col4.metric("Liberado (Base)", format_number(convert_amount(base_lib), currency=currency))
    col5.metric("Liberado (Opt)",  format_number(convert_amount(opt_lib),  currency=currency),
                delta=f"{varlib*100:.2f}%")
    col6.metric("Δ Liberado",      format_number(convert_amount(dlib),     currency=currency))

    with st.expander("ℹ️ Definición y método"):
        st.markdown(
            "- *KPI*: Capital = EAD × RW × Ratio (aprox estándar).\n"
            "- *Optimización*: Guardrails de capital (no crece vs base) y repricing/EAD para liberar.\n"
            "- *Ventaja: Más capacidad de crecimiento **con el mismo* capital regulatorio."
        )
