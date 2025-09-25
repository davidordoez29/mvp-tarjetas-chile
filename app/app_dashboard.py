# app_dashboard.py — WDOF Motor Clásico (v1.0 bloqueado)
# ------------------------------------------------------
# Requisitos:
#   streamlit, pandas, numpy
# Ejecución (Colab):
#   !streamlit run app/app_dashboard.py --server.port=7860 --server.address=0.0.0.0
# Bundle esperado por defecto: /content/out/dashboard_bundle
# ------------------------------------------------------

import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Configuración y constantes
# -----------------------------
st.set_page_config(page_title="WDOF – Motor Clásico", layout="wide")

DEFAULT_BUNDLE = os.environ.get("BUNDLE_DIR", "/content/out/dashboard_bundle")

# Hints de columnas porcentuales y montos
_PCT_HINTS   = ["pd", "prob", "ratio", "roa", "roe", "margen", "apr", "tasa", "porc", "pct"]
_AMT_HINTS   = ["ead", "el", "rwa", "k", "util", "ingreso", "costo", "cap", "saldo", "limite"]
_RATE_HINTS  = ["apr", "tasa", "rate", "cof"]
_ID_COLS     = ["id", "id_cliente", "rut", "segmento"]

# Columnas por arista
FILES = {
    "A1": {
        "portfolio": ("default_portfolio{}.csv", ["EAD_actual","EAD_optimizado","EL_actual","EL_optimizado","PD_pond_actual","PD_pond_optimizado"]),
        "segment"  : ("default_segment{}.csv",   ["segmento","EAD_actual","EAD_optimizado","EL_actual","EL_optimizado","PD_pond_actual","PD_pond_optimizado"]),
        "detail"   : ("default_detail{}.csv",    None)  # libre
    },
    "A2": {
        "portfolio": ("yield_portfolio{}.csv", ["ingreso_base","ingreso_opt","utilidad_base","utilidad_opt","EAD_in","EAD_out"]),
        "segment"  : ("yield_segment{}.csv",   ["segmento","ingreso_opt","utilidad_opt","EAD_in","EAD_out"]),
        "detail"   : ("yield_detail{}.csv",    None),
        "curve"    : ("yield_curve_segment{}.csv", None),  # opcional
    },
    "A3": {
        "portfolio": ("incentives_portfolio{}.csv", None),  # si existe
        "detail"   : ("incentives_detail{}.csv",    None),
        "diag"     : ("incentives_diag_summary{}.csv", None),
        "sensitivity": ("incentives_sensitivity{}.csv", None),
    },
    "A4": {
        "portfolio": ("capital_portfolio{}.csv", ["EAD_base","EAD_opt","EL_base","EL_opt","RWA_base","RWA_opt","K_base","K_opt"]),
        "segment"  : ("capital_segment{}.csv",   ["segmento","EAD_base","EAD_opt","EL_base","EL_opt","RWA_base","RWA_opt","K_base","K_opt"]),
        "detail"   : ("capital_detail{}.csv",    None),
    },
    "GUARD": {
        "portfolio": ("guardrails_portfolio.csv", None),
        "segment"  : ("guardrails_segment.csv", None),
        "eval"     : ("guardrails_eval_portfolio.csv", None),
    },
    "UNIFIED": {
        "kpis": ("kpis_unificados.csv", None)
    }
}

# --------------------------------
# Utilidades de lectura y formateo
# --------------------------------
def read_csv_safe(path: Path, **kw) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path, **kw)
    except Exception as e:
        st.warning(f"No pude leer {path.name}: {e}")
    return None

def cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    df.columns = [c.strip() for c in df.columns]
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df.columns = [c.lower() for c in df.columns]
    return df

def apply_currency(df: pd.DataFrame, moneda: str, usdclp: float) -> pd.DataFrame:
    """Convierte montos a CLP o USD según selección. Assume entrada en CLP salvo que el bundle ya tenga mezcla."""
    if df is None or df.empty:
        return df
    out = df.copy()
    # Heurística: convertir todas columnas numéricas con hint de monto
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    to_scale = [c for c in num_cols if any(h in c for h in _AMT_HINTS)]
    if not to_scale:
        return out
    if moneda == "USD":
        if usdclp <= 0:
            st.warning("Tipo de cambio USDCLP inválido; se mantiene CLP.")
            return out
        out[to_scale] = out[to_scale] / usdclp
    # Si CLP, dejamos como está (asumimos CLP base)
    return out

def _fmt_number(x, moneda):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return f"{x:,}".replace(",", ".")
    if isinstance(x, (float, np.floating)):
        # Mostrar dos decimales en montos
        return f"{x:,.2f}".replace(",", "").replace(".", ",").replace("", ".")
    return str(x)

def format_df_auto(df: pd.DataFrame, moneda: str, usdclp: float) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out = cols_lower(out)

    # Conversión de moneda en columnas de montos
    out = apply_currency(out, moneda, usdclp)

    # Porcentajes: detectar y formatear a %
    for c in out.columns:
        lc = c.lower()
        series = out[c]
        if pd.api.types.is_numeric_dtype(series):
            if any(h in lc for h in _PCT_HINTS) and series.max() <= 1.2:
                out[c] = (series * 100.0).round(2)
    # Formato final string en montos y tasas
    display = pd.DataFrame(index=out.index)
    for c in out.columns:
        series = out[c]
        if pd.api.types.is_numeric_dtype(series):
            if any(h in c for h in _PCT_HINTS) and series.max() <= 120.0:
                display[c] = series.apply(lambda v: "" if pd.isna(v) else f"{v:.2f}%")
            elif any(h in c for h in _RATE_HINTS) and series.max() <= 5.0:
                display[c] = (series * 100.0).apply(lambda v: f"{v:.2f}%")
            elif any(h in c for h in _AMT_HINTS):
                display[c] = series.apply(lambda v: _fmt_number(v, moneda))
            else:
                display[c] = series.apply(lambda v: _fmt_number(v, moneda))
        else:
            display[c] = series
    # Restaurar mayúsculas iniciales “bonitas”
    display.columns = [c.replace("_", " ").title() for c in display.columns]
    return display

def load_by_arista(bundle: Path, arista: str, sfx: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Carga todos los CSVs de una arista con sufijo de escenario."""
    out = {}
    spec = FILES.get(arista, {})
    for key, (pattern, _) in spec.items():
        fname = pattern.format(sfx)
        df = read_csv_safe(bundle / fname)
        out[key] = df
    return out

# -----------------------------
# Sidebar (entrada de usuario)
# -----------------------------
st.sidebar.title("WDOF – Configuración")

bundle_dir_in = st.sidebar.text_input(
    "Ruta del bundle (dashboard_bundle)",
    value=str(DEFAULT_BUNDLE),
    key="bundle_path_input"
)
bundle_dir = Path(bundle_dir_in)
exists = bundle_dir.exists()
if not exists:
    st.sidebar.error(f"No se encuentra: {bundle_dir}")
else:
    st.sidebar.success(f"Bundle OK → {bundle_dir}")

escenario = st.sidebar.selectbox(
    "Escenario",
    ["Conservador", "Potenciado"],
    index=0,
    key="escenario_select"
)
SFX = "" if escenario == "Conservador" else "_agresivo"

moneda = st.sidebar.selectbox(
    "Moneda de visualización",
    ["CLP", "USD"],
    index=0,
    key="moneda_select"
)
usdclp = st.sidebar.number_input(
    "USDCLP para conversión (si seleccionas USD)",
    min_value=1.0, value=900.0, step=1.0,
    key="usdclp_input"
)

st.sidebar.markdown("---")
st.sidebar.caption("WDOF Motor Clásico v1.0 · Build bloqueado")

# -----------------------------
# Encabezado
# -----------------------------
st.title("WDOF — Optimización de Cartera (Motor Clásico)")
st.write(
    "Este dashboard muestra los resultados del motor *sin IA* (clásico), "
    "para los escenarios *Conservador* y *Potenciado*. Los KPIs se explican en cada arista."
)

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["Arista 1 – Default", "Arista 2 – Yield", "Arista 3 – Incentivos", "Arista 4 – Capital", "Guardrails", "KPIs Unificados"])

# -----------------------------
# Arista 1 – Default
# -----------------------------
with tabs[0]:
    st.subheader("Arista 1 – Riesgo de Crédito (Redistribución de EAD)")
    st.markdown(
        "- *Qué resolvemos aquí:* Reasignar exposición (EAD) hacia perfiles más sanos para *reducir EL* y *bajar PD ponderado*, manteniendo el tamaño total de cartera.\n"
        "- *Cálculos clave:* EL = PD × LGD × EAD. Se compara situación *Actual* vs *Optimizada* por cliente/segmento y en portafolio."
    )
    A1 = load_by_arista(bundle_dir, "A1", SFX)
    port, seg, det = A1.get("portfolio"), A1.get("segment"), A1.get("detail")

    if port is not None:
        st.write("*KPIs de Portafolio*")
        st.dataframe(format_df_auto(port, moneda, usdclp), use_container_width=True, height=180)
    if seg is not None:
        st.write("*KPIs por Segmento*")
        st.dataframe(format_df_auto(seg, moneda, usdclp), use_container_width=True, height=280)
    if det is not None:
        st.write("*Detalle por Cliente*")
        st.dataframe(format_df_auto(det.head(5000), moneda, usdclp), use_container_width=True, height=360)

# -----------------------------
# Arista 2 – Yield
# -----------------------------
with tabs[1]:
    st.subheader("Arista 2 – Rendimiento (Pricing & Mix)")
    st.markdown(
        "- *Qué resolvemos aquí:* Ajuste de tasa y mezcla para *incrementar utilidad* respetando límites (bandas de tasa, costo de fondos, sensibilidad).\n"
        "- *Cálculos clave:* Utilidad ≈ Ingreso por interés – COF – EL. Compara *Base* vs *Optimizado*."
    )
    A2 = load_by_arista(bundle_dir, "A2", SFX)
    port, seg, det, curve = A2.get("portfolio"), A2.get("segment"), A2.get("detail"), A2.get("curve")

    if port is not None:
        st.write("*KPIs de Portafolio*")
        st.dataframe(format_df_auto(port, moneda, usdclp), use_container_width=True, height=200)
    if seg is not None:
        st.write("*KPIs por Segmento*")
        st.dataframe(format_df_auto(seg, moneda, usdclp), use_container_width=True, height=280)
    if det is not None:
        st.write("*Detalle por Cliente*")
        st.dataframe(format_df_auto(det.head(5000), moneda, usdclp), use_container_width=True, height=360)
    if curve is not None and not curve.empty:
        st.write("*Curva de referencia por segmento*")
        st.dataframe(format_df_auto(curve, moneda, usdclp), use_container_width=True, height=240)

# -----------------------------
# Arista 3 – Incentivos
# -----------------------------
with tabs[2]:
    st.subheader("Arista 3 – Incentivos Comerciales")
    st.markdown(
        "- *Qué resolvemos aquí:* Seleccionar incentivos *positivos ROI* que aumentan uso/ingreso sin deteriorar riesgo, bajo un *presupuesto*.\n"
        "- *Cálculos clave:* Ingreso incremental estimado vs Costo ⇒ *ROI*; diagnóstico y sensibilidad."
    )
    A3 = load_by_arista(bundle_dir, "A3", SFX)
    diag, sens, det = A3.get("diag"), A3.get("sensitivity"), A3.get("detail")

    if diag is not None:
        st.write("*Diagnóstico Ejecutivo*")
        st.dataframe(format_df_auto(diag, moneda, usdclp), use_container_width=True, height=240)
    if sens is not None:
        st.write("*Sensibilidad*")
        st.dataframe(format_df_auto(sens, moneda, usdclp), use_container_width=True, height=240)
    if det is not None:
        st.write("*Detalle por Cliente*")
        st.dataframe(format_df_auto(det.head(5000), moneda, usdclp), use_container_width=True, height=360)

# -----------------------------
# Arista 4 – Capital (Basilea)
# -----------------------------
with tabs[3]:
    st.subheader("Arista 4 – Capital Requerido (Basilea)")
    st.markdown(
        "- *Qué resolvemos aquí:* Minimizar *RWA* y *K* manteniendo retorno y riesgo en rangos regulados.\n"
        "- *Cálculos clave:* RWA = RW × EAD; K = k_ratio × RWA. Se compara *Base* vs *Optimizado*."
    )
    A4 = load_by_arista(bundle_dir, "A4", SFX)
    port, seg, det = A4.get("portfolio"), A4.get("segment"), A4.get("detail")

    if port is not None:
        st.write("*KPIs de Portafolio*")
        st.dataframe(format_df_auto(port, moneda, usdclp), use_container_width=True, height=180)
    if seg is not None:
        st.write("*KPIs por Segmento*")
        st.dataframe(format_df_auto(seg, moneda, usdclp), use_container_width=True, height=280)
    if det is not None:
        st.write("*Detalle por Cliente*")
        st.dataframe(format_df_auto(det.head(5000), moneda, usdclp), use_container_width=True, height=360)

# -----------------------------
# Guardrails
# -----------------------------
with tabs[4]:
    st.subheader("Guardrails (Controles y Restricciones)")
    st.markdown(
        "Verificación de límites (concentración, bandas, exposure caps, tracking-error vs base, etc.)."
    )
    G = load_by_arista(bundle, "GUARD", "") if (bundle := bundle_dir) else {"portfolio": None, "segment": None, "eval": None}
    gport, gseg, geval = G.get("portfolio"), G.get("segment"), G.get("eval")

    if gport is not None:
        st.write("*Guardrails – Portafolio*")
        st.dataframe(format_df_auto(gport, moneda, usdclp), use_container_width=True, height=240)
    if gseg is not None:
        st.write("*Guardrails – Segmento*")
        st.dataframe(format_df_auto(gseg, moneda, usdclp), use_container_width=True, height=240)
    if geval is not None:
        st.write("*Guardrails – Evaluación*")
        st.dataframe(format_df_auto(geval, moneda, usdclp), use_container_width=True, height=240)

# -----------------------------
# KPIs Unificados
# -----------------------------
with tabs[5]:
    st.subheader("KPIs Unificados (Resumen Ejecutivo)")
    st.markdown(
        "Comparación transversal de KPIs por arista y escenario para discurso ejecutivo y comité."
    )
    kpi_path = bundle_dir / FILES["UNIFIED"]["kpis"][0]
    kpis = read_csv_safe(kpi_path)
    if kpis is not None:
        st.dataframe(format_df_auto(kpis, moneda, usdclp), use_container_width=True, height=360)
    else:
        st.info("No se encontró kpis_unificados.csv en el bundle.")

# Footer
st.markdown("---")
st.caption("© WDOF — Motor Clásico (Bloqueado) • Cumple IFRS9/Basilea (modo demo).")
