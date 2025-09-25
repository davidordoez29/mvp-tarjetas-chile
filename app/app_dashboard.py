# app_dashboard.py — WDOF Motor Clásico (restaurado + mejoras conservadoras)
# -------------------------------------------------------------------------
# Cómo ejecutar (Colab):
# !streamlit run app/app_dashboard.py --server.port=7860 --server.address=0.0.0.0
#
# Supone bundle en:
#   - BUNDLE_DIR (env)  o
#   - /content/out/dashboard_bundle  o
#   - ./out/dashboard_bundle
# No escanea /proc. No cambia tu Notebook ni tus CSV.
# -------------------------------------------------------------------------

import os
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Config UI ----------
st.set_page_config(page_title="WDOF – Motor Clásico", layout="wide")

# Hints de formateo (solo presentación)
_PCT_HINTS  = ["pd", "prob", "ratio", "porc", "pct", "margen", "apr", "tasa"]
_AMT_HINTS  = ["ead", "el", "rwa", "k", "util", "ingreso", "costo", "cap", "saldo", "limite"]
_RATE_HINTS = ["apr", "tasa", "rate", "cof"]

# Mapa de archivos por arista (no tocar naming de tus CSV)
FILES = {
    "A1": {  # Default
        "portfolio": "default_portfolio{SFX}.csv",
        "segment":   "default_segment{SFX}.csv",
        "detail":    "default_detail{SFX}.csv",
    },
    "A2": {  # Yield
        "portfolio": "yield_portfolio{SFX}.csv",
        "segment":   "yield_segment{SFX}.csv",
        "detail":    "yield_detail{SFX}.csv",
        "curve":     "yield_curve_segment{SFX}.csv",  # opcional
    },
    "A3": {  # Incentivos
        "portfolio": "incentives_portfolio{SFX}.csv",          # opcional
        "detail":    "incentives_detail{SFX}.csv",
        "diag":      "incentives_diag_summary{SFX}.csv",
        "sens":      "incentives_sensitivity{SFX}.csv",
    },
    "A4": {  # Capital
        "portfolio": "capital_portfolio{SFX}.csv",
        "segment":   "capital_segment{SFX}.csv",
        "detail":    "capital_detail{SFX}.csv",
    },
    "GUARD": {
        "portfolio": "guardrails_portfolio.csv",
        "segment":   "guardrails_segment.csv",
        "eval":      "guardrails_eval_portfolio.csv",
    },
    "UNI": {
        "kpis": "kpis_unificados.csv"
    }
}

# ---------- Utilidades de lectura ----------
def read_csv_safe(path: Path, **kw) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path, **kw)
    except Exception as e:
        st.warning(f"No pude leer {path.name}: {e}")
    return None

def autodetect_bundle() -> Path:
    cand = []
    if os.environ.get("BUNDLE_DIR"):
        cand.append(Path(os.environ["BUNDLE_DIR"]))
    cand += [Path("/content/out/dashboard_bundle"), Path("./out/dashboard_bundle")]
    for p in cand:
        if p.exists() and p.is_dir():
            return p
    return Path("/content/out/dashboard_bundle")  # por defecto

def load_by_arista(bundle: Path, arista: str, sfx: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Carga dict con dataframes por sub-sección de la arista."""
    out: Dict[str, Optional[pd.DataFrame]] = {}
    spec = FILES.get(arista, {})
    for key, pattern in spec.items():
        fname = pattern.format(SFX=sfx)
        out[key] = read_csv_safe(bundle / fname)
    return out

# ---------- Formateo (presentación) ----------
def _fmt_num(x, as_money=False):
    if pd.isna(x): return ""
    if isinstance(x, (int, np.integer)):
        return f"{x:,}".replace(",", ".")
    if isinstance(x, (float, np.floating)):
        return f"{x:,.2f}".replace(",", "").replace(".", ",").replace("", ".")
    return str(x)

def apply_currency(df: pd.DataFrame, moneda: str, usdclp: float) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    # columnas numéricas con hint de monto
    numcols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    scale = [c for c in numcols if any(h in c.lower() for h in _AMT_HINTS)]
    if moneda == "USD" and usdclp and usdclp > 0:
        out[scale] = out[scale] / usdclp
    return out

def format_df_auto(df: pd.DataFrame, moneda: str, usdclp: float) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    # normalizar nombres a minúsculas solo para detección; mantenemos columnas originales para display bonito
    lower_map = {c: c.lower() for c in out.columns}
    # monedas
    out = apply_currency(out, moneda, usdclp)
    # convertir porcentajes si vienen 0–1
    for c in out.columns:
        lc = lower_map[c]
        if pd.api.types.is_numeric_dtype(out[c]):
            if any(h in lc for h in _PCT_HINTS) and out[c].max() <= 1.2:
                out[c] = (out[c] * 100.0).round(2)
    # render amigable
    disp = pd.DataFrame(index=out.index)
    for c in out.columns:
        lc = lower_map[c]
        if pd.api.types.is_numeric_dtype(out[c]):
            if any(h in lc for h in _PCT_HINTS) and out[c].max() <= 120:
                disp[c] = out[c].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}%")
            elif any(h in lc for h in _RATE_HINTS) and out[c].max() <= 5.0:
                disp[c] = (out[c] * 100.0).apply(lambda v: f"{v:.2f}%")
            elif any(h in lc for h in _AMT_HINTS):
                disp[c] = out[c].apply(lambda v: _fmt_num(v, as_money=True))
            else:
                disp[c] = out[c].apply(_fmt_num)
        else:
            disp[c] = out[c]
    return disp

# ---------- Sidebar ----------
st.sidebar.title("WDOF – Configuración")

bundle_path = st.sidebar.text_input(
    "Ruta del bundle (dashboard_bundle)",
    value=str(autodetect_bundle()),
    key="bundle_path_text"
)
bundle_dir = Path(bundle_path)
if bundle_dir.exists():
    st.sidebar.success(f"Bundle OK → {bundle_dir}")
else:
    st.sidebar.error(f"No se encontró {bundle_dir}")

escenario = st.sidebar.selectbox("Escenario", ["Conservador", "Potenciado"], index=0, key="escenario_sel")
SFX = "" if escenario == "Conservador" else "_agresivo"

moneda = st.sidebar.selectbox("Moneda", ["CLP", "USD"], index=0, key="moneda_sel")
usdclp = st.sidebar.number_input("USDCLP (si eliges USD)", min_value=1.0, value=900.0, step=1.0, key="usdclp_input")

st.sidebar.markdown("---")
st.sidebar.caption("Motor clásico bloqueado · Visualización estable")

# ---------- Encabezado ----------
st.title("WDOF — Optimización de Cartera (Motor Clásico)")
st.markdown(
    "Resultados del motor *sin IA* por arista y escenario. "
    "Se muestran KPIs y detalle respetando el contrato de archivos del bundle."
)

# ---------- Tabs ----------
tabs = st.tabs([
    "Arista 1 – Default",
    "Arista 2 – Yield",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital",
    "Guardrails",
    "KPIs Unificados",
])

# ---------- A1 ----------
with tabs[0]:
    st.subheader("Arista 1 – Riesgo de Crédito (Redistribución EAD)")
    st.markdown(
        "- *Objetivo: Reducir EL y PD ponderado **manteniendo* el EAD total.\n"
        "- *Métrica base*: EL = PD × LGD × EAD (comparativo Actual vs Optimizado)."
    )
    A1 = load_by_arista(bundle_dir, "A1", SFX)
    if A1["portfolio"] is not None:
        st.write("*KPIs de Portafolio*")
        st.dataframe(format_df_auto(A1["portfolio"], moneda, usdclp), use_container_width=True, height=180)
    if A1["segment"] is not None:
        st.write("*KPIs por Segmento*")
        st.dataframe(format_df_auto(A1["segment"], moneda, usdclp), use_container_width=True, height=260)
    if A1["detail"] is not None:
        st.write("*Detalle por Cliente*")
        st.dataframe(format_df_auto(A1["detail"].head(5000), moneda, usdclp), use_container_width=True, height=360)

# ---------- A2 ----------
with tabs[1]:
    st.subheader("Arista 2 – Rendimiento (Pricing & Mix)")
    st.markdown(
        "- *Objetivo*: Incrementar utilidad dentro de bandas y respetando COF.\n"
        "- *Utilidad* ≈ Ingreso por interés – COF – EL."
    )
    A2 = load_by_arista(bundle_dir, "A2", SFX)
    if A2["portfolio"] is not None:
        st.write("*KPIs de Portafolio*")
        st.dataframe(format_df_auto(A2["portfolio"], moneda, usdclp), use_container_width=True, height=200)
    if A2["segment"] is not None:
        st.write("*KPIs por Segmento*")
        st.dataframe(format_df_auto(A2["segment"], moneda, usdclp), use_container_width=True, height=260)
    if A2["detail"] is not None:
        st.write("*Detalle por Cliente*")
        st.dataframe(format_df_auto(A2["detail"].head(5000), moneda, usdclp), use_container_width=True, height=360)
    if A2.get("curve") is not None:
        st.write("*Curva de Referencia por Segmento*")
        st.dataframe(format_df_auto(A2["curve"], moneda, usdclp), use_container_width=True, height=220)

# ---------- A3 ----------
with tabs[2]:
    st.subheader("Arista 3 – Incentivos Comerciales")
    st.markdown(
        "- *Objetivo: Incentivos con **ROI positivo* bajo un *presupuesto*.\n"
        "- *Indicadores*: Ingreso incremental estimado, costo y ROI."
    )
    A3 = load_by_arista(bundle_dir, "A3", SFX)
    if A3.get("diag") is not None:
        st.write("*Diagnóstico*")
        st.dataframe(format_df_auto(A3["diag"], moneda, usdclp), use_container_width=True, height=220)
    if A3.get("sens") is not None:
        st.write("*Sensibilidad*")
        st.dataframe(format_df_auto(A3["sens"], moneda, usdclp), use_container_width=True, height=220)
    if A3.get("portfolio") is not None:
        st.write("*Resumen de Portafolio*")
        st.dataframe(format_df_auto(A3["portfolio"], moneda, usdclp), use_container_width=True, height=200)
    if A3.get("detail") is not None:
        st.write("*Detalle por Cliente*")
        st.dataframe(format_df_auto(A3["detail"].head(5000), moneda, usdclp), use_container_width=True, height=360)

# ---------- A4 ----------
with tabs[3]:
    st.subheader("Arista 4 – Capital (Basilea Estándar)")
    st.markdown(
        "- *Objetivo*: Disminuir RWA y K manteniendo retornos y límites regulatorios.\n"
        "- *Fórmulas*: RWA = RW × EAD; K = k_ratio × RWA."
    )
    A4 = load_by_arista(bundle_dir, "A4", SFX)
    if A4["portfolio"] is not None:
        st.write("*KPIs de Portafolio*")
        st.dataframe(format_df_auto(A4["portfolio"], moneda, usdclp), use_container_width=True, height=180)
    if A4["segment"] is not None:
        st.write("*KPIs por Segmento*")
        st.dataframe(format_df_auto(A4["segment"], moneda, usdclp), use_container_width=True, height=260)
    if A4["detail"] is not None:
        st.write("*Detalle por Cliente*")
        st.dataframe(format_df_auto(A4["detail"].head(5000), moneda, usdclp), use_container_width=True, height=360)

# ---------- Guardrails ----------
with tabs[4]:
    st.subheader("Guardrails (Controles)")
    G = load_by_arista(bundle_dir, "GUARD", "")
    if G.get("portfolio") is not None:
        st.write("*Portafolio*")
        st.dataframe(format_df_auto(G["portfolio"], moneda, usdclp), use_container_width=True, height=200)
    if G.get("segment") is not None:
        st.write("*Segmento*")
        st.dataframe(format_df_auto(G["segment"], moneda, usdclp), use_container_width=True, height=200)
    if G.get("eval") is not None:
        st.write("*Evaluación*")
        st.dataframe(format_df_auto(G["eval"], moneda, usdclp), use_container_width=True, height=220)

# ---------- KPIs Unificados ----------
with tabs[5]:
    st.subheader("KPIs Unificados")
    kpis = read_csv_safe(bundle_dir / FILES["UNI"]["kpis"])
    if kpis is not None:
        st.dataframe(format_df_auto(kpis, moneda, usdclp), use_container_width=True, height=360)
    else:
        st.info("No se encontró kpis_unificados.csv en el bundle.")

st.markdown("---")
st.caption("© WDOF – Motor Clásico (visualización restaurada)")
