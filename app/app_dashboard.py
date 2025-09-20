# app/app_dashboard.py — MVP Bancario (4 Aristas) v2.1
# Robusta a esquemas Conservador / Potenciado y variaciones menores de columnas.
import os, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ==========================
# Config de archivos
# ==========================
REQ_FILES_BASE = {
    # Arista 1
    "a1_port": "default_portfolio{S}.csv",
    "a1_seg":  "default_segment{S}.csv",
    "a1_det":  "default_detail{S}.csv",
    # Arista 2
    "a2_port": "yield_portfolio{S}.csv",
    "a2_seg":  "yield_segment{S}.csv",
    "a2_det":  "yield_detail{S}.csv",
    "a2_cur":  "yield_curve_segment{S}.csv",
    # Arista 3
    "a3_det":  "incentives_detail{S}.csv",
    "a3_sum":  "incentives_diag_summary{S}.csv",
    "a3_sens": "incentives_sensitivity{S}.csv",
    # Arista 4
    "a4_port": "capital_portfolio{S}.csv",
    "a4_seg":  "capital_segment{S}.csv",
    "a4_det":  "capital_detail{S}.csv",
    # Guardrails (catálogo export real)
    "gr_port": "guardrails_portfolio.csv",
    "gr_seg":  "guardrails_segment.csv",
    # Guardrails (evaluación celda 16)
    "gr_eval": "guardrails_eval_portfolio.csv",
}

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    "/content/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
    "/content/out",
    "./out",
]

# ==========================
# Utilidades de formato
# ==========================
_num_like = re.compile(r"^-?\d+(\.\d+)?$")

def _to_float_or_nan(v):
    if v is None: return np.nan
    if isinstance(v, (int, float)): 
        try: return float(v)
        except: return np.nan
    if isinstance(v, str):
        s = v.strip().replace("%","").replace(",", ".")
        return float(s) if _num_like.match(s) else np.nan
    return np.nan

def fmt_pct_val(val):
    if isinstance(val, str):
        s = val.strip()
        if s.endswith("%"): return s.replace(".", ",")
        x = _to_float_or_nan(s)
        if np.isnan(x): return "—"
        return f"{x:.2f}%".replace(".", ",")
    x = _to_float_or_nan(val)
    if np.isnan(x): return "—"
    return f"{x:.2f}%".replace(".", ",")

def _to_display_currency(val: float, target: str, usdclp: float) -> float:
    if pd.isna(val): return np.nan
    return float(val) / float(usdclp) if target.upper() == "USD" and usdclp else float(val)

def fmt_money_val(val, moneda: str, usdclp: float):
    # strings no-numéricas se devuelven tal cual; NaN -> "—"
    if isinstance(val, str):
        v = val.strip()
        if v == "" or v.upper() == "N/A": return "—"
        # si aparenta ser número en string, lo convertimos para formatear
        tmp = _to_float_or_nan(v)
        if not np.isnan(tmp): val = tmp
        else: return v
    if val is None or (isinstance(val, float) and math.isnan(val)): return "—"
    x = _to_display_currency(val, moneda, usdclp)
    if x is None or (isinstance(x, float) and math.isnan(x)): return "—"
    neg = x < 0; x = abs(x)
    ent = int(x); dec = int(round((x - ent) * 100))
    if dec == 100: ent += 1; dec = 0
    ent_str = f"{ent:,}".replace(",", ".")
    return f"-{ent_str},{dec:02d}" if neg else f"{ent_str},{dec:02d}"

def var_pct(actual, opt):
    a = _to_float_or_nan(actual); o = _to_float_or_nan(opt)
    if np.isnan(a) or a == 0: return None
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

# ==========================
# Utilidades de carga
# ==========================
def _dir_ok(d: str, suf: str) -> bool:
    if not d or not os.path.isdir(d): return False
    # Basta con hallar al menos un archivo clave del escenario
    keys = ["a1_port","a2_port","a4_port"]
    hits = 0
    for k in keys:
        fname = REQ_FILES_BASE[k].format(S=suf)
        if os.path.exists(os.path.join(d, fname)):
            hits += 1
    return hits >= 1

def autodetect_bundle(suf: str) -> str | None:
    for d in CANDIDATE_DIRS:
        if _dir_ok(d, suf): return d
    return None

def load_csv(bundle_dir: str, pattern: str, suf: str) -> pd.DataFrame | None:
    path = os.path.join(bundle_dir, pattern.format(S=suf))
    if not os.path.exists(path): return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def g0(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            try:
                return float(df[c].iloc[0])
            except Exception:
                try:
                    return _to_float_or_nan(df[c].iloc[0])
                except Exception:
                    return None
    return None

# ==========================
# App
# ==========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")
st.sidebar.title("⚙️ Configuración")

escenario = st.sidebar.radio("Escenario", ["Conservador", "Potenciado"], horizontal=True)
suf = "" if escenario == "Conservador" else "_agresivo"

moneda = st.sidebar.radio("Moneda", ["CLP", "USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))

default_dir = autodetect_bundle(suf)
bundle_dir = st.sidebar.text_input("📦 Ruta del bundle", value=(default_dir or ""), help="Ej: /content/out/dashboard_bundle").strip() or default_dir

st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Modelo matemático con cumplimiento (IFRS9 + Basel proxy), comparando *Actual vs Optimizado* por arista y escenario.")

if not bundle_dir:
    st.error("No encuentro el bundle. Revisa la ruta o genera el paquete en el notebook.")
    st.stop()

tabs = st.tabs([
    "Arista 1 – Default/Impago",
    "Arista 2 – Yield/Pricing",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital/Provisiones",
    "Guardrails (Resguardos)"
])

# ==============================
# Arista 1 – Default/Impago
# ==============================
with tabs[0]:
    st.header("Arista 1 – Default/Impago")
    st.markdown("*Objetivo:* Mantener el *EAD* total, recomponiendo mix hacia menor *PD* para *reducir EL* sin frenar el negocio.")
    st.markdown("""
*KPIs clave*
- *EAD*: Exposición en riesgo (total estable; cambia la composición).  
- *Pérdida Esperada (EL): PD × LGD × EAD (12m IFRS9); buscamos *↓**.  
- *Interés Devengado Bruto*: APR × EAD (proxy del margen antes de pérdidas).  
- *Utilidad: Interés Bruto – EL – (costos si aplicaran). Buscamos *↑**.  
- *PD ponderado: PD promedio ponderado por EAD; buscamos *↓**.
    """)

    a1p = load_csv(bundle_dir, REQ_FILES_BASE["a1_port"], suf)
    if a1p is not None and not a1p.empty:
        # robustez a nombres alternativos
        kpi_row_money("Interés Devengado Bruto",
            g0(a1p, ["Interes_devengado_bruto_actual", "Ingreso_actual", "ingreso_base"]),
            g0(a1p, ["Interes_devengado_bruto_optimizado", "Ingreso_optimizado", "ingreso_final", "ingreso_opt"]),
            moneda, usdclp, "APR×EAD (proxy de margen).")
        kpi_row_money("Utilidad",
            g0(a1p, ["Utilidad_actual","utilidad_base","Util_actual"]),
            g0(a1p, ["Utilidad_optimizada","utilidad_final","utilidad_opt","Util_opt"]),
            moneda, usdclp)
        kpi_row_money("Pérdida Esperada (EL)",
            g0(a1p, ["EL_actual","EL_base"]),
            g0(a1p, ["EL_optimizado","EL_final","EL_opt"]),
            moneda, usdclp)
        kpi_row_money("EAD",
            g0(a1p, ["EAD_actual","EAD_base"]),
            g0(a1p, ["EAD_optimizado","EAD_final","EAD_opt"]),
            moneda, usdclp)
        # PD ponderado (0–1 → %)
        pd_act = g0(a1p, ["PD_pond_actual","PDpond_actual"])
        pd_opt = g0(a1p, ["PD_pond_optimizado","PDpond_optimizado"])
        kpi_row_pct("PD Ponderado",
            (pd_act*100 if pd_act is not None else None),
            (pd_opt*100 if pd_opt is not None else None))

    a1d = load_csv(bundle_dir, REQ_FILES_BASE["a1_det"], suf)
    if a1d is not None and not a1d.empty:
        st.markdown("*Detalle por cliente*")
        money_cols = ["e_base","e_final","income_base","income_final","util_base","util_final","EL_base","EL_final"]
        pct_cols = ["pd_base","pd_final","lgd_base","lgd_final","PD_12m","LGD_adj"]
        df_fmt = format_df_currency(a1d, money_cols, moneda, usdclp)
        df_fmt = format_df_pct(df_fmt, pct_cols)
        st.dataframe(df_fmt, use_container_width=True, height=360)

# ==============================
# Arista 2 – Yield/Pricing
# ==============================
with tabs[1]:
    st.header("Arista 2 – Yield/Pricing")
    st.markdown("*Objetivo:* Encontrar el *APR óptimo por segmento* que maximiza *Utilidad* equilibrando *precio vs volumen* (elasticidades).")
    st.markdown("""
*KPIs clave*
- *Utilidad Total: Interés Bruto – EL – COF. Objetivo *↑**.  
- *Interés Bruto Total*: APR × EAD_out.  
- *EAD_in / EAD_out*: Volumen antes/después por elasticidad.  
- *APR óptimo (promedio seg.)* dentro de bandas/caps.
    """)

    a2p = load_csv(bundle_dir, REQ_FILES_BASE["a2_port"], suf)
    a2s = load_csv(bundle_dir, REQ_FILES_BASE["a2_seg"], suf)
    a2d = load_csv(bundle_dir, REQ_FILES_BASE["a2_det"], suf)

    if a2p is not None and not a2p.empty:
        kpi_row_money("Utilidad Total",
            g0(a2p, ["utilidad_base","Utilidad_base"]),
            g0(a2p, ["utilidad_opt","Utilidad_opt"]),
            moneda, usdclp)
        kpi_row_money("Interés Bruto Total",
            g0(a2p, ["ingreso_base","Ingreso_base"]),
            g0(a2p, ["ingreso_opt","Ingreso_opt"]),
            moneda, usdclp)
        # EAD in/out como KPI de volumen
        kpi_row_money("EAD (in → out)",
            g0(a2p, ["EAD_in","EAD_base","EAD"]),
            g0(a2p, ["EAD_out","EAD_opt"]),
            moneda, usdclp,
            "Volumen afectado por elasticidad de demanda vs APR.")

    if a2s is not None and not a2s.empty:
        st.markdown("*Resumen por segmento (APR_opt y resultados)*")
        df = a2s.copy()
        money_cols = ["EAD_in","EAD_out","ingreso_opt","utilidad_opt"]
        df1 = format_df_currency(df, money_cols, moneda, usdclp)
        st.dataframe(df1, use_container_width=True, height=340)

    if a2d is not None and not a2d.empty:
        st.markdown("*Detalle por cliente (pricing)*")
        money_cols = ["ead_in","e_out","income_opt","EL_opt","COF_opt","util_opt"]
        df2 = format_df_currency(a2d, money_cols, moneda, usdclp)
        st.dataframe(df2, use_container_width=True, height=340)

# ==============================
# Arista 3 – Incentivos
# ==============================
with tabs[2]:
    st.header("Arista 3 – Incentivos")
    st.markdown("*Objetivo:* Asignar incentivos *sólo* donde el *ROI* esperado es *> 0*, bajo presupuesto global.")
    st.markdown("""
*KPIs clave*
- *Costo de incentivos*: gasto ejecutado.  
- *Ingreso incremental*: aumento de ingresos atribuible a la acción.  
- *ROI: Ingreso incremental / Costo. Objetivo *> 1** y creciente.  
- *Sensibilidad*: cómo varían resultados al exigir mayor ROI mínimo.
    """)

    a3d = load_csv(bundle_dir, REQ_FILES_BASE["a3_det"], suf)
    a3s = load_csv(bundle_dir, REQ_FILES_BASE["a3_sum"], suf)
    a3x = load_csv(bundle_dir, REQ_FILES_BASE["a3_sens"], suf)

    if a3s is not None and not a3s.empty:
        r = a3s.iloc[0].to_dict()
        costo = r.get("budget_usado", r.get("costo", 0.0))
        ingr  = r.get("ingreso_incremental", r.get("ingreso_inc", 0.0))
        roi   = (ingr / costo * 100.0) if costo not in (None, 0, "0") else np.nan
        kpi_row_money("Costo de incentivos", costo, costo, moneda, usdclp)
        kpi_row_money("Ingreso incremental", ingr, ingr, moneda, usdclp)
        st.metric("ROI", fmt_pct_val(roi))

    if a3d is not None and not a3d.empty:
        st.markdown("*Detalle seleccionado* (top por ROI dentro de presupuesto)")
        df = a3d.copy()
        # nombres alternativos
        if "roi" not in df.columns and "ROI" in df.columns:
            df = df.rename(columns={"ROI":"roi"})
        money_cols = ["costo_incentivo","ingreso_incremental","budget_usado"]
        pct_cols = ["roi"]
        df1 = format_df_currency(df, money_cols, moneda, usdclp)
        df1 = format_df_pct(df1, pct_cols)
        st.dataframe(df1, use_container_width=True, height=360)

    if a3x is not None and not a3x.empty:
        st.markdown("*Sensibilidades (ROI mínimo)*")
        df = a3x.copy()
        df2 = format_df_currency(df, ["budget","costo","ingreso_inc"], moneda, usdclp)
        df2 = format_df_pct(df2, ["ROI"])
        st.dataframe(df2, use_container_width=True, height=280)

# ==============================
# Arista 4 – Capital/Provisiones
# ==============================
with tabs[3]:
    st.header("Arista 4 – Capital/Provisiones")
    st.markdown("*Objetivo:* Hacer más eficiente el *consumo de capital* (RWA, K) y reducir *provisiones* (≈ EL) sin deteriorar la calidad.")
    st.markdown("""
*KPIs clave (portafolio)*
- *EAD (base vs opt)*.  
- *RWA (proxy Basel): RW × EAD. Buscamos *↓**.  
- *Capital (K): K_ratio × RWA. Buscamos *↓**.  
- *Provisiones (≈ EL): consistentes con IFRS9. Buscamos *↓**.
    """)

    a4p = load_csv(bundle_dir, REQ_FILES_BASE["a4_port"], suf)
    a4d = load_csv(bundle_dir, REQ_FILES_BASE["a4_det"], suf)
    a4s = load_csv(bundle_dir, REQ_FILES_BASE["a4_seg"], suf)

    if a4p is not None and not a4p.empty:
        # robusto a nombres: nuevo (EAD_base/opt, etc.)
        kpi_row_money("EAD",
            g0(a4p, ["EAD_base","EAD"]),
            g0(a4p, ["EAD_opt","EAD"]),
            moneda, usdclp)
        kpi_row_money("RWA (proxy Basel)",
            g0(a4p, ["RWA_base","RWA"]),
            g0(a4p, ["RWA_opt","RWA"]),
            moneda, usdclp)
        kpi_row_money("Capital (K)",
            g0(a4p, ["K_base","K"]),
            g0(a4p, ["K_opt","K"]),
            moneda, usdclp)
        kpi_row_money("Provisiones (≈ EL)",
            g0(a4p, ["EL_base","EL"]),
            g0(a4p, ["EL_opt","EL"]),
            moneda, usdclp)

    if a4d is not None and not a4d.empty:
        st.markdown("*Detalle por cliente*")
        money_cols = ["EAD_base","EAD_opt","RWA_base","RWA_opt","K_base","K_opt","EL_base","EL_opt","Util_base","Util_opt"]
        df1 = format_df_currency(a4d, money_cols, moneda, usdclp)
        st.dataframe(df1, use_container_width=True, height=360)

    if a4s is not None and not a4s.empty:
        st.markdown("*Resumen por segmento*")
        df2 = format_df_currency(a4s, ["EAD_base","EAD_opt","RWA_base","RWA_opt","K_base","K_opt","EL_base","EL_opt","Util_base","Util_opt"], moneda, usdclp)
        st.dataframe(df2, use_container_width=True, height=320)

# ==============================
# Guardrails
# ==============================
with tabs[4]:
    st.header("Guardrails (Resguardos)")
    st.markdown("Catálogo de *límites* (regulatorios y de negocio) y *evaluación* de consistencia (Celda 16).")

    gport = load_csv(bundle_dir, REQ_FILES_BASE["gr_port"], "")
    gseg  = load_csv(bundle_dir, REQ_FILES_BASE["gr_seg"], "")
    geval = load_csv(bundle_dir, REQ_FILES_BASE["gr_eval"], "")

    if gport is None and gseg is None:
        st.info("No se encontraron archivos de catálogo de guardrails.")
    else:
        if gport is not None and not gport.empty:
            st.subheader("Catálogo – Portafolio")
            df = gport.copy()
            for c in ["umbral","observado_actual","observado_optimizado"]:
                if c in df.columns: df[c] = df[c].apply(fmt_pct_val)
            st.dataframe(df, use_container_width=True, height=280)
        if gseg is not None and not gseg.empty:
            st.subheader("Catálogo – Segmento")
            df = gseg.copy()
            if "observado" in df.columns:
                df["observado"] = df["observado"].apply(fmt_pct_val)
            st.dataframe(df, use_container_width=True, height=280)

    st.markdown("---")
    st.subheader("Evaluación automática (Celda 16)")
    if geval is None or geval.empty:
        st.info("No se encontró *guardrails_eval_portfolio.csv*. Ejecuta la Celda 16 en el Notebook.")
    else:
        st.dataframe(geval, use_container_width=True, height=320)

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (IFRS9 + Basel proxy). Estructura lista para piloto IT.")
