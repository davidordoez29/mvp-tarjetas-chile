# app/app_dashboard.py — MVP Bancario (4 Aristas) v2.3 (root-fix)
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
    # Guardrails
    "gr_port": "guardrails_portfolio.csv",
    "gr_seg":  "guardrails_segment.csv",
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
# Parsing / Formato numérico
# ==========================
_num_like = re.compile(r"^-?\d+(\.\d+)?$")

def parse_num_any(v):
    if v is None: return np.nan
    if isinstance(v, (int, float)):
        try: return float(v)
        except: return np.nan
    if isinstance(v, str):
        s = v.strip().replace(" ", "").replace("−","-").replace("%","")
        if s == "" or s.upper() in {"N/A","NA","NULL","NONE","—"}: return np.nan
        last_dot, last_com = s.rfind("."), s.rfind(",")
        if last_dot == -1 and last_com == -1:
            return float(s) if s.lstrip("-").isdigit() else np.nan
        if last_com > last_dot:
            s = s.replace(".","").replace(",",".")
        else:
            s = s.replace(",","")
        try: return float(s)
        except: return np.nan
    return np.nan

def _to_display_currency(val: float, target: str, usdclp: float) -> float:
    if pd.isna(val): return np.nan
    return float(val) / float(usdclp) if target.upper() == "USD" and usdclp else float(val)

def fmt_money_val(val, moneda: str, usdclp: float):
    x = parse_num_any(val)
    if np.isnan(x): return "—"
    x = _to_display_currency(x, moneda, usdclp)
    neg = x < 0; x = abs(x)
    ent = int(x); dec = int(round((x - ent) * 100))
    if dec == 100: ent += 1; dec = 0
    ent_str = f"{ent:,}".replace(",", ".")
    return f"-{ent_str},{dec:02d}" if neg else f"{ent_str},{dec:02d}"

def fmt_pct_val(val):
    if isinstance(val, str) and val.strip().endswith("%"):
        return val.strip().replace(".", ",")
    x = parse_num_any(val)
    if np.isnan(x): return "—"
    return f"{x:.2f}%".replace(".", ",")

def var_pct(actual, opt):
    a = parse_num_any(actual); o = parse_num_any(opt)
    if np.isnan(a) or a == 0: return None
    return (o - a) / a * 100.0

def kpi_row_money(label: str, actual, opt, moneda: str, usdclp: float, help_text: str = ""):
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        st.metric(f"{label} – Actual", fmt_money_val(actual, moneda, usdclp))
        if help_text: st.caption(help_text)
    with c2:
        st.metric(f"{label} – Optimizado", fmt_money_val(opt, moneda, usdclp))
    with c3:
        vp = var_pct(actual, opt)
        st.metric("VAR %", fmt_pct_val(vp) if vp is not None else "—")

def kpi_row_pct(label: str, actual_pct, opt_pct, help_text: str = ""):
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        st.metric(f"{label} – Actual", fmt_pct_val(actual_pct)); 
        if help_text: st.caption(help_text)
    with c2:
        st.metric(f"{label} – Optimizado", fmt_pct_val(opt_pct))
    with c3:
        vp = var_pct(actual_pct, opt_pct)
        st.metric("VAR %", fmt_pct_val(vp) if vp is not None else "—")

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
# Bundle helpers
# ==========================
def _exists(bundle_dir: str, key: str, suf: str) -> bool:
    p = os.path.join(bundle_dir, REQ_FILES_BASE[key].format(S=suf))
    return os.path.exists(p)

def _scenario_autodetect(bundle_dir: str) -> str:
    """Retorna sufijo "" (Conservador) o "_agresivo" (Potenciado) según lo disponible.
       Si sólo hay _agresivo -> fuerza Potenciado."""
    has_cons = any(_exists(bundle_dir, k, "") for k in ["a1_port","a2_port","a4_port"])
    has_aggr = any(_exists(bundle_dir, k, "_agresivo") for k in ["a1_port","a2_port","a4_port"])
    if has_aggr and not has_cons:
        return "_agresivo"
    if has_cons and not has_aggr:
        return ""
    # Si hay ambos o ninguno, por defecto conservador
    return ""

def autodetect_bundle(suf: str) -> str | None:
    for d in CANDIDATE_DIRS:
        if d and os.path.isdir(d):
            # si no hay nada con suf pedido, intentamos el autodetect interno
            if any(_exists(d, k, suf) for k in ["a1_port","a2_port","a4_port"]):
                return d
            # caso: usaron otro sufijo al arrancar → probar autodetect interno
            s2 = _scenario_autodetect(d)
            if any(_exists(d, k, s2) for k in ["a1_port","a2_port","a4_port"]):
                return d
    return None

def load_csv(bundle_dir: str, pattern: str, suf: str) -> pd.DataFrame | None:
    path = os.path.join(bundle_dir, pattern.format(S=suf))
    if not os.path.exists(path): return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def pick0(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns and not df.empty:
            return parse_num_any(df[c].iloc[0])
    return None

# ==========================
# App
# ==========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")
st.sidebar.title("⚙️ Configuración")

# Paso 1: escenario (pero lo validaremos con autodetección del bundle)
escenario_ui = st.sidebar.radio("Escenario", ["Conservador", "Potenciado"], horizontal=True)
suf_ui = "" if escenario_ui == "Conservador" else "_agresivo"

# Paso 2: moneda
moneda = st.sidebar.radio("Moneda", ["CLP", "USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))

# Paso 3: bundle + escenario real disponible
default_dir = autodetect_bundle(suf_ui)
bundle_dir = st.sidebar.text_input("📦 Ruta del bundle", value=(default_dir or ""), help="Ej: /content/out/dashboard_bundle").strip() or default_dir

st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Modelo matemático con cumplimiento (IFRS9 + Basel proxy). Comparación *Actual vs Optimizado* por arista y escenario.")

if not bundle_dir:
    st.error("No encuentro el bundle. Revisa la ruta o genera el paquete en el Notebook.")
    st.stop()

# Si el escenario elegido no existe en el bundle, forzamos el disponible
suf_real = suf_ui
if not any(_exists(bundle_dir, k, suf_real) for k in ["a1_port","a2_port","a4_port"]):
    suf_real = _scenario_autodetect(bundle_dir)
    if suf_real == "_agresivo":
        st.info("⚠️ Sólo se encontraron archivos del escenario *Potenciado*. Se ajustó automáticamente.")
    else:
        st.info("⚠️ Sólo se encontraron archivos del escenario *Conservador*. Se ajustó automáticamente.")

# Diagnóstico del bundle
with st.expander("🔎 Diagnóstico del bundle", expanded=False):
    found = []
    for key, pat in REQ_FILES_BASE.items():
        p = os.path.join(bundle_dir, pat.format(S=suf_real))
        if os.path.exists(p):
            found.append(f"✅ {os.path.basename(p)}")
    st.write("\n".join(found) if found else "No se encontraron archivos para el escenario seleccionado.")

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
    st.markdown("*Objetivo:* Mantener *EAD* total estable, recomponiendo mix hacia menor *PD* para *reducir EL* sin frenar el negocio.")
    st.markdown("""
*KPIs clave*
- *EAD*: Exposición en riesgo (total estable; cambia la composición).  
- *Pérdida Esperada (EL): PD × LGD × EAD (12m IFRS9); buscamos *↓**.  
- *Interés Devengado Bruto*: APR × EAD (proxy del margen antes de pérdidas).  
- *Utilidad: Interés Bruto – EL – (costos si aplicaran). Buscamos *↑**.  
- *PD ponderado: PD promedio ponderado por EAD; buscamos *↓**.
    """)

    a1p = load_csv(bundle_dir, REQ_FILES_BASE["a1_port"], suf_real)
    a1d = load_csv(bundle_dir, REQ_FILES_BASE["a1_det"], suf_real)

    # KPIs (si falta en portafolio, intentar sumar del detalle)
    if a1p is None or a1p.empty:
        if a1d is None or a1d.empty:
            st.warning("No hay datos para Arista 1 (ni portfolio ni detail).")
        else:
            df = a1d.copy()
            # intentamos construir KPIs básicos desde el detalle
            ead_act = parse_num_any(df.get("e_base", np.nan)).sum() if "e_base" in df else np.nan
            ead_opt = parse_num_any(df.get("e_final", np.nan)).sum() if "e_final" in df else np.nan
            el_act  = parse_num_any(df.get("EL_base", np.nan)).sum() if "EL_base" in df else np.nan
            el_opt  = parse_num_any(df.get("EL_final", np.nan)).sum() if "EL_final" in df else np.nan
            inc_act = parse_num_any(df.get("income_base", np.nan)).sum() if "income_base" in df else np.nan
            inc_opt = parse_num_any(df.get("income_final", np.nan)).sum() if "income_final" in df else np.nan
            util_a  = parse_num_any(df.get("util_base", np.nan)).sum() if "util_base" in df else np.nan
            util_o  = parse_num_any(df.get("util_final", np.nan)).sum() if "util_final" in df else np.nan
            kpi_row_money("Interés Devengado Bruto", inc_act, inc_opt, moneda, usdclp)
            kpi_row_money("Utilidad", util_a, util_o, moneda, usdclp)
            kpi_row_money("Pérdida Esperada (EL)", el_act, el_opt, moneda, usdclp)
            kpi_row_money("EAD", ead_act, ead_opt, moneda, usdclp)
    else:
        kpi_row_money("Interés Devengado Bruto",
            pick0(a1p, ["Interes_devengado_bruto_actual","Ingreso_actual","ingreso_base"]),
            pick0(a1p, ["Interes_devengado_bruto_optimizado","Ingreso_optimizado","ingreso_final","ingreso_opt"]),
            moneda, usdclp, "APR×EAD (proxy de margen).")
        kpi_row_money("Utilidad",
            pick0(a1p, ["Utilidad_actual","utilidad_base","Util_actual"]),
            pick0(a1p, ["Utilidad_optimizada","utilidad_final","utilidad_opt","Util_opt"]),
            moneda, usdclp)
        kpi_row_money("Pérdida Esperada (EL)",
            pick0(a1p, ["EL_actual","EL_base"]),
            pick0(a1p, ["EL_optimizado","EL_final","EL_opt"]),
            moneda, usdclp)
        kpi_row_money("EAD",
            pick0(a1p, ["EAD_actual","EAD_base","EAD"]),
            pick0(a1p, ["EAD_optimizado","EAD_final","EAD_opt","EAD"]),
            moneda, usdclp)

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
    st.markdown("*Objetivo:* Encontrar el *APR óptimo por segmento* que maximiza *Utilidad* equilibrando *precio vs volumen*.")
    st.markdown("""
*KPIs clave*
- *Utilidad Total*: Interés Bruto – EL – COF (↑).  
- *Interés Bruto Total*: APR × EAD_out.  
- *EAD_in / EAD_out*: Volumen antes/después por elasticidad.  
- *APR óptimo (promedio seg.)* dentro de bandas/caps.
    """)

    a2p = load_csv(bundle_dir, REQ_FILES_BASE["a2_port"], suf_real)
    a2s = load_csv(bundle_dir, REQ_FILES_BASE["a2_seg"], suf_real)
    a2d = load_csv(bundle_dir, REQ_FILES_BASE["a2_det"], suf_real)

    if a2p is None or a2p.empty:
        # Intentar sintetizar desde detalle
        if a2d is None or a2d.empty:
            st.warning("No hay datos para Arista 2 (ni portfolio ni detail).")
        else:
            df = a2d.copy()
            util_b = parse_num_any(df.get("util_base", np.nan)).sum() if "util_base" in df else np.nan
            util_o = parse_num_any(df.get("util_opt", np.nan)).sum() if "util_opt" in df else np.nan
            ing_b  = parse_num_any(df.get("income_base", np.nan)).sum() if "income_base" in df else np.nan
            ing_o  = parse_num_any(df.get("income_opt", np.nan)).sum() if "income_opt" in df else np.nan
            e_in   = parse_num_any(df.get("ead_in", np.nan)).sum() if "ead_in" in df else np.nan
            e_out  = parse_num_any(df.get("e_out", np.nan)).sum() if "e_out" in df else np.nan
            kpi_row_money("Utilidad Total", util_b, util_o, moneda, usdclp)
            kpi_row_money("Interés Bruto Total", ing_b, ing_o, moneda, usdclp)
            kpi_row_money("EAD (in → out)", e_in, e_out, moneda, usdclp,
                          "Volumen afectado por elasticidad de demanda vs APR.")
    else:
        kpi_row_money("Utilidad Total",
            pick0(a2p, ["utilidad_base","Utilidad_base"]),
            pick0(a2p, ["utilidad_opt","Utilidad_opt"]),
            moneda, usdclp)
        kpi_row_money("Interés Bruto Total",
            pick0(a2p, ["ingreso_base","Ingreso_base"]),
            pick0(a2p, ["ingreso_opt","Ingreso_opt"]),
            moneda, usdclp)
        kpi_row_money("EAD (in → out)",
            pick0(a2p, ["EAD_in","EAD_base","EAD"]),
            pick0(a2p, ["EAD_out","EAD_opt","EAD"]),
            moneda, usdclp,
            "Volumen afectado por elasticidad de demanda vs APR.")

    if a2s is not None and not a2s.empty:
        st.markdown("*Resumen por segmento (APR_opt y resultados)*")
        df = a2s.copy()
        df1 = format_df_currency(df, ["EAD_in","EAD_out","ingreso_opt","utilidad_opt"], moneda, usdclp)
        st.dataframe(df1, use_container_width=True, height=340)

    if a2d is not None and not a2d.empty:
        st.markdown("*Detalle por cliente (pricing)*")
        df2 = format_df_currency(a2d.copy(), ["ead_in","e_out","income_opt","EL_opt","COF_opt","util_opt"], moneda, usdclp)
        st.dataframe(df2, use_container_width=True, height=340)

# ==============================
# Arista 3 – Incentivos
# ==============================
with tabs[2]:
    st.header("Arista 3 – Incentivos")
    st.markdown("*Objetivo:* Asignar incentivos *sólo* donde *ROI > 0*, bajo presupuesto global.")
    st.markdown("""
*KPIs clave*
- *Costo de incentivos* (gasto).  
- *Ingreso incremental* (uplift atribuible).  
- *ROI* = Ingreso inc. / Costo (↑).  
- *Sensibilidad* por umbrales de ROI.
    """)

    a3d = load_csv(bundle_dir, REQ_FILES_BASE["a3_det"], suf_real)
    a3s = load_csv(bundle_dir, REQ_FILES_BASE["a3_sum"], suf_real)
    a3x = load_csv(bundle_dir, REQ_FILES_BASE["a3_sens"], suf_real)

    if (a3s is None or a3s.empty) and (a3d is None or a3d.empty):
        st.warning("No hay datos para Arista 3.")
    else:
        if a3s is not None and not a3s.empty:
            r = a3s.iloc[0].to_dict()
            costo = r.get("budget_usado", r.get("costo", 0.0))
            ingr  = r.get("ingreso_incremental", r.get("ingreso_inc", 0.0))
            c, i  = parse_num_any(costo), parse_num_any(ingr)
            roi   = (i / c * 100.0) if (c not in (None, 0, np.nan)) else np.nan
            kpi_row_money("Costo de incentivos", c, c, moneda, usdclp)
            kpi_row_money("Ingreso incremental", i, i, moneda, usdclp)
            st.metric("ROI", fmt_pct_val(roi))

        if a3d is not None and not a3d.empty:
            st.markdown("*Detalle seleccionado* (top por ROI dentro de presupuesto)")
            df = a3d.copy()
            if "roi" not in df.columns and "ROI" in df.columns:
                df = df.rename(columns={"ROI":"roi"})
            df1 = format_df_currency(df, ["costo_incentivo","ingreso_incremental","budget_usado"], moneda, usdclp)
            df1 = format_df_pct(df1, ["roi"])
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
    st.markdown("*Objetivo:* Hacer más eficiente el *consumo de capital* (RWA, K) y reducir *provisiones* (≈ EL).")
    st.markdown("""
*KPIs clave (portafolio)*
- *EAD (base vs opt)*.  
- *RWA* = RW × EAD (↓).  
- *Capital (K)* = K_ratio × RWA (↓).  
- *Provisiones (≈ EL)* (↓).
    """)

    a4p = load_csv(bundle_dir, REQ_FILES_BASE["a4_port"], suf_real)
    a4d = load_csv(bundle_dir, REQ_FILES_BASE["a4_det"], suf_real)
    a4s = load_csv(bundle_dir, REQ_FILES_BASE["a4_seg"], suf_real)

    if a4p is None or a4p.empty:
        if a4d is None or a4d.empty:
            st.warning("No hay datos para Arista 4.")
        else:
            df = a4d.copy()
            ead_b = parse_num_any(df.get("EAD_base", np.nan)).sum() if "EAD_base" in df else np.nan
            ead_o = parse_num_any(df.get("EAD_opt",  np.nan)).sum() if "EAD_opt"  in df else np.nan
            rwa_b = parse_num_any(df.get("RWA_base", np.nan)).sum() if "RWA_base" in df else np.nan
            rwa_o = parse_num_any(df.get("RWA_opt",  np.nan)).sum() if "RWA_opt"  in df else np.nan
            k_b   = parse_num_any(df.get("K_base",   np.nan)).sum() if "K_base"   in df else np.nan
            k_o   = parse_num_any(df.get("K_opt",    np.nan)).sum() if "K_opt"    in df else np.nan
            el_b  = parse_num_any(df.get("EL_base",  np.nan)).sum() if "EL_base"  in df else np.nan
            el_o  = parse_num_any(df.get("EL_opt",   np.nan)).sum() if "EL_opt"   in df else np.nan
            kpi_row_money("EAD", ead_b, ead_o, moneda, usdclp)
            kpi_row_money("RWA (proxy Basel)", rwa_b, rwa_o, moneda, usdclp)
            kpi_row_money("Capital (K)", k_b, k_o, moneda, usdclp)
            kpi_row_money("Provisiones (≈ EL)", el_b, el_o, moneda, usdclp)
    else:
        kpi_row_money("EAD",
            pick0(a4p, ["EAD_base","EAD"]), pick0(a4p, ["EAD_opt","EAD"]), moneda, usdclp)
        kpi_row_money("RWA (proxy Basel)",
            pick0(a4p, ["RWA_base","RWA"]), pick0(a4p, ["RWA_opt","RWA"]), moneda, usdclp)
        kpi_row_money("Capital (K)",
            pick0(a4p, ["K_base","K"]),     pick0(a4p, ["K_opt","K"]),     moneda, usdclp)
        kpi_row_money("Provisiones (≈ EL)",
            pick0(a4p, ["EL_base","EL"]),   pick0(a4p, ["EL_opt","EL"]),   moneda, usdclp)

    if a4d is not None and not a4d.empty:
        st.markdown("*Detalle por cliente*")
        df1 = format_df_currency(a4d, ["EAD_base","EAD_opt","RWA_base","RWA_opt","K_base","K_opt","EL_base","EL_opt","Util_base","Util_opt"], moneda, usdclp)
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
    st.markdown("Catálogo y *evaluación automática* (Celda 16).")

    gport = load_csv(bundle_dir, REQ_FILES_BASE["gr_port"], "")
    gseg  = load_csv(bundle_dir, REQ_FILES_BASE["gr_seg"], "")
    geval = load_csv(bundle_dir, REQ_FILES_BASE["gr_eval"], "")

    if gport is None and gseg is None and (geval is None or geval.empty):
        st.info("No se encontraron archivos de guardrails. Ejecuta Celdas 15–16 y exporta al bundle.")
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
