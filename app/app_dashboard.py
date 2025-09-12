# app/app_dashboard.py
import os, glob, json, math
import pandas as pd
import numpy as np
import streamlit as st

# ==========================
# Archivos requeridos (nombres corregidos)
# ==========================
REQ_FILES = {
    # Arista 1 (Default)
    "def_port": "default_portfolio.csv",
    "def_seg":  "default_segment.csv",
    "def_det":  "default_detail.csv",
    # Arista 2 (Yield) — NOMBRES CORREGIDOS
    "yld_port": "yield_portfolio.csv",
    "yld_seg":  "yield_segment.csv",
    "yld_det":  "yield_detail.csv",
    "yld_curv": "yield_curve_segment.csv",
    # Arista 3 (Incentivos)
    "inc_det":  "incentives_detail.csv",
    "inc_sum":  "incentives_diag_summary.csv",
    "inc_sens": "incentives_sensitivity.csv",
    # Arista 4 (Capital / Provisiones)
    "cap_port": "capital_portafolio.csv",  # “portafolio” en español
    "cap_seg":  "capital_segment.csv",
    "cap_det":  "capital_detail.csv",
    # Meta
    "kpi_defs": "kpi_defs.json",
    "seg_defs": "segment_defs.json",
    "meta":     "bundle_meta.json",
}

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    "/content/mvp-tarjetas-chile/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
    "/content/out/dashboard_bundle",
]

def _dir_ok(d: str) -> bool:
    try:
        if not d or not os.path.isdir(d):
            return False
        hits = sum(os.path.exists(os.path.join(d, v)) for v in REQ_FILES.values())
        return hits >= 6
    except Exception:
        return False

def autodetect_bundle() -> str | None:
    for d in CANDIDATE_DIRS:
        if _dir_ok(d):
            return d
    # búsqueda amplia por si cambió la ruta
    try:
        bases = ["/content/mvp-tarjetas-chile", "/content", "."]
        candidates = []
        for base in bases:
            for p in glob.glob(os.path.join(base, "**", "dashboard_bundle"), recursive=True):
                if _dir_ok(p):
                    candidates.append((p, os.path.getmtime(p)))
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
    except Exception:
        pass
    return None

def load_bundle(bundle_dir: str):
    dfs, missing = {}, []
    for key, fname in REQ_FILES.items():
        path = os.path.join(bundle_dir, fname)
        if not os.path.exists(path):
            missing.append(fname); dfs[key] = None; continue
        try:
            if fname.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    dfs[key] = json.load(f)
            else:
                dfs[key] = pd.read_csv(path)
        except Exception as e:
            missing.append(f"{fname} (error: {e})")
            dfs[key] = None
    return dfs, missing

# ==========================
# Formato: miles con punto, decimales con coma (2)
# ==========================
def _to_display_currency(val: float, target: str, usdclp: float) -> float:
    if pd.isna(val):
        return np.nan
    if target.upper() == "USD":
        return float(val) / float(usdclp) if usdclp else np.nan
    return float(val)

def fmt_money(val: float, target: str, usdclp: float) -> str:
    """
    Formatea como 1.234.567,89 evitando reemplazos que dañen el separador.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    x = _to_display_currency(val, target, usdclp)
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    neg = x < 0
    x = abs(x)
    ent = int(x)
    dec = int(round((x - ent) * 100))
    if dec == 100:
        ent += 1
        dec = 0
    ent_str = f"{ent:,}".replace(",", ".")     # miles → punto
    s = f"{ent_str},{dec:02d}"                 # decimales → coma
    return f"-{s}" if neg else s

def fmt_pct(val: float) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"{val:.2f}%".replace(".", ",")

def var_pct(actual: float, opt: float) -> float | None:
    if actual is None or pd.isna(actual) or actual == 0:
        return None
    return (opt - actual) / actual * 100.0

def kpi_row(label: str, actual: float, opt: float, moneda: str, usdclp: float, help_text: str = ""):
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        st.metric(label=f"{label} – Actual", value=fmt_money(actual, moneda, usdclp))
        if help_text:
            st.caption(help_text)
    with c2:
        st.metric(label=f"{label} – Optimizado", value=fmt_money(opt, moneda, usdclp))
    with c3:
        vp = var_pct(actual, opt)
        st.metric(label="VAR %", value=fmt_pct(vp) if vp is not None else "—")

def kpi_row_pct(label: str, actual_pct: float, opt_pct: float, help_text: str = ""):
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        st.metric(label=f"{label} – Actual", value=fmt_pct(actual_pct))
        if help_text:
            st.caption(help_text)
    with c2:
        st.metric(label=f"{label} – Optimizado", value=fmt_pct(opt_pct))
    with c3:
        vp = var_pct(actual_pct, opt_pct)
        st.metric(label="VAR %", value=fmt_pct(vp) if vp is not None else "—")

# ==========================
# App
# ==========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

st.sidebar.title("⚙️ Configuración")
default_dir = autodetect_bundle()
bundle_dir = st.sidebar.text_input(
    "📦 Ruta del bundle",
    value=(default_dir or ""),
    help="Ej: /content/mvp-tarjetas-chile/out/dashboard_bundle"
).strip() or default_dir

if not bundle_dir:
    st.error("No encuentro el bundle. Genera el paquete (Celda 12) y vuelve a cargar.")
    st.stop()

dfs, missing = load_bundle(bundle_dir)
if missing:
    st.warning("Faltan archivos en el bundle (o nombres distintos):\n- " + "\n- ".join(missing))

moneda = st.sidebar.radio("Moneda a visualizar", ["CLP", "USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))
st.sidebar.caption("Aplica a todos los montos del dashboard.")

st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Portafolio de tarjetas. Comparación *Actual vs. Optimizado* con KPIs clave por arista.")

tabs = st.tabs([
    "Arista 1 – Default/Impago",
    "Arista 2 – Yield/Pricing",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital/Provisiones"
])

# ================
# Arista 1 – Default
# ================
with tabs[0]:
    st.subheader("Arista 1 – Default/Impago")
    st.markdown("""
*Qué ves aquí:* impacto del modelo en pérdidas esperadas (EL), exposición (EAD), ingreso y utilidad neta.<br>
*KPIs:*
- *EAD*: Exposición en riesgo.
- *EL*: Pérdida Esperada = PD × LGD × EAD.
- *Ingreso*: APR × EAD.
- *Costos*: Financieros + Operativos.
- *Utilidad*: Ingreso − EL − Costos.
- *PD ponderado*: PD promedio ponderado por EAD.
""", unsafe_allow_html=True)

    port = dfs.get("def_port")
    if port is None or (isinstance(port, pd.DataFrame) and port.empty):
        st.error("No se encontró *default_portfolio.csv*.")
    else:
        def g0(df, col): 
            return df[col].iloc[0] if (isinstance(df, pd.DataFrame) and col in df.columns and not df.empty) else np.nan

        EAD_act = g0(port, "EAD_actual")
        EAD_opt = g0(port, "EAD_optimizado")
        EL_act  = g0(port, "EL_actual")
        EL_opt  = g0(port, "EL_optimizado")
        Ing_act = g0(port, "Ingreso_actual")
        Ing_opt = g0(port, "Ingreso_optimizado")
        Cost_act= g0(port, "Costos_actual")
        Cost_opt= g0(port, "Costos_optimizado")
        Uti_act = g0(port, "Utilidad_actual")
        Uti_opt = g0(port, "Utilidad_optimizada")
        PDw_act = g0(port, "PD_pond_actual")
        PDw_opt = g0(port, "PD_pond_optimizado")

        kpi_row("EAD", EAD_act, EAD_opt, moneda, usdclp, "Exposición total")
        kpi_row("EL (Pérdida Esperada)", EL_act, EL_opt, moneda, usdclp, "PD × LGD × EAD")
        kpi_row("Ingreso", Ing_act, Ing_opt, moneda, usdclp, "APR × EAD")
        kpi_row("Costos Totales", Cost_act, Cost_opt, moneda, usdclp, "Financieros + Operativos")
        kpi_row("Utilidad", Uti_act, Uti_opt, moneda, usdclp, "Ingreso − EL − Costos")
        if pd.notna(PDw_act) or pd.notna(PDw_opt):
            kpi_row_pct("PD Ponderado (EAD)", PDw_act*100 if pd.notna(PDw_act) else np.nan,
                        PDw_opt*100 if pd.notna(PDw_opt) else np.nan,
                        "Probabilidad de default promedio ponderada por EAD")

    st.markdown("---")
    st.caption("Nuestro modelo reduce EL manteniendo control de EAD y maximizando utilidad bajo restricciones de negocio.")

# ================
# Arista 2 – Yield / Pricing
# ================
with tabs[1]:
    st.subheader("Arista 2 – Yield/Pricing")
    st.markdown("""
*Qué ves aquí:* efecto del pricing (APR) en ingreso y utilidad, aislando impacto de precio vs. precio+volumen.<br>
*KPIs:*
- *Ingreso/Utilidad (Total)*: usando r_opt y e_opt.
- *Ingreso/Utilidad (Solo Pricing)*: usando r_opt con EAD = baseline.
""", unsafe_allow_html=True)

    port = dfs.get("yld_port")
    if port is None or (isinstance(port, pd.DataFrame) and port.empty):
        st.error("No se encontraron archivos de Yield (verifica nombres).")
    else:
        def g0(df, name):
            return df[name].iloc[0] if name in df.columns and not df.empty else np.nan

        Ing_base = g0(port, "ingreso_base")
        Ing_iso  = g0(port, "ingreso_iso")
        Ing_opt  = g0(port, "ingreso_opt")

        Uti_base = g0(port, "utilidad_base")
        Uti_iso  = g0(port, "utilidad_iso")
        Uti_opt  = g0(port, "utilidad_opt")

        EL_base  = g0(port, "EL_baseline")
        EL_iso   = g0(port, "el_iso")
        EL_opt   = g0(port, "el_opt")

        kpi_row("Ingreso (Total)", Ing_base, Ing_opt, moneda, usdclp, "Precio + Volumen")
        kpi_row("Ingreso (Solo Pricing)", Ing_base, Ing_iso, moneda, usdclp, "EAD fijado en baseline")
        kpi_row("Utilidad (Total)", Uti_base, Uti_opt, moneda, usdclp, "Precio + Volumen")
        kpi_row("Utilidad (Solo Pricing)", Uti_base, Uti_iso, moneda, usdclp, "EAD fijado en baseline")
        kpi_row("EL", EL_base, EL_opt, moneda, usdclp, "Pérdida esperada total")

    st.markdown("---")
    st.caption("El pricing óptimo mueve el margen sin deteriorar desproporcionadamente el riesgo ni los costos.")

# ================
# Arista 3 – Incentivos
# ================
with tabs[2]:
    st.subheader("Arista 3 – Incentivos")
    st.markdown("""
*Qué ves aquí:* costo de incentivos vs. incremento esperado de ingreso/utilidad. ROI del esquema propuesto.<br>
*KPIs (portafolio):*
- *Costo Incentivos*.
- *Ingreso Incremental*.
- *ROI* = Ingreso Incremental / Costo.
""", unsafe_allow_html=True)

    det = dfs.get("inc_det")
    summ = dfs.get("inc_sum")

    if det is None or (isinstance(det, pd.DataFrame) and det.empty):
        st.error("No se encontró *incentives_detail.csv*.")
        total_cost = 0.0; uplift = 0.0
    else:
        # columnas candidatas
        cost_col = None
        for c in ["inc_cost","costo_incentivo","costo_incentivo_monto","costo_incentivo_total"]:
            if c in det.columns:
                cost_col = c; break
        if cost_col is None and ("costo_incentivo_tasa" in det.columns and "ead_baseline" in det.columns):
            det["_inc_cost_"] = pd.to_numeric(det["costo_incentivo_tasa"], errors="coerce").fillna(0)\
                                   * pd.to_numeric(det["ead_baseline"], errors="coerce").fillna(0)
            cost_col = "_inc_cost_"
        if cost_col is None:
            det["_inc_cost_"] = 0.0
            cost_col = "_inc_cost_"

        uplift_col = None
        for c in ["uplift_ingreso","ingreso_uplift","delta_ingreso","uplift_utilidad"]:
            if c in det.columns:
                uplift_col = c; break
        if uplift_col is None:
            det["_uplift_"] = 0.0
            uplift_col = "_uplift_"

        total_cost = pd.to_numeric(det[cost_col], errors="coerce").fillna(0).sum()
        uplift     = pd.to_numeric(det[uplift_col], errors="coerce").fillna(0).sum()

    roi = (uplift / total_cost) if total_cost > 0 else np.nan

    kpi_row("Costo de Incentivos", total_cost, total_cost, moneda, usdclp, "Suma de costos de beneficios")
    kpi_row("Ingreso Incremental", uplift, uplift, moneda, usdclp, "Suma de incrementos")
    c1, c2 = st.columns([1.2, 2.0])
    with c1:
        st.metric("ROI (Ingreso/Costo)", fmt_pct(roi*100 if pd.notna(roi) else np.nan))
    with c2:
        st.caption("Objetivo: ROI > 0 y rentabilidad positiva del esquema de beneficios.")

    if isinstance(summ, pd.DataFrame) and not summ.empty:
        with st.expander("Diagnóstico (resumen)"):
            st.dataframe(summ.head(50))

    st.markdown("---")
    st.caption("El motor asigna incentivos con ROI positivo por segmento/cliente, maximizando utilidad.")

# ================
# Arista 4 – Capital / Provisiones
# ================
with tabs[3]:
    st.subheader("Arista 4 – Capital / Provisiones")
    st.markdown("""
*Qué ves aquí:* requerimientos de capital y provisiones (proxy) antes y después de la optimización.<br>
*KPIs:*
- *Capital Requerido* (proxy RW×K×EAD).
- *Provisiones* ~ EL.
- *Liberación* = Actual − Optimizado.
""", unsafe_allow_html=True)

    cap = dfs.get("cap_port")
    if cap is None or (isinstance(cap, pd.DataFrame) and cap.empty):
        st.error("No se encontró *capital_portafolio.csv*.")
    else:
        def g0(df, name):
            return df[name].iloc[0] if name in df.columns and not df.empty else np.nan

        cap_base = g0(cap, "capital_req_base")
        cap_opt  = g0(cap, "capital_req_opt")
        prov_base= g0(cap, "prov_base")
        prov_opt = g0(cap, "prov_opt")

        kpi_row("Capital Requerido", cap_base, cap_opt, moneda, usdclp, "Proxy RW×K×EAD")
        kpi_row("Provisiones", prov_base, prov_opt, moneda, usdclp, "≈ EL")

        lib_cap  = cap_base - cap_opt  if pd.notna(cap_base) and pd.notna(cap_opt)  else np.nan
        lib_prov = prov_base - prov_opt if pd.notna(prov_base) and pd.notna(prov_opt) else np.nan

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Liberación de Capital", fmt_money(lib_cap, moneda, usdclp))
        with col2:
            st.metric("Liberación de Provisiones", fmt_money(lib_prov, moneda, usdclp))

    st.markdown("---")
    st.caption("La optimización reduce consumo de capital y estabiliza provisiones a lo largo del ciclo.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas integradas).")
