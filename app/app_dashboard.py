# app/app_dashboard.py — v3 (Executive Pitch + Guardrails derivados + formato numérico)

import os, glob, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ==========================
# Archivos requeridos (por arista / QA)
# ==========================
REQ_FILES = {
    # Arista 1 (Default)
    "def_port": "default_portfolio.csv",   # EAD/EL/Ingreso/Costos/Utilidad/PD_pond (actual/opt)
    "def_seg":  "default_segment.csv",
    "def_det":  "default_detail.csv",
    # Arista 2 (Yield)
    "yld_port": "yield_portfolio.csv",
    "yld_seg":  "yield_segment.csv",
    "yld_det":  "yield_detail.csv",
    "yld_curv": "yield_curve_segment.csv",
    # Arista 3 (Incentivos)
    "inc_det":  "incentives_detail.csv",
    "inc_sum":  "incentives_diag_summary.csv",
    "inc_sens": "incentives_sensitivity.csv",
    # Arista 4 (Capital / Provisiones)
    "cap_port": "capital_portfolio.csv",
    "cap_seg":  "capital_segment.csv",
    "cap_det":  "capital_detail.csv",
    # QA/Regulatorio
    "el_brk":   "el_breakdown.csv",
    "rwa_det":  "rwa_detail.csv",
    "rwa_seg":  "rwa_segment.csv",
    "qa_stage": "qa_ifrs_stage.csv",
}

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    "/content/mvp-tarjetas-chile/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
    "/content/out/dashboard_bundle",
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
# Formato de números (CLP/USD)
# ==========================
def _to_display_currency(val: float, target: str, usdclp: float) -> float:
    if pd.isna(val): return np.nan
    if target.upper() == "USD":
        return float(val) / float(usdclp) if usdclp else np.nan
    return float(val)

def fmt_money_val(val: float, target: str, usdclp: float) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)): return "—"
    x = _to_display_currency(val, target, usdclp)
    if x is None or (isinstance(x, float) and math.isnan(x)): return "—"
    neg = x < 0; x = abs(x)
    ent = int(x); dec = int(round((x - ent) * 100))
    if dec == 100: ent += 1; dec = 0
    ent_str = f"{ent:,}".replace(",", ".")
    return f"-{ent_str},{dec:02d}" if neg else f"{ent_str},{dec:02d}"

def fmt_pct_val(val: float) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)): return "—"
    return f"{val:.2f}%".replace(".", ",")

def var_pct(actual: float, opt: float) -> float | None:
    if actual is None or pd.isna(actual) or actual == 0: return None
    return (opt - actual) / actual * 100.0

def kpi_row(label: str, actual: float, opt: float, moneda: str, usdclp: float, help_text: str = ""):
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        st.metric(label=f"{label} – Actual", value=fmt_money_val(actual, moneda, usdclp))
        if help_text: st.caption(help_text)
    with c2:
        st.metric(label=f"{label} – Optimizado", value=fmt_money_val(opt, moneda, usdclp))
    with c3:
        vp = var_pct(actual, opt)
        st.metric(label="VAR %", value=fmt_pct_val(vp) if vp is not None else "—")

def kpi_row_pct(label: str, actual_pct: float, opt_pct: float, help_text: str = ""):
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
            df2[c] = df2[c].apply(lambda v: fmt_pct_val(v))
    return df2

def format_df_mixed(df: pd.DataFrame, money_cols: list[str], pct_cols: list[str], moneda: str, usdclp: float):
    out = df.copy()
    for c in money_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: fmt_money_val(v, moneda, usdclp))
    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: fmt_pct_val(v))
    return out

# ==========================
# App
# ==========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

# Sidebar: bundle + moneda + guardrails
st.sidebar.title("⚙️ Configuración")
default_dir = autodetect_bundle()
bundle_dir = st.sidebar.text_input(
    "📦 Ruta del bundle",
    value=(default_dir or ""),
    help="Ej: /content/mvp-tarjetas-chile/out/dashboard_bundle"
).strip() or default_dir

moneda = st.sidebar.radio("Moneda a visualizar", ["CLP", "USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))
st.sidebar.caption("Aplica a todos los montos del dashboard.")

st.sidebar.markdown("---")
st.sidebar.markdown("*Resguardos (umbrales de referencia)*")
apr_floor = float(st.sidebar.number_input("APR mínima (floor)", min_value=0.0, max_value=1.0, value=0.05, step=0.01))
apr_ceil  = float(st.sidebar.number_input("APR máxima (ceil)",  min_value=0.0, max_value=1.0, value=0.60, step=0.01))
max_ead_shift_pct = float(st.sidebar.number_input("Máx. cambio EAD por segmento (%)", min_value=0.0, value=25.0, step=1.0))
max_top1_conc_pct = float(st.sidebar.number_input("Máx. concentración Top-1 segmento (%)", min_value=0.0, value=40.0, step=1.0))

if not bundle_dir:
    st.error("No encuentro el bundle. Genera el paquete y vuelve a cargar.")
    st.stop()

dfs, missing = load_bundle(bundle_dir)
if missing:
    st.warning("Faltan archivos en el bundle (o nombres distintos):\n- " + "\n- ".join(missing))

st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Comparación *Actual vs. Optimizado* del portafolio con KPIs por arista. Valores en CLP/USD según selección.")

# ==========================
# Pestañas (incluye Análisis Ejecutivo y Glosario)
# ==========================
tabs = st.tabs([
    "Análisis ejecutivo",
    "Arista 1 – Default/Impago",
    "Arista 2 – Yield/Pricing",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital/Provisiones",
    "Guardrails (resguardos)",
    "Glosario"
])

# ==========================
# TAB 0 — Análisis ejecutivo (pitch)
# ==========================
with tabs[0]:
    st.subheader("Análisis ejecutivo")
    st.markdown("""
*Resumen (qué resolvemos):*
- *Default/Impago*: reducir EL reubicando EAD hacia perfiles más sanos.
- *Yield/Pricing*: hallar la tasa (APR) que maximiza utilidad combinando precio y volumen.
- *Incentivos*: asignar beneficios solo donde el ROI es positivo y significativo.
- *Capital/Provisiones*: disminuir consumo de capital regulatorio y estabilizar provisiones.

*Cómo leer los resultados* (Actual vs. Optimizado): cada arista muestra KPIs clave y su variación porcentual (VAR%), en la moneda elegida (CLP/USD).
""")

    # KPIs ejecutivos: tomar 1–3 métricas por arista si están disponibles
    def g0(df, name): 
        return (df[name].iloc[0] if isinstance(df, pd.DataFrame) and (name in df.columns) and not df.empty else np.nan)

    # Arista 1
    defp = dfs.get("def_port")
    EAD_act = g0(defp,"EAD_actual");  EAD_opt = g0(defp,"EAD_optimizado")
    EL_act  = g0(defp,"EL_actual");   EL_opt  = g0(defp,"EL_optimizado")
    Uti_act = g0(defp,"Utilidad_actual"); Uti_opt = g0(defp,"Utilidad_optimizada")

    # Arista 2
    yldp = dfs.get("yld_port")
    Y_ing_base = g0(yldp,"ingreso_base"); Y_ing_opt = g0(yldp,"ingreso_opt")
    Y_uti_base = g0(yldp,"utilidad_base"); Y_uti_opt = g0(yldp,"utilidad_opt")

    # Arista 3
    inc_det = dfs.get("inc_det")
    inc_cost_tot, inc_uplift_tot, inc_roi = np.nan, np.nan, np.nan
    if isinstance(inc_det, pd.DataFrame) and not inc_det.empty:
        cost_col = next((c for c in inc_det.columns if c.lower() in
                        {"inc_cost","costo_incentivo","costo_incentivo_monto","costo_incentivo_total"}), None)
        uplift_col = next((c for c in inc_det.columns if c.lower() in
                        {"ingreso_uplift","uplift_ingreso","delta_ingreso","ingreso_incremental"}), None)
        if cost_col is None: inc_det["_inc_cost"]=0.0; cost_col="inc_cost_"
        if uplift_col is None: inc_det["_uplift"]=0.0; uplift_col="uplift_"
        inc_cost_tot  = pd.to_numeric(inc_det[cost_col], errors="coerce").fillna(0).sum()
        inc_uplift_tot= pd.to_numeric(inc_det[uplift_col], errors="coerce").fillna(0).sum()
        inc_roi       = (inc_uplift_tot/inc_cost_tot) if inc_cost_tot>0 else np.nan

    # Arista 4
    capp = dfs.get("cap_port")
    cap_base = g0(capp,"capital_req_base"); cap_opt = g0(capp,"capital_req_opt")
    prov_base= g0(capp,"prov_base");        prov_opt= g0(capp,"prov_opt")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("*Arista 1 — Default/Impago*")
        kpi_row("EL (Pérdida Esperada)", EL_act, EL_opt, moneda, usdclp)
        kpi_row("Utilidad", Uti_act, Uti_opt, moneda, usdclp)
    with col2:
        st.markdown("*Arista 2 — Yield/Pricing*")
        kpi_row("Ingreso (Total)", Y_ing_base, Y_ing_opt, moneda, usdclp)
        kpi_row("Utilidad (Total)", Y_uti_base, Y_uti_opt, moneda, usdclp)

    col3,col4 = st.columns(2)
    with col3:
        st.markdown("*Arista 3 — Incentivos*")
        kpi_row("Costo de Incentivos", inc_cost_tot, inc_cost_tot, moneda, usdclp)
        kpi_row("Ingreso Incremental", inc_uplift_tot, inc_uplift_tot, moneda, usdclp)
        st.metric("ROI (Ingreso/Costo)", fmt_pct_val((inc_roi*100) if pd.notna(inc_roi) else np.nan))
    with col4:
        st.markdown("*Arista 4 — Capital/Provisiones*")
        kpi_row("Capital Requerido", cap_base, cap_opt, moneda, usdclp, "Proxy RW×K×EAD")
        kpi_row("Provisiones",       prov_base, prov_opt, moneda, usdclp, "≈ EL")

    st.markdown("---")
    st.caption("Historia: reubicamos exposición, ajustamos pricing, focalizamos incentivos y aliviamos capital/provisiones dentro de resguardos.")

# ================
# TAB 1 — Arista 1 (igual que v2, con intro por arista)
# ================
with tabs[1]:
    st.subheader("Arista 1 — Default/Impago")
    st.markdown("""
*¿Qué resolvemos aquí?* Reducimos la *pérdida esperada (EL)* reubicando exposición (EAD) hacia perfiles más sanos, sin sacrificar ingresos.  
*Historia del resultado:* Al mover exposición desde clientes con alta PD×LGD hacia perfiles medianos/sanos, el *EL baja* y la *utilidad sube*.
""")

    port = dfs.get("def_port")
    if port is None or (isinstance(port, pd.DataFrame) and port.empty):
        st.error("No se encontró *default_portfolio.csv*.")
    else:
        def g0(df, col): return df[col].iloc[0] if (isinstance(df, pd.DataFrame) and col in df.columns and not df.empty) else np.nan

        EAD_act = g0(port, "EAD_actual");  EAD_opt = g0(port, "EAD_optimizado")
        EL_act  = g0(port, "EL_actual");   EL_opt  = g0(port, "EL_optimizado")
        Ing_act = g0(port, "Ingreso_actual"); Ing_opt = g0(port, "Ingreso_optimizado")
        Cost_act= g0(port, "Costos_actual");   Cost_opt= g0(port, "Costos_optimizado")
        Uti_act = g0(port, "Utilidad_actual"); Uti_opt = g0(port, "Utilidad_optimizada")
        PDw_act = g0(port, "PD_pond_actual");  PDw_opt = g0(port, "PD_pond_optimizado")

        kpi_row("EAD", EAD_act, EAD_opt, moneda, usdclp, "Exposure at Default (exposición en riesgo)")
        kpi_row("EL (Pérdida Esperada)", EL_act, EL_opt, moneda, usdclp, "PD × LGD × EAD")
        kpi_row("Ingreso", Ing_act, Ing_opt, moneda, usdclp, "APR × EAD")
        kpi_row("Costos Totales", Cost_act, Cost_opt, moneda, usdclp, "Financieros + Operativos")
        kpi_row("Utilidad", Uti_act, Uti_opt, moneda, usdclp, "Ingreso − EL − Costos")

        if pd.notna(PDw_act) or pd.notna(PDw_opt):
            kpi_row_pct("PD Ponderado (EAD)", PDw_act*100 if pd.notna(PDw_act) else np.nan,
                        PDw_opt*100 if pd.notna(PDw_opt) else np.nan,
                        "PD promedio ponderado por EAD")

    seg = dfs.get("def_seg")
    if isinstance(seg, pd.DataFrame) and not seg.empty:
        st.markdown("*Detalle por segmento*")
        seg_fmt = format_df_currency(seg, [
            "EAD_actual","EAD_optimizado","EL_actual","EL_optimizado",
            "Ingreso_actual","Ingreso_optimizado","Costos_actual","Costos_optimizado",
            "Utilidad_actual","Utilidad_optimizada"
        ], moneda, usdclp)
        st.dataframe(seg_fmt, use_container_width=True)

# ================
# TAB 2 — Arista 2
# ================
with tabs[2]:
    st.subheader("Arista 2 — Yield/Pricing")
    st.markdown("""
*¿Qué resolvemos aquí?* Buscamos la *tasa óptima (APR)* por segmento balanceando precio y volumen (elasticidad del EAD).  
*Historia del resultado:* Ajustando tasa por segmento, sube el *ingreso* y la *utilidad* sin disparar el riesgo.
""")

    port = dfs.get("yld_port")
    if port is None or (isinstance(port, pd.DataFrame) and port.empty):
        st.error("No se encontraron archivos de Yield.")
    else:
        def g0(df, name): return df[name].iloc[0] if name in df.columns and not df.empty else np.nan
        Ing_base = g0(port,"ingreso_base"); Ing_iso = g0(port,"ingreso_iso"); Ing_opt = g0(port,"ingreso_opt")
        Uti_base = g0(port,"utilidad_base"); Uti_iso = g0(port,"utilidad_iso"); Uti_opt = g0(port,"utilidad_opt")
        EL_base  = g0(port,"EL_baseline");   EL_iso  = g0(port,"el_iso");        EL_opt  = g0(port,"el_opt")

        kpi_row("Ingreso (Total)", Ing_base, Ing_opt, moneda, usdclp, "Precio + Volumen")
        kpi_row("Ingreso (Solo Pricing)", Ing_base, Ing_iso, moneda, usdclp, "EAD fijado en baseline")
        kpi_row("Utilidad (Total)", Uti_base, Uti_opt, moneda, usdclp, "Precio + Volumen")
        kpi_row("Utilidad (Solo Pricing)", Uti_base, Uti_iso, moneda, usdclp, "EAD fijado en baseline")
        kpi_row("EL", EL_base, EL_opt, moneda, usdclp, "Pérdida esperada total")

    # Tablas formateadas
    seg = dfs.get("yld_seg")
    if isinstance(seg, pd.DataFrame) and not seg.empty:
        st.markdown("*Segmentos (pricing y efecto total)*")
        money_cols = ["ingreso_base","ingreso_iso","ingreso_opt","utilidad_base","utilidad_iso","utilidad_opt"]
        seg_fmt = format_df_currency(seg, money_cols, moneda, usdclp)
        st.dataframe(seg_fmt, use_container_width=True)

    det = dfs.get("yld_det")
    if isinstance(det, pd.DataFrame) and not det.empty:
        with st.expander("Detalle por cliente (muestra)"):
            money_cols = ["ingreso_base","ingreso_iso","ingreso_opt","utilidad_base","utilidad_iso","utilidad_opt"]
            det_fmt = format_df_currency(det.head(200).copy(), money_cols, moneda, usdclp)
            st.dataframe(det_fmt, use_container_width=True)

# ================
# TAB 3 — Arista 3
# ================
with tabs[3]:
    st.subheader("Arista 3 — Incentivos")
    st.markdown("""
*¿Qué resolvemos aquí?* Asignamos beneficios solo donde el *ROI* (ingreso incremental/costo) es significativo.  
*Historia del resultado:* Al focalizar, maximizamos *ingreso incremental* sin inflar el *costo de incentivos*.
""")

    det  = dfs.get("inc_det")
    summ = dfs.get("inc_sum")

    total_cost = np.nan; uplift = np.nan; roi = np.nan
    if isinstance(det, pd.DataFrame) and not det.empty:
        cost_col = next((c for c in det.columns if c.lower() in
                        {"inc_cost","costo_incentivo","costo_incentivo_monto","costo_incentivo_total"}), None)
        uplift_col = next((c for c in det.columns if c.lower() in
                        {"ingreso_uplift","uplift_ingreso","delta_ingreso","ingreso_incremental"}), None)
        if cost_col is None: det["_inc_cost"] = 0.0; cost_col="inc_cost_"
        if uplift_col is None: det["_uplift"] = 0.0; uplift_col="uplift_"
        total_cost = pd.to_numeric(det[cost_col], errors="coerce").fillna(0).sum()
        uplift     = pd.to_numeric(det[uplift_col], errors="coerce").fillna(0).sum()
        roi        = (uplift/total_cost) if total_cost>0 else np.nan

    kpi_row("Costo de Incentivos", total_cost, total_cost, moneda, usdclp, "Suma de costos de beneficios")
    kpi_row("Ingreso Incremental", uplift, uplift, moneda, usdclp, "Suma de incrementos estimados")
    st.metric("ROI (Ingreso/Costo)", fmt_pct_val((roi*100) if pd.notna(roi) else np.nan))

    if isinstance(summ, pd.DataFrame) and not summ.empty:
        st.markdown("*Resumen por segmento*")
        money_cols = ["inc_cost_total","ingreso_uplift_total"]
        pct_cols = ["roi"]
        df = summ.copy()
        if "roi" in df.columns and df["roi"].dropna().abs().max() <= 2.0:
            df["roi"] = df["roi"]*100.0
        df_fmt = format_df_mixed(df, money_cols, pct_cols, moneda, usdclp)
        st.dataframe(df_fmt, use_container_width=True)

# ================
# TAB 4 — Arista 4
# ================
with tabs[4]:
    st.subheader("Arista 4 — Capital/Provisiones")
    st.markdown("""
*¿Qué resolvemos aquí?* Reducimos consumo de *capital regulatorio* (RWA×K) y *provisiones* (~EL) manteniendo ingresos saludables.  
*Historia del resultado:* Reasignando EAD hacia perfiles con menor riesgo, la *provisión* y el *capital requerido* se estabilizan y pueden bajar.
""")

    cap_port = dfs.get("cap_port")
    cap_seg  = dfs.get("cap_seg")
    cap_det  = dfs.get("cap_det")

    if cap_port is None or (isinstance(cap_port, pd.DataFrame) and cap_port.empty):
        st.error("No se encontró *capital_portfolio.csv*.")
    else:
        def g0(df, name): return df[name].iloc[0] if name in df.columns and not df.empty else np.nan
        cap_base = g0(cap_port, "capital_req_base")
        cap_opt  = g0(cap_port, "capital_req_opt")
        prov_base= g0(cap_port, "prov_base")
        prov_opt = g0(cap_port, "prov_opt")

        kpi_row("Capital Requerido", cap_base, cap_opt, moneda, usdclp, "Proxy RW×K×EAD")
        kpi_row("Provisiones", prov_base, prov_opt, moneda, usdclp, "≈ EL (pérdida esperada)")

        lib_cap  = cap_base - cap_opt  if pd.notna(cap_base) and pd.notna(cap_opt)  else np.nan
        lib_prov = prov_base - prov_opt if pd.notna(prov_base) and pd.notna(prov_opt) else np.nan

        col1, col2 = st.columns(2)
        with col1: st.metric("Liberación de Capital", fmt_money_val(lib_cap, moneda, usdclp))
        with col2: st.metric("Liberación de Provisiones", fmt_money_val(lib_prov, moneda, usdclp))

    colA, colB = st.columns(2)
    with colA:
        st.markdown("*Capital por segmento*")
        if isinstance(cap_seg, pd.DataFrame) and not cap_seg.empty:
            seg_fmt = format_df_currency(
                cap_seg.copy(),
                ["capital_req_base","capital_req_opt","prov_base","prov_opt"],
                moneda, usdclp
            )
            st.dataframe(seg_fmt, use_container_width=True)
        else:
            st.info("No hay capital_segment.csv en el bundle.")
    with colB:
        st.markdown("*Detalle de capital (muestra)*")
        if isinstance(cap_det, pd.DataFrame) and not cap_det.empty:
            det_fmt = format_df_currency(
                cap_det.copy(),
                ["capital_req_base","capital_req_opt","prov_base","prov_opt","ead_baseline","e_opt"],
                moneda, usdclp
            )
            st.dataframe(det_fmt.head(200), use_container_width=True)
        else:
            st.info("No hay capital_detail.csv en el bundle.")

# ================
# TAB 5 — Guardrails (resguardos) con derivación si faltan archivos
# ================
with tabs[5]:
    st.subheader("Guardrails (resguardos)")
    st.markdown("""
*¿Qué son?* Reglas de resguardo para mantener límites de riesgo/negocio: bandas de *APR, límites de **EAD* por segmento, *concentración* máxima, etc.  
Los umbrales (referencia) se configuran en la barra lateral.
""")

    # Si existen tablas explícitas, mostrarlas; si no, derivar de lo disponible
    gr_paths = [
        Path(bundle_dir)/"guardrails_portfolio.csv",
        Path(bundle_dir)/"guardrails_segment.csv"
    ]
    showed = False
    for p in gr_paths:
        if p.exists():
            try:
                g = pd.read_csv(p)
                money_cols = [c for c in g.columns if any(k in c.lower() for k in ["monto","clp","capital","ead","utilidad","ingreso","provision"])]
                pct_cols   = [c for c in g.columns if any(k in c.lower() for k in ["%", "pct","ratio","apr","pd","lgd","k","rw"])]
                g_fmt = format_df_mixed(g, money_cols, pct_cols, moneda, usdclp)
                st.markdown(f"*{p.name}*")
                st.dataframe(g_fmt, use_container_width=True)
                showed = True
            except Exception as e:
                st.info(f"No pude leer {p.name}: {e}")

    if not showed:
        # Derivar guardrails de lo que haya en el bundle
        # 1) Banda APR observada
        apr_min_obs, apr_max_obs = np.nan, np.nan
        ydet = dfs.get("yld_det")
        if isinstance(ydet, pd.DataFrame) and not ydet.empty:
            apr_cols = [c for c in ydet.columns if any(k in c.lower() for k in ["apr","r_opt","tasa","rate"])]
            if apr_cols:
                vals = pd.to_numeric(ydet[apr_cols].stack(), errors="coerce").dropna()
                if not vals.empty:
                    apr_min_obs = float(vals.min()); apr_max_obs = float(vals.max())

        # 2) Cambio de EAD por segmento (|Δ| / EAD_base)
        ead_shift_pct = np.nan
        dseg = dfs.get("def_seg")
        if isinstance(dseg, pd.DataFrame) and not dseg.empty and \
           all(c in dseg.columns for c in ["EAD_actual","EAD_optimizado"]):
            base_ead = pd.to_numeric(dseg["EAD_actual"], errors="coerce").fillna(0.0)
            opt_ead  = pd.to_numeric(dseg["EAD_optimizado"], errors="coerce").fillna(0.0)
            denom = base_ead.replace(0, np.nan)
            rel = ((opt_ead - base_ead).abs() / denom * 100).replace([np.inf,-np.inf], np.nan).dropna()
            if not rel.empty:
                ead_shift_pct = float(rel.max())

        # 3) Concentración Top-1 (share de EAD por segmento)
        conc_top1 = np.nan
        if isinstance(dseg, pd.DataFrame) and not dseg.empty and "EAD_optimizado" in dseg.columns:
            tot = pd.to_numeric(dseg["EAD_optimizado"], errors="coerce").fillna(0.0).sum()
            if tot > 0:
                shares = pd.to_numeric(dseg["EAD_optimizado"], errors="coerce").fillna(0.0) / tot * 100.0
                conc_top1 = float(shares.max())

        # 4) PD ponderado no se dispare (proxy)
        pdw_base = pdw_opt = np.nan
        dport = dfs.get("def_port")
        if isinstance(dport, pd.DataFrame) and not dport.empty:
            if "PD_pond_actual" in dport.columns: pdw_base = float(pd.to_numeric(dport["PD_pond_actual"], errors="coerce").iloc[0])
            if "PD_pond_optimizado" in dport.columns: pdw_opt = float(pd.to_numeric(dport["PD_pond_optimizado"], errors="coerce").iloc[0])

        data = []
        data.append({
            "guardrail": "Banda APR",
            "definición": "La tasa efectiva debe permanecer dentro del rango permitido",
            "umbral": f"[{apr_floor:.2f}, {apr_ceil:.2f}]".replace(".", ","),
            "observado_actual": "—",
            "observado_optimizado": f"[{(apr_min_obs if not pd.isna(apr_min_obs) else '—')}, {(apr_max_obs if not pd.isna(apr_max_obs) else '—')}]",
            "cumple": ("Sí" if (not pd.isna(apr_min_obs) and not pd.isna(apr_max_obs) and apr_min_obs>=apr_floor and apr_max_obs<=apr_ceil) else "N/A")
        })
        data.append({
            "guardrail": "Máx. cambio EAD por segmento",
            "definición": "|ΔEAD|/EAD_base por segmento, medido en %",
            "umbral": f"≤ {max_ead_shift_pct:.2f}%".replace(".", ","),
            "observado_actual": "—",
            "observado_optimizado": (f"{ead_shift_pct:.2f}%" if not pd.isna(ead_shift_pct) else "—").replace(".", ","),
            "cumple": ("Sí" if not pd.isna(ead_shift_pct) and ead_shift_pct <= max_ead_shift_pct else ("No" if not pd.isna(ead_shift_pct) else "N/A"))
        })
        data.append({
            "guardrail": "Concentración Top-1 segmento",
            "definición": "Participación del segmento más grande sobre EAD optimizado",
            "umbral": f"≤ {max_top1_conc_pct:.2f}%".replace(".", ","),
            "observado_actual": "—",
            "observado_optimizado": (f"{conc_top1:.2f}%" if not pd.isna(conc_top1) else "—").replace(".", ","),
            "cumple": ("Sí" if not pd.isna(conc_top1) and conc_top1 <= max_top1_conc_pct else ("No" if not pd.isna(conc_top1) else "N/A"))
        })
        data.append({
            "guardrail": "PD ponderado (no sube)",
            "definición": "PD promedio ponderado por EAD no debe aumentar tras la optimización",
            "umbral": "PD_opt ≤ PD_base",
            "observado_actual": (fmt_pct_val(pdw_base*100) if not pd.isna(pdw_base) else "—"),
            "observado_optimizado": (fmt_pct_val(pdw_opt*100) if not pd.isna(pdw_opt) else "—"),
            "cumple": ("Sí" if (not pd.isna(pdw_base) and not pd.isna(pdw_opt) and pdw_opt <= pdw_base) else "N/A")
        })

        df_gr = pd.DataFrame(data)
        # columnas % ya están como string; no aplicar formato de moneda aquí
        st.markdown("*Resguardos derivados (bundle actual)*")
        st.dataframe(df_gr, use_container_width=True)

# ================
# TAB 6 — Glosario (Definiciones de KPIs)
# ================
with tabs[6]:
    st.subheader("Glosario — Definiciones de KPIs")
    st.markdown("""
- *EAD (Exposure at Default):* Exposición del crédito cuando ocurre el default (monto en riesgo).
- *PD (Probability of Default):* Probabilidad de incumplimiento de pago en el horizonte considerado.
- *LGD (Loss Given Default):* Severidad de la pérdida dado el incumplimiento.
- *EL (Expected Loss):* Pérdida esperada = *PD × LGD × EAD*.
- *APR (Annual Percentage Rate):* Tasa efectiva anual.
- *Ingreso:* Aproximación a *APR × EAD* (neto de costo financiero en comparativas específicas).
- *Costos:* Suma de costos financieros (funding) y operativos.
- *Utilidad:* *Ingreso − EL − Costos*.
- *RWA (Risk-Weighted Assets):* Activos ponderados por riesgo (proxy estandarizado).
- *Capital requerido:* *RWA × K* (K: ratio regulatorio).
- *Provisiones:* Proxy contable proporcional a EL.
- *ROI (Incentivos):* *Ingreso incremental / Costo de incentivos*.
""")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas integradas).")
