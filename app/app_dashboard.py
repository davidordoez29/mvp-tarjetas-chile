# app/app_dashboard.py — versión robusta compatible con Bundle v2 (Celda 14 NUEVA)

import os, glob, json, math
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# ==========================
# Archivos requeridos (nombres exactos del Bundle v2)
# ==========================
REQ_FILES = {
    # Arista 1 (Default)
    "def_port": "default_portfolio.csv",
    "def_seg": "default_segment.csv",
    "def_det": "default_detail.csv",
    # Arista 2 (Yield)
    "yld_port": "yield_portfolio.csv",
    "yld_seg": "yield_segment.csv",
    "yld_det": "yield_detail.csv",
    "yld_curv": "yield_curve_segment.csv",
    # Arista 3 (Incentivos)
    "inc_det": "incentives_detail.csv",
    "inc_sum": "incentives_diag_summary.csv",
    "inc_sens": "incentives_sensitivity.csv",
    # Arista 4 (Capital / Provisiones)
    "cap_port": "capital_portfolio.csv",
    "cap_seg": "capital_segment.csv",
    "cap_det": "capital_detail.csv",
    # Meta
    "kpi_defs": "kpi_defs.json",
    "seg_defs": "segment_defs.json",
    "meta": "bundle_meta.json",
}

CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR", "").strip(),
    "/content/mvp-tarjetas-chile/out/dashboard_bundle",
    "/content/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
]

# ==========================
# Utilidades generales
# ==========================
def _dir_ok(d: str) -> bool:
    try:
        if not d or not os.path.isdir(d):
            return False
        hits = sum(os.path.exists(os.path.join(d, v)) for v in REQ_FILES.values())
        # con 6+ archivos ya lo consideramos un bundle válido
        return hits >= 6
    except Exception:
        return False

def autodetect_bundle() -> str | None:
    # 1) candidates directos
    for d in CANDIDATE_DIRS:
        if _dir_ok(d):
            return d
    # 2) búsqueda recursiva en ubicaciones comunes
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

@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def load_bundle(bundle_dir: str):
    dfs, missing = {}, []
    for key, fname in REQ_FILES.items():
        path = os.path.join(bundle_dir, fname)
        if not os.path.exists(path):
            missing.append(fname)
            dfs[key] = None
            continue
        try:
            if fname.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    dfs[key] = json.load(f)
            else:
                dfs[key] = _read_csv(path)
                if dfs[key] is None:
                    missing.append(f"{fname} (no se pudo leer)")
        except Exception as e:
            missing.append(f"{fname} (error: {e})")
            dfs[key] = None
    return dfs, missing

# ==========================
# Formato de números (CLP/USD)
# ==========================
def _to_display_currency(val: float, target: str, usdclp: float) -> float:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return np.nan
    if target.upper() == "USD":
        return float(val) / float(usdclp) if usdclp else np.nan
    return float(val)

def fmt_money_val(val: float, target: str, usdclp: float) -> str:
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
    ent_str = f"{ent:,}".replace(",", ".")
    return f"-{ent_str},{dec:02d}" if neg else f"{ent_str},{dec:02d}"

def fmt_pct_val(val: float) -> str:
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
        st.metric(label=f"{label} – Actual", value=fmt_money_val(actual, moneda, usdclp))
        if help_text:
            st.caption(help_text)
    with c2:
        st.metric(label=f"{label} – Optimizado", value=fmt_money_val(opt, moneda, usdclp))
    with c3:
        vp = var_pct(actual, opt)
        st.metric(label="VAR %", value=fmt_pct_val(vp) if vp is not None else "—")

def kpi_row_pct(label: str, actual_pct: float, opt_pct: float, help_text: str = ""):
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        st.metric(label=f"{label} – Actual", value=fmt_pct_val(actual_pct))
        if help_text:
            st.caption(help_text)
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

def g0(df: pd.DataFrame | None, col: str):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return np.nan
    return df[col].iloc[0]

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
    st.error("No se encontró el bundle. Genera el paquete (Celda 14 NUEVA) y vuelve a cargar.")
    st.stop()

dfs, missing = load_bundle(bundle_dir)
if missing:
    st.warning("Faltan archivos en el bundle (o nombres distintos):\n- " + "\n- ".join(missing))

moneda = st.sidebar.radio("Moneda a visualizar", ["CLP", "USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))
st.sidebar.caption("Aplica a todos los montos del dashboard.")

st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Comparación **Actual vs. Optimizado** con KPIs clave por arista. Datos del bundle: " + (bundle_dir or "—"))

tabs = st.tabs([
    "Arista 1 – Default/Impago",
    "Arista 2 – Yield/Pricing",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital/Provisiones"
])

with st.expander("¿Qué resuelve cada arista?"):
    st.markdown("""
**Arista 1 (Default/Impago):** Reduce la **Pérdida Esperada (EL)** administrando tasa y exposición (EAD) según riesgo, con guardrails.<br>
**Arista 2 (Yield/Pricing):** Optimiza el **APR** para maximizar utilidad equilibrando precio y volumen (elasticidades).<br>
**Arista 3 (Incentivos):** Asigna beneficios donde el **ROI** (ingreso incremental / costo) es mayor, con presupuesto y saturación.<br>
**Arista 4 (Capital/Provisiones):** Minimiza consumo de **capital** y estabiliza **provisiones** (proxies) manteniendo el riesgo acotado.
""", unsafe_allow_html=True)

# ================
# Arista 1 – Default
# ================
with tabs[0]:
    st.subheader("Arista 1 – Default/Impago")
    st.markdown("""
**KPIs:**
- **EAD** (Exposure at Default)
- **EL** (Expected Loss) = PD × LGD × EAD
- **Ingreso** = (APR − costo_fondeo − costo_ops) × EAD
- **Utilidad** = Ingreso − EL
- **PD ponderado** por EAD
""", unsafe_allow_html=True)

    def_port = dfs.get("def_port")
    def_seg = dfs.get("def_seg")
    def_det = dfs.get("def_det")

    if def_port is None or (isinstance(def_port, pd.DataFrame) and def_port.empty):
        st.error("No se encontró **default_portfolio.csv**.")
    else:
        EAD_act = g0(def_port, "EAD_actual"); EAD_opt = g0(def_port, "EAD_optimizado")
        EL_act = g0(def_port, "EL_actual"); EL_opt = g0(def_port, "EL_optimizado")
        Ing_act = g0(def_port, "Ingreso_actual"); Ing_opt = g0(def_port, "Ingreso_optimizado")
        Cost_act= g0(def_port, "Costos_actual"); Cost_opt= g0(def_port, "Costos_optimizado")
        Uti_act = g0(def_port, "Utilidad_actual");Uti_opt = g0(def_port, "Utilidad_optimizada")
        PDw_act = g0(def_port, "PD_pond_actual"); PDw_opt = g0(def_port, "PD_pond_optimizado")

        kpi_row("EAD", EAD_act, EAD_opt, moneda, usdclp, "Exposición total")
        kpi_row("EL (Pérdida Esperada)", EL_act, EL_opt, moneda, usdclp, "PD × LGD × EAD")
        kpi_row("Ingreso", Ing_act, Ing_opt, moneda, usdclp, "APR neta × EAD")
        kpi_row("Costos Totales", Cost_act, Cost_opt, moneda, usdclp, "Fondeo + Operación")
        kpi_row("Utilidad", Uti_act, Uti_opt, moneda, usdclp, "Ingreso − EL")
        if pd.notna(PDw_act) or pd.notna(PDw_opt):
            kpi_row_pct("PD Ponderado (EAD)", PDw_act*100 if pd.notna(PDw_act) else np.nan,
                        PDw_opt*100 if pd.notna(PDw_opt) else np.nan,
                        "Promedio ponderado por EAD")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Default por segmento**")
        if isinstance(def_seg, pd.DataFrame) and not def_seg.empty:
            df = def_seg.copy()
            money_cols = [
                "EAD_actual","EAD_optimizado","EL_actual","EL_optimizado",
                "Ingreso_actual","Ingreso_optimizado","Costos_actual","Costos_optimizado",
                "Utilidad_actual","Utilidad_optimizada"
            ]
            df = format_df_currency(df, money_cols, moneda, usdclp)
            if "PD_pond_actual" in def_seg.columns:
                df["PD_pond_actual"] = def_seg["PD_pond_actual"].apply(lambda v: fmt_pct_val(v*100))
            if "PD_pond_optimizado" in def_seg.columns:
                df["PD_pond_optimizado"] = def_seg["PD_pond_optimizado"].apply(lambda v: fmt_pct_val(v*100))
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No hay default_segment.csv en el bundle.")
    with colB:
        st.markdown("**Detalle (muestra)**")
        if isinstance(def_det, pd.DataFrame) and not def_det.empty:
            df = def_det.head(300).copy()
            money_cols = [
                "ead_baseline","ead_pricing","EL_actual","EL_optimizado",
                "ingreso_actual","ingreso_optimizado","costos_actual","costos_optimizado",
                "Utilidad_actual","Utilidad_optimizada"
            ]
            df = format_df_currency(df, money_cols, moneda, usdclp)
            if "pd_base" in def_det.columns:
                df["pd_base"] = def_det["pd_base"].apply(lambda v: fmt_pct_val(v*100))
            if "pd_opt" in def_det.columns:
                df["pd_opt"] = def_det["pd_opt"].apply(lambda v: fmt_pct_val(v*100))
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No hay default_detail.csv en el bundle.")

# ================
# Arista 2 – Yield / Pricing
# ================
with tabs[1]:
    st.subheader("Arista 2 – Yield/Pricing")
    st.markdown("""
**KPIs:**
- **Ingreso/Utilidad (Total)**: pricing + volumen
- **Ingreso/Utilidad (Solo Pricing)**: EAD fijado en baseline
""", unsafe_allow_html=True)

    yld_port = dfs.get("yld_port")
    yld_seg = dfs.get("yld_seg")
    yld_det = dfs.get("yld_det")
    yld_curv = dfs.get("yld_curv")

    if yld_port is None or (isinstance(yld_port, pd.DataFrame) and yld_port.empty):
        st.error("No se encontró **yield_portfolio.csv**.")
    else:
        Ing_base = g0(yld_port,"ingreso_base"); Ing_iso = g0(yld_port,"ingreso_iso"); Ing_opt = g0(yld_port,"ingreso_opt")
        Uti_base = g0(yld_port,"utilidad_base"); Uti_iso = g0(yld_port,"utilidad_iso"); Uti_opt = g0(yld_port,"utilidad_opt")
        EL_base = g0(yld_port,"EL_baseline"); EL_iso = g0(yld_port,"el_iso"); EL_opt = g0(yld_port,"el_opt")

        kpi_row("Ingreso (Total)", Ing_base, Ing_opt, moneda, usdclp, "Precio + Volumen")
        kpi_row("Ingreso (Solo Pricing)", Ing_base, Ing_iso, moneda, usdclp, "EAD fijado en baseline")
        kpi_row("Utilidad (Total)", Uti_base, Uti_opt, moneda, usdclp)
        kpi_row("Utilidad (Solo Pricing)", Uti_base, Uti_iso, moneda, usdclp)
        kpi_row("EL", EL_base, EL_opt, moneda, usdclp, "Pérdida esperada total")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Yield por segmento**")
        if isinstance(yld_seg, pd.DataFrame) and not yld_seg.empty:
            df = format_df_currency(
                yld_seg.copy(),
                ["ingreso_base","ingreso_iso","ingreso_opt","utilidad_base","utilidad_iso","utilidad_opt","EL_baseline","el_iso","el_opt"],
                moneda, usdclp
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No hay yield_segment.csv en el bundle.")
    with colB:
        st.markdown("**Detalle (muestra)**")
        if isinstance(yld_det, pd.DataFrame) and not yld_det.empty:
            df = format_df_currency(
                yld_det.head(300).copy(),
                ["ead_baseline","ead_pricing","ingreso_base","ingreso_iso","ingreso_opt","utilidad_base","utilidad_iso","utilidad_opt","EL_baseline","el_iso","el_opt"],
                moneda, usdclp
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No hay yield_detail.csv en el bundle.")

    st.markdown("**Curva r → ingreso por segmento**")
    if isinstance(yld_curv, pd.DataFrame) and not yld_curv.empty and {"segmento","r_grid","ingreso_grid"}.issubset(yld_curv.columns):
        # Mostrar tabla simple (sin gráficos para evitar dependencias)
        df = yld_curv.copy()
        df["ingreso_grid_fmt"] = df["ingreso_grid"].apply(lambda v: fmt_money_val(v, moneda, usdclp))
        st.dataframe(df[["segmento","r_grid","ingreso_grid_fmt"]], use_container_width=True)
    else:
        st.info("No hay yield_curve_segment.csv en el bundle.")

# ================
# Arista 3 – Incentivos
# ================
with tabs[2]:
    st.subheader("Arista 3 – Incentivos")
    st.markdown("""
**KPIs (portafolio):**
- **Costo de Incentivos** (CLP/USD)
- **Ingreso Incremental**
- **ROI** = Ingreso Incremental / Costo
""", unsafe_allow_html=True)

    inc_det = dfs.get("inc_det")
    inc_sum = dfs.get("inc_sum")
    inc_sens = dfs.get("inc_sens")

    total_cost = np.nan
    uplift = np.nan
    roi = np.nan

    # Si hay resumen, úsalo para KPIs (fila TOTAL); si no, derive de detalle
    if isinstance(inc_sum, pd.DataFrame) and not inc_sum.empty:
        summ = inc_sum.copy()
        # fila TOTAL si existe
        row_total = None
        if "segmento" in summ.columns:
            mask = summ["segmento"].astype(str).str.upper().eq("TOTAL")
            if mask.any():
                row_total = summ[mask].iloc[0].to_dict()
        if row_total is None:
            # sumar todo
            tot_cost = pd.to_numeric(summ.get("inc_cost_total", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
            tot_upl = pd.to_numeric(summ.get("ingreso_uplift_total", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
            total_cost, uplift = float(tot_cost), float(tot_upl)
            roi = (uplift/total_cost) if total_cost>0 else np.nan
        else:
            total_cost = float(pd.to_numeric(pd.Series([row_total.get("inc_cost_total", np.nan)]), errors="coerce").fillna(0).iloc[0])
            uplift = float(pd.to_numeric(pd.Series([row_total.get("ingreso_uplift_total", np.nan)]), errors="coerce").fillna(0).iloc[0])
            roi = float(row_total.get("roi", np.nan)) if not pd.isna(row_total.get("roi", np.nan)) else ((uplift/total_cost) if total_cost>0 else np.nan)
    elif isinstance(inc_det, pd.DataFrame) and not inc_det.empty:
        # derivar de detalle
        if "inc_cost" in inc_det.columns:
            total_cost = pd.to_numeric(inc_det["inc_cost"], errors="coerce").fillna(0).sum()
        else:
            total_cost = 0.0
        if "ingreso_uplift" in inc_det.columns:
            uplift = pd.to_numeric(inc_det["ingreso_uplift"], errors="coerce").fillna(0).sum()
        else:
            uplift = 0.0
        roi = (uplift/total_cost) if total_cost>0 else np.nan

    # KPIs portafolio
    kpi_row("Costo de Incentivos", total_cost, total_cost, moneda, usdclp, "Suma de costos de beneficios")
    kpi_row("Ingreso Incremental", uplift, uplift, moneda, usdclp, "Suma de incrementos estimados")
    st.metric("ROI (Ingreso/Costo)", fmt_pct_val(roi*100 if pd.notna(roi) else np.nan))

    # Tablas
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Resumen por segmento**")
        if isinstance(inc_sum, pd.DataFrame) and not inc_sum.empty:
            df = inc_sum.copy()
            # Formatos
            for c in ["inc_cost_total","ingreso_uplift_total"]:
                if c in df.columns:
                    df[c] = df[c].apply(lambda v: fmt_money_val(v, moneda, usdclp))
            if "roi" in df.columns:
                df["roi"] = df["roi"].apply(lambda x: fmt_pct_val(x*100 if pd.notna(x) else np.nan))
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No hay incentives_diag_summary.csv en el bundle.")
    with colB:
        st.markdown("**Detalle (muestra)**")
        if isinstance(inc_det, pd.DataFrame) and not inc_det.empty:
            df = inc_det.copy()
            money_cols = [c for c in df.columns if any(k in c.lower() for k in ["cost","uplift","monto","ingreso"]) ]
            df = format_df_currency(df, money_cols, moneda, usdclp)
            if "roi_individual" in df.columns:
                df["roi_individual"] = df["roi_individual"].apply(lambda x: fmt_pct_val(x*100 if pd.notna(x) else np.nan))
            st.dataframe(df.head(300), use_container_width=True)
        else:
            st.info("No hay incentives_detail.csv en el bundle.")

    with st.expander("Sensibilidades (parámetros usados en la corrida)"):
        if isinstance(inc_sens, pd.DataFrame) and not inc_sens.empty:
            st.dataframe(inc_sens, use_container_width=True)
        else:
            st.info("No hay incentives_sensitivity.csv en el bundle.")

# ================
# Arista 4 – Capital / Provisiones
# ================
with tabs[3]:
    st.subheader("Arista 4 – Capital / Provisiones")
    st.markdown("""
**KPIs:**
- **Capital Requerido** (proxy RW×K×EAD)
- **Provisiones** ~ EL
- **Liberación** = Actual − Optimizado
""", unsafe_allow_html=True)

    cap_port = dfs.get("cap_port")
    cap_seg = dfs.get("cap_seg")
    cap_det = dfs.get("cap_det")

    if cap_port is None or (isinstance(cap_port, pd.DataFrame) and cap_port.empty):
        st.error("No se encontró **capital_portfolio.csv**.")
    else:
        cap_base = g0(cap_port, "capital_req_base")
        cap_opt = g0(cap_port, "capital_req_opt")
        prov_base= g0(cap_port, "prov_base")
        prov_opt = g0(cap_port, "prov_opt")

        kpi_row("Capital Requerido", cap_base, cap_opt, moneda, usdclp, "Proxy RW×K×EAD")
        kpi_row("Provisiones", prov_base, prov_opt, moneda, usdclp, "≈ EL")

        lib_cap  = cap_base - cap_opt  if pd.notna(cap_base) and pd.notna(cap_opt)  else np.nan
        lib_prov = prov_base - prov_opt if pd.notna(prov_base) and pd.notna(prov_opt) else np.nan

        colX, colY = st.columns(2)
        with colX: st.metric("Liberación de Capital", fmt_money_val(lib_cap, moneda, usdclp))
        with colY: st.metric("Liberación de Provisiones", fmt_money_val(lib_prov, moneda, usdclp))

    colA, colB = st.columns(2)
    with colA:
        st.markdown("*Capital por segmento*")
        if isinstance(cap_seg, pd.DataFrame) and not cap_seg.empty:
            df = format_df_currency(
                cap_seg.copy(),
                ["capital_req_base","capital_req_opt","prov_base","prov_opt"],
                moneda, usdclp
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No hay capital_segment.csv en el bundle.")
    with colB:
        st.markdown("*Detalle (muestra)*")
        if isinstance(cap_det, pd.DataFrame) and not cap_det.empty:
            df = format_df_currency(
                cap_det.head(300).copy(),
                ["capital_req_base","capital_req_opt","prov_base","prov_opt","ead_baseline","e_opt"],
                moneda, usdclp
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No hay capital_detail.csv en el bundle.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas integradas).")
