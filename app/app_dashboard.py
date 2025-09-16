# >>> WDOF_BUNDLE_PATCH START
import os as _os
from pathlib import Path as _Path

def _resolve_bundle_dir():
    # Prioridad 1: entorno
    _env = _os.environ.get("BUNDLE_DIR","").strip()
# >>> WDOF_FORCE_BUNDLE_PATHS START
clients_p = BUNDLE_DIR / "dashboard_bundle_clients.csv"
segs_p    = BUNDLE_DIR / "dashboard_bundle_segments.csv"
# <<< WDOF_FORCE_BUNDLE_PATHS END
# >>> WDOF_FORCE_BUNDLE_PATHS START

# <<< WDOF_FORCE_BUNDLE_PATHS END
# >>> WDOF_FORCE_BUNDLE_PATHS START

# <<< WDOF_FORCE_BUNDLE_PATHS END
# >>> WDOF_FORCE_BUNDLE_PATHS START

# <<< WDOF_FORCE_BUNDLE_PATHS END
# >>> WDOF_FORCE_BUNDLE_PATHS START

# <<< WDOF_FORCE_BUNDLE_PATHS END
# >>> WDOF_FORCE_BUNDLE_PATHS START

# <<< WDOF_FORCE_BUNDLE_PATHS END
# >>> WDOF_FORCE_BUNDLE_PATHS START

# <<< WDOF_FORCE_BUNDLE_PATHS END
# >>> WDOF_FORCE_BUNDLE_PATHS START

# <<< WDOF_FORCE_BUNDLE_PATHS END
# >>> WDOF_FORCE_BUNDLE_PATHS START

# <<< WDOF_FORCE_BUNDLE_PATHS END
# >>> WDOF_FORCE_BUNDLE_PATHS START

# <<< WDOF_FORCE_BUNDLE_PATHS END
# >>> WDOF_FORCE_BUNDLE_PATHS START

# <<< WDOF_FORCE_BUNDLE_PATHS END
# >>> WDOF_FORCE_BUNDLE_PATHS START

# <<< WDOF_FORCE_BUNDLE_PATHS END
# >>> WDOF_FORCE_BUNDLE_PATHS START

# <<< WDOF_FORCE_BUNDLE_PATHS END
# >>> WDOF_DIAG START
try:
    import streamlit as st
    from pathlib import Path as _P
    with _st.sidebar:
        _st.markdown("### Diagnóstico Bundle")
        _st.write("BUNDLE_DIR (app):", str(BUNDLE_DIR))
        _c = _P(BUNDLE_DIR)/"dashboard_bundle_clients.csv"
        _s = _P(BUNDLE_DIR)/"dashboard_bundle_segments.csv"
        _st.write("clients_csv:", "✅" if _c.exists() else "❌", str(_c))
        _st.write("segments_csv:", "✅" if _s.exists() else "❌", str(_s))
except Exception:
    pass
# <<< WDOF_DIAG END
    _cand = []
    if _env:
        _cand.append(_Path(_env))
    try:
        repo_root = _Path(file_).resolve().parents[1]
        _cand += [
            _repo_root/"out"/"dashboard_bundle",
            _repo_root/"out",
            _Path("/content")/"mvp-tarjetas-chile"/"out"/"dashboard_bundle",
            _Path("/content")/"out"/"dashboard_bundle"
        ]
    except Exception:
        pass
    req = {"dashboard_bundle_clients.csv","dashboard_bundle_segments.csv"}
    for c in _cand:
        try:
            if c and c.exists() and req.issubset({p.name for p in c.glob("*.csv")}):
                return c
        except Exception:
            pass
    # fallback: si viene env sin validar, úsalo; si no, cwd
    return _Path(_env) if _env else _Path(".")
# <<< WDOF_BUNDLE_PATCH END
# app/app_dashboard.py — robusto, con formateo y corrección KeyError en Incentivos

import os, glob, json, math
import pandas as pd
import numpy as np
import streamlit as st

# ==========================
# Archivos requeridos (nombres exactos)
# ==========================
REQ_FILES = {
    # Arista 1 (Default)
    "def_port": "default_portfolio.csv",
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

# ==========================
# App
# ==========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

st.sidebar.title("⚙️ Configuración")
default_dir = autodetect_bundle()
bundle_dir = st.sidebar.text_input("📦 Ruta del bundle", value=(default_dir or ""), help="Ej: /content/mvp-tarjetas-chile/out/dashboard_bundle").strip() or default_dir

if not bundle_dir:
    st.error("No encuentro el bundle. Genera el paquete (Notebook 02, Celda 7.5) y vuelve a cargar.")
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

with st.expander("¿Qué resuelve cada arista?"):
    st.markdown("""
*Arista 1 (Default/Impago):* Reduce la *Pérdida Esperada (EL)* administrando la exposición (EAD) por cliente/segmento sin frenar innecesariamente el negocio.<br>
*Arista 2 (Yield/Pricing):* Encuentra el *APR* que maximiza utilidad equilibrando precio y volumen (elasticidad), con bandas y restricciones de negocio.<br>
*Arista 3 (Incentivos):* Asigna beneficios donde el *ROI* (ingreso incremental / costo) es positivo y significativo, para acelerar adopción y uso.<br>
*Arista 4 (Capital/Provisiones):* Minimiza consumo de *capital* y *provisiones* (proxies) manteniendo riesgo acotado y los ingresos del portafolio.
""", unsafe_allow_html=True)

# ================
# Arista 1 – Default
# ================
with tabs[0]:
    st.subheader("Arista 1 – Default/Impago")
    st.markdown("""
*KPIs:*
- *EAD* (Exposure at Default): Exposición en riesgo.
- *EL* (Expected Loss): PD × LGD × EAD.
- *Ingreso: APR × EAD. **Costos*: Financieros + Operativos.
- *Utilidad*: Ingreso − EL − Costos.
- *PD ponderado*: PD promedio ponderado por EAD.
""", unsafe_allow_html=True)

    port = dfs.get("def_port")
    if port is None or (isinstance(port, pd.DataFrame) and port.empty):
        st.error("No se encontró *default_portfolio.csv*.")
    else:
        def g0(df, col):
            return df[col].iloc[0] if (isinstance(df, pd.DataFrame) and col in df.columns and not df.empty) else np.nan

        EAD_act = g0(port, "EAD_actual");  EAD_opt = g0(port, "EAD_optimizado")
        EL_act  = g0(port, "EL_actual");   EL_opt  = g0(port, "EL_optimizado")
        Ing_act = g0(port, "Ingreso_actual"); Ing_opt = g0(port, "Ingreso_optimizado")
        Cost_act= g0(port, "Costos_actual");   Cost_opt= g0(port, "Costos_optimizado")
        Uti_act = g0(port, "Utilidad_actual"); Uti_opt = g0(port, "Utilidad_optimizada")
        PDw_act = g0(port, "PD_pond_actual");  PDw_opt = g0(port, "PD_pond_optimizado")

        kpi_row("EAD", EAD_act, EAD_opt, moneda, usdclp, "Exposición total (Exposure at Default)")
        kpi_row("EL (Pérdida Esperada)", EL_act, EL_opt, moneda, usdclp, "PD × LGD × EAD")
        kpi_row("Ingreso", Ing_act, Ing_opt, moneda, usdclp, "APR × EAD")
        kpi_row("Costos Totales", Cost_act, Cost_opt, moneda, usdclp, "Financieros + Operativos")
        kpi_row("Utilidad", Uti_act, Uti_opt, moneda, usdclp, "Ingreso − EL − Costos")
        if pd.notna(PDw_act) or pd.notna(PDw_opt):
            kpi_row_pct("PD Ponderado (EAD)", PDw_act*100 if pd.notna(PDw_act) else np.nan,
                        PDw_opt*100 if pd.notna(PDw_opt) else np.nan,
                        "Probabilidad de default ponderada por EAD")

    st.markdown("---")
    st.caption("Historia: bajamos EL en segmentos de alto riesgo y reubicamos exposición hacia perfiles con mejor retorno → sube la utilidad total.")

# ================
# Arista 2 – Yield / Pricing
# ================
with tabs[1]:
    st.subheader("Arista 2 – Yield/Pricing")
    st.markdown("""
*KPIs:*
- *Ingreso/Utilidad (Total)*: usando r_opt y e_opt.
- *Ingreso/Utilidad (Solo Pricing)*: usando r_opt con EAD = baseline.
""", unsafe_allow_html=True)

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

    st.markdown("---")
    st.caption("Historia: ajustamos APR por segmento según elasticidad; el precio correcto maximiza margen sin disparar EL.")

# ================
# Arista 3 – Incentivos (blindada)
# ================
with tabs[2]:
    st.subheader("Arista 3 – Incentivos")
    st.markdown("""
*KPIs (portafolio):*
- *Costo Incentivos* (CLP/USD).
- *Ingreso Incremental* (CLP/USD).
- *ROI* = Ingreso Incremental / Costo.
""", unsafe_allow_html=True)

    det  = dfs.get("inc_det")
    summ = dfs.get("inc_sum")

    # Totales de portafolio desde detalle (si existe)
    total_cost = np.nan; uplift = np.nan; roi = np.nan
    cost_col = None; uplift_col = None

    if isinstance(det, pd.DataFrame) and not det.empty:
        # Detectar columnas de costo
        for c in ["inc_cost","costo_incentivo","costo_incentivo_monto","costo_incentivo_total"]:
            if c in det.columns: cost_col = c; break
        if cost_col is None and ("costo_incentivo_tasa" in det.columns):
            # preferimos e_opt si existe; si no, ead_baseline
            base_e = "e_opt" if "e_opt" in det.columns else ("ead_baseline" if "ead_baseline" in det.columns else None)
            if base_e is not None:
                det["_inc_cost_"] = pd.to_numeric(det["costo_incentivo_tasa"], errors="coerce").fillna(0)\
                                     * pd.to_numeric(det[base_e], errors="coerce").fillna(0)
                cost_col = "_inc_cost_"
        if cost_col is None:
            det["_inc_cost"] = 0.0; cost_col = "inc_cost_"

        # Detectar columnas de uplift
        for c in ["ingreso_uplift","uplift_ingreso","delta_ingreso","ingreso_incremental"]:
            if c in det.columns: uplift_col = c; break
        if uplift_col is None:
            det["_uplift"] = 0.0; uplift_col = "uplift_"

        total_cost = pd.to_numeric(det[cost_col], errors="coerce").fillna(0).sum()
        uplift     = pd.to_numeric(det[uplift_col], errors="coerce").fillna(0).sum()
        roi        = (uplift/total_cost) if total_cost>0 else np.nan

    # Normalizar columnas del resumen si existe
    if isinstance(summ, pd.DataFrame) and not summ.empty:
        rename_map = {}
        for c in list(summ.columns):
            lc = str(c).strip().lower()
            if lc in ["inc_cost_total","inc_cost","costo_total","costo_incentivo_total","costo_incentivos_total"]:
                rename_map[c] = "inc_cost_total"
            elif lc in ["ingreso_uplift_total","ingreso_uplift","uplift_ingreso","ingreso_incremental","delta_ingreso_total","delta_ingreso"]:
                rename_map[c] = "ingreso_uplift_total"
            elif lc in ["roi","roi_total","retorno","retorno_beneficio"]:
                rename_map[c] = "roi"
            elif lc in ["segmento","segment","segment_name"]:
                rename_map[c] = "segmento"
        if rename_map:
            summ = summ.rename(columns=rename_map)

    # Tabla de resumen preferida:
    showed_table = False
    if isinstance(summ, pd.DataFrame) and not summ.empty:
        wanted = ["segmento","inc_cost_total","ingreso_uplift_total","roi"]
        inter  = [c for c in wanted if c in summ.columns]
        if inter:
            dfshow = summ.loc[:, inter].copy()
            # Formateo
            money_cols = [c for c in inter if c in ["inc_cost_total","ingreso_uplift_total"]]
            if money_cols:
                dfshow = format_df_currency(dfshow, money_cols, moneda, usdclp)
            if "roi" in dfshow.columns:
                dfshow["roi"] = dfshow["roi"].apply(lambda x: fmt_pct_val(x*100 if pd.notna(x) else np.nan))
            st.markdown("*Resumen de incentivos*")
            st.dataframe(dfshow, use_container_width=True)
            showed_table = True

    # Si no logramos mostrar summary, intentamos por segmento desde el detalle:
    if not showed_table and isinstance(det, pd.DataFrame) and not det.empty and ("segmento" in det.columns):
        g = det.groupby("segmento", as_index=False).agg(
            inc_cost_total = (cost_col,   lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
            ingreso_uplift_total = (uplift_col, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
        )
        g["roi"] = np.where(g["inc_cost_total"]>0, g["ingreso_uplift_total"]/g["inc_cost_total"], np.nan)
        g_fmt = format_df_currency(g, ["inc_cost_total","ingreso_uplift_total"], moneda, usdclp)
        g_fmt["roi"] = g["roi"].apply(lambda x: fmt_pct_val(x*100 if pd.notna(x) else np.nan))
        st.markdown("*Resumen de incentivos por segmento (derivado del detalle)*")
        st.dataframe(g_fmt, use_container_width=True)
        showed_table = True

    # KPIs (siempre mostramos)
    kpi_row("Costo de Incentivos", total_cost, total_cost, moneda, usdclp, "Suma de costos de beneficios")
    kpi_row("Ingreso Incremental", uplift, uplift, moneda, usdclp, "Suma de incrementos estimados")
    st.metric("ROI (Ingreso/Costo)", fmt_pct_val(roi*100 if pd.notna(roi) else np.nan))

    with st.expander("Detalle por cliente (vista rápida)"):
        if isinstance(det, pd.DataFrame) and not det.empty:
            det_fmt = det.copy()
            money_cols = [c for c in det_fmt.columns if any(k in c.lower() for k in ["cost","uplift","monto","ingreso"]) ]
            det_fmt = format_df_currency(det_fmt, money_cols, moneda, usdclp)
            st.dataframe(det_fmt.head(200), use_container_width=True)
        else:
            st.info("No hay detalle de incentivos disponible.")

    st.markdown("---")
    st.caption("Historia: los incentivos se focalizan donde el ROI es positivo; movemos la aguja en uso/ingreso sin disparar costos.")

# ================
# Arista 4 – Capital / Provisiones
# ================
with tabs[3]:
    st.subheader("Arista 4 – Capital / Provisiones")
    st.markdown("""
*KPIs:*
- *Capital Requerido* (proxy RW×K×EAD).
- *Provisiones* ~ EL.
- *Liberación* = Actual − Optimizado.
""", unsafe_allow_html=True)

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
        kpi_row("Provisiones", prov_base, prov_opt, moneda, usdclp, "≈ EL")

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

    st.markdown("---")
    st.caption("Historia: reubicar EAD hacia perfiles más sanos reduce consumo de capital y estabiliza provisiones.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas integradas).")