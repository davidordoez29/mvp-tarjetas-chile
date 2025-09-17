# app/app_dashboard.py — MVP Bancario (4 Aristas) — versión ejecutiva con formatos y guardrails
import os, glob, json, math
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# ==========================
# Archivos requeridos (bundle)
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
    # Meta / defs (opcionales)
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
        if not d or not os.path.isdir(d):
            return False
        # al menos 6 archivos esperados para considerarlo bundle válido
        hits = sum(os.path.exists(os.path.join(d, v)) for v in REQ_FILES.values())
        return hits >= 6
    except Exception:
        return False

def autodetect_bundle() -> str | None:
    for d in CANDIDATE_DIRS:
        if _dir_ok(d):
            return d
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
            missing.append(fname)
            dfs[key] = None
            continue
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
# Formato de números (CLP / USD)
# ==========================
def _to_display_currency(val: float, target: str, usdclp: float) -> float:
    if pd.isna(val):
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

def format_df_auto(df: pd.DataFrame, money_cols=None, pct_cols=None, moneda="CLP", usdclp=900.0):
    """Aplica formato a columnas típicas si existen."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    money_kw = ["monto","ingreso","utilidad","el","capital","prov","ead","cost"]
    pct_kw   = ["apr","rate","tasa","pd","lgd","roi","roc"]
    if money_cols is None:
        money_cols = [c for c in df.columns if any(k in c.lower() for k in money_kw)]
    if pct_cols is None:
        pct_cols = [c for c in df.columns if any(k in c.lower() for k in pct_kw)]
    out = format_df_currency(df, money_cols, moneda, usdclp)
    out = format_df_pct(out, pct_cols)
    return out

# ==========================
# App
# ==========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

# --- Sidebar
st.sidebar.title("⚙️ Configuración")
default_dir = autodetect_bundle()
bundle_dir = st.sidebar.text_input(
    "📦 Ruta del bundle",
    value=(default_dir or ""),
    help="Ej: /content/mvp-tarjetas-chile/out/dashboard_bundle"
).strip() or default_dir

if not bundle_dir:
    st.error("No se encontró el bundle con ambos CSV. Sube/Sync a out/dashboard_bundle o define BUNDLE_DIR en el despliegue.")
    st.stop()

dfs, missing = load_bundle(bundle_dir)
if missing:
    st.warning("Faltan archivos en el bundle (o nombres distintos):\n- " + "\n- ".join(missing))

moneda = st.sidebar.radio("Moneda", ["CLP", "USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))
st.sidebar.caption("Aplica a todos los montos del dashboard.")

# Guardrails (desde meta si existe; si no, valores ejemplo)
st.sidebar.markdown("### 🧱 Guardrails (Resguardos / Límites)")
guardrails = {}
meta = dfs.get("meta") or {}
if isinstance(meta, dict) and "guardrails" in meta:
    guardrails = meta["guardrails"] or {}
else:
    # fallback ilustrativo
    guardrails = {
        "PD máxima (prom.)": 0.08,      # 8%
        "LGD máxima (prom.)": 0.45,     # 45%
        "APR mínima": 0.12,             # 12%
        "APR máxima": 0.45,             # 45%
        "Utilidad mínima (mensual)": 50_000_000,  # CLP
    }

# Mostrar guardrails con formato amigable
gr_rows = []
for k, v in guardrails.items():
    if isinstance(v, (int, float)):
        # heurística: si el nombre sugiere porcentaje, muestro como %
        if any(w in k.lower() for w in ["pd", "lgd", "apr", "tasa", "porcentaje", "%"]):
            gr_rows.append((k, fmt_pct_val(v*100 if v <= 1 else v)))
        elif any(w in k.lower() for w in ["utilidad","ingreso","capital","monto","clp","$"]):
            gr_rows.append((k, fmt_money_val(v, moneda, usdclp)))
        else:
            # por defecto número con 2 decimales y miles
            s = f"{v:,.2f}".replace(",", ".").replace(".", ",", 1)
            gr_rows.append((k, s))
    else:
        gr_rows.append((k, str(v)))
if gr_rows:
    for k, v in gr_rows:
        st.sidebar.write(f"- *{k}*: {v}")

# ==========================
# Header
# ==========================
st.title("📊 MVP Bancario — Motor de Optimización (4 Aristas)")
st.caption("Comparación *Actual vs Optimizado* con KPIs por arista. Enfoque ejecutivo para directorio.")

tabs = st.tabs([
    "Arista 1 — Default / Impago",
    "Arista 2 — Yield / Pricing",
    "Arista 3 — Incentivos",
    "Arista 4 — Capital / Provisiones",
])

# ==========================================================
# Arista 1 — Default / Impago
# ==========================================================
with tabs[0]:
    st.subheader("Arista 1 — Default / Impago")
    st.markdown("""
*¿Qué resolvemos?*  
Reducimos la *Pérdida Esperada (EL)* moviendo exposición (EAD) hacia clientes con mejor perfil, sin frenar innecesariamente el negocio.

*KPIs clave*  
- *EAD*: exposición total en riesgo.  
- *EL*: PD × LGD × EAD.  
- *Ingreso* y *Costos* (financieros + operativos).  
- *Utilidad* = Ingreso − EL − Costos.  
- *PD ponderado*: probabilidad de impago promedio ponderada por EAD.
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

        kpi_row("EAD", EAD_act, EAD_opt, moneda, usdclp, "Exposure at Default")
        kpi_row("EL (Pérdida Esperada)", EL_act, EL_opt, moneda, usdclp, "PD × LGD × EAD")
        kpi_row("Ingreso", Ing_act, Ing_opt, moneda, usdclp)
        kpi_row("Costos Totales", Cost_act, Cost_opt, moneda, usdclp, "Financieros + Operativos")
        kpi_row("Utilidad", Uti_act, Uti_opt, moneda, usdclp)
        if pd.notna(PDw_act) or pd.notna(PDw_opt):
            kpi_row_pct("PD Ponderado (EAD)", PDw_act*100 if pd.notna(PDw_act) else np.nan,
                        PDw_opt*100 if pd.notna(PDw_opt) else np.nan)

    st.markdown("#### Análisis Ejecutivo")
    st.write(
        "Movimos exposición hacia perfiles menos riesgosos, lo que reduce EL sin sacrificar ingreso. "
        "El portafolio queda más estable (PD ponderado menor) y la utilidad total mejora al bajar las pérdidas esperadas."
    )

# ==========================================================
# Arista 2 — Yield / Pricing
# ==========================================================
with tabs[1]:
    st.subheader("Arista 2 — Yield / Pricing")
    st.markdown("""
*¿Qué resolvemos?*  
Encontramos el *precio (APR)* que maximiza utilidad equilibrando margen y volumen, con elasticidad y límites de negocio.

*KPIs clave*  
- *Ingreso/Utilidad (Total)*: con precio y volumen optimizados.  
- *Ingreso/Utilidad (Solo Pricing)*: cambio de precio manteniendo EAD baseline.  
- *EL*: para controlar el riesgo al ajustar precio.
    """, unsafe_allow_html=True)

    port = dfs.get("yld_port")
    seg  = dfs.get("yld_seg")
    det  = dfs.get("yld_det")
    curv = dfs.get("yld_curv")

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
        kpi_row("EL", EL_base, EL_opt, moneda, usdclp, "Pérdida esperada")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("*Resumen por segmento*")
        if isinstance(seg, pd.DataFrame) and not seg.empty:
            seg_fmt = format_df_auto(seg, moneda=moneda, usdclp=usdclp)
            st.dataframe(seg_fmt, use_container_width=True)
        else:
            st.info("No hay yield_segment.csv en el bundle.")
    with colB:
        st.markdown("*Detalle de pricing (muestra)*")
        if isinstance(det, pd.DataFrame) and not det.empty:
            det_fmt = format_df_auto(det.head(300), moneda=moneda, usdclp=usdclp)
            st.dataframe(det_fmt, use_container_width=True)
        else:
            st.info("No hay yield_detail.csv en el bundle.")

    st.markdown("#### Análisis Ejecutivo")
    st.write(
        "Ajustamos tasas por segmento según elasticidad: donde subir precio no destruye volumen, capturamos margen; "
        "donde la sensibilidad es alta, priorizamos retención de volumen. Esto maximiza utilidad total sin disparar EL."
    )

# ==========================================================
# Arista 3 — Incentivos
# ==========================================================
with tabs[2]:
    st.subheader("Arista 3 — Incentivos")
    st.markdown("""
*¿Qué resolvemos?*  
Asignamos beneficios donde el *ROI* (ingreso incremental / costo) es alto y sostenible, acelerando adopción y uso rentables.

*KPIs clave*  
- *Costo de Incentivos*.  
- *Ingreso Incremental*.  
- *ROI* = Ingreso / Costo.
    """, unsafe_allow_html=True)

    det  = dfs.get("inc_det")
    summ = dfs.get("inc_sum")

    # Calcular totales de portafolio de la forma más robusta
    total_cost = np.nan; uplift = np.nan; roi = np.nan
    cost_col = None; uplift_col = None

    if isinstance(det, pd.DataFrame) and not det.empty:
        # columnas probables de costo
        for c in ["inc_cost","costo_incentivo","costo_incentivo_total","costo_incentivo_monto"]:
            if c in det.columns: cost_col = c; break
        if cost_col is None:
            det["inc_cost"] = 0.0; cost_col = "inc_cost"

        # columnas probables de uplift
        for c in ["ingreso_uplift","uplift_ingreso","delta_ingreso","ingreso_incremental"]:
            if c in det.columns: uplift_col = c; break
        if uplift_col is None:
            det["ingreso_uplift"] = 0.0; uplift_col = "ingreso_uplift"

        total_cost = pd.to_numeric(det[cost_col], errors="coerce").fillna(0).sum()
        uplift     = pd.to_numeric(det[uplift_col], errors="coerce").fillna(0).sum()
        roi        = (uplift/total_cost) if total_cost>0 else np.nan

    # Mostrar resumen si existe
    showed_table = False
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
        summ = summ.rename(columns=rename_map) if rename_map else summ

        wanted = ["segmento","inc_cost_total","ingreso_uplift_total","roi"]
        inter  = [c for c in wanted if c in summ.columns]
        if inter:
            dfshow = summ.loc[:, inter].copy()
            dfshow_fmt = dfshow.copy()
            if "inc_cost_total" in dfshow_fmt.columns:
                dfshow_fmt["inc_cost_total"] = dfshow_fmt["inc_cost_total"].apply(lambda v: fmt_money_val(v, moneda, usdclp))
            if "ingreso_uplift_total" in dfshow_fmt.columns:
                dfshow_fmt["ingreso_uplift_total"] = dfshow_fmt["ingreso_uplift_total"].apply(lambda v: fmt_money_val(v, moneda, usdclp))
            if "roi" in dfshow_fmt.columns:
                dfshow_fmt["roi"] = dfshow["roi"].apply(lambda v: fmt_pct_val(v*100 if pd.notna(v) else np.nan))
            st.markdown("*Resumen de incentivos*")
            st.dataframe(dfshow_fmt, use_container_width=True)
            showed_table = True

    if not showed_table and isinstance(det, pd.DataFrame) and not det.empty and ("segmento" in det.columns):
        g = det.groupby("segmento", as_index=False).agg(
            inc_cost_total = (cost_col,   lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
            ingreso_uplift_total = (uplift_col, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
        )
        g["roi"] = np.where(g["inc_cost_total"]>0, g["ingreso_uplift_total"]/g["inc_cost_total"], np.nan)
        g_fmt = g.copy()
        g_fmt["inc_cost_total"] = g_fmt["inc_cost_total"].apply(lambda v: fmt_money_val(v, moneda, usdclp))
        g_fmt["ingreso_uplift_total"] = g_fmt["ingreso_uplift_total"].apply(lambda v: fmt_money_val(v, moneda, usdclp))
        g_fmt["roi"] = g["roi"].apply(lambda x: fmt_pct_val(x*100 if pd.notna(x) else np.nan))
        st.markdown("*Resumen de incentivos por segmento (derivado del detalle)*")
        st.dataframe(g_fmt, use_container_width=True)

    # KPIs
    kpi_row("Costo de Incentivos", total_cost, total_cost, moneda, usdclp, "Suma de costos de beneficios")
    kpi_row("Ingreso Incremental", uplift, uplift, moneda, usdclp, "Suma de incrementos estimados")
    st.metric("ROI (Ingreso/Costo)", fmt_pct_val(roi*100 if pd.notna(roi) else np.nan))

    st.markdown("#### Análisis Ejecutivo")
    st.write(
        "Focalizamos beneficios donde el ROI es más alto. El gasto se concentra en clientes y segmentos con mayor probabilidad "
        "de generar ingreso incremental sostenido."
    )

# ==========================================================
# Arista 4 — Capital / Provisiones
# ==========================================================
with tabs[3]:
    st.subheader("Arista 4 — Capital / Provisiones")
    st.markdown("""
*¿Qué resolvemos?*  
Hacemos más *eficiente* el uso del capital regulatorio y estabilizamos provisiones, manteniendo el riesgo bajo control.  
Puede subir el capital total si el portafolio crece, pero el *retorno por unidad de capital* mejora.

*KPIs clave*  
- *Capital Requerido* (proxy: RW × K × EAD).  
- *Provisiones* ≈ EL.  
- *Liberación* = Actual − Optimizado (cuando corresponde).  
- *Eficiencia*: Utilidad / Capital (indicativo del retorno por capital).
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
        uti_base = g0(cap_port, "utilidad_base") if "utilidad_base" in cap_port.columns else np.nan
        uti_opt  = g0(cap_port, "utilidad_opt")  if "utilidad_opt"  in cap_port.columns else np.nan

        kpi_row("Capital Requerido", cap_base, cap_opt, moneda, usdclp, "Proxy RW × K × EAD")
        kpi_row("Provisiones",       prov_base, prov_opt, moneda, usdclp, "≈ EL")

        lib_cap  = cap_base - cap_opt  if pd.notna(cap_base) and pd.notna(cap_opt)  else np.nan
        lib_prov = prov_base - prov_opt if pd.notna(prov_base) and pd.notna(prov_opt) else np.nan

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Liberación de Capital", fmt_money_val(lib_cap, moneda, usdclp))
        with col2: st.metric("Liberación de Provisiones", fmt_money_val(lib_prov, moneda, usdclp))
        with col3:
            # Indicador simple de eficiencia: utilidad/capital
            roc_base = (uti_base / cap_base * 100.0) if (pd.notna(uti_base) and pd.notna(cap_base) and cap_base>0) else np.nan
            roc_opt  = (uti_opt  / cap_opt  * 100.0) if (pd.notna(uti_opt)  and pd.notna(cap_opt)  and cap_opt>0) else np.nan
            st.metric("Eficiencia (Utilidad/Capital)", fmt_pct_val(roc_opt) if pd.notna(roc_opt) else "—",
                      delta=fmt_pct_val((roc_opt-roc_base) if (pd.notna(roc_opt) and pd.notna(roc_base)) else np.nan))

    colA, colB = st.columns(2)
    with colA:
        st.markdown("*Capital por segmento*")
        if isinstance(cap_seg, pd.DataFrame) and not cap_seg.empty:
            seg_fmt = cap_seg.copy()
            # Formateos explícitos
            seg_fmt = format_df_currency(seg_fmt,
                                         ["capital_req_base","capital_req_opt","prov_base","prov_opt","ead_baseline","e_opt"],
                                         moneda, usdclp)
            seg_fmt = format_df_pct(seg_fmt,
                                    [c for c in ["pd_base","pd_opt","lgd_base","lgd_opt","apr_efectiva","r_opt"]
                                     if c in cap_seg.columns])
            st.dataframe(seg_fmt, use_container_width=True)
        else:
            st.info("No hay capital_segment.csv en el bundle.")

    with colB:
        st.markdown("*Detalle de capital (muestra)*")
        if isinstance(cap_det, pd.DataFrame) and not cap_det.empty:
            det_fmt = cap_det.copy()
            det_fmt = format_df_currency(det_fmt,
                                         ["capital_req_base","capital_req_opt","prov_base","prov_opt","ead_baseline","e_opt"],
                                         moneda, usdclp)
            det_fmt = format_df_pct(det_fmt,
                                    [c for c in ["pd_base","pd_opt","lgd_base","lgd_opt","apr_efectiva","r_opt"]
                                     if c in cap_det.columns])
            st.dataframe(det_fmt.head(300), use_container_width=True)
        else:
            st.info("No hay capital_detail.csv en el bundle.")

    st.markdown("#### Análisis Ejecutivo")
    st.write(
        "El capital puede aumentar si ampliamos el portafolio hacia clientes sanos; aun así, mejora la *eficiencia* "
        "porque la *utilidad por unidad de capital* sube. Las provisiones totales pueden crecer en monto, pero bajan "
        "proporcionalmente al riesgo del portafolio."
    )

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas integradas).")
