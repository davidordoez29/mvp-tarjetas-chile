# app/app_dashboard.py — Versión pitch-ready con pestañas, storytelling por arista y formateo CLP/USD

import os, glob, json, math
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# ==========================
# Archivos requeridos (nombres exactos en el bundle)
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
        if not d or not os.path.isdir(d):
            return False
        hits = sum(os.path.exists(os.path.join(d, v)) for v in REQ_FILES.values())
        return hits >= 6
    except Exception:
        return False

def autodetect_bundle() -> str | None:
    # 1) candidatos directos
    for d in CANDIDATE_DIRS:
        if _dir_ok(d):
            return d
    # 2) búsqueda recursiva
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
    neg = x < 0; x = abs(x)
    ent = int(x); dec = int(round((x - ent) * 100))
    if dec == 100:
        ent += 1; dec = 0
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

def g0(df, col):
    try:
        return df[col].iloc[0] if (isinstance(df, pd.DataFrame) and col in df.columns and not df.empty) else np.nan
    except Exception:
        return np.nan

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
    st.error("No se encontró el bundle. Genera el paquete y vuelve a cargar.")
    st.stop()

dfs, missing = load_bundle(bundle_dir)
if missing:
    st.warning("Faltan archivos en el bundle (o nombres distintos):\n- " + "\n- ".join(missing))

moneda = st.sidebar.radio("Moneda a visualizar", ["CLP", "USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))
st.sidebar.caption("Aplica a todos los montos del dashboard.")

st.title("📊 MVP Bancario – Optimización en 4 Aristas")
st.caption("Comparación *Actual vs. Optimizado* con KPIs clave por arista. Modelo con repricing, elasticidades, incentivos y reasignación de capital.")

tabs = st.tabs([
    "Arista 1 – Default/Impago",
    "Arista 2 – Yield/Pricing",
    "Arista 3 – Incentivos",
    "Arista 4 – Capital/Provisiones"
])

# ================
# Arista 1 – Default / Impago
# ================
with tabs[0]:
    st.subheader("Arista 1 – Default/Impago")
    st.markdown("""
*¿Qué resuelve?*  
Reducimos *Pérdida Esperada (EL)* moviendo dos palancas:  
1) *Repricing continuo* por riesgo (tasa menor a alto riesgo, mayor a bajo riesgo).  
2) *Throttle de EAD* en los percentiles de *PD* más riesgosos.  
Guardrail: EL optimizado no supera el objetivo del portafolio.
""")

    port = dfs.get("def_port")
    seg  = dfs.get("def_seg")
    det  = dfs.get("def_det")

    if port is None or (isinstance(port, pd.DataFrame) and port.empty):
        st.error("No se encontró *default_portfolio.csv*.")
    else:
        EAD_act = g0(port, "EAD_actual");  EAD_opt = g0(port, "EAD_optimizado")
        EL_act  = g0(port, "EL_actual");   EL_opt  = g0(port, "EL_optimizado")
        Ing_act = g0(port, "Ingreso_actual"); Ing_opt = g0(port, "Ingreso_optimizado")
        Cost_act= g0(port, "Costos_actual");   Cost_opt= g0(port, "Costos_optimizado")
        Uti_act = g0(port, "Utilidad_actual"); Uti_opt = g0(port, "Utilidad_optimizada")
        PDw_act = g0(port, "PD_pond_actual");  PDw_opt = g0(port, "PD_pond_optimizado")

        # Storytelling cuantitativo
        st.markdown("*Storytelling*")
        delta_el = var_pct(EL_act, EL_opt)
        delta_uti = var_pct(Uti_act, Uti_opt)
        st.write(
            f"• El *EL* cae {fmt_pct_val(delta_el) if delta_el is not None else '—'} "
            f"al aplicar repricing por riesgo y throttle de EAD en alto riesgo."
        )
        st.write(
            f"• La *Utilidad* sube {fmt_pct_val(delta_uti) if delta_uti is not None else '—'} "
            f"porque reducimos pérdidas sin frenar los ingresos."
        )

        # KPIs
        kpi_row("EAD", EAD_act, EAD_opt, moneda, usdclp, "Exposición total (Exposure at Default)")
        kpi_row("EL (Pérdida Esperada)", EL_act, EL_opt, moneda, usdclp, "PD × LGD × EAD")
        kpi_row("Ingreso", Ing_act, Ing_opt, moneda, usdclp, "APR×EAD menos costos financieros y operativos")
        kpi_row("Costos Totales", Cost_act, Cost_opt, moneda, usdclp, "Financieros + Operativos")
        kpi_row("Utilidad", Uti_act, Uti_opt, moneda, usdclp, "Ingreso − EL − Costos")
        if pd.notna(PDw_act) or pd.notna(PDw_opt):
            kpi_row_pct("PD Ponderado (EAD)", PDw_act*100 if pd.notna(PDw_act) else np.nan,
                        PDw_opt*100 if pd.notna(PDw_opt) else np.nan,
                        "Probabilidad de default ponderada por EAD")

        st.markdown("*Detalle por segmento*")
        if isinstance(seg, pd.DataFrame) and not seg.empty:
            seg_fmt = seg.copy()
            seg_fmt = format_df_currency(seg_fmt,
                ["EAD_actual","EAD_optimizado","EL_actual","EL_optimizado",
                 "Ingreso_actual","Ingreso_optimizado","Costos_actual","Costos_optimizado",
                 "Utilidad_actual","Utilidad_optimizada"],
                moneda, usdclp)
            if "PD_pond_actual" in seg_fmt.columns:
                seg_fmt["PD_pond_actual"] = seg["PD_pond_actual"].apply(lambda x: fmt_pct_val(x*100))
            if "PD_pond_optimizado" in seg_fmt.columns:
                seg_fmt["PD_pond_optimizado"] = seg["PD_pond_optimizado"].apply(lambda x: fmt_pct_val(x*100))
            st.dataframe(seg_fmt, use_container_width=True)
        else:
            st.info("No hay default_segment.csv en el bundle.")

        with st.expander("Detalle por cliente (vista rápida)"):
            if isinstance(det, pd.DataFrame) and not det.empty:
                det_fmt = det.copy()
                money_cols = [c for c in det_fmt.columns if any(k in c.lower() for k in
                                ["ead","ingreso","utilidad","costo","el","capital"])]
                det_fmt = format_df_currency(det_fmt, money_cols, moneda, usdclp)
                if "pd_base" in det_fmt.columns:
                    det_fmt["pd_base"] = det["pd_base"].apply(lambda x: fmt_pct_val(x*100))
                if "pd_opt" in det_fmt.columns:
                    det_fmt["pd_opt"]  = det["pd_opt"].apply(lambda x: fmt_pct_val(x*100))
                st.dataframe(det_fmt.head(300), use_container_width=True)
            else:
                st.info("No hay default_detail.csv en el bundle.")

    st.markdown("---")
    st.caption("Historia A1: alinear precio con riesgo y reducir EAD en colas riesgosas baja EL sin castigar el negocio.")

# ================
# Arista 2 – Yield / Pricing
# ================
with tabs[1]:
    st.subheader("Arista 2 – Yield/Pricing")
    st.markdown("""
*¿Qué resuelve?*  
Encuentra el *APR* por segmento que *maximiza utilidad*, equilibrando precio y volumen (elasticidad).  
Guardrail: EL del portafolio se mantiene dentro del objetivo.
""")

    port = dfs.get("yld_port")
    seg  = dfs.get("yld_seg")
    det  = dfs.get("yld_det")
    curv = dfs.get("yld_curv")

    if port is None or (isinstance(port, pd.DataFrame) and port.empty):
        st.error("No se encontraron archivos de Yield.")
    else:
        Ing_base = g0(port,"ingreso_base"); Ing_iso = g0(port,"ingreso_iso"); Ing_opt = g0(port,"ingreso_opt")
        Uti_base = g0(port,"utilidad_base"); Uti_iso = g0(port,"utilidad_iso"); Uti_opt = g0(port,"utilidad_opt")
        EL_base  = g0(port,"EL_baseline");   EL_iso  = g0(port,"el_iso");        EL_opt  = g0(port,"el_opt")

        # Storytelling
        st.markdown("*Storytelling*")
        d_ing_total = var_pct(Ing_base, Ing_opt)
        d_uti_total = var_pct(Uti_base, Uti_opt)
        st.write(
            f"• *Utilidad total* ↑ {fmt_pct_val(d_uti_total) if d_uti_total is not None else '—'} "
            f"al fijar la tasa óptima por segmento (grid search)."
        )
        st.write(
            f"• La variante *solo-pricing* (EAD fijo) también mejora: "
            f"{fmt_pct_val(var_pct(Ing_base, Ing_iso)) if Ing_base not in [None,0] else '—'} en ingreso."
        )

        # KPIs
        kpi_row("Ingreso (Total)", Ing_base, Ing_opt, moneda, usdclp, "Precio + Volumen")
        kpi_row("Ingreso (Solo Pricing)", Ing_base, Ing_iso, moneda, usdclp, "EAD fijado en baseline")
        kpi_row("Utilidad (Total)", Uti_base, Uti_opt, moneda, usdclp, "Precio + Volumen")
        kpi_row("Utilidad (Solo Pricing)", Uti_base, Uti_iso, moneda, usdclp, "EAD fijado en baseline")
        kpi_row("EL", EL_base, EL_opt, moneda, usdclp, "Pérdida esperada total")

        cols = st.columns(2)
        with cols[0]:
            st.markdown("*Resumen por segmento*")
            if isinstance(seg, pd.DataFrame) and not seg.empty:
                seg_fmt = format_df_currency(seg.copy(),
                    ["ingreso_base","ingreso_iso","ingreso_opt","utilidad_base","utilidad_iso","utilidad_opt",
                     "EL_baseline","el_iso","el_opt"], moneda, usdclp)
                st.dataframe(seg_fmt, use_container_width=True)
            else:
                st.info("No hay yield_segment.csv en el bundle.")
        with cols[1]:
            st.markdown("*Curva r → ingreso (malla)*")
            if isinstance(curv, pd.DataFrame) and not curv.empty:
                # mostramos 2000 filas para no saturar
                st.dataframe(curv.head(2000), use_container_width=True)
            else:
                st.info("No hay yield_curve_segment.csv en el bundle.")

    st.markdown("---")
    st.caption("Historia A2: cada segmento tiene su tasa óptima; el precio correcto maximiza margen sin disparar EL.")

# ================
# Arista 3 – Incentivos
# ================
with tabs[2]:
    st.subheader("Arista 3 – Incentivos")
    st.markdown("""
*¿Qué resuelve?*  
Asigna incentivos donde el *ROI* es *positivo y alto*, incluyendo saturación y topes por cliente/segmento para evitar sobre-inversión.  
El objetivo es *subir ingreso incremental* con costo controlado.
""")

    det  = dfs.get("inc_det")
    summ = dfs.get("inc_sum")
    sens = dfs.get("inc_sens")

    # Totales desde detalle
    total_cost = np.nan; uplift = np.nan; roi = np.nan
    cost_col = None; uplift_col = None

    if isinstance(det, pd.DataFrame) and not det.empty:
        for c in ["inc_cost","costo_incentivo","costo_incentivo_monto","costo_incentivo_total"]:
            if c in det.columns: cost_col = c; break
        if cost_col is None and ("costo_incentivo_tasa" in det.columns):
            base_e = "e_opt" if "e_opt" in det.columns else ("ead_pricing" if "ead_pricing" in det.columns else None)
            if base_e is not None:
                det["_inc_cost_"] = pd.to_numeric(det["costo_incentivo_tasa"], errors="coerce").fillna(0)\
                                     * pd.to_numeric(det[base_e], errors="coerce").fillna(0)
                cost_col = "_inc_cost_"
        if cost_col is None:
            det["_inc_cost"] = 0.0; cost_col = "inc_cost_"

        for c in ["ingreso_uplift","uplift_ingreso","delta_ingreso","ingreso_incremental"]:
            if c in det.columns: uplift_col = c; break
        if uplift_col is None:
            det["_uplift"] = 0.0; uplift_col = "uplift_"

        total_cost = pd.to_numeric(det[cost_col], errors="coerce").fillna(0).sum()
        uplift     = pd.to_numeric(det[uplift_col], errors="coerce").fillna(0).sum()
        roi        = (uplift/total_cost) if total_cost>0 else np.nan

    # Storytelling
    st.markdown("*Storytelling*")
    st.write(
        f"• Se invierten *{fmt_money_val(total_cost, moneda, usdclp)}* en incentivos, "
        f"generando *{fmt_money_val(uplift, moneda, usdclp)}* de ingreso incremental."
    )
    st.write(
        f"• *ROI* portafolio: *{fmt_pct_val((roi or np.nan)*100)}*; se asigna capital sólo donde el retorno marginal es positivo."
    )

    # Tabla de resumen
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
        if rename_map:
            summ = summ.rename(columns=rename_map)

        wanted = ["segmento","inc_cost_total","ingreso_uplift_total","roi"]
        inter  = [c for c in wanted if c in summ.columns]
        if inter:
            dfshow = summ.loc[:, inter].copy()
            money_cols = [c for c in inter if c in ["inc_cost_total","ingreso_uplift_total"]]
            if money_cols:
                dfshow = format_df_currency(dfshow, money_cols, moneda, usdclp)
            if "roi" in dfshow.columns:
                dfshow["roi"] = dfshow["roi"].apply(lambda x: fmt_pct_val(x*100 if pd.notna(x) else np.nan))
            st.markdown("*Resumen por segmento*")
            st.dataframe(dfshow, use_container_width=True)
            showed_table = True

    # Fallback: agrupar desde detalle
    if not showed_table and isinstance(det, pd.DataFrame) and not det.empty and ("segmento" in det.columns):
        g = det.groupby("segmento", as_index=False).agg(
            inc_cost_total = (cost_col,   lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
            ingreso_uplift_total = (uplift_col, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
        )
        g["roi"] = np.where(g["inc_cost_total"]>0, g["ingreso_uplift_total"]/g["inc_cost_total"], np.nan)
        g_fmt = format_df_currency(g, ["inc_cost_total","ingreso_uplift_total"], moneda, usdclp)
        g_fmt["roi"] = g["roi"].apply(lambda x: fmt_pct_val(x*100 if pd.notna(x) else np.nan))
        st.markdown("*Resumen por segmento (derivado del detalle)*")
        st.dataframe(g_fmt, use_container_width=True)
        showed_table = True

    with st.expander("Detalle por cliente (vista rápida)"):
        if isinstance(det, pd.DataFrame) and not det.empty:
            det_fmt = det.copy()
            money_cols = [c for c in det_fmt.columns if any(k in c.lower() for k in ["cost","uplift","monto","ingreso"]) ]
            det_fmt = format_df_currency(det_fmt, money_cols, moneda, usdclp)
            st.dataframe(det_fmt.head(300), use_container_width=True)
        else:
            st.info("No hay detail de incentivos disponible.")

    st.markdown("---")
    st.caption("Historia A3: sprint de adopción con foco en retorno; caps evitan sobre-inversión en un segmento o cliente.")

# ================
# Arista 4 – Capital / Provisiones
# ================
with tabs[3]:
    st.subheader("Arista 4 – Capital / Provisiones")
    st.markdown("""
*¿Qué resuelve?*  
Reduce *consumo de capital* (proxy RW×K×EAD) y *provisiones* (~EL) al *reubicar EAD* desde colas de alto riesgo a perfiles más sanos, con guardrails de portafolio.
""")

    cap_port = dfs.get("cap_port")
    cap_seg  = dfs.get("cap_seg")
    cap_det  = dfs.get("cap_det")

    if cap_port is None or (isinstance(cap_port, pd.DataFrame) and cap_port.empty):
        st.error("No se encontró *capital_portfolio.csv*.")
    else:
        cap_base = g0(cap_port, "capital_req_base")
        cap_opt  = g0(cap_port, "capital_req_opt")
        prov_base= g0(cap_port, "prov_base")
        prov_opt = g0(cap_port, "prov_opt")

        # Storytelling
        st.markdown("*Storytelling*")
        st.write(
            f"• *Capital requerido* cae {fmt_pct_val(var_pct(cap_base, cap_opt))} "
            f"al trasladar EAD del decil más riesgoso hacia el más sano."
        )
        st.write(
            f"• *Provisiones* bajan {fmt_pct_val(var_pct(prov_base, prov_opt))} "
            f"acompasando la reducción del EL."
        )

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
            st.dataframe(det_fmt.head(300), use_container_width=True)
        else:
            st.info("No hay capital_detail.csv en el bundle.")

    st.markdown("---")
    st.caption("Historia A4: aligerar el consumo de capital libera capacidad para crecer rentable con riesgo acotado.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas integradas).")
