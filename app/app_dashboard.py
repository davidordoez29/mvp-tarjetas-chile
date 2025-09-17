# app/app_dashboard.py — Periodo (Anual/Trim/Mes) + Escenario + Guardrails + Attribution + Formato CLP/USD

import os, glob, json, math
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# ============== Time scaling (consistente con notebook) ==============
def pd_annual_to_monthly(pd_a: float) -> float:
    pd_a = float(pd_a)
    if pd_a <= 0: return 0.0
    if pd_a >= 1: return 1.0
    return 1.0 - (1.0 - pd_a) ** (1.0/12.0)

def scale_amount_by_period(x: float, period: str) -> float:
    period = (period or "ANUAL").upper()
    if period == "ANUAL":      return float(x)
    elif period == "TRIMESTRE":return float(x) / 4.0
    elif period == "MES":      return float(x) / 12.0
    else:                      return float(x)

# ============== Bundle requirements ==============
REQ_FILES = {
    "def_port": "default_portfolio.csv",
    "def_seg":  "default_segment.csv",
    "def_det":  "default_detail.csv",
    "yld_port": "yield_portfolio.csv",
    "yld_seg":  "yield_segment.csv",
    "yld_det":  "yield_detail.csv",
    "yld_curv": "yield_curve_segment.csv",
    "inc_det":  "incentives_detail.csv",
    "inc_sum":  "incentives_diag_summary.csv",
    "inc_sens": "incentives_sensitivity.csv",
    "cap_port": "capital_portfolio.csv",
    "cap_seg":  "capital_segment.csv",
    "cap_det":  "capital_detail.csv",
}

# ============== Formato CLP/USD ==============
def _to_display_currency(val: float, target: str, usdclp: float) -> float:
    if pd.isna(val): return np.nan
    if target.upper()=="USD":
        return float(val)/float(usdclp) if usdclp else np.nan
    return float(val)

def fmt_money_val(val: float, target: str, usdclp: float) -> str:
    if val is None or (isinstance(val,float) and math.isnan(val)): return "—"
    x = _to_display_currency(val, target, usdclp)
    if x is None or (isinstance(x,float) and math.isnan(x)): return "—"
    neg = x < 0; x = abs(x)
    ent = int(x); dec = int(round((x - ent) * 100))
    if dec == 100: ent += 1; dec = 0
    ent_str = f"{ent:,}".replace(",", ".")
    return f"-{ent_str},{dec:02d}" if neg else f"{ent_str},{dec:02d}"

def fmt_pct_val(val: float) -> str:
    if val is None or (isinstance(val,float) and math.isnan(val)): return "—"
    return f"{val:.2f}%".replace(".", ",")

def var_pct(actual: float, opt: float) -> float | None:
    if actual is None or pd.isna(actual) or actual==0: return None
    return (opt-actual)/actual*100.0

def format_df_currency(df: pd.DataFrame, cols: list[str], moneda: str, usdclp: float):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda v: fmt_money_val(v, moneda, usdclp))
    return df2

# ============== Carga bundle y escenarios ==============
CANDIDATE_DIRS = [
    os.environ.get("BUNDLE_DIR","").strip(),
    "/content/mvp-tarjetas-chile/out/dashboard_bundle",
    "./out/dashboard_bundle",
    "./dashboard_bundle",
    "/content/out/dashboard_bundle",
]

def _dir_ok(d: str) -> bool:
    try:
        if not d or not os.path.isdir(d): return False
        hits = 0
        for v in REQ_FILES.values():
            if os.path.exists(os.path.join(d, v)): hits += 1
        # toleramos que falte alguno; chequeamos guardrails aparte
        return hits >= 4
    except Exception:
        return False

def autodetect_bundle_root() -> str | None:
    for d in CANDIDATE_DIRS:
        if _dir_ok(d): return d
    # búsqueda recursiva
    bases = ["/content/mvp-tarjetas-chile", "/content", "."]
    candidates=[]
    for base in bases:
        for p in glob.glob(os.path.join(base,"**","dashboard_bundle"), recursive=True):
            if _dir_ok(p): candidates.append((p, os.path.getmtime(p)))
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return None

def list_scenarios(bundle_root: str) -> list[str]:
    out=[]
    for cand in ["base","estres","optimista"]:
        if os.path.isdir(os.path.join(bundle_root, cand)):
            out.append(cand)
    if not out: out=["."]  # bundle en raíz
    return out

def load_bundle(bundle_dir: str):
    dfs = {}
    missing = []
    for key, fname in REQ_FILES.items():
        p = os.path.join(bundle_dir, fname)
        if os.path.exists(p):
            try:
                dfs[key] = pd.read_csv(p)
            except Exception as e:
                dfs[key] = None
                missing.append(f"{fname} (error: {e})")
        else:
            dfs[key] = None
            missing.append(fname)
    # Guardrails/manifest opcionales
    grd = os.path.join(bundle_dir, "resumen_guardrails.csv")
    man = os.path.join(bundle_dir, "bundle_manifest.json")
    dfs["guardrails"] = pd.read_csv(grd) if os.path.exists(grd) else None
    dfs["manifest"]   = json.load(open(man,"r",encoding="utf-8")) if os.path.exists(man) else None
    return dfs, missing

# ============== App ==============
st.set_page_config(page_title="MVP Bancario — 4 Aristas", layout="wide")

st.sidebar.title("⚙️ Configuración")
root = autodetect_bundle_root()
bundle_root = st.sidebar.text_input("📦 Carpeta del bundle (raíz)", value=(root or "")).strip() or root

if not bundle_root:
    st.error("No se encontró el bundle. Genera el paquete y vuelve a cargar.")
    st.stop()

scenarios = list_scenarios(bundle_root)
scenario  = st.sidebar.selectbox("Escenario", scenarios, index=0)
bundle_dir = bundle_root if scenario=="." else os.path.join(bundle_root, scenario)

period = st.sidebar.radio("Periodo", ["ANUAL","TRIMESTRE","MES"], horizontal=True, index=0)
moneda = st.sidebar.radio("Moneda", ["CLP","USD"], horizontal=True, index=0)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))
st.sidebar.caption("Reescala montos y PD a periodo; convierte CLP↔USD en visualización.")

dfs, missing = load_bundle(bundle_dir)
if len(missing)>=len(REQ_FILES):  # casi nada
    st.error(f"Bundle incompleto en {bundle_dir}. Faltan varios archivos.")
    st.stop()

st.title("📊 MVP Bancario — Optimización en 4 Aristas")
st.caption(f"Escenario: *{scenario}* · Periodo: *{period}* · Formato monetario: *{moneda}*")

# Helper
def g0(df, col): 
    try: return float(df[col].iloc[0]) if (isinstance(df, pd.DataFrame) and col in df.columns and not df.empty) else np.nan
    except: return np.nan

def scale_if_notnan(x): 
    return scale_amount_by_period(x, period) if not pd.isna(x) else x

# Guardrails (si existen)
if isinstance(dfs.get("guardrails"), pd.DataFrame):
    gr = dfs["guardrails"].copy()
    # Escalamos valores monetarios (por periodo)
    for c in ["valor_base","valor_opt"]:
        if c in gr.columns:
            gr[c] = gr[c].apply(lambda v: scale_if_notnan(v))
    with st.expander("Guardrails del portafolio"):
        st.dataframe(gr, use_container_width=True)

tabs = st.tabs([
    "Arista 1 — Default/Impago",
    "Arista 2 — Yield/Pricing",
    "Arista 3 — Incentivos",
    "Arista 4 — Capital/Provisiones"
])

# =============== A1
with tabs[0]:
    st.subheader("Arista 1 — Default/Impago")
    st.markdown("""
*¿Qué resuelve?* Reduce *EL* (pérdida esperada) combinando *repricing por riesgo* y *throttle de EAD* en colas riesgosas.
""")

    port = dfs.get("def_port"); seg = dfs.get("def_seg"); det = dfs.get("def_det")
    if port is None or (isinstance(port,pd.DataFrame) and port.empty):
        st.info("No se encontró default_portfolio.csv")
    else:
        # Reescalar montos a periodo
        for c in ["EAD_actual","EAD_optimizado","EL_actual","EL_optimizado","Ingreso_actual","Ingreso_optimizado",
                  "Costos_actual","Costos_optimizado","Utilidad_actual","Utilidad_optimizada"]:
            if c in port.columns: port[c] = port[c].apply(scale_if_notnan)

        EAD_act = g0(port, "EAD_actual");  EAD_opt = g0(port, "EAD_optimizado")
        EL_act  = g0(port, "EL_actual");   EL_opt  = g0(port, "EL_optimizado")
        Ing_act = g0(port, "Ingreso_actual"); Ing_opt = g0(port, "Ingreso_optimizado")
        Cost_act= g0(port, "Costos_actual");   Cost_opt= g0(port, "Costos_optimizado")
        Uti_act = g0(port, "Utilidad_actual"); Uti_opt = g0(port, "Utilidad_optimizada")

        st.markdown("*Storytelling*")
        st.write(
            f"• *EL* cae {fmt_pct_val(var_pct(EL_act, EL_opt) or np.nan)} gracias a reasignar exposición y ajustar tasas."
        )
        st.write(
            f"• *Utilidad* sube {fmt_pct_val(var_pct(Uti_act, Uti_opt) or np.nan)} al reducir pérdidas sin frenar ingreso."
        )

        c1,c2,c3 = st.columns(3)
        with c1: st.metric("EAD – Actual", fmt_money_val(EAD_act, moneda, usdclp))
        with c2: st.metric("EAD – Optimizado", fmt_money_val(EAD_opt, moneda, usdclp))
        with c3: st.metric("VAR % EAD", fmt_pct_val((var_pct(EAD_act, EAD_opt) or np.nan)))

        c1,c2,c3 = st.columns(3)
        with c1: st.metric("EL – Actual", fmt_money_val(EL_act, moneda, usdclp))
        with c2: st.metric("EL – Optimizado", fmt_money_val(EL_opt, moneda, usdclp))
        with c3: st.metric("VAR % EL", fmt_pct_val((var_pct(EL_act, EL_opt) or np.nan)))

        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Ingreso – Actual", fmt_money_val(Ing_act, moneda, usdclp))
        with c2: st.metric("Ingreso – Optimizado", fmt_money_val(Ing_opt, moneda, usdclp))
        with c3: st.metric("Utilidad – Actual", fmt_money_val(Uti_act, moneda, usdclp))
        st.metric("Utilidad – Optimizada", fmt_money_val(Uti_opt, moneda, usdclp))

        # Seg / Det
        st.markdown("*Segmentos*")
        if isinstance(seg,pd.DataFrame) and not seg.empty:
            seg2 = seg.copy()
            for c in ["EAD_actual","EAD_optimizado","EL_actual","EL_optimizado",
                      "Ingreso_actual","Ingreso_optimizado","Costos_actual","Costos_optimizado",
                      "Utilidad_actual","Utilidad_optimizada"]:
                if c in seg2.columns: seg2[c] = seg2[c].apply(scale_if_notnan)
            money_cols = [c for c in seg2.columns if any(k in c.lower() for k in 
                            ["ead","ingreso","utilidad","costo","el","capital"])]
            st.dataframe(format_df_currency(seg2, money_cols, moneda, usdclp), use_container_width=True)
        else:
            st.info("Sin default_segment.csv")

        if isinstance(dfs.get("default_attribution"), pd.DataFrame):
            st.markdown("*Descomposición EL (ΔPD vs ΔEAD)*")
            st.dataframe(dfs["default_attribution"], use_container_width=True)

# =============== A2
with tabs[1]:
    st.subheader("Arista 2 — Yield/Pricing")
    st.markdown("""
*¿Qué resuelve?* Encuentra la *tasa óptima* por segmento equilibrando precio y volumen (elasticidad) con EL acotada.
""")

    port = dfs.get("yld_port"); seg = dfs.get("yld_seg"); det = dfs.get("yld_det"); curv = dfs.get("yld_curv")

    if port is None or (isinstance(port,pd.DataFrame) and port.empty):
        st.info("No se encontraron archivos de Yield.")
    else:
        for c in ["ingreso_base","ingreso_iso","ingreso_opt","utilidad_base","utilidad_iso","utilidad_opt",
                  "EL_baseline","el_iso","el_opt"]:
            if c in port.columns: port[c] = port[c].apply(scale_if_notnan)

        Ing_base = g0(port,"ingreso_base"); Ing_iso=g0(port,"ingreso_iso"); Ing_opt=g0(port,"ingreso_opt")
        Uti_base = g0(port,"utilidad_base"); Uti_iso=g0(port,"utilidad_iso"); Uti_opt=g0(port,"utilidad_opt")
        EL_base  = g0(port,"EL_baseline");   EL_iso=g0(port,"el_iso");        EL_opt=g0(port,"el_opt")

        st.markdown("*Storytelling*")
        st.write(
            f"• *Utilidad total* ↑ {fmt_pct_val((var_pct(Uti_base, Uti_opt) or np.nan))} fijando r óptima por segmento."
        )
        st.write(
            f"• Efecto *solo-pricing*: ingreso mejora {fmt_pct_val((var_pct(Ing_base, Ing_iso) or np.nan))} sin cambiar EAD."
        )

        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Ingreso (Total) – Base", fmt_money_val(Ing_base, moneda, usdclp))
        with c2: st.metric("Ingreso (Total) – Opt", fmt_money_val(Ing_opt, moneda, usdclp))
        with c3: st.metric("Δ%", fmt_pct_val((var_pct(Ing_base, Ing_opt) or np.nan)))

        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Utilidad – Base", fmt_money_val(Uti_base, moneda, usdclp))
        with c2: st.metric("Utilidad – Opt", fmt_money_val(Uti_opt, moneda, usdclp))
        with c3: st.metric("Δ%", fmt_pct_val((var_pct(Uti_base, Uti_opt) or np.nan)))

        st.markdown("*Segmentos*")
        if isinstance(seg,pd.DataFrame) and not seg.empty:
            seg2=seg.copy()
            for c in ["ingreso_base","ingreso_iso","ingreso_opt","utilidad_base","utilidad_iso","utilidad_opt",
                      "EL_baseline","el_iso","el_opt"]:
                if c in seg2.columns: seg2[c] = seg2[c].apply(scale_if_notnan)
            mcols=[c for c in seg2.columns if any(k in c.lower() for k in ["ingreso","utilidad","el"])]
            st.dataframe(format_df_currency(seg2, mcols, moneda, usdclp), use_container_width=True)

        if isinstance(dfs.get("yield_attribution"), pd.DataFrame):
            st.markdown("*Descomposición (Precio vs Volumen)*")
            st.dataframe(dfs["yield_attribution"], use_container_width=True)

        st.markdown("*Curva r→ingreso (malla, muestra)*")
        if isinstance(curv,pd.DataFrame) and not curv.empty:
            st.dataframe(curv.head(2000), use_container_width=True)
        else:
            st.info("Sin yield_curve_segment.csv")

# =============== A3
with tabs[2]:
    st.subheader("Arista 3 — Incentivos")
    st.markdown("""
*¿Qué resuelve?* Invierte incentivos sólo donde el *ROI* es alto, con *topes* por cliente/segmento para evitar sobre-inversión.
""")

    det  = dfs.get("inc_det")
    summ = dfs.get("inc_sum")

    total_cost=np.nan; uplift=np.nan; roi=np.nan
    cost_col = None; uplift_col=None

    if isinstance(det,pd.DataFrame) and not det.empty:
        for c in ["inc_cost","costo_incentivo","costo_incentivo_total","costo_incentivo_monto"]:
            if c in det.columns: cost_col=c; break
        for c in ["ingreso_uplift","ingreso_incremental","delta_ingreso","uplift_ingreso"]:
            if c in det.columns: uplift_col=c; break
        total_cost = float(pd.to_numeric(det.get(cost_col,0), errors="coerce").fillna(0).sum())
        uplift     = float(pd.to_numeric(det.get(uplift_col,0), errors="coerce").fillna(0).sum())
        # reescalar a periodo
        total_cost = scale_amount_by_period(total_cost, period)
        uplift     = scale_amount_by_period(uplift, period)
        roi        = (uplift/total_cost) if total_cost>0 else np.nan

    st.markdown("*Storytelling*")
    st.write(
        f"• Se invierten *{fmt_money_val(total_cost, moneda, usdclp)}* para generar "
        f"*{fmt_money_val(uplift, moneda, usdclp)}* de ingreso incremental."
    )
    st.write(
        f"• *ROI portafolio*: {fmt_pct_val((roi or np.nan)*100)}."
    )

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Costo Incentivos", fmt_money_val(total_cost, moneda, usdclp))
    with c2: st.metric("Ingreso Incremental", fmt_money_val(uplift, moneda, usdclp))
    with c3: st.metric("ROI", fmt_pct_val((roi or np.nan)*100))

    if isinstance(summ,pd.DataFrame) and not summ.empty:
        rename_map={}
        for c in list(summ.columns):
            lc = str(c).strip().lower()
            if lc in ["inc_cost_total","inc_cost","costo_total","costo_incentivo_total","costo_incentivos_total"]:
                rename_map[c]="inc_cost_total"
            elif lc in ["ingreso_uplift_total","ingreso_uplift","ingreso_incremental","delta_ingreso_total","delta_ingreso"]:
                rename_map[c]="ingreso_uplift_total"
            elif lc in ["roi","roi_total","retorno","retorno_beneficio"]:
                rename_map[c]="roi"
            elif lc in ["segmento","segment","segment_name"]:
                rename_map[c]="segmento"
        summ = summ.rename(columns=rename_map) if rename_map else summ
        if "inc_cost_total" in summ.columns:
            summ["inc_cost_total"] = summ["inc_cost_total"].apply(lambda v: scale_amount_by_period(v, period))
        if "ingreso_uplift_total" in summ.columns:
            summ["ingreso_uplift_total"] = summ["ingreso_uplift_total"].apply(lambda v: scale_amount_by_period(v, period))
        dfshow = summ.copy()
        dfshow = format_df_currency(dfshow, ["inc_cost_total","ingreso_uplift_total"], moneda, usdclp)
        if "roi" in dfshow.columns:
            dfshow["roi"] = dfshow["roi"].apply(lambda x: fmt_pct_val(x*100 if pd.notna(x) else np.nan))
        st.markdown("*Resumen por segmento*")
        st.dataframe(dfshow, use_container_width=True)

    with st.expander("Detalle (muestra)"):
        if isinstance(det,pd.DataFrame) and not det.empty:
            det2 = det.copy()
            money_cols = [c for c in det2.columns if any(k in c.lower() for k in ["cost","uplift","monto","ingreso"])]
            # Nota: no reescalo fila a fila (usualmente vienen annual); si quisieras, aplica scale_amount_by_period col a col.
            st.dataframe(format_df_currency(det2, money_cols, moneda, usdclp).head(300), use_container_width=True)
        else:
            st.info("Sin incentives_detail.csv")

# =============== A4
with tabs[3]:
    st.subheader("Arista 4 — Capital/Provisiones")
    st.markdown("""
*¿Qué resuelve?* Reduce *capital* (RW×K×EAD) y *provisiones* (~EL) al reubicar EAD desde colas riesgosas a perfiles sanos.
""")

    cap_port = dfs.get("cap_port"); cap_seg = dfs.get("cap_seg"); cap_det = dfs.get("cap_det")

    if cap_port is None or (isinstance(cap_port,pd.DataFrame) and cap_port.empty):
        st.info("No se encontró capital_portfolio.csv")
    else:
        for c in ["capital_req_base","capital_req_opt","prov_base","prov_opt"]:
            if c in cap_port.columns: cap_port[c] = cap_port[c].apply(scale_if_notnan)

        cap_base = g0(cap_port,"capital_req_base")
        cap_opt  = g0(cap_port,"capital_req_opt")
        prov_base= g0(cap_port,"prov_base")
        prov_opt = g0(cap_port,"prov_opt")

        st.markdown("*Storytelling*")
        st.write(
            f"• *Capital requerido* cae {fmt_pct_val((var_pct(cap_base, cap_opt) or np.nan))} "
            f"gracias a la reasignación de EAD."
        )
        st.write(
            f"• *Provisiones* bajan {fmt_pct_val((var_pct(prov_base, prov_opt) or np.nan))} en línea con EL."
        )

        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Capital – Base", fmt_money_val(cap_base, moneda, usdclp))
        with c2: st.metric("Capital – Opt", fmt_money_val(cap_opt, moneda, usdclp))
        with c3: st.metric("Δ%", fmt_pct_val((var_pct(cap_base, cap_opt) or np.nan)))

        c1,c2 = st.columns(2)
        with c1: st.metric("Provisiones – Base", fmt_money_val(prov_base, moneda, usdclp))
        with c2: st.metric("Provisiones – Opt", fmt_money_val(prov_opt, moneda, usdclp))

        st.markdown("*Capital por segmento*")
        if isinstance(cap_seg,pd.DataFrame) and not cap_seg.empty:
            seg2 = cap_seg.copy()
            for c in ["capital_req_base","capital_req_opt","prov_base","prov_opt"]:
                if c in seg2.columns: seg2[c] = seg2[c].apply(scale_if_notnan)
            mcols=[c for c in seg2.columns if any(k in c.lower() for k in ["capital","prov","ead"])]
            st.dataframe(format_df_currency(seg2, mcols, moneda, usdclp), use_container_width=True)

        if isinstance(dfs.get("capital_attribution"), pd.DataFrame):
            st.markdown("*Descomposición (ΔCapital/ΔProvisiones)*")
            st.dataframe(dfs["capital_attribution"], use_container_width=True)

        with st.expander("Detalle (muestra)"):
            if isinstance(cap_det,pd.DataFrame) and not cap_det.empty:
                det2 = cap_det.copy()
                money_cols=[c for c in det2.columns if any(k in c.lower() for k in ["capital","prov","ead"])]
                for c in money_cols:
                    if c in det2.columns: det2[c] = det2[c].apply(scale_if_notnan)
                st.dataframe(format_df_currency(det2, money_cols, moneda, usdclp).head(300), use_container_width=True)
            else:
                st.info("Sin capital_detail.csv")

st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas). Periodo reescalado para visualización; base anual.")
