# app/app_dashboard.py
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
    "cap_port": "capital_portfolio.csv",   # ← nombre corregido
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
# Formato de moneda/porcentaje
# ==========================
def _to_display_currency(val: float, target: str, usdclp: float) -> float:
    if pd.isna(val):
        return np.nan
    if target.upper() == "USD":
        return float(val) / float(usdclp) if usdclp else np.nan
    return float(val)

def fmt_money(val: float, target: str, usdclp: float) -> str:
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
    s = f"{ent_str},{dec:02d}"
    return f"-{s}" if neg else s

def fmt_pct(val: float) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"{val:.2f}%".replace(".", ",")

def var_pct(actual: float, opt: float) -> float | None:
    if actual is None or pd.isna(actual) or actual == 0:
        return None
    return (opt - actual) / actual * 100.0

def g0(df, col):
    try:
        return df[col].iloc[0] if (isinstance(df, pd.DataFrame) and col in df.columns and not df.empty) else np.nan
    except Exception:
        return np.nan

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

def styler_money(df: pd.DataFrame, money_cols, moneda: str, usdclp: float, pct_cols=None):
    fmt_map = {c: (lambda x, _c=c: fmt_money(pd.to_numeric(x, errors="coerce"), moneda, usdclp)) for c in money_cols}
    if pct_cols:
        for c in pct_cols:
            fmt_map[c] = (lambda x, _c=c: fmt_pct(pd.to_numeric(x, errors="coerce")))
    return df.style.format(fmt_map)

# ==========================
# Narrativas ejecutivas (storytelling sencillo)
# ==========================
def story_default(EAD_act, EAD_opt, EL_act, EL_opt, Ing_act, Ing_opt, Uti_act, Uti_opt, moneda, usdclp):
    de = var_pct(EAD_act, EAD_opt)
    dl = var_pct(EL_act, EL_opt)
    di = var_pct(Ing_act, Ing_opt)
    du = var_pct(Uti_act, Uti_opt)

    def pct_txt(v):
        return fmt_pct(v) if v is not None else "—"
    return st.markdown(f"""
*Cómo se conecta todo:*
- *EAD* (Exposure at Default / Exposición en Riesgo): {pct_txt(de)}. Más exposición bien asignada habilita mayor *Ingreso*.
- *EL* (Expected Loss / *Pérdida Esperada*): {pct_txt(dl)}. Aun con cambios en EAD, contenemos la pérdida esperada vía selección y precio.
- *Ingreso* (APR×EAD): {pct_txt(di)}.
- *Utilidad* (Ingreso − EL − Costos): {pct_txt(du)}.

*Traducción ejecutiva:* incrementamos exposición en clientes y tramos donde el riesgo es controlado, lo que eleva ingresos; al mismo tiempo, la pérdida esperada se mantiene bajo control, resultando en *más utilidad* neta para el banco.
""")

def story_yield(Ing_base, Ing_iso, Ing_opt, Uti_base, Uti_iso, Uti_opt):
    di_iso = var_pct(Ing_base, Ing_iso)
    di_tot = var_pct(Ing_base, Ing_opt)
    du_iso = var_pct(Uti_base, Uti_iso)
    du_tot = var_pct(Uti_base, Uti_opt)
    return st.markdown(f"""
*Precio vs. Volumen:*
- *Solo pricing* (misma EAD): Ingreso {fmt_pct(di_iso)} · Utilidad {fmt_pct(du_iso)}.
- *Total (precio + volumen)*: Ingreso {fmt_pct(di_tot)} · Utilidad {fmt_pct(du_tot)}.

*Lectura ejecutiva:* la mejora de precio ya aporta por sí sola; al combinarla con el nivel óptimo de exposición, el resultado total en utilidad es mayor.
""")

def story_incentivos(roi, total_cost, uplift, moneda, usdclp):
    roi_txt = fmt_pct(roi*100) if roi is not None and not pd.isna(roi) else "—"
    return st.markdown(f"""
*Qué significa:*
- *Costo de Incentivos*: {fmt_money(total_cost, moneda, usdclp)}
- *Ingreso Incremental*: {fmt_money(uplift, moneda, usdclp)}
- *ROI* (Ingreso/Costo): {roi_txt}

*Lectura ejecutiva:* el esquema asigna beneficios donde *cada peso invertido* retorna ingresos incrementales medibles. Si ves ROI bajo o en 0, revisa los parámetros de la Celda 11 (definición de incentivos y su conversión a monto).
""")

def story_capital(cap_base, cap_opt, prov_base, prov_opt, moneda, usdclp):
    lib_cap  = (cap_base - cap_opt) if (pd.notna(cap_base) and pd.notna(cap_opt)) else np.nan
    lib_prov = (prov_base - prov_opt) if (pd.notna(prov_base) and pd.notna(prov_opt)) else np.nan
    return st.markdown(f"""
*Qué significa:*
- *Capital requerido* (proxy): de {fmt_money(cap_base, moneda, usdclp)} a {fmt_money(cap_opt, moneda, usdclp)}.
- *Provisiones* (≈ EL): de {fmt_money(prov_base, moneda, usdclp)} a {fmt_money(prov_opt, moneda, usdclp)}.
- *Liberación*: Capital {fmt_money(lib_cap, moneda, usdclp)} · Provisiones {fmt_money(lib_prov, moneda, usdclp)}.

*Lectura ejecutiva:* al optimizar precio y exposición, el portafolio *consume menos capital* y *requiere menores provisiones*, habilitando crecimiento rentable.
""")

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

with st.sidebar.expander("📂 Archivos del bundle detectados", expanded=False):
    for key, fname in REQ_FILES.items():
        ok = dfs.get(key) is not None
        st.write(("✅ " if ok else "❌ ") + fname)
if missing:
    st.warning("Faltan o fallaron archivos:\n- " + "\n- ".join(missing))

moneda = st.sidebar.radio("Moneda a visualizar", ["CLP", "USD"], horizontal=True)
usdclp = float(st.sidebar.number_input("USDCLP (1 USD = ? CLP)", min_value=1.0, value=900.0, step=1.0))
st.sidebar.caption("Aplica a todos los montos del dashboard.")

st.title("📊 MVP Bancario — Optimización en 4 Aristas (Executive View)")
st.caption("Formato de montos: *miles con punto* y *dos decimales; porcentajes con **dos decimales*.")

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
    st.subheader("Arista 1 — Default/Impago")
    st.markdown("""
*KPIs (definiciones):*
- *EAD (Exposure at Default / Exposición en Riesgo):* monto expuesto si el cliente cae en impago.
- *EL (Expected Loss / Pérdida Esperada):* probabilidad × severidad × exposición (≈ PD×LGD×EAD).
- *Ingreso:* tasa anual aplicada a la exposición (*APR × EAD*).
- *Costos:* financieros + operativos.
- *Utilidad:* Ingreso − EL − Costos.
- *PD ponderado (por EAD):* PD promedio, pesando más donde hay más exposición.
""")

    port = dfs.get("def_port")
    if isinstance(port, pd.DataFrame) and not port.empty:
        EAD_act = g0(port, "EAD_actual"); EAD_opt = g0(port, "EAD_optimizado")
        EL_act  = g0(port, "EL_actual");  EL_opt  = g0(port, "EL_optimizado")
        Ing_act = g0(port, "Ingreso_actual"); Ing_opt = g0(port, "Ingreso_optimizado")
        Cost_act= g0(port, "Costos_actual");  Cost_opt= g0(port, "Costos_optimizado")
        Uti_act = g0(port, "Utilidad_actual");Uti_opt = g0(port, "Utilidad_optimizada")
        PDw_act = g0(port, "PD_pond_actual"); PDw_opt = g0(port, "PD_pond_optimizado")

        kpi_row("EAD", EAD_act, EAD_opt, moneda, usdclp, "Exposición total (Exposure at Default)")
        kpi_row("EL (Pérdida Esperada)", EL_act, EL_opt, moneda, usdclp, "PD × LGD × EAD")
        kpi_row("Ingreso", Ing_act, Ing_opt, moneda, usdclp, "APR × EAD")
        kpi_row("Costos Totales", Cost_act, Cost_opt, moneda, usdclp, "Financieros + Operativos")
        kpi_row("Utilidad", Uti_act, Uti_opt, moneda, usdclp, "Ingreso − EL − Costos")
        if pd.notna(PDw_act) or pd.notna(PDw_opt):
            kpi_row_pct("PD Ponderado (EAD)",
                        PDw_act*100 if pd.notna(PDw_act) else np.nan,
                        PDw_opt*100 if pd.notna(PDw_opt) else np.nan,
                        "Promedio ponderado por exposición")

        st.markdown("---")
        story_default(EAD_act, EAD_opt, EL_act, EL_opt, Ing_act, Ing_opt, Uti_act, Uti_opt, moneda, usdclp)
    else:
        st.error("No se encontró *default_portfolio.csv*.")

# ================
# Arista 2 – Yield / Pricing
# ================
with tabs[1]:
    st.subheader("Arista 2 — Yield/Pricing")
    st.markdown("""
*KPIs (definiciones):*
- *Ingreso/Utilidad (Total):* con r_opt y e_opt (precio + exposición).
- *Ingreso/Utilidad (Solo Pricing):* con r_opt y EAD fijo en baseline (aisla el efecto precio).
""")

    port = dfs.get("yld_port")
    if isinstance(port, pd.DataFrame) and not port.empty:
        Ing_base = g0(port, "ingreso_base"); Ing_iso = g0(port, "ingreso_iso"); Ing_opt = g0(port, "ingreso_opt")
        Uti_base = g0(port, "utilidad_base"); Uti_iso = g0(port, "utilidad_iso"); Uti_opt = g0(port, "utilidad_opt")
        EL_base  = g0(port, "EL_baseline");   EL_iso  = g0(port, "el_iso");        EL_opt = g0(port, "el_opt")

        kpi_row("Ingreso (Total)", Ing_base, Ing_opt, moneda, usdclp, "Precio + Volumen")
        kpi_row("Ingreso (Solo Pricing)", Ing_base, Ing_iso, moneda, usdclp, "EAD fijado en baseline")
        kpi_row("Utilidad (Total)", Uti_base, Uti_opt, moneda, usdclp, "Precio + Volumen")
        kpi_row("Utilidad (Solo Pricing)", Uti_base, Uti_iso, moneda, usdclp, "EAD fijado en baseline")
        kpi_row("EL", EL_base, EL_opt, moneda, usdclp, "Pérdida esperada total")

        st.markdown("---")
        story_yield(Ing_base, Ing_iso, Ing_opt, Uti_base, Uti_iso, Uti_opt)
    else:
        st.error("No se encontraron archivos de Yield.")

# ================
# Arista 3 – Incentivos
# ================
with tabs[2]:
    st.subheader("Arista 3 — Incentivos")
    st.markdown("""
*KPIs (definiciones):*
- *Costo de Incentivos:* gasto total en beneficios/descuentos.
- *Ingreso Incremental:* ingresos atribuibles al esquema de incentivos.
- *ROI:* Ingreso Incremental / Costo de Incentivos.
""")

    det = dfs.get("inc_det"); summ = dfs.get("inc_sum")

    total_cost = 0.0; uplift = 0.0

    if isinstance(det, pd.DataFrame) and not det.empty:
        cost_col = next((c for c in det.columns if c.lower() in [
            "inc_cost","costo_incentivo","costo_incentivo_monto","costo_incentivo_total","costo_beneficios"
        ]), None)
        if cost_col is None and "costo_incentivo_tasa" in det.columns:
            base_e = det["e_opt"] if "e_opt" in det.columns else det.get("ead_baseline", pd.Series(0, index=det.index))
            det["_inc_cost_"] = pd.to_numeric(det["costo_incentivo_tasa"], errors="coerce").fillna(0) * \
                                  pd.to_numeric(base_e, errors="coerce").fillna(0)
            cost_col = "_inc_cost_"
        if cost_col:
            total_cost = pd.to_numeric(det[cost_col], errors="coerce").fillna(0).sum()

        uplift_col = next((c for c in det.columns if c.lower() in [
            "uplift_ingreso","ingreso_uplift","delta_ingreso","ingreso_incremental","uplift_total"
        ]), None)
        if uplift_col:
            uplift = pd.to_numeric(det[uplift_col], errors="coerce").fillna(0).sum()

        # Tabla detalle con formato
        money_cols = [c for c in det.columns if any(k in c.lower() for k in ["cost","costo","monto","ingreso","utilidad"])]
        pct_cols   = [c for c in det.columns if ("tasa" in c.lower()) or c.lower().endswith("_pct")]
        st.markdown("*Detalle (muestra):*")
        try:
            st.dataframe(styler_money(det.head(200), money_cols, moneda, usdclp, pct_cols=pct_cols),
                         use_container_width=True)
        except Exception:
            st.dataframe(det.head(200), use_container_width=True)

    # Fallback desde summary si falta algo
    if (total_cost == 0.0 or uplift == 0.0) and isinstance(summ, pd.DataFrame) and not summ.empty:
        for cc in ["cost_total","costo_total","costo_incentivo_total","total_cost"]:
            if cc in summ.columns:
                total_cost = float(pd.to_numeric(summ[cc], errors="coerce").fillna(0).sum()); break
        for uu in ["uplift_ingreso_total","ingreso_incremental_total","ingreso_total_uplift","uplift_total"]:
            if uu in summ.columns:
                uplift = float(pd.to_numeric(summ[uu], errors="coerce").fillna(0).sum()); break

    roi = (uplift / total_cost) if total_cost > 0 else np.nan

    kpi_row("Costo de Incentivos", total_cost, total_cost, moneda, usdclp, "Suma de costos de beneficios")
    kpi_row("Ingreso Incremental", uplift, uplift, moneda, usdclp, "Suma de incrementos")
    st.metric("ROI (Ingreso/Costo)", fmt_pct(roi*100 if pd.notna(roi) else np.nan))

    st.markdown("---")
    story_incentivos(roi, total_cost, uplift, moneda, usdclp)

# ================
# Arista 4 – Capital / Provisiones
# ================
with tabs[3]:
    st.subheader("Arista 4 — Capital / Provisiones")
    st.markdown("""
*KPIs (definiciones):*
- *Capital Requerido (proxy):* RW × K × EAD.
- *Provisiones (≈ EL):* pérdida esperada del portafolio.
- *Liberación:* Actual − Optimizado (clip en 0 si no hay liberación).
""")

    cap = dfs.get("cap_port"); cap_seg = dfs.get("cap_seg"); cap_det = dfs.get("cap_det")

    if isinstance(cap, pd.DataFrame) and not cap.empty:
        cap_base = g0(cap, "capital_req_base"); cap_opt = g0(cap, "capital_req_opt")
        prov_base= g0(cap, "prov_base");        prov_opt= g0(cap, "prov_opt")

        kpi_row("Capital Requerido", cap_base, cap_opt, moneda, usdclp, "Proxy RW×K×EAD")
        kpi_row("Provisiones",       prov_base, prov_opt, moneda, usdclp, "≈ EL")

        lib_cap  = (cap_base - cap_opt) if (pd.notna(cap_base) and pd.notna(cap_opt)) else np.nan
        lib_prov = (prov_base - prov_opt) if (pd.notna(prov_base) and pd.notna(prov_opt)) else np.nan

        c1, c2 = st.columns(2)
        with c1: st.metric("Liberación de Capital", fmt_money(lib_cap, moneda, usdclp))
        with c2: st.metric("Liberación de Provisiones", fmt_money(lib_prov, moneda, usdclp))

        st.markdown("---")
        story_capital(cap_base, cap_opt, prov_base, prov_opt, moneda, usdclp)

        # Tablas con formato
        if isinstance(cap_seg, pd.DataFrame) and not cap_seg.empty:
            st.markdown("*Capital por segmento:*")
            money_cols = [c for c in cap_seg.columns if any(k in c.lower() for k in ["capital","prov","monto"])]
            st.dataframe(styler_money(cap_seg, money_cols, moneda, usdclp), use_container_width=True)

        if isinstance(cap_det, pd.DataFrame) and not cap_det.empty:
            with st.expander("Detalle de capital (muestra)"):
                money_cols = [c for c in cap_det.columns if any(k in c.lower() for k in ["capital","prov","monto"])]
                st.dataframe(styler_money(cap_det.head(200), money_cols, moneda, usdclp), use_container_width=True)
    else:
        st.error("No se encontró *capital_portfolio.csv*.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas). Explicaciones ejecutivas y KPIs conectados.")
