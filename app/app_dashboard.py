# app/app_dashboard.py
import os, glob, json, math
import pandas as pd
import numpy as np
import streamlit as st

# ==========================
# Archivos requeridos (nombres exactos del bundle)
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
    "cap_port": "capital_portfolio.csv",   # nombre correcto
    "cap_seg":  "capital_segment.csv",
    "cap_det":  "capital_detail.csv",
    # Meta
    "kpi_defs": "kpi_defs.json",
    "seg_defs": "segment_defs.json",
    "meta":     "bundle_meta.json",
}

# Aceptar también 'capital_portafolio.csv' si existe (tolerancia)
CAP_ALT = "capital_portafolio.csv"

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
        hits = 0
        for v in REQ_FILES.values():
            if os.path.exists(os.path.join(d, v)):
                hits += 1
        # tolerancia capital_portafolio
        if not os.path.exists(os.path.join(d, REQ_FILES["cap_port"])) and \
           os.path.exists(os.path.join(d, CAP_ALT)):
            hits += 1
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
        # capital_portfolio con tolerancia a "portafolio"
        if key == "cap_port" and not os.path.exists(path):
            alt = os.path.join(bundle_dir, CAP_ALT)
            if os.path.exists(alt):
                path = alt
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
# Formato: miles con punto, decimales con coma
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

def df_money_cols(df, cols, moneda, usdclp):
    for c in cols:
        if c in df.columns:
            df[c + "_clp"] = df[c].apply(lambda v: fmt_money(v, "CLP", usdclp))
            df[c + "_usd"] = df[c].apply(lambda v: fmt_money(v, "USD", usdclp))
    return df

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
    st.error("No encuentro el bundle. Genera el paquete (Celda 12/14) y vuelve a cargar.")
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

# ==========================
# Intro general por arista (storytelling corto)
# ==========================
INTRO = {
    0: ("Reducimos pérdidas esperadas sin sacrificar crecimiento.",
        "Con PD y LGD constantes, ajustamos la exposición (EAD) donde es rentable para bajar EL y subir la utilidad."),
    1: ("Encontramos el precio que maximiza margen neto.",
        "Movemos APR dentro de bandas; el volumen (EAD) responde vía elasticidad, resguardando riesgo/costos."),
    2: ("Asignamos incentivos con ROI positivo.",
        "Usamos presupuesto para activar clientes con mayor retorno marginal y límites de exposición por riesgo."),
    3: ("Liberamos capital y estabilizamos provisiones.",
        "Al bajar EL y EAD donde no rinde, reduce RW×K×EAD y provisiones esperadas."),
}

# ================
# Arista 1 – Default
# ================
with tabs[0]:
    st.subheader("Arista 1 – Default/Impago")
    st.markdown(f"*Qué ves aquí:* {INTRO[0][0]}  \n{INTRO[0][1]}")
    st.markdown("""
*KPIs:*
- *EAD* (Exposure at Default / Exposición en Riesgo).
- *EL* (Expected Loss / Pérdida Esperada = PD × LGD × EAD).
- *Ingreso* (APR × EAD), *Costos* (financieros + operativos), *Utilidad* (Ingreso − EL − Costos).
- *PD ponderado* (PD medio ponderado por EAD).
""")

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

        kpi_row("EAD", EAD_act, EAD_opt, moneda, usdclp, "Exposición total (Exposure at Default)")
        kpi_row("EL (Pérdida Esperada)", EL_act, EL_opt, moneda, usdclp, "PD × LGD × EAD")
        kpi_row("Ingreso", Ing_act, Ing_opt, moneda, usdclp, "APR × EAD")
        kpi_row("Costos Totales", Cost_act, Cost_opt, moneda, usdclp, "Financieros + Operativos")
        kpi_row("Utilidad", Uti_act, Uti_opt, moneda, usdclp, "Ingreso − EL − Costos")
        if pd.notna(PDw_act) or pd.notna(PDw_opt):
            kpi_row_pct("PD Ponderado (EAD)", PDw_act*100 if pd.notna(PDw_act) else np.nan,
                        PDw_opt*100 if pd.notna(PDw_opt) else np.nan,
                        "Prob. de default promedio ponderada por EAD")

    st.markdown("---")
    st.caption("Nuestro modelo reduce EL concentrando EAD donde el retorno riesgo/beneficio es superior.")

# ================
# Arista 2 – Yield / Pricing
# ================
with tabs[1]:
    st.subheader("Arista 2 – Yield/Pricing")
    st.markdown(f"*Qué ves aquí:* {INTRO[1][0]}  \n{INTRO[1][1]}")
    st.markdown("""
*KPIs:*
- *Total: usa *r_opt y e_opt (precio + volumen).
- *Solo Pricing: usa *r_opt manteniendo EAD en baseline (aisla efecto de precio).
""")

    port = dfs.get("yld_port")
    if port is None or (isinstance(port, pd.DataFrame) and port.empty):
        st.error("No se encontraron archivos de Yield.")
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
    st.caption("El pricing óptimo eleva el margen sin deteriorar desproporcionadamente riesgo y costos.")

# ================
# Arista 3 – Incentivos (corregido KPIs: Actual=0 vs Opt=totales)
# ================
with tabs[2]:
    st.subheader("Arista 3 – Incentivos")
    st.markdown(f"*Qué ves aquí:* {INTRO[2][0]}  \n{INTRO[2][1]}")
    st.markdown("""
*KPIs (portafolio):*
- *Costo Incentivos* (CLP/USD): gasto del esquema.
- *Ingreso Incremental* (CLP/USD): mayor ingreso por activación/uso.
- *ROI* = Ingreso Incremental / Costo.
> En el *escenario actual* no hay incentivo nuevo ⇒ costo e ingresos incrementales *= 0*.
""")

    det = dfs.get("inc_det")
    summ = dfs.get("inc_sum")

    total_cost = 0.0
    uplift_ing = 0.0
    util_upl   = 0.0

    if isinstance(det, pd.DataFrame) and not det.empty:
        # Columnas estándar que genera la Celda 4 (incentives)
        cost_col = "inc_cost" if "inc_cost" in det.columns else None
        ing_col  = "ingreso_uplift" if "ingreso_uplift" in det.columns else None
        utl_col  = "utilidad_uplift" if "utilidad_uplift" in det.columns else None

        if cost_col is None and "costo_incentivo_tasa" in det.columns and "ead_baseline" in det.columns:
            det["_inc_cost_"] = pd.to_numeric(det["costo_incentivo_tasa"], errors="coerce").fillna(0)\
                                   * pd.to_numeric(det["ead_baseline"], errors="coerce").fillna(0)
            cost_col = "_inc_cost_"

        total_cost = pd.to_numeric(det[cost_col], errors="coerce").fillna(0).sum() if cost_col else 0.0
        uplift_ing = pd.to_numeric(det[ing_col],  errors="coerce").fillna(0).sum() if ing_col  else 0.0
        util_upl   = pd.to_numeric(det[utl_col],  errors="coerce").fillna(0).sum() if utl_col  else 0.0

    # KPIs correctos: Actual = 0, Optimizado = totales
    kpi_row("Costo de Incentivos", 0.0, total_cost, moneda, usdclp, "Gasto del esquema propuesto")
    kpi_row("Ingreso Incremental", 0.0, uplift_ing, moneda, usdclp, "Mayor ingreso por activación/uso")
    # ROI sobre ingreso (también puedes mostrar ROI utilidad)
    roi_ing = (uplift_ing / total_cost) if total_cost > 0 else np.nan
    c1, c2 = st.columns([1.0, 2.0])
    with c1:
        st.metric("ROI (Ingreso/Costo)", fmt_pct(roi_ing*100 if pd.notna(roi_ing) else np.nan))
    with c2:
        st.caption("Priorizamos clientes con mejor rendimiento marginal hasta agotar presupuesto (si aplica).")

    # Tabla resumen por segmento (si existe)
    if isinstance(summ, pd.DataFrame) and not summ.empty:
        st.markdown("##### Resumen por segmento")
        # Formatear columnas de dinero
        money_cols = ["Costo_inc","Ingreso_uplift","Utilidad_uplift"]
        for c in money_cols:
            if c in summ.columns:
                summ[c + "_fmt"] = summ[c].apply(lambda v: fmt_money(v, moneda, usdclp))
        show_cols = ["segmento"] + [c+"_fmt" for c in money_cols if c in summ.columns]
        st.dataframe(summ[show_cols], use_container_width=True)

    # Detalle (debajo de KPIs y explicación)
    if isinstance(det, pd.DataFrame) and not det.empty:
        st.markdown("##### Detalle de asignación (muestra)")
        det_show = det.copy()
        # columnas monetarias típicas
        money_cols = ["inc_cost","ingreso_uplift","el_uplift","cfin_uplift","cop_uplift","utilidad_uplift"]
        for c in money_cols:
            if c in det_show.columns:
                det_show[c + "_fmt"] = det_show[c].apply(lambda v: fmt_money(v, moneda, usdclp))
        col_keep = ["id_cliente","segmento","s_opt","e_opt","e_inc"] + [c+"_fmt" for c in money_cols if c in det_show.columns]
        col_keep = [c for c in col_keep if c in det_show.columns]
        st.dataframe(det_show[col_keep].head(1000), use_container_width=True)

    st.markdown("---")
    st.caption("El motor asigna incentivos con ROI positivo por cliente/segmento, respetando límites de exposición y presupuesto.")

# ================
# Arista 4 – Capital / Provisiones (con tolerancia de nombre)
# ================
with tabs[3]:
    st.subheader("Arista 4 – Capital / Provisiones")
    st.markdown(f"*Qué ves aquí:* {INTRO[3][0]}  \n{INTRO[3][1]}")
    st.markdown("""
*KPIs:*
- *Capital Requerido* (proxy *RW×K×EAD*).
- *Provisiones* ≈ *EL*.
- *Liberación* = Actual − Optimizado.
""")

    cap = dfs.get("cap_port")
    if cap is None or (isinstance(cap, pd.DataFrame) and cap.empty):
        st.error("No se encontró *capital_portfolio.csv* (o 'capital_portafolio.csv').")
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

    # Tablas con formato
    cap_seg = dfs.get("cap_seg")
    if isinstance(cap_seg, pd.DataFrame) and not cap_seg.empty:
        st.markdown("##### Capital por segmento")
        cs = cap_seg.copy()
        for c in ["capital_req_base","capital_req_opt","prov_base","prov_opt"]:
            if c in cs.columns:
                cs[c + "_fmt"] = cs[c].apply(lambda v: fmt_money(v, moneda, usdclp))
        show = ["segmento"] + [c+"_fmt" for c in ["capital_req_base","capital_req_opt","prov_base","prov_opt"] if c+"_fmt" in cs.columns]
        st.dataframe(cs[show], use_container_width=True)

    cap_det = dfs.get("cap_det")
    if isinstance(cap_det, pd.DataFrame) and not cap_det.empty:
        st.markdown("##### Detalle de capital (muestra)")
        cd = cap_det.copy()
        for c in ["capital_req_base","capital_req_opt","prov_base","prov_opt"]:
            if c in cd.columns:
                cd[c + "_fmt"] = cd[c].apply(lambda v: fmt_money(v, moneda, usdclp))
        keep = [c for c in ["id_cliente","segmento","capital_req_base_fmt","capital_req_opt_fmt","prov_base_fmt","prov_opt_fmt"] if c in cd.columns]
        st.dataframe(cd[keep].head(1000), use_container_width=True)

    st.markdown("---")
    st.caption("La optimización libera capital y estabiliza provisiones al reubicar exposición hacia perfiles con mejor retorno ajustado por riesgo.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("© MVP Bancario — Motor de Optimización (4 aristas integradas).")
