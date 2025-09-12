import os
import json
import math
import pandas as pd
import streamlit as st

# =========================
# Utilidades de formato
# =========================
def fmt_number(x, nd=2):
    """Formatea con miles por punto y decimales con coma."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        s = f"{float(x):,.{nd}f}"  # 1,234,567.89
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return s
    except Exception:
        return "—"

def fmt_money(x, moneda="CLP", nd=0):
    suf = " CLP" if moneda.upper()=="CLP" else " USD"
    return fmt_number(x, nd) + suf

def fmt_pct(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return fmt_number(x, nd) + " %"
    except Exception:
        return "—"

def safe_pct_var(opt, act):
    try:
        act = float(act); opt = float(opt)
        if act == 0:
            return float("nan")
        return (opt - act) / act * 100.0
    except Exception:
        return float("nan")

def money_convert(val, moneda, usdclp):
    """Convierte montos a la moneda elegida (dataset base en CLP)."""
    try:
        v = float(val)
    except Exception:
        return float("nan")
    if moneda.upper() == "USD":
        return v / float(usdclp)
    return v

def load_csv_candidates(candidates):
    """Devuelve el primer CSV existente de la lista de rutas."""
    for p in candidates:
        if p and os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

# =========================
# Setup y carga de datos
# =========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

OUT_DIR   = os.environ.get("OUT_DIR", "/content/mvp-tarjetas-chile/out")
BUNDLE    = os.path.join(OUT_DIR, "dashboard_bundle")

# Candidatos por arista (canónicos + alias)
paths = {
    "def_port": [
        os.path.join(BUNDLE,"default_portfolio.csv"),
        os.path.join(OUT_DIR,"default_portfolio.csv"),
        os.path.join(OUT_DIR,"default_compare_portfolio.csv"),
    ],
    "yield_port": [
        os.path.join(BUNDLE,"yield_portfolio.csv"),
        os.path.join(OUT_DIR,"yield_portfolio.csv"),
        os.path.join(OUT_DIR,"yield_compare_portfolio.csv"),
    ],
    "cap_port": [
        os.path.join(BUNDLE,"capital_portfolio.csv"),
        os.path.join(OUT_DIR,"capital_portfolio.csv"),
        os.path.join(OUT_DIR,"capital_portafolio.csv"),
        os.path.join(OUT_DIR,"capital_compare_portfolio.csv"),
        os.path.join(OUT_DIR,"capital_compare_portafolio.csv"),
    ],
    "inc_det": [
        os.path.join(BUNDLE,"incentives_detail.csv"),
        os.path.join(OUT_DIR,"incentives_detail.csv"),
    ],
    "inc_sum": [
        os.path.join(BUNDLE,"incentives_diag_summary.csv"),
        os.path.join(OUT_DIR,"incentives_diag_summary.csv"),
    ],
}

df_def_port   = load_csv_candidates(paths["def_port"])
df_yield_port = load_csv_candidates(paths["yield_port"])
df_cap_port   = load_csv_candidates(paths["cap_port"])
df_inc_det    = load_csv_candidates(paths["inc_det"])
df_inc_sum    = load_csv_candidates(paths["inc_sum"])

# =========================
# Sidebar — Moneda
# =========================
st.sidebar.title("⚙️ Configuración")
moneda = st.sidebar.radio("Moneda", ["CLP", "USD"], index=0, horizontal=True)
usdclp = st.sidebar.number_input("1 USD equivale a (CLP)", min_value=1.0, value=900.0, step=1.0)
st.sidebar.caption("Todos los montos del dashboard se convierten usando este tipo de cambio.")

# =========================
# Encabezado
# =========================
st.title("📊 MVP Bancario — 4 Aristas")
st.caption("Visual ejecutivo con comparación *Actual vs Optimizaciones* por arista y VAR %.")

# =========================
# Helpers KPI blocks
# =========================
def kpi_triplet(label, actual, optim):
    c1, c2, c3 = st.columns([1,1,1])
    with c1: st.metric(label=label+" • Actual", value=actual)
    with c2: st.metric(label=label+" • Optimizado", value=optim)
    var_pct = "—"
    try:
        # extraer flotantes del texto (antes del sufijo)
        a = float(str(actual).split(" ")[0].replace(".","").replace(",",".")) if isinstance(actual,str) else None
        o = float(str(optim).split(" ")[0].replace(".","").replace(",",".")) if isinstance(optim,str) else None
        if a is not None and o is not None and a != 0:
            var_pct = fmt_pct((o-a)/a*100.0)
    except Exception:
        pass
    with c3: st.metric(label=label+" • VAR %", value=var_pct)

def explain_block(text):
    st.markdown(
        f"""
<div style="background:#0f172a10;border-left:4px solid #0ea5e9;padding:12px 12px;border-radius:8px;">
{text}
</div>
""", unsafe_allow_html=True)

def defs_table(rows):
    df = pd.DataFrame(rows, columns=["KPI", "Definición"])
    st.table(df)

# =========================
# Tab layout (4 aristas)
# =========================
tabs = st.tabs(["Arista 1: Default/Impago", "Arista 2: Yield/Pricing", "Arista 3: Incentivos", "Arista 4: Capital/Provisiones"])

# -------------------------
# Arista 1 — Default
# -------------------------
with tabs[0]:
    st.subheader("Qué vas a ver")
    explain_block(
        "Impacto de la optimización en *riesgo de crédito*: EAD, Expected Loss (EL), "
        "ingresos y *utilidad neta* del portafolio."
    )

    st.markdown("*Definiciones rápidas*")
    defs_table([
        ("EAD", "Exposure at Default (exposición en caso de impago)."),
        ("EL", "Expected Loss = PD × LGD × EAD."),
        ("Utilidad", "Ingreso financiero − EL − Costos (financieros y operativos)."),
        ("PD ponderado", "Promedio ponderado de PD por EAD del portafolio."),
    ])

    if df_def_port is None or df_def_port.empty:
        st.warning("No se encontró default_portfolio.csv. Genera primero el bundle (Celda 12/16).")
    else:
        r = df_def_port.iloc[0].copy()

        # convertir montos a la moneda elegida
        for col in ["EAD_actual","EAD_optimizado","EL_actual","EL_optimizado",
                    "Ingreso_actual","Ingreso_optimizado","Costos_actual","Costos_optimizado",
                    "Utilidad_actual","Utilidad_optimizada","Reduccion_EL_monto","Delta_Utilidad"]:
            if col in r:
                r[col] = money_convert(r[col], moneda, usdclp)

        # KPIs
        kpi_triplet("EAD", fmt_money(r.get("EAD_actual"), moneda), fmt_money(r.get("EAD_optimizado"), moneda))
        kpi_triplet("Expected Loss", fmt_money(r.get("EL_actual"), moneda), fmt_money(r.get("EL_optimizado"), moneda))
        kpi_triplet("Ingreso", fmt_money(r.get("Ingreso_actual"), moneda), fmt_money(r.get("Ingreso_optimizado"), moneda))
        kpi_triplet("Costos", fmt_money(r.get("Costos_actual"), moneda), fmt_money(r.get("Costos_optimizado"), moneda))
        kpi_triplet("Utilidad", fmt_money(r.get("Utilidad_actual"), moneda), fmt_money(r.get("Utilidad_optimizada"), moneda))

        # Boxes de insight
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"*Reducción EL (monto):* {fmt_money(r.get('Reduccion_EL_monto'), moneda)}")
        with c2:
            st.info(f"*Reducción EL (%):* {fmt_pct(r.get('Reduccion_EL_pct'))}")
        with c3:
            st.info(f"*Δ Utilidad:* {fmt_money(r.get('Delta_Utilidad'), moneda)}")

    st.markdown("*¿Por qué mejora?*")
    explain_block(
        "El motor re-asigna exposición y tasa dentro de *bandas gobernadas* por reglas del banco, "
        "reduciendo EL (via EAD y composición) sin sacrificar ingreso, y priorizando clientes "
        "con mejor relación *retorno/riesgo. Esto incrementa la **utilidad total*."
    )

# -------------------------
# Arista 2 — Yield/Pricing
# -------------------------
with tabs[1]:
    st.subheader("Qué vas a ver")
    explain_block(
        "Efecto del *pricing por cliente* en ingreso y utilidad. Separamos el impacto de precio puro "
        "(manteniendo EAD constante) del impacto combinado (precio + cambio de EAD)."
    )

    st.markdown("*Definiciones rápidas*")
    defs_table([
        ("Ingreso", "Tasa (APR) × EAD."),
        ("Utilidad", "Ingreso − EL − Costos."),
        ("Δ Pricing puro", "Variación de ingreso/utilidad cambiando solo la tasa (EAD fijo)."),
    ])

    if df_yield_port is None or df_yield_port.empty:
        st.warning("No se encontró yield_portfolio.csv. Genera primero el bundle (Celda 12/16).")
    else:
        r = df_yield_port.iloc[0].copy()

        # algunos bundles usan nombres 'ingreso_base'/'ingreso_opt'… homogenizamos
        # intentamos leer claves esperadas; si no están, probamos aliases
        def alias(row, main, alts):
            if main in row: return row.get(main)
            for a in alts:
                if a in row: return row.get(a)
            return float("nan")

        act_ing = alias(r, "ingreso_base", ["Ingreso_actual"])
        opt_ing = alias(r, "ingreso_opt",  ["Ingreso_optimizado"])
        act_util= alias(r, "utilidad_base",["Utilidad_actual"])
        opt_util= alias(r, "utilidad_opt", ["Utilidad_optimizada"])

        # conversión moneda
        vals = {
            "EAD_actual": alias(r,"EAD_actual",["ead_baseline"]),
            "EAD_optim": alias(r,"EAD_optimizado",["EAD_optimizado"]),
            "Ingreso_actual": act_ing,
            "Ingreso_opt": opt_ing,
            "Utilidad_actual": act_util,
            "Utilidad_opt": opt_util
        }
        for k in list(vals.keys()):
            vals[k] = money_convert(vals[k], moneda, usdclp)

        kpi_triplet("EAD", fmt_money(vals["EAD_actual"], moneda), fmt_money(vals["EAD_optim"], moneda))
        kpi_triplet("Ingreso", fmt_money(vals["Ingreso_actual"], moneda), fmt_money(vals["Ingreso_opt"], moneda))
        kpi_triplet("Utilidad", fmt_money(vals["Utilidad_actual"], moneda), fmt_money(vals["Utilidad_opt"], moneda))

        # deltas si existen
        d_income_pr = r.get("delta_ingreso_prc")
        d_income_tt = r.get("delta_ingreso_total")
        d_util_pr   = r.get("delta_util_prc")
        d_util_tt   = r.get("delta_util_total")

        c1, c2 = st.columns(2)
        with c1:
            if pd.notna(d_income_pr):
                st.info(f"*Δ Ingreso (solo pricing):* {fmt_money(money_convert(d_income_pr, moneda, usdclp), moneda)}")
            if pd.notna(d_util_pr):
                st.info(f"*Δ Utilidad (solo pricing):* {fmt_money(money_convert(d_util_pr, moneda, usdclp), moneda)}")
        with c2:
            if pd.notna(d_income_tt):
                st.info(f"*Δ Ingreso (total):* {fmt_money(money_convert(d_income_tt, moneda, usdclp), moneda)}")
            if pd.notna(d_util_tt):
                st.info(f"*Δ Utilidad (total):* {fmt_money(money_convert(d_util_tt, moneda, usdclp), moneda)}")

    st.markdown("*¿Por qué mejora?*")
    explain_block(
        "El pricing se ajusta por *riesgo individual* y *elasticidad estimada*: subimos tasas donde el cliente mantiene uso, "
        "y ofrecemos condiciones dentro de bandas donde la elasticidad mejora el volumen, maximizando *ingreso marginal* y *utilidad*."
    )

# -------------------------
# Arista 3 — Incentivos
# -------------------------
with tabs[2]:
    st.subheader("Qué vas a ver")
    explain_block(
        "Eficiencia de *beneficios/campañas* a nivel cliente: cuánto ingreso incremental generan por cada unidad de costo, "
        "y cómo priorizar para maximizar ROI."
    )

    st.markdown("*Definiciones rápidas*")
    defs_table([
        ("INC", "Ingreso incremental atribuible al incentivo."),
        ("Costo incentivo", "Costo monetizado del beneficio/acción."),
        ("ROI", "INC / Costo incentivo."),
    ])

    if (df_inc_sum is None or df_inc_sum.empty) and (df_inc_det is None or df_inc_det.empty):
        st.warning("No se encontraron archivos de incentivos. Genera incentives_detail.csv o incentives_diag_summary.csv.")
    else:
        # Si tenemos resumen, úsalo; si no, construimos uno desde el detalle
        if df_inc_sum is not None and not df_inc_sum.empty:
            inc = df_inc_sum.copy()
            # normalizamos nombres posibles
            inc_cols = {c.lower(): c for c in inc.columns}
            inc_monto = inc[inc_cols.get("ingreso_incremental","ingreso_incremental")] if "ingreso_incremental" in inc_cols else None
            costo     = inc[inc_cols.get("costo_incentivo","costo_incentivo")] if "costo_incentivo" in inc_cols else None
            # fallback si faltan columnas
            if inc_monto is None or costo is None:
                if df_inc_det is not None and not df_inc_det.empty:
                    det = df_inc_det.copy()
                    inc = pd.DataFrame({
                        "ingreso_incremental":[det.get("ingreso_incremental", det.get("INC", pd.Series([0]))).sum()],
                        "costo_incentivo":[det.get("costo_incentivo", pd.Series([0])).sum()]
                    })
                else:
                    inc = pd.DataFrame({"ingreso_incremental":[0.0], "costo_incentivo":[0.0]})
        else:
            det = df_inc_det.copy()
            inc = pd.DataFrame({
                "ingreso_incremental":[det.get("ingreso_incremental", det.get("INC", pd.Series([0]))).sum()],
                "costo_incentivo":[det.get("costo_incentivo", pd.Series([0])).sum()]
            })

        inc_total = float(inc["ingreso_incremental"].sum()) if "ingreso_incremental" in inc else 0.0
        costo_tot = float(inc["costo_incentivo"].sum()) if "costo_incentivo" in inc else 0.0
        roi = inc_total / costo_tot if costo_tot > 0 else float("nan")

        kpi_triplet("Ingreso Incremental (INC)",
                    fmt_money(money_convert(inc_total, moneda, usdclp), moneda),
                    fmt_money(money_convert(inc_total, moneda, usdclp), moneda))
        kpi_triplet("Costo Incentivo",
                    fmt_money(money_convert(costo_tot, moneda, usdclp), moneda),
                    fmt_money(money_convert(costo_tot, moneda, usdclp), moneda))
        c1, c2, _ = st.columns([1,1,1])
        with c1:
            st.metric("ROI (INC / Costo)", value=fmt_number(roi, 2))

    st.markdown("*¿Por qué mejora?*")
    explain_block(
        "Priorizamos incentivos por *ROI esperado* usando elasticidades y propensión. "
        "El banco invierte donde el *ING incremental por peso* es mayor, reduciendo desperdicio y elevando *rentabilidad*."
    )

# -------------------------
# Arista 4 — Capital/Provisiones
# -------------------------
with tabs[3]:
    st.subheader("Qué vas a ver")
    explain_block(
        "Efecto de la optimización en *capital requerido* (proxy Basilea/IFRS) y *provisiones* (derivadas del EL). "
        "La liberación de capital/provisiones mejora el *ROE* y habilita crecimiento."
    )

    st.markdown("*Definiciones rápidas*")
    defs_table([
        ("Capital requerido", "Proxy: EAD × RW × K (factores regulat. aproximados)."),
        ("Provisiones", "Basadas en EL (PD×LGD×EAD)."),
        ("Liberación", "Diferencia entre actual y optimizado: monto y %."),
    ])

    if df_cap_port is None or df_cap_port.empty:
        st.warning("No se encontró capital_portfolio.csv (o capital_portafolio.csv). Genera Capital (Celda 10) y re-bundle (Celda 12/16).")
    else:
        r = df_cap_port.iloc[0].copy()
        # nombres esperados (normalizamos por si hay alias)
        capital_act = r.get("capital_req_base", r.get("Capital_actual", float("nan")))
        capital_opt = r.get("capital_req_opt",  r.get("Capital_optimizado", float("nan")))
        prov_act    = r.get("prov_base",       r.get("Provisiones_actual", float("nan")))
        prov_opt    = r.get("prov_opt",        r.get("Provisiones_optimizado", float("nan")))
        cap_lib_m   = r.get("capital_lib_monto", r.get("Capital_liberado_monto", float("nan")))
        cap_lib_p   = r.get("capital_lib_pct",   r.get("Capital_liberado_pct", float("nan")))
        prov_lib_m  = r.get("prov_lib_monto",    r.get("Provisiones_liberado_monto", float("nan")))
        prov_lib_p  = r.get("prov_lib_pct",      r.get("Provisiones_liberado_pct", float("nan")))

        # conversión
        capital_act = money_convert(capital_act, moneda, usdclp)
        capital_opt = money_convert(capital_opt, moneda, usdclp)
        prov_act    = money_convert(prov_act, moneda, usdclp)
        prov_opt    = money_convert(prov_opt, moneda, usdclp)
        cap_lib_m   = money_convert(cap_lib_m, moneda, usdclp)
        prov_lib_m  = money_convert(prov_lib_m, moneda, usdclp)

        kpi_triplet("Capital requerido", fmt_money(capital_act, moneda), fmt_money(capital_opt, moneda))
        kpi_triplet("Provisiones", fmt_money(prov_act, moneda), fmt_money(prov_opt, moneda))

        c1, c2 = st.columns(2)
        with c1:
            st.info(f"*Capital liberado:* {fmt_money(cap_lib_m, moneda)} · {fmt_pct(cap_lib_p)}")
        with c2:
            st.info(f"*Provisiones liberadas:* {fmt_money(prov_lib_m, moneda)} · {fmt_pct(prov_lib_p)}")

    st.markdown("*¿Por qué mejora?*")
    explain_block(
        "Al reconfigurar EAD y precios con restricciones, el modelo reduce *EL* y el *capital absorbido*. "
        "Esto libera recursos y mejora el *ROE*, manteniendo cumplimiento bajo reglas y auditoría."
    )
