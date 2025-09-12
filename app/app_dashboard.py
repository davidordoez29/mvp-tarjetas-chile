# app/app_dashboard.py
import json
import os
import math
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime

# =========================
# Configuración general
# =========================
st.set_page_config(page_title="MVP Bancario – 4 Aristas", layout="wide")

# Ubicación por defecto del bundle de CSV
DEFAULT_BUNDLE_DIR = os.environ.get("BUNDLE_DIR", "out/dashboard_bundle")

# =========================
# Utilidades de formato
# =========================
def _swap_commas(s: str) -> str:
    # Cambia "1,234,567.89" -> "1.234.567,89"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_money(x: float, currency: str = "CLP", decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return f"0 {currency}"
    try:
        s = f"{x:,.{decimals}f}"
        s = _swap_commas(s)
        return f"{s} {currency}"
    except Exception:
        return f"0 {currency}"

def fmt_pct(x: float, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "0,00 %"
    try:
        s = f"{x*100:,.{decimals}f}"
        s = _swap_commas(s)
        return f"{s} %"
    except Exception:
        return "0,00 %"

def fmt_rate(x: float, decimals: int = 2) -> str:
    # para tasas mostradas como 0,12 -> 12,00 %
    return fmt_pct(x, decimals=decimals)

def safe_num(s, default=0.0):
    try:
        return pd.to_numeric(s, errors="coerce").fillna(default)
    except Exception:
        return default

# =========================
# Sidebar: controles globales
# =========================
with st.sidebar:
    st.header("⚙️ Configuración")
    bundle_dir = st.text_input("Carpeta con CSV (bundle):", value=DEFAULT_BUNDLE_DIR)
    moneda = st.selectbox("Moneda de visualización", options=["CLP", "USD"], index=0)
    usdclp = st.number_input("Tipo de cambio (1 USD = X CLP)", min_value=1.0, value=900.0, step=1.0)
    st.caption("Todos los montos se convertirán automáticamente dependiendo de la moneda.")

def to_view_currency(series: pd.Series, moneda: str, usdclp: float) -> pd.Series:
    if moneda.upper() == "CLP":
        return series
    else:  # USD
        return series / max(usdclp, 1e-9)

# =========================
# Carga de insumos del bundle
# =========================
def load_csv(name):
    p = Path(bundle_dir) / name
    if p.exists():
        return pd.read_csv(p)
    return None

def load_json(name):
    p = Path(bundle_dir) / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}

port_default = load_csv("default_portfolio.csv")
seg_default  = load_csv("default_segment.csv")
det_default  = load_csv("default_detail.csv")

port_yield = load_csv("yield_portfolio.csv")
seg_yield  = load_csv("yield_segment.csv")
det_yield  = load_csv("yield_detail.csv")
curve_yield = load_csv("yield_curve_segment.csv")

port_cap = load_csv("capital_portafolio.csv")  or load_csv("capital_portfolio.csv")
seg_cap  = load_csv("capital_segment.csv")
det_cap  = load_csv("capital_detail.csv")

det_inc  = load_csv("incentives_detail.csv")
sum_inc  = load_csv("incentives_diag_summary.csv")
sens_inc = load_csv("incentives_sensitivity.csv")

kpi_defs = load_json("kpi_defs.json")
seg_defs = load_json("segment_defs.json")
meta     = load_json("bundle_meta.json")

# =========================
# Header
# =========================
st.title("📊 MVP Bancario – Optimización de Rentabilidad")
st.caption("4 aristas integradas: (1) Default/Impago, (2) Yield/Pricing, (3) Incentivos, (4) Capital/Provisiones")

if meta:
    gen = meta.get("generado_en") or datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"Bundle: *{bundle_dir}* | Generado: *{gen}* | Versión: *{meta.get('version','N/A')}*")

# =========================
# Tarjeta KPI helper
# =========================
def kpi_triplet(title: str, actual: float, opt: float, currency: str = "CLP", usdclp: float = 900.0, pct=False, money=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"{title} (Actual)", fmt_pct(actual) if pct else fmt_money(to_view_currency(pd.Series([actual]), currency, usdclp).iloc[0], currency))
    with c2:
        st.metric(f"{title} (Optimizado)", fmt_pct(opt) if pct else fmt_money(to_view_currency(pd.Series([opt]), currency, usdclp).iloc[0], currency))
    with c3:
        delta = None
        if actual not in (0, None) and isinstance(actual, (int, float)) and not math.isclose(actual, 0.0):
            delta = (opt - actual) / actual
        else:
            # si no hay base, mostramos la diferencia absoluta con signo
            delta = None
        if pct:
            st.metric("Δ VAR (%)", fmt_pct((opt - actual)) if actual == 0 else fmt_pct((opt - actual) / actual))
        else:
            # variación en %
            var_pct = (opt - actual) / actual if actual else 0.0
            st.metric("Δ VAR (%)", fmt_pct(var_pct))

# =========================
# Pestañas
# =========================
tabs = st.tabs([
    "Arista 1 — Default/Impago",
    "Arista 2 — Yield / Pricing",
    "Arista 3 — Incentivos",
    "Arista 4 — Capital / Provisiones"
])

# =========================
# Arista 1 — Default/Impago
# =========================
with tabs[0]:
    st.subheader("Arista 1 — Default/Impago")
    st.write(
        "Esta vista compara el *riesgo esperado de pérdida (EL), el **EAD* y la *utilidad* "
        "entre el método actual y el optimizado. Los KPIs se muestran agregados a nivel portafolio."
    )
    st.info(kpi_defs.get("default", "EL = PD × LGD × EAD. La utilidad es Ingreso – (EL + Costos)."))

    if port_default is None:
        st.warning("No encontré default_portfolio.csv en el bundle.")
    else:
        row = port_default.iloc[0].copy()
        # Convertir a moneda
        ead_act = to_view_currency(row["EAD_actual"], moneda, usdclp)
        ead_opt = to_view_currency(row["EAD_optimizado"], moneda, usdclp)
        el_act  = to_view_currency(row["EL_actual"], moneda, usdclp)
        el_opt  = to_view_currency(row["EL_optimizado"], moneda, usdclp)
        uti_act = to_view_currency(row["Utilidad_actual"], moneda, usdclp)
        uti_opt = to_view_currency(row["Utilidad_optimizada"], moneda, usdclp)

        st.markdown("*KPIs de Portafolio*")
        kpi_triplet("EAD", ead_act, ead_opt, currency=moneda, usdclp=usdclp, pct=False)
        kpi_triplet("Expected Loss (EL)", el_act, el_opt, currency=moneda, usdclp=usdclp, pct=False)
        kpi_triplet("Utilidad", uti_act, uti_opt, currency=moneda, usdclp=usdclp, pct=False)

        # PD ponderado (formateado %)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("PD ponderado (Actual)", fmt_rate(row.get("PD_pond_actual", 0.0)))
        with c2:
            st.metric("PD ponderado (Optimizado)", fmt_rate(row.get("PD_pond_optimizado", 0.0)))

        st.markdown("---")
        st.markdown("*Segmentos (ordenado por reducción de EL)*")
        if seg_default is not None and not seg_default.empty:
            df = seg_default.copy()
            # Conversiones de moneda para columnas monetarias
            for col in ["EAD_actual","EAD_optimizado","EL_actual","EL_optimizado","Ingreso_actual","Ingreso_optimizado","Costos_actual","Costos_optimizado","Utilidad_actual","Utilidad_optimizada","Reduccion_EL_monto"]:
                if col in df.columns:
                    df[col] = to_view_currency(df[col], moneda, usdclp)
            st.dataframe(df)
        else:
            st.caption("No hay segmentación para mostrar.")

    st.markdown("---")
    st.success("*¿Por qué nuestro modelo es mejor?* Integramos PD/LGD/EAD con restricciones por segmento y topes de exposición para "
               "maximizar utilidad *reduciendo EL sin sacrificar ingreso*, y manteniendo gobernanza (bandas, límites, champion–challenger).")

# =========================
# Arista 2 — Yield / Pricing
# =========================
with tabs[1]:
    st.subheader("Arista 2 — Yield / Pricing")
    st.write(
        "Comparamos ingreso y utilidad con el precio optimizado. Además, aislamos el efecto *solo pricing* manteniendo EAD constante, "
        "para mostrar el impacto puro del cambio de tasa."
    )
    st.info(kpi_defs.get("yield", "Yield = r × EAD. Mostramos Δ por precio y Δ total (precio + EAD)."))

    if port_yield is None:
        st.warning("No encontré yield_portfolio.csv.")
    else:
        row = port_yield.iloc[0].copy()
        # Totales
        uti_b = to_view_currency(row.get("utilidad_base", 0.0), moneda, usdclp)
        uti_iso = to_view_currency(row.get("utilidad_iso", 0.0), moneda, usdclp)
        uti_opt = to_view_currency(row.get("utilidad_opt", 0.0), moneda, usdclp)

        st.markdown("*KPIs de Portafolio (Utilidad)*")
        kpi_triplet("Utilidad (Total)", uti_b, uti_opt, currency=moneda, usdclp=usdclp, pct=False)

        st.markdown("*Efecto Solo Pricing (EAD fijo)*")
        kpi_triplet("Utilidad (Solo pricing)", uti_b, uti_iso, currency=moneda, usdclp=usdclp, pct=False)

        st.markdown("---")
        st.markdown("*Segmentos (Δ Utilidad)*")
        if seg_yield is not None and not seg_yield.empty:
            df = seg_yield.copy()
            for col in ["utilidad_base","utilidad_iso","utilidad_opt","delta_util_prc","delta_util_total",
                        "ingreso_base","ingreso_iso","ingreso_opt","EL_baseline","el_iso","el_opt",
                        "cost_fin_base","cost_op_base","cost_fin_iso","cost_op_iso","cost_fin_opt","cost_op_opt",
                        "EAD_actual","EAD_optimizado"]:
                if col in df.columns:
                    df[col] = to_view_currency(df[col], moneda, usdclp)
            st.dataframe(df.sort_values("delta_util_total", ascending=False))
        else:
            st.caption("No hay segmentación de yield para mostrar.")

    st.markdown("---")
    st.success("*¿Por qué nuestro modelo es mejor?* Aprende elasticidades implícitas y respeta bandas/reglas, "
               "permitiendo subir tasas donde el riesgo/competencia lo permite y bajarlas donde maximiza margen neto.")

# =========================
# Arista 3 — Incentivos
# =========================
with tabs[2]:
    st.subheader("Arista 3 — Incentivos")
    st.write("Mostramos una vista de *ROI por incentivo* y diagnósticos de sensibilidad (placeholder listo para datos reales).")
    st.info(kpi_defs.get("incentives", "Comparar costo de incentivo vs. ingreso incremental (uplift) → ROI."))

    if sum_inc is not None:
        st.markdown("*Resumen diagnóstico*")
        st.dataframe(sum_inc)
    else:
        st.caption("No se encontró incentives_diag_summary.csv.")

    st.markdown("---")
    if sens_inc is not None:
        st.markdown("*Sensibilidad (placeholder)*")
        st.dataframe(sens_inc.head(50))
    else:
        st.caption("No se encontró incentives_sensitivity.csv.")

    st.markdown("---")
    st.success("*¿Por qué nuestro modelo es mejor?* Enlaza incentivos a elasticidad y rentabilidad real, "
               "evitando promociones que canibalizan margen sin generar valor.")

# =========================
# Arista 4 — Capital / Provisiones
# =========================
with tabs[3]:
    st.subheader("Arista 4 — Capital / Provisiones")
    st.write(
        "Calculamos proxies de *capital requerido* (Basilea/IFRS9 simplificado) y *provisiones*, comparando actual vs. optimizado. "
        "Esto permite evidenciar capital liberado y su impacto en ROE."
    )
    st.info(kpi_defs.get("capital", "Capital requerido ≈ RW × K × EAD; Provisiones ≈ EL."))

    if port_cap is None:
        st.warning("No encontré capital_portafolio.csv (o capital_portfolio.csv).")
    else:
        row = port_cap.iloc[0].copy()

        cap_b = to_view_currency(row.get("capital_req_base", 0.0), moneda, usdclp)
        cap_o = to_view_currency(row.get("capital_req_opt", 0.0), moneda, usdclp)
        kpi_triplet("Capital requerido", cap_b, cap_o, currency=moneda, usdclp=usdclp, pct=False)

        prov_b = to_view_currency(row.get("prov_base", 0.0), moneda, usdclp)
        prov_o = to_view_currency(row.get("prov_opt", 0.0), moneda, usdclp)
        kpi_triplet("Provisiones", prov_b, prov_o, currency=moneda, usdclp=usdclp, pct=False)

        st.markdown("---")
        if seg_cap is not None and not seg_cap.empty:
            df = seg_cap.copy()
            for col in ["capital_req_base","capital_req_opt","prov_base","prov_opt","capital_lib_monto","prov_lib_monto"]:
                if col in df.columns:
                    df[col] = to_view_currency(df[col], moneda, usdclp)
            st.dataframe(df)
        else:
            st.caption("No se encontró segmentación de capital/provisiones.")

    st.markdown("---")
    st.success("*¿Por qué nuestro modelo es mejor?* Optimiza con mirada de consumo de capital y provisiones, "
               "no sólo P&L corriente. Muestra *capital liberado* y su potencial de reasignación rentable.")
