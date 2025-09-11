# app_dashboard.py
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ─────────────────────────────────────────────────────────────
# Config general
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="MVP Bancario – Arista 1 (Default/Impago)", layout="wide")

OUT_DIR = "out"  # ajusta si tu build usa otra carpeta para CSV
SCHEMA_FILE = os.path.join(OUT_DIR, "dashboard_schema.json")

# ─────────────────────────────────────────────────────────────
# Sidebar: Moneda y TC
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")
moneda = st.sidebar.radio("Moneda", ["CLP", "USD"], horizontal=True)
tc = st.sidebar.number_input(
    "Tipo de cambio (CLP por 1 USD)",
    min_value=1.0, value=900.0, step=1.0, format="%.2f",
    help="Se usa para convertir todos los montos del dashboard."
)

# ─────────────────────────────────────────────────────────────
# Utilidades (conversión y formato)
# ─────────────────────────────────────────────────────────────
def _to_money_numeric(x, moneda, tc):
    try:
        v = float(x)
    except:
        return np.nan
    return v / float(tc) if moneda == "USD" else v

def _fmt_money(x, moneda):
    try:
        v = float(x)
    except:
        return ""
    s = f"{int(round(v,0)):,}".replace(",", ".")
    return f"{s} {moneda}"

def _fmt_pct(x):
    try:
        v = float(x)
    except:
        return ""
    # admitimos tanto 0.12 como 12
    if 0 <= v <= 1:
        v *= 100
    return f"{v:.2f}%"

def _fmt_num(x):
    try:
        v = float(x)
    except:
        return ""
    return f"{v:.2f}"

@st.cache_data(show_spinner=False)
def load_schema(schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_df(csv_path):
    return pd.read_csv(csv_path)

def auto_prepare(df, meta, moneda, tc):
    """
    df: DataFrame original
    meta: dict con 'money', 'percent', 'numeric', 'categorical'
    return: (df_num, df_fmt)
      - df_num: agrega columnas *_cur (money convertidos a CLP/USD como número)
      - df_fmt: agrega columnas *_fmt (money/percent/numeric formateados como string)
    """
    df_num = df.copy()
    df_fmt = df.copy()

    for c in meta.get("money", []):
        if c in df_num.columns:
            cur_col = f"{c}_cur"
            fmt_col = f"{c}_fmt"
            df_num[cur_col] = df_num[c].apply(lambda x: _to_money_numeric(x, moneda, tc))
            df_fmt[fmt_col] = df_num[cur_col].apply(lambda x: _fmt_money(x, moneda))

    for c in meta.get("percent", []):
        if c in df_fmt.columns:
            df_fmt[f"{c}_fmt"] = df_fmt[c].apply(_fmt_pct)

    for c in meta.get("numeric", []):
        if c in df_fmt.columns:
            df_fmt[f"{c}_fmt"] = df_fmt[c].apply(_fmt_num)

    return df_num, df_fmt

def kpi_card(col, label, value, delta=None, help_text=None):
    with col:
        st.metric(label, value, delta=delta, help=help_text)

def safe_get(df, col_fmt, fallback_cols):
    """Devuelve df[col_fmt] si existe, si no, busca en fallback_cols sin formato."""
    if col_fmt in df.columns:
        return df[col_fmt]
    for c in fallback_cols:
        if c in df.columns:
            return df[c]
    return pd.Series([""] * len(df))

# ─────────────────────────────────────────────────────────────
# Cargar esquema y datasets
# ─────────────────────────────────────────────────────────────
st.title("📊 Arista 1 — Default/Impago")
st.caption("Comparación método actual vs. optimizado. Todo se adapta automáticamente a CLP/USD y formato a partir del esquema.")

if not os.path.exists(SCHEMA_FILE):
    st.error(f"No encuentro el esquema {SCHEMA_FILE}. Genera Celda 7.bis en el notebook.")
    st.stop()

schema = load_schema(SCHEMA_FILE)

# PORTAFOLIO
try:
    meta_port = schema["default_compare_portfolio"]
    df_port = load_df(meta_port["path"])
    port_num, port_fmt = auto_prepare(df_port, meta_port, moneda, tc)
except Exception as e:
    st.error(f"Error cargando portafolio: {e}")
    st.stop()

# SEGMENTOS
try:
    meta_seg = schema["default_compare_segment"]
    df_seg = load_df(meta_seg["path"])
    seg_num, seg_fmt = auto_prepare(df_seg, meta_seg, moneda, tc)
except Exception as e:
    st.error(f"Error cargando segmentos: {e}")
    st.stop()

# DETALLE
try:
    meta_det = schema["default_compare_detail"]
    df_det = load_df(meta_det["path"])
    det_num, det_fmt = auto_prepare(df_det, meta_det, moneda, tc)
except Exception as e:
    st.error(f"Error cargando detalle: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# KPIs — 3 filas como pediste
# ─────────────────────────────────────────────────────────────

# Fila 1: PD prom actual / PD prom optimizado / Reducción EL %
row1 = st.columns(3)
kpi_card(
    row1[0], "PD ponderado (Actual)",
    safe_get(port_fmt, "PD_pond_actual_fmt", ["PD_pond_actual"]).iloc[0]
)
kpi_card(
    row1[1], "PD ponderado (Optimizado)",
    safe_get(port_fmt, "PD_pond_optimizado_fmt", ["PD_pond_optimizado"]).iloc[0]
)
kpi_card(
    row1[2], "Reducción EL (%)",
    safe_get(port_fmt, "Reduccion_EL_pct_fmt", ["Reduccion_EL_pct"]).iloc[0]
)

# Fila 2: EL total actual / EL total optimizada / Reducción EL monto
row2 = st.columns(3)
kpi_card(
    row2[0], "EL Total (Actual)",
    safe_get(port_fmt, "EL_actual_fmt", ["EL_actual"]).iloc[0]
)
kpi_card(
    row2[1], "EL Total (Optimizado)",
    safe_get(port_fmt, "EL_optimizado_fmt", ["EL_optimizado"]).iloc[0]
)
kpi_card(
    row2[2], "Reducción EL (monto)",
    safe_get(port_fmt, "Reduccion_EL_monto_fmt", ["Reduccion_EL_monto"]).iloc[0]
)

# Fila 3: EAD total actual / EAD total optimizada / explicación EAD
row3 = st.columns([1,1,1.2])
kpi_card(
    row3[0], "EAD Total (Actual)",
    safe_get(port_fmt, "EAD_actual_fmt", ["EAD_actual"]).iloc[0]
)
kpi_card(
    row3[1], "EAD Total (Optimizado)",
    safe_get(port_fmt, "EAD_optimizado_fmt", ["EAD_optimizado"]).iloc[0]
)
with row3[2]:
    st.info(
        "ℹ️ *EAD (Exposure at Default)*: monto expuesto si el cliente cae en mora. "
        "Nuestro optimizador puede *reubicar EAD* (subir/bajar límites o asignación) entre segmentos/clientes para "
        "reducir EL = PD×LGD×EAD manteniendo rentabilidad. Más EAD *no implica* más pérdida si cae en perfiles "
        "con *menor PD×LGD* o mejor margen."
    )

st.divider()

# ─────────────────────────────────────────────────────────────
# Gráfico: EL Actual vs Optimizado
# ─────────────────────────────────────────────────────────────
try:
    el_act = float(port_num["EL_actual_cur"].iloc[0]) if "EL_actual_cur" in port_num else float(df_port["EL_actual"].iloc[0])
    el_opt = float(port_num["EL_optimizado_cur"].iloc[0]) if "EL_optimizado_cur" in port_num else float(df_port["EL_optimizado"].iloc[0])
    chart_df = pd.DataFrame({
        "Escenario": ["Actual","Optimizado"],
        "EL_cur": [el_act, el_opt]
    })
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Escenario:N", title=""),
            y=alt.Y("EL_cur:Q", title=f"Expected Loss ({moneda})"),
            tooltip=[alt.Tooltip("Escenario:N"), alt.Tooltip("EL_cur:Q", format=",.0f")]
        )
    )
    st.subheader("EL Actual vs Optimizado")
    st.altair_chart(chart, use_container_width=True)
except Exception as e:
    st.warning(f"No se pudo renderizar el gráfico EL Actual vs Optimizado: {e}")

st.divider()

# ─────────────────────────────────────────────────────────────
# Comparación por segmento
# ─────────────────────────────────────────────────────────────
st.subheader("Comparación por segmento")

# Uppercase de segmento + glosario al lado
seg_lay = st.columns([3,2])
seg_show = seg_fmt.copy()
if "segmento" in seg_show.columns:
    seg_show["segmento"] = seg_show["segmento"].astype(str).str.upper()

# Mapeo significado de segmentos
seg_gloss = {
    "MASS": "Clientes de banca masiva",
    "MASS_AFFLUENT": "Masivos con mayor capacidad/ingreso",
    "AFFLUENT": "Alta renta",
    "PYME": "Pequeñas y medianas empresas"
}

with seg_lay[0]:
    # Selección de columnas amigables
    cols = []
    for k in [
        ("EAD_actual","EAD_actual_fmt"),
        ("EAD_optimizado","EAD_optimizado_fmt"),
        ("EL_actual","EL_actual_fmt"),
        ("EL_optimizado","EL_optimizado_fmt"),
        ("Utilidad_actual","Utilidad_actual_fmt"),
        ("Utilidad_optimizada","Utilidad_optimizada_fmt"),
        ("Reduccion_EL_monto","Reduccion_EL_monto_fmt"),
        ("Reduccion_EL_pct","Reduccion_EL_pct_fmt"),
        ("PD_pond_actual","PD_pond_actual_fmt"),
        ("PD_pond_optimizado","PD_pond_optimizado_fmt"),
    ]:
        raw, fmt = k
        if fmt in seg_show.columns:
            seg_show[raw] = seg_show[fmt]
        # si no hay fmt, deja el raw original
        if raw in seg_fmt.columns and raw not in seg_show.columns:
            seg_show[raw] = seg_fmt[raw]

    show_cols = ["segmento","EAD_actual","EAD_optimizado","EL_actual","EL_optimizado",
                 "Utilidad_actual","Utilidad_optimizada","Reduccion_EL_monto","Reduccion_EL_pct",
                 "PD_pond_actual","PD_pond_optimizado"]
    show_cols = [c for c in show_cols if c in seg_show.columns]
    st.dataframe(seg_show[show_cols], use_container_width=True, height=360)

with seg_lay[1]:
    st.info(
        "*Significado de segmentos*\n\n" +
        "\n".join([f"- *{k}*: {v}" for k,v in seg_gloss.items()])
    )

st.divider()

# ─────────────────────────────────────────────────────────────
# Top segmentos por reducción de EL (gráfico)
# ─────────────────────────────────────────────────────────────
try:
    if {"segmento","Reduccion_EL_monto_cur"}.issubset(seg_num.columns):
        seg_red = seg_num[["segmento","Reduccion_EL_monto_cur"]].copy()
    else:
        seg_red = seg_fmt[["segmento","Reduccion_EL_monto"]].copy()
        if "segmento" in seg_red.columns:
            seg_red["Reduccion_EL_monto_cur"] = pd.to_numeric(seg_red["Reduccion_EL_monto"], errors="coerce")

    seg_red["segmento"] = seg_red["segmento"].astype(str).str.upper()
    seg_red = seg_red.sort_values("Reduccion_EL_monto_cur", ascending=False)
    seg_red = seg_red.head(15)

    ch2 = (
        alt.Chart(seg_red)
        .mark_bar()
        .encode(
            x=alt.X("Reduccion_EL_monto_cur:Q", title=f"Reducción EL ({moneda})"),
            y=alt.Y("segmento:N", sort="-x", title=""),
            tooltip=[alt.Tooltip("segmento:N"),
                     alt.Tooltip("Reduccion_EL_monto_cur:Q", format=",.0f")]
        )
    )
    st.subheader("Segmentos por Reducción de EL")
    st.altair_chart(ch2, use_container_width=True)
except Exception as e:
    st.warning(f"No se pudo renderizar el gráfico de segmentos: {e}")

st.divider()

# ─────────────────────────────────────────────────────────────
# Detalle por cliente + Glosario
# ─────────────────────────────────────────────────────────────
st.subheader("Detalle por cliente")
det_show = det_fmt.copy()

# Reemplazar por columnas formateadas si existen
relabels = {
    "ead_baseline": "ead_baseline_fmt",
    "e_opt": "e_opt_fmt",
    "EL_baseline": "EL_baseline_fmt",
    "el_opt": "el_opt_fmt",
    "ingreso_base": "ingreso_base_fmt",
    "ingreso_opt": "ingreso_opt_fmt",
    "cost_fin_base": "cost_fin_base_fmt",
    "cost_op_base": "cost_op_base_fmt",
    "cost_fin_opt": "cost_fin_opt_fmt",
    "cost_op_opt": "cost_op_opt_fmt",
    "utilidad_base": "utilidad_base_fmt",
    "utilidad_opt": "utilidad_opt_fmt",
    "pd_score": "pd_score_fmt",
    "lgd_pred": "lgd_pred_fmt",
    "apr_efectiva": "apr_efectiva_fmt",
    "r_opt": "r_opt_fmt",
}

for raw, fmt in relabels.items():
    if fmt in det_show.columns:
        det_show[raw] = det_show[fmt]

show_cols_det = [
    "id_cliente","segmento",
    "ead_baseline","e_opt",
    "EL_baseline","el_opt",
    "ingreso_base","ingreso_opt",
    "cost_fin_base","cost_op_base",
    "cost_fin_opt","cost_op_opt",
    "utilidad_base","utilidad_opt",
    "pd_score","lgd_pred","apr_efectiva","r_opt"
]
show_cols_det = [c for c in show_cols_det if c in det_show.columns]
st.dataframe(det_show[show_cols_det], use_container_width=True, height=420)

with st.expander("📘 Glosario de columnas"):
    st.markdown("""
- *EAD: Exposición en caso de Incumplimiento (*Exposure at Default).  
- *PD: Probabilidad de Default (*Probability of Default).  
- *LGD: Pérdida dado Default (*Loss Given Default).  
- *EL*: Pérdida Esperada = PD × LGD × EAD.  
- *APR*: Tasa anual efectiva de interés (pricing).  
- *Utilidad*: Ingreso financiero – EL – Costos (financieros + operativos).  
- *\_base**: Escenario actual (sin optimización).  
- *\_opt**: Escenario optimizado por el modelo.  
    """)

st.divider()

# ─────────────────────────────────────────────────────────────
# Panel: ¿Por qué nuestro método es mejor?
# ─────────────────────────────────────────────────────────────
st.subheader("¿Por qué nuestro método es mejor que el actual?")
st.markdown("""
*Actual (reglas estáticas)*  
- Límites y pricing iguales por segmentos amplios.  
- *EAD* mal distribuida entre perfiles con distinta *PD* (Probability of Default) y *LGD* (Loss Given Default).  
- Se pierde margen por no ajustar *APR* a riesgo.  

*Nuestro método (optimización + guardarraíles)*  
- Reasigna *EAD* hacia clientes/segmentos con menor *PD×LGD* manteniendo ingresos, reduciendo *EL* (Expected Loss).  
- Ajusta *APR* (pricing) dentro de bandas y con restricciones de negocio.  
- Mide impacto en *Utilidad, **EL, **PD ponderada* y capital/provisiones.  
""")
