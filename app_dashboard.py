# app_dashboard.py — Dashboard ejecutivo (Streamlit) con formato CLP/USD y %
import os, glob
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# Reglas de formato global
# =========================
def _swap_locale(s: str) -> str:
    # '1,234,567.89' -> '1.234.567,89'
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_num(x, decimals=2):
    """Número con miles y 'decimals' decimales; ej: 1112459.25 -> 1.112.459,25"""
    try:
        x = float(x)
        s = f"{x:,.{decimals}f}"
        return _swap_locale(s)
    except Exception:
        return str(x)

def fmt_money(x, currency="CLP"):
    """Moneda CLP/USD sin decimales; ej: 1112459 -> '1.112.459 CLP'."""
    try:
        x = float(x)
    except:
        return f"{x} {currency.upper()}"
    s = f"{x:,.0f}"  # 0 decimales
    return f"{_swap_locale(s)} {currency.upper()}"

def fmt_pct(x, decimals=2):
    """Porcentaje con 2 decimales; 0.1234 -> '12,34%'."""
    try:
        return f"{fmt_num(100*float(x), decimals)}%"
    except Exception:
        return str(x)

def to_currency(series, currency="CLP", fx=900.0):
    """Convierte serie a la moneda seleccionada (USD divide por fx)."""
    if currency.upper() == "USD":
        return series.astype(float) / float(fx)
    return series.astype(float)

def format_df_for_display(df, moneda, cols_money=(), cols_pct=(), cols_int2=()):
    """Devuelve copia con columnas formateadas como string según reglas."""
    dff = df.copy()
    for c in cols_money:
        if c in dff.columns:
            dff[c] = dff[c].apply(lambda v: fmt_money(v, moneda))
    for c in cols_pct:
        if c in dff.columns:
            dff[c] = dff[c].apply(lambda v: fmt_pct(v, 2))
    for c in cols_int2:
        if c in dff.columns:
            dff[c] = dff[c].apply(lambda v: fmt_num(v, 2))  # enteros con 2 decimales
    return dff

# =========================
# Configuración Streamlit
# =========================
st.set_page_config(page_title="MVP Bancario — Dashboard", layout="wide")
st.title("MVP Bancario — Dashboard Ejecutivo")
st.caption("Piso 4 · Optimización de rentabilidad ajustada por riesgo · CLP/USD sin decimales · % con 2 decimales")

# =========================
# Controles en Sidebar
# =========================
st.sidebar.header("Controles")
OUT_DIR = st.sidebar.text_input("Ruta OUT_DIR", value="out")
mode = st.sidebar.selectbox("Modo", ["sample", "segment"])
fx = st.sidebar.number_input("Tipo de cambio USD/CLP", min_value=100.0, max_value=2000.0, value=900.0, step=1.0)
moneda = st.sidebar.selectbox("Moneda", ["CLP", "USD"])

# Detectar escenarios disponibles
pattern = os.path.join(OUT_DIR, f"consolidado_piso3_*_{mode}.csv")
scenarios = sorted([os.path.basename(p).replace(f"consolidado_piso3_","").replace(f"_{mode}.csv","")
                    for p in glob.glob(pattern)])
escen = st.sidebar.selectbox("Escenario", scenarios if scenarios else ["(no hay archivos)"])

st.sidebar.markdown("---")
with st.sidebar.expander("Ayuda"):
    st.write("""
- OUT_DIR: carpeta donde Celda 23 guardó los CSV finalizados (Piso 3).
- Moneda: aplica a TODAS las métricas y tablas (CLP/USD sin decimales).
- % (PD/APR): siempre se muestran con 2 decimales.
- Enteros / conteos: se muestran con 2 decimales (50.000,00).
    """)

# =========================
# Carga de datos
# =========================
def load_consolidado(out_dir, scen_name, mode):
    path = os.path.join(out_dir, f"consolidado_piso3_{scen_name}_{mode}.csv")
    if not os.path.exists(path):
        st.error(f"No encontré {path}. Genera el consolidado (Celda 23 del Piso 3).")
        st.stop()
    return pd.read_csv(path)

if not scenarios:
    st.warning(f"No se encontraron archivos con patrón {pattern}.")
    st.stop()

df = load_consolidado(OUT_DIR, escen, mode)

# =========================
# Conversiones de moneda
# =========================
for base in ["e_opt","ingreso_opt","el_opt","cost_opt","utilidad_opt","ingreso_mensual"]:
    if base in df.columns:
        df[base + "_disp"] = to_currency(df[base], moneda, fx)

# =========================
# KPIs
# =========================
EAD_total  = df["e_opt_disp"].sum() if "e_opt_disp" in df.columns else 0
Util_total = df["utilidad_opt_disp"].sum() if "utilidad_opt_disp" in df.columns else 0
Ing_total  = df["ingreso_opt_disp"].sum() if "ingreso_opt_disp" in df.columns else 0
EL_total   = df["el_opt_disp"].sum() if "el_opt_disp" in df.columns else 0
Cost_total = df["cost_opt_disp"].sum() if "cost_opt_disp" in df.columns else 0

EAD_total_raw = df["e_opt"].sum() if "e_opt" in df.columns else 0
pd_pond = (df["pd_score"]*df["e_opt"]).sum()/EAD_total_raw if ("pd_score" in df.columns and EAD_total_raw>0) else np.nan
apr_pond= (df["r_opt"]*df["e_opt"]).sum()/EAD_total_raw if ("r_opt" in df.columns and EAD_total_raw>0) else (df["r_opt"].mean() if "r_opt" in df.columns else np.nan)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("EAD total", fmt_money(EAD_total, moneda))
col2.metric("Utilidad",  fmt_money(Util_total, moneda))
col3.metric("Ingreso",   fmt_money(Ing_total, moneda))
col4.metric("EL",        fmt_money(EL_total, moneda))
col5.metric("Costos",    fmt_money(Cost_total, moneda))
st.metric("PD ponderado (EAD)", fmt_pct(pd_pond, 2))
st.metric("APR promedio ponderado", fmt_pct(apr_pond, 2))

with st.expander("¿Qué significa cada KPI?"):
    st.markdown(f"""
- *EAD (Exposure At Default):* monto expuesto al incumplimiento (*{moneda}* sin decimales).
- *Utilidad:* Ingreso − Pérdida Esperada (EL) − Costos (*{moneda}* sin decimales).
- *PD ponderado (EAD):* promedio de PD ponderado por EAD (porcentaje, 2 decimales).
- *APR promedio ponderado:* tasa anual promedio ponderada por EAD (porcentaje, 2 decimales).
""")

st.markdown("---")

# =========================
# Top 20 por utilidad
# =========================
cols_show = []
for c in ["id_cliente","segmento","region"]:
    if c in df.columns:
        cols_show.append(c)

ren = {
    "pd_score": "PD",
    "r_opt": "APR",
    "e_opt_disp": f"EAD ({moneda})",
    "utilidad_opt_disp": f"Utilidad ({moneda})",
    "ingreso_mensual_disp": f"Ingreso mensual ({moneda})",
}
for c in ["pd_score","r_opt","e_opt_disp","utilidad_opt_disp","ingreso_mensual_disp"]:
    if c in df.columns:
        cols_show.append(c)

if cols_show:
    top = df.sort_values("utilidad_opt_disp", ascending=False)[cols_show].head(20).copy()
    # Formateo por tipo
    money_cols = [c for c in top.columns if c.endswith("(CLP)") or c.endswith("(USD)") or c.endswith("_disp")]
    pct_cols   = [c for c in ["PD","APR"] if c in [ren.get(k,k) for k in top.columns]]  # renombrados
    # Renombrar columnas
    top = top.rename(columns=ren)
    # Aplicar formato
    top_fmt = top.copy()
    for c in top_fmt.columns:
        if c in ["PD","APR"]:
            top_fmt[c] = top_fmt[c].apply(lambda v: fmt_pct(v, 2))
        elif c in [f"EAD ({moneda})", f"Utilidad ({moneda})", f"Ingreso mensual ({moneda})"]:
            top_fmt[c] = top_fmt[c].apply(lambda v: fmt_money(v, moneda))
        elif c.lower().endswith("_disp"):
            top_fmt[c] = top_fmt[c].apply(lambda v: fmt_money(v, moneda))
    st.subheader("Top 20 por utilidad")
    st.dataframe(top_fmt, use_container_width=True)

# =========================
# Gráficos
# =========================
st.subheader("Distribución de APR óptima")
if "r_opt" in df.columns:
    fig1 = px.histogram(pd.DataFrame({"APR%": 100*df["r_opt"].dropna()}), x="APR%", nbins=30, title="Histograma de r_opt (APR óptima)")
    fig1.update_layout(xaxis_title="APR [%]", yaxis_title="Frecuencia")
    fig1.update_xaxes(tickformat=".2f")  # 2 decimales
    st.plotly_chart(fig1, use_container_width=True)

st.subheader("PD vs APR (tamaño ∝ EAD)")
if set(["pd_score","r_opt","e_opt"]).issubset(df.columns):
    size = np.clip(df["e_opt"].fillna(0).values, 0, np.percentile(df["e_opt"].fillna(0), 95))
    aux = pd.DataFrame({
        "PD%": 100*df["pd_score"],
        "APR%": 100*df["r_opt"],
        "EAD": df["e_opt_disp"] if "e_opt_disp" in df.columns else df["e_opt"],
    })
    fig2 = px.scatter(aux, x="PD%", y="APR%", size=size, opacity=0.7, title="Relación PD vs APR (tamaño proporcional a EAD)")
    fig2.update_xaxes(title_text="PD [%]", tickformat=".2f")
    fig2.update_yaxes(title_text="APR [%]", tickformat=".2f")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Utilidad por segmento")
if set(["segmento","utilidad_opt_disp"]).issubset(df.columns):
    seg = (df.groupby("segmento")
           .agg(Utilidad=("utilidad_opt_disp","sum"),
                N=("segmento","count"))
           .reset_index()
           .sort_values("Utilidad", ascending=False))
    # Tabla formateada debajo
    seg_table = format_df_for_display(seg, moneda,
                                      cols_money=["Utilidad"],
                                      cols_int2=["N"])
    st.dataframe(seg_table, use_container_width=True)
    # Barras
    fig3 = px.bar(seg, x="segmento", y="Utilidad", title=f"Utilidad por segmento ({moneda})")
    # Etiquetas del eje Y en miles sin decimales: lo resolvemos en hover y tabla; el eje numérico queda estándar en Plotly.
    st.plotly_chart(fig3, use_container_width=True)

# =========================
# Descarga
# =========================
st.markdown("---")
st.subheader("Descargas")
# Vista actual "top" si existe, si no el consolidado
download_df = top if 'top' in locals() else df
csv_bytes = download_df.to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV de la vista", data=csv_bytes, file_name=f"vista_{escen}_{mode}.csv", mime="text/csv")

st.caption(f"Fuente: consolidado_piso3_{escen}_{mode}.csv · Moneda: {moneda} (sin decimales) · % con 2 decimales · USDCLP={fx:.0f}")
