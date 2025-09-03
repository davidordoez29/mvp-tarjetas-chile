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
# =========================
# === ARISTA 1 (RESTAURADA, SIN GUARDARRÁIL) ===
# =========================
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

def _conv_scalar(v):
    """Convierte un escalar CLP→USD si corresponde, usando helpers existentes."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))): 
        return v
    return to_currency(pd.Series([v]), moneda, fx).iloc[0]

def _big_card(title, value, subtitle=None):
    st.markdown(
        f"""
        <div style="padding:14px;border:1px solid #eee;border-radius:12px;margin-bottom:10px;">
          <div style="font-size:13px;color:#666;margin-bottom:6px;">{title}</div>
          <div style="font-size:22px;font-weight:700;line-height:1.1;word-break:break-word;">{value}</div>
          {f'<div style="font-size:12px;color:#888;margin-top:6px;">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_arista1_default_impago():
    st.markdown("## Arista 1 — Default / Impago (Actual vs Optimizado)")

    port_path = os.path.join(OUT_DIR, "default_compare_portfolio.csv")
    segm_path = os.path.join(OUT_DIR, "default_compare_segment.csv")
    detl_path = os.path.join(OUT_DIR, "default_compare_detail.csv")

    missing = [p for p in [port_path, segm_path, detl_path] if not os.path.exists(p)]
    if missing:
        st.info(
            "No encontré los archivos comparativos de la Arista 1.\n\n"
            "Asegúrate de tener en OUT_DIR:\n"
            "- default_compare_portfolio.csv\n"
            "- default_compare_segment.csv\n"
            "- default_compare_detail.csv"
        )
        return

    port = pd.read_csv(port_path)
    seg  = pd.read_csv(segm_path)
    det  = pd.read_csv(detl_path)

    # ---- KPIs portafolio
    kpis = {row["kpi"]: row["value"] for _, row in port.iterrows()}
    PD_b = float(kpis.get("PD promedio ponderado (Actual)", float("nan")))
    PD_o = float(kpis.get("PD promedio ponderado (Optimizado)", float("nan")))
    EL_b = float(kpis.get("EL total (Actual)", 0.0))
    EL_o = float(kpis.get("EL total (Optimizado)", 0.0))
    EL_drop_abs = float(kpis.get("Reducción EL (monto)", EL_b - EL_o))
    EL_drop_pct = float(kpis.get("Reducción EL (%)", (EL_b - EL_o) / EL_b if EL_b else float("nan")))
    EAD_b = float(kpis.get("EAD total (Actual)", 0.0))
    EAD_o = float(kpis.get("EAD total (Optimizado)", 0.0))

    # Conversión visual (usa helpers existentes)
    EL_b_disp, EL_o_disp = _conv_scalar(EL_b), _conv_scalar(EL_o)
    EL_drop_disp         = _conv_scalar(EL_drop_abs)
    EAD_b_disp, EAD_o_disp = _conv_scalar(EAD_b), _conv_scalar(EAD_o)

    # === FILA 1: PD prom Actual | PD prom Optimizado | Reducción EL %
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1: _big_card("PD prom. (Actual)",     fmt_pct(PD_b, 2))
    with r1c2: _big_card("PD prom. (Optimizado)", fmt_pct(PD_o, 2))
    with r1c3: _big_card("Reducción EL (%)",      fmt_pct(EL_drop_pct, 2))

    # === FILA 2: EL total Actual | EL total Optimizado | Reducción EL monto
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1: _big_card(f"EL total (Actual)",     fmt_money(EL_b_disp, moneda))
    with r2c2: _big_card(f"EL total (Optimizado)", fmt_money(EL_o_disp, moneda))
    with r2c3: _big_card("Reducción EL (monto)",   fmt_money(EL_drop_disp, moneda))

    # === FILA 3: EAD total Actual | EAD total Optimizado | Cuadro explicativo
    r3c1, r3c2, r3c3 = st.columns([1,1,1.3])
    with r3c1: _big_card(f"EAD total (Actual)",     fmt_money(EAD_b_disp, moneda))
    with r3c2: _big_card(f"EAD total (Optimizado)", fmt_money(EAD_o_disp, moneda))
    with r3c3:
        st.markdown(
            f"""
            <div style="padding:14px;border:1px solid #d0e3ff;background:#f4f9ff;border-radius:12px;">
              <div style="font-weight:700;margin-bottom:6px;">¿Por qué puede subir la EAD optimizada?</div>
              <div style="font-size:13px;line-height:1.35;">
                <b>EAD</b> (<i>Exposure at Default</i>) es la exposición si un cliente incumple.
                En el método optimizado reasignamos exposición hacia <b>clientes más sanos</b> (menor PD×LGD).
                Puede subir la EAD agregada y aun así <b>bajar la EL (Expected Loss)</b>, lo que mejora la rentabilidad:
                <b>más negocio donde duele menos</b>.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.expander("Definiciones rápidas de KPIs"):
        st.markdown(f"""
- *PD (Probability of Default)*: probabilidad de impago (porcentaje).
- *LGD (Loss Given Default)*: porcentaje de pérdida si el cliente incumple.
- *EAD (Exposure at Default): exposición en caso de impago ({moneda}*).
- *EL (Expected Loss): PD × LGD × EAD ({moneda}*).
- *Reducción EL*: cuánto baja la pérdida esperada al optimizar (monto y %).
""")

    st.markdown("---")

    # ---- Gráfica EL Actual vs Optimizado (con etiquetas)
    comp_df = pd.DataFrame({
        "Escenario": ["ACTUAL","OPTIMIZADO"],
        f"EL ({moneda})": [EL_b_disp, EL_o_disp]
    })
    fig_el = px.bar(
        comp_df, x="Escenario", y=f"EL ({moneda})",
        title="Pérdida Esperada (EL) — Actual vs Optimizado",
        text=f"EL ({moneda})",
        color="Escenario",
        color_discrete_map={"ACTUAL":"#636EFA","OPTIMIZADO":"#00CC96"}
    )
    fig_el.update_traces(
        texttemplate='%{text:,.0f}',
        textposition='outside',
        hovertemplate=f"%{{x}}<br>EL: %{{y:,.0f}} {moneda}<extra></extra>"
    )
    fig_el.update_layout(
        uniformtext_minsize=12, uniformtext_mode='hide',
        margin=dict(l=20,r=20,t=60,b=20),
        yaxis_title=f"EL ({moneda})",
        xaxis_title=""
    )
    st.plotly_chart(fig_el, use_container_width=True)

    # ---- Comparación por segmento (SIN guardarraíl)
    st.subheader("Comparación por segmento")
    if "segmento" in seg.columns:
        seg["segmento"] = seg["segmento"].astype(str).str.upper()

    # Formateo visual (usa helpers)
    seg_disp = seg.copy()
    if "EL_reduccion_abs" in seg_disp.columns:
        seg_disp = seg_disp.sort_values("EL_reduccion_abs", ascending=False)

    # Moneda
    for col in ["EAD_actual","EAD_opt","EL_actual","EL_opt","EL_reduccion_abs"]:
        if col in seg_disp.columns:
            seg_disp[col] = to_currency(seg_disp[col], moneda, fx).apply(lambda v: fmt_money(v, moneda))
    # Porcentajes
    for col in ["PD_actual","PD_opt","EL_reduccion_pct"]:
        if col in seg_disp.columns:
            seg_disp[col] = seg_disp[col].apply(lambda v: fmt_pct(v, 2))
    # Conteos
    if "n_clientes" in seg_disp.columns:
        seg_disp["n_clientes"] = seg_disp["n_clientes"].apply(lambda v: fmt_num(v, 2))

    st.dataframe(seg_disp, use_container_width=True)

    with st.expander("¿Qué significa cada segmento?"):
        st.markdown("*Catálogo de segmentos (nombre → significado):*")
        seg_map = {
            "ALTO INGRESO": "Clientes con mayor ingreso y menor riesgo.",
            "MEDIO INGRESO": "Base masiva, riesgo medio; sensibilidad a tasa.",
            "BAJO INGRESO": "Mayor PD; requiere límites prudentes.",
            "PYME": "Flujos variables; límites/tasas diferenciadas.",
            "EMPRENDEDOR": "Ingresos volátiles; histórico corto.",
            "NUEVO": "Poca historia crediticia; PD incierto.",
            "MASS": "Segmento masivo; heterogéneo, requiere microsegmentación.",
        }
        detected = seg["segmento"].dropna().unique().tolist() if "segmento" in seg.columns else []
        if detected:
            for name in sorted(detected):
                meaning = seg_map.get(name, "Definición pendiente (ajustar con el banco).")
                st.markdown(f"- *{name}* → {meaning}")
        else:
            st.info("No se detectaron segmentos; revisa el archivo de entrada.")

    # ---- Gráfica: Segmentos por Reducción de EL (sin 'Top 15')
    if "EL_reduccion_abs" in seg.columns:
        seg_plot = seg.copy()
        seg_plot["EL_reduccion_abs_disp"] = to_currency(seg_plot["EL_reduccion_abs"], moneda, fx)
        fig_seg = px.bar(
            seg_plot.sort_values("EL_reduccion_abs", ascending=False),
            y="segmento", x="EL_reduccion_abs_disp",
            orientation="h",
            title=f"Segmentos por Reducción de EL ({moneda})",
            text="EL_reduccion_abs_disp",
            color="EL_reduccion_abs_disp",
            color_continuous_scale="Blues"
        )
        fig_seg.update_traces(
            texttemplate='%{text:,.0f}',
            textposition='outside',
            hovertemplate=f"Segmento: %{{y}}<br>Reducción EL: %{{x:,.0f}} {moneda}<extra></extra>"
        )
        fig_seg.update_layout(
            coloraxis_showscale=False,
            yaxis=dict(categoryorder='total ascending'),
            margin=dict(l=20,r=20,t=60,b=20),
            xaxis_title=f"Reducción EL ({moneda})", yaxis_title="Segmento"
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    # ---- Detalle por cliente
    st.subheader("Detalle por cliente (muestra)")
    det_view = det.copy()
    if "segmento" in det_view.columns:
        det_view["segmento"] = det_view["segmento"].astype(str).str.upper()

    for col in ["pd_baseline","pd_opt"]:
        if col in det_view.columns:
            det_view[col] = det_view[col].apply(lambda v: fmt_pct(v, 2))
    for col in ["ead_baseline","ead_opt","EL_baseline","EL_opt"]:
        if col in det_view.columns:
            det_view[col] = to_currency(det_view[col], moneda, fx).apply(lambda v: fmt_money(v, moneda))

    st.dataframe(det_view.head(50), use_container_width=True)
    st.download_button(
        "Descargar detalle completo (CSV)",
        data=det.to_csv(index=False).encode("utf-8"),
        file_name="default_compare_detail.csv",
        mime="text/csv",
    )

    with st.expander("Glosario de columnas (fila 1 del cuadro)"):
        st.markdown(f"""
- *id_cliente*: identificador único del cliente.
- *segmento*: grupo del cliente (en MAYÚSCULA).
- *region*: zona geográfica.
- *pd_baseline (Probability of Default): PD del **método actual* (promedio por segmento).
- *pd_opt (Probability of Default): PD de **nuestro modelo* (por cliente).
- *lgd_baseline (Loss Given Default): % de pérdida si incumple (método actual*).
- *lgd_opt (Loss Given Default): % de pérdida si incumple (nuestro modelo*).
- *ead_baseline (Exposure at Default): exposición al default en el **método actual* (*{moneda}*).
- *ead_opt (Exposure at Default): exposición al default en **nuestro modelo* (*{moneda}*).
- *EL_baseline (Expected Loss): PD×LGD×EAD en **método actual* (*{moneda}*).
- *EL_opt (Expected Loss): PD×LGD×EAD en **nuestro modelo* (*{moneda}*).
""")

# Llama al render
render_arista1_default_impago()
# =========================
# === FIN ARISTA 1 ===
# =========================
