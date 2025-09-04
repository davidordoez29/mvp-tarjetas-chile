# app_dashboard.py
# MVP Tarjetas Chile — Arista 1 (Default / Expected Loss)
# Dashboard comparativo Baseline vs Optimizado, ejecutivo y didáctico.

import os
import numpy as np
import pandas as pd
import streamlit as st

# Intento usar Plotly (mejor rotulación); si no existe, caigo a st.bar_chart
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="MVP Tarjetas Chile — Arista 1 (Default/EL)",
    layout="wide",
)

# -----------------------------
# HELPERS (formatos y utilidades)
# -----------------------------
def num(s):
    return pd.to_numeric(s, errors="coerce")

def to_currency(v, moneda="CLP", usdclp=900.0):
    try:
        v = float(v)
    except:
        return np.nan
    return v if moneda.upper() == "CLP" else v / float(usdclp)

def fmt_money(x, moneda="CLP"):
    try:
        return f"{float(x):,.0f} {moneda}"
    except:
        return f"N/A {moneda}"

def fmt_pct(x):
    try:
        return f"{float(x) * 100:.2f}%"
    except:
        return "N/A"

def fmt_num(x):
    try:
        return f"{float(x):,.2f}"
    except:
        return "N/A"

def coalesce_series(*series):
    out = None
    for s in series:
        if s is None:
            continue
        s = num(s)
        out = s if out is None else out.where(out.notna(), s)
    return out

def read_csv_safe(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"No pude leer {path}: {e}")
        return None

def normalize_keys(df):
    if df is None: return None
    for c in ["id_cliente", "segmento"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "segmento" in df.columns:
        df["segmento"] = df["segmento"].astype(str).str.upper()
    return df

# -----------------------------
# SIDEBAR — Opciones
# -----------------------------
st.sidebar.title("⚙️ Opciones")
data_dir = st.sidebar.text_input("Carpeta de datos", value="out")
moneda   = st.sidebar.radio("Moneda", options=["CLP", "USD"], index=0, horizontal=True)
usdclp   = st.sidebar.number_input("Tipo de cambio USD/CLP", min_value=100.0, max_value=5000.0, value=900.0, step=10.0)
cost_is_rate = st.sidebar.checkbox(
    "Tratar costos operativos como TASA (por defecto: MONTO)",
    value=False,
    help="Nuestro modelo (Celda 20) trata costos como MONTO; marca esta opción sólo si tu fuente de costos es tasa."
)
st.sidebar.markdown("---")
st.sidebar.markdown("*Archivos esperados* en la carpeta seleccionada:")
st.sidebar.code(
    "\n".join([
        "default_compare_detail.csv",
        "default_compare_detail_conv.csv (opcional)",
        "default_compare_segment.csv",
        "opt_input_ready.csv",
        "opt_output_piso3.csv",
    ])
)

# -----------------------------
# CARGA DE DATOS (robusta)
# -----------------------------
detail_base_path = os.path.join(data_dir, "default_compare_detail.csv")
detail_conv_path = os.path.join(data_dir, "default_compare_detail_conv.csv")
segment_path     = os.path.join(data_dir, "default_compare_segment.csv")
ready_path       = os.path.join(data_dir, "opt_input_ready.csv")
opt_path         = os.path.join(data_dir, "opt_output_piso3.csv")

df_detail = read_csv_safe(detail_base_path)
df_conv   = read_csv_safe(detail_conv_path)
df_seg    = read_csv_safe(segment_path)
df_ready  = read_csv_safe(ready_path)
df_optout = read_csv_safe(opt_path)

for ref in ["df_detail", "df_conv", "df_seg", "df_ready", "df_optout"]:
    locals()[ref] = normalize_keys(locals()[ref])

# Si no existe el conv, lo creamos a partir de detail (según moneda)
if df_conv is None and df_detail is not None:
    df_conv = df_detail.copy()
    suffix = moneda.lower()
    for c in ["ead_baseline","e_opt","ingreso_opt","el_opt","cost_opt","utilidad_opt"]:
        if c in df_conv.columns and f"{c}_{suffix}" not in df_conv.columns:
            df_conv[f"{c}_{suffix}"] = df_conv[c].apply(lambda v: to_currency(v, moneda, usdclp))
    try:
        df_conv.to_csv(detail_conv_path, index=False)
    except Exception:
        pass

# Validaciones mínimas
missing = [n for n, df in [
    ("default_compare_detail.csv", df_detail),
    ("default_compare_segment.csv", df_seg),
    ("opt_input_ready.csv", df_ready),
    ("opt_output_piso3.csv", df_optout),
] if df is None]
if missing:
    st.error("Faltan archivos requeridos: " + ", ".join(missing))
    st.stop()

# -----------------------------
# KPIs (robusto y consistente con Piso 3)
# -----------------------------
def compute_kpis(df_detail, df_conv, df_ready, df_optout, moneda="CLP", usdclp=900.0, cost_is_rate=False):
    dfb = df_detail.copy()
    dfr = df_ready.copy()
    opt = df_optout.copy()

    # ead_baseline
    if "ead_baseline" not in dfb.columns:
        fallback = None
        if "ead_pred" in dfr.columns: fallback = "ead_pred"
        elif "limite_ref" in dfr.columns: fallback = "limite_ref"
        if fallback:
            dfb = dfb.merge(dfr[["id_cliente","segmento",fallback]],
                            on=["id_cliente","segmento"], how="left")
            dfb["ead_baseline"] = dfb[fallback]; dfb.drop(columns=[fallback], inplace=True)
        else:
            dfb["ead_baseline"] = 0.0

    # PD/LGD/APR/Costos
    for col in ["pd_score","lgd_pred","apr_efectiva","apr_base","costos_operativos"]:
        if col not in dfb.columns: dfb[col] = np.nan

    # costos_operativos (prioridad df_detail -> df_ready -> df_optout -> mediana)
    tmp = dfb.merge(dfr[["id_cliente","costos_operativos"]].rename(columns={"costos_operativos":"cop_ready"}),
                    on="id_cliente", how="left")
    tmp = tmp.merge(opt[["id_cliente","costos_operativos"]].rename(columns={"costos_operativos":"cop_opt"}),
                    on="id_cliente", how="left")
    cop_dfb   = num(dfb["costos_operativos"]) if "costos_operativos" in dfb.columns else None
    cop_ready = num(tmp["cop_ready"]) if "cop_ready" in tmp.columns else None
    cop_opt   = num(tmp["cop_opt"]) if "cop_opt" in tmp.columns else None

    cop_all = coalesce_series(cop_dfb, cop_ready, cop_opt)
    if cop_all is None:
        cop_all = pd.Series(0.0, index=dfb.index)
    else:
        med = float(pd.concat([s.dropna() for s in [cop_dfb, cop_ready, cop_opt] if s is not None], axis=0).median()) \
              if any([(s is not None and s.dropna().size>0) for s in [cop_dfb, cop_ready, cop_opt]]) else 0.0
        cop_all = cop_all.fillna(med)

    pd_b  = num(dfb["pd_score"]).clip(0,1).fillna(0.0)
    lgd_b = num(dfb["lgd_pred"]).clip(0,1).fillna(0.0)
    ead_b = num(dfb["ead_baseline"]).clip(lower=0).fillna(0.0)
    apr_u = num(dfb["apr_efectiva"].where(dfb["apr_efectiva"].notna(), dfb["apr_base"])).fillna(0.0)

    EL_b   = pd_b * lgd_b * ead_b
    ING_b  = apr_u * ead_b
    COST_b = cop_all * ead_b if cost_is_rate else cop_all
    UTIL_b = ING_b - (EL_b + COST_b)

    def cur(x): return to_currency(x, moneda, usdclp)

    port = {}
    port["EAD_ACT"]  = cur(ead_b.sum())
    port["EL_ACT"]   = cur(EL_b.sum())
    port["ING_ACT"]  = cur(ING_b.sum())
    port["COST_ACT"] = cur(COST_b.sum())
    port["UTIL_ACT"] = cur(UTIL_b.sum())
    port["PDW_ACT"]  = float((pd_b * ead_b).sum() / ead_b.sum()) if ead_b.sum() > 0 else np.nan

    suffix = moneda.lower()
    def sum_safe(col): return pd.to_numeric(df_conv[col], errors="coerce").sum() if col in df_conv.columns else np.nan
    port["EAD_OPT"]  = sum_safe(f"e_opt_{suffix}")
    port["EL_OPT"]   = sum_safe(f"el_opt_{suffix}")
    port["ING_OPT"]  = sum_safe(f"ingreso_opt_{suffix}")
    port["COST_OPT"] = sum_safe(f"cost_opt_{suffix}")
    port["UTIL_OPT"] = sum_safe(f"utilidad_opt_{suffix}")

    pd_ref  = dfr[["id_cliente","pd_score"]].copy()
    pd_ref["pd_score"] = num(pd_ref["pd_score"]).clip(0,1)
    eopt_key = df_optout[["id_cliente","e_opt"]].copy()
    pd_opt = eopt_key.merge(pd_ref, on="id_cliente", how="left")
    port["PDW_OPT"] = float((pd_opt["pd_score"].fillna(0) * pd_opt["e_opt"].fillna(0)).sum() / pd_opt["e_opt"].sum()) \
                      if pd_opt["e_opt"].sum() > 0 else np.nan

    if pd.notna(port["EL_ACT"]) and pd.notna(port["EL_OPT"]):
        port["RED_EL_MONTO"] = port["EL_ACT"] - port["EL_OPT"]
        port["RED_EL_PCT"]   = (port["RED_EL_MONTO"] / port["EL_ACT"]) if port["EL_ACT"] > 0 else np.nan
    else:
        port["RED_EL_MONTO"] = np.nan
        port["RED_EL_PCT"]   = np.nan

    return port

PORT = compute_kpis(df_detail, df_conv, df_ready, df_optout,
                    moneda=moneda, usdclp=usdclp, cost_is_rate=cost_is_rate)

# -----------------------------
# HEADER
# -----------------------------
st.title("📈 Arista 1 — Default / Expected Loss (EL)")
st.caption("Comparativo *Baseline vs Optimizado* — claro, didáctico y accionable para dirección.")

# -----------------------------
# KPIs (3 filas)
# -----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("PD prom ACTUAL", fmt_pct(PORT["PDW_ACT"]))
with c2:
    st.metric("PD prom OPTIMIZADO", fmt_pct(PORT["PDW_OPT"]))
with c3:
    st.metric("Reducción EL (%)", fmt_pct(PORT["RED_EL_PCT"]))

c4, c5, c6 = st.columns(3)
with c4:
    st.metric("EL TOTAL ACTUAL", fmt_money(PORT["EL_ACT"], moneda))
with c5:
    st.metric("EL TOTAL OPTIMIZADO", fmt_money(PORT["EL_OPT"], moneda))
with c6:
    st.metric("Reducción EL (monto)", fmt_money(PORT["RED_EL_MONTO"], moneda))

c7, c8, c9 = st.columns(3)
with c7:
    st.metric("EAD TOTAL ACTUAL", fmt_money(PORT["EAD_ACT"], moneda))
with c8:
    st.metric("EAD TOTAL OPTIMIZADO", fmt_money(PORT["EAD_OPT"], moneda))
with c9:
    with st.expander("🧠 ¿Por qué puede subir EAD y bajar EL?"):
        st.write(
            """
            - *EAD*: Exposición al momento de incumplimiento (saldo expuesto).  
            - *EL* = *PD × LGD × EAD*.  
            - El modelo *reasigna EAD* desde clientes de *alto PD×LGD* hacia clientes *más sanos*:  
              baja la *EL* total aunque la *EAD* se mantenga o suba levemente.  
            - Resultado: *mejor rentabilidad* (más ingreso con riesgo esperado menor o similar).
            """
        )

# -----------------------------
# GRÁFICO EL ACTUAL VS OPTIMIZADO
# -----------------------------
st.subheader("🔍 EL actual vs optimizado")
bar_df = pd.DataFrame({
    "Escenario": ["ACTUAL", "OPTIMIZADO"],
    f"EL ({moneda})": [PORT["EL_ACT"], PORT["EL_OPT"]],
})
if HAS_PLOTLY:
    fig = px.bar(
        bar_df, x="Escenario", y=f"EL ({moneda})", text=f"EL ({moneda})",
        title=f"EL por escenario ({moneda})"
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(yaxis_title=f"EL ({moneda})", xaxis_title="", uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.bar_chart(bar_df.set_index("Escenario"))

# -----------------------------
# COMPARACIÓN POR SEGMENTO
# -----------------------------
st.subheader("🏷️ Comparación por segmento")
seg = df_seg.copy()
for c in ["ead_baseline","EL_baseline","ingreso_base","costo_base","utilidad_base",
          "e_opt","el_opt","ingreso_opt","cost_opt","utilidad_opt"]:
    if c in seg.columns:
        seg[c] = num(seg[c]).fillna(0.0)

seg["RED_EL_MONTO"] = seg["EL_baseline"] - seg["el_opt"]
seg["RED_EL_PCT"]   = np.where(seg["EL_baseline"] > 0, seg["RED_EL_MONTO"]/seg["EL_baseline"], np.nan)

# Ordenar por mayor reducción
seg = seg.sort_values("RED_EL_MONTO", ascending=False)

# Mostrar con formato
seg_show = seg.copy()
money_cols = ["ead_baseline","EL_baseline","ingreso_base","costo_base","utilidad_base",
              "e_opt","el_opt","ingreso_opt","cost_opt","utilidad_opt","RED_EL_MONTO"]
pct_cols   = ["RED_EL_PCT"]
for c in money_cols:
    if c in seg_show.columns:
        seg_show[c] = seg_show[c].apply(lambda x: fmt_money(to_currency(x, moneda, usdclp), moneda))
for c in pct_cols:
    if c in seg_show.columns:
        seg_show[c] = seg_show[c].apply(fmt_pct)

st.dataframe(seg_show, use_container_width=True)

with st.expander("ℹ️ Glosario de segmentos"):
    st.json({
        "MASS": "Masivo banca personas",
        "MASS_AFFLUENT": "Personas con ingresos medios-altos",
        "AFFLUENT": "Altos ingresos / alta vinculación",
        "PYME": "Pequeña y mediana empresa",
        "CORP": "Empresas corporativas / grandes",
    })

# -----------------------------
# GRÁFICO — Segmentos por reducción de EL
# -----------------------------
st.subheader("📊 Segmentos por reducción de EL (monto)")
seg_plot = seg.copy()
seg_plot["RED_EL_MONTO_CUR"] = seg_plot["RED_EL_MONTO"].apply(lambda x: to_currency(x, moneda, usdclp))
seg_plot = seg_plot[["segmento","RED_EL_MONTO_CUR"]].rename(columns={"RED_EL_MONTO_CUR": f"Reducción EL ({moneda})"})

if HAS_PLOTLY:
    fig2 = px.bar(
        seg_plot, x="segmento", y=f"Reducción EL ({moneda})", text=f"Reducción EL ({moneda})",
        title=f"Reducción de EL por segmento ({moneda})"
    )
    fig2.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig2.update_layout(xaxis_title="Segmento", yaxis_title=f"Reducción EL ({moneda})", uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.bar_chart(seg_plot.set_index("segmento"))

# -----------------------------
# DETALLE POR CLIENTE (tabla + filtros)
# -----------------------------
st.subheader("🧾 Detalle por cliente (optimizado)")
dfc = df_conv.copy()
suffix = moneda.lower()

detail_cols = ["id_cliente","segmento",
               f"ead_baseline_{suffix}", f"e_opt_{suffix}",
               f"el_opt_{suffix}", f"ingreso_opt_{suffix}",
               f"cost_opt_{suffix}", f"utilidad_opt_{suffix}"]
detail_cols = [c for c in detail_cols if c in dfc.columns]

seg_list = sorted(list(dfc["segmento"].dropna().unique())) if "segmento" in dfc.columns else []
sel_segs = st.multiselect("Filtrar por segmento", options=seg_list, default=seg_list)
dfc_f = dfc[dfc["segmento"].isin(sel_segs)] if seg_list else dfc

st.dataframe(dfc_f[detail_cols].head(1000), use_container_width=True)

with st.expander("ℹ️ Glosario de columnas (detalle)"):
    st.markdown("""
    - *id_cliente*: identificador anónimo del cliente  
    - *segmento*: MASS / MASS_AFFLUENT / AFFLUENT / PYME / CORP  
    - *ead_baseline*: exposición base (antes de optimizar)  
    - *e_opt*: exposición optimizada  
    - *el_opt*: pérdida esperada optimizada  
    - *ingreso_opt*: ingreso financiero optimizado  
    - *cost_opt*: costos operativos (modelo)  
    - *utilidad_opt*: ingreso_opt – (el_opt + cost_opt)
    """)

# -----------------------------
# MÉTODO ACTUAL vs NUESTRO MÉTODO (panel explicativo)
# -----------------------------
st.subheader("🧠 ¿Por qué nuestro método es mejor que el actual?")
st.markdown("""
*Método actual (simplificado):* reglas fijas por segmento, límites homogéneos, tasas poco sensibles al riesgo.  
*Nuestro método:* optimización matemática con envelopes de McCormick que asigna *r* (APR efectiva) y *EAD* por cliente maximizando utilidad con restricciones por segmento y totales.

- *PD (Probability of Default = Probabilidad de Incumplimiento)*  
- *LGD (Loss Given Default = Severidad de Pérdida)*  
- *EAD (Exposure At Default = Exposición en Incumplimiento)*  
- *EL (Expected Loss = Pérdida Esperada) = PD × LGD × EAD*  
- *APR (Annual Percentage Rate = Tasa Efectiva)*  
- *Costos operativos: tratados como **MONTO* (consistente con nuestro modelo).  

*Beneficio clave:* reasignamos exposición hacia clientes con menor *PD×LGD* manteniendo o mejorando ingreso; por eso *EL baja* aunque *EAD* total pueda subir levemente → *mejor utilidad*.
""")

# -----------------------------
# DESCARGAS (sidebar)
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Descargas")
def download_if_exists(path, label):
    if os.path.exists(path):
        with open(path, "rb") as f:
            st.sidebar.download_button(label=label, data=f, file_name=os.path.basename(path))

download_if_exists(detail_base_path, "default_compare_detail.csv")
download_if_exists(detail_conv_path, "default_compare_detail_conv.csv")
download_if_exists(segment_path, "default_compare_segment.csv")
download_if_exists(ready_path, "opt_input_ready.csv")
download_if_exists(opt_path, "opt_output_piso3.csv")

st.success("Dashboard listo. Cambia moneda/TC en la barra lateral para ver el impacto inmediato.")
