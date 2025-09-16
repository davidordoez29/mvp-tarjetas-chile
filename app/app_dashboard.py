# app/app_dashboard.py
# =============================================================================
# Dashboard WDOF – MVP Tarjetas (Default, Yield, Incentivos, Capital)
# Lee bundle CSVs, calcula KPIs y presenta resumen ejecutivo con Streamlit.
# =============================================================================

from pathlib import Path
import os
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------- Config & Constantes ----------------------------
DEFAULT_FUNDING_COST = 0.03     # costo financiero base
DEFAULT_OPS_COST     = 0.01     # costo operacional unitario (si aplica)
RISK_WEIGHT          = 0.75     # retail ilustrativo
CAPITAL_RATIO        = 0.105    # 10.5% (8% + buffer 2.5%)

# ------------------------------- Utilidades ---------------------------------
def nz(s, lo=None, hi=None, fill=0.0):
    """Coerce a numeric con límites y fill para NaN/strings."""
    x = pd.to_numeric(s, errors="coerce").fillna(fill)
    if lo is not None or hi is not None:
        x = x.clip(lower=lo, upper=hi)
    return x

def fmt_clp(v):
    try:
        return f"$ {v:,.0f}".replace(",", ".")
    except Exception:
        return str(v)

@st.cache_data(show_spinner=False)
def _resolve_bundle_dir():
    """
    Resolver robusto de carpeta bundle:
    1) BUNDLE_DIR (env), 2) rutas más comunes en Colab/GitHub.
    Requires: dashboard_bundle_clients.csv, dashboard_bundle_segments.csv
    """
    env = os.environ.get("BUNDLE_DIR", "").strip()
    cands = []
    if env:
        cands.append(Path(env))

    try:
        # repo_root = app/.. (dos niveles arriba del archivo actual)
        repo_root = Path(_file_).resolve().parents[1]
        cands += [
            repo_root / "out" / "dashboard_bundle",
            repo_root / "out",
            Path("/content") / "mvp-tarjetas-chile" / "out" / "dashboard_bundle",
            Path("/content") / "out" / "dashboard_bundle",
        ]
    except Exception:
        pass

    required = {"dashboard_bundle_clients.csv", "dashboard_bundle_segments.csv"}
    for c in cands:
        try:
            if c.exists():
                names = {p.name for p in c.glob("*.csv")}
                if required.issubset(names):
                    return c
        except Exception:
            continue

    # fallback: usar env si existe, o cwd
    return Path(env) if env else Path(".")

@st.cache_data(show_spinner=False)
def load_bundle():
    """
    Carga los CSV del bundle y garantiza columnas mínimas.
    Devuelve (clients_df, segments_df, bundle_dir, diagnostics)
    """
    bdir = _resolve_bundle_dir()
    cpath = bdir / "dashboard_bundle_clients.csv"
    spath = bdir / "dashboard_bundle_segments.csv"

    diags = {
        "bundle_dir": str(bdir),
        "clients_exists": cpath.exists(),
        "segments_exists": spath.exists(),
        "clients_path": str(cpath),
        "segments_path": str(spath),
    }

    if not cpath.exists() or not spath.exists():
        # Dataframes vacíos (para que la app no explote)
        empty_cols_clients = ["id_cliente","apr_efectiva","ead_baseline","pd_score","lgd_pred","r_opt","e_opt",
                              "ingreso_base","ingreso_opt","EL_base","EL_opt"]
        empty_cols_segments = ["segmento","n","ead_base","ead_opt","ingreso_base","ingreso_opt","EL_base","EL_opt"]
        return (pd.DataFrame(columns=empty_cols_clients),
                pd.DataFrame(columns=empty_cols_segments),
                bdir,
                diags)

    clients = pd.read_csv(cpath)
    segments = pd.read_csv(spath)

    # Asegurar columnas mínimas para cálculos
    need_clients = ["id_cliente","apr_efectiva","ead_baseline","pd_score","lgd_pred","r_opt","e_opt"]
    for c in need_clients:
        if c not in clients.columns:
            clients[c] = np.nan

    # Derivados si faltan
    if "ingreso_base" not in clients.columns:
        clients["ingreso_base"] = (nz(clients["apr_efectiva"])-DEFAULT_FUNDING_COST) * nz(clients["ead_baseline"])
    if "ingreso_opt" not in clients.columns:
        clients["ingreso_opt"] = (nz(clients["r_opt"])-DEFAULT_FUNDING_COST) * nz(clients["e_opt"])
    if "EL_base" not in clients.columns:
        clients["EL_base"] = nz(clients["pd_score"], lo=1e-6, hi=1.0) * nz(clients["lgd_pred"], lo=1e-6, hi=1.0) * nz(clients["ead_baseline"])
    if "EL_opt" not in clients.columns:
        clients["EL_opt"] = nz(clients["pd_score"], lo=1e-6, hi=1.0) * nz(clients["lgd_pred"], lo=1e-6, hi=1.0) * nz(clients["e_opt"])

    # Segments mínimos
    need_segments = ["segmento","n","ead_base","ead_opt","ingreso_base","ingreso_opt","EL_base","EL_opt"]
    for c in need_segments:
        if c not in segments.columns:
            segments[c] = np.nan

    # Tipos
    for col in ["apr_efectiva","r_opt","pd_score","lgd_pred"]:
        if col in clients.columns:
            clients[col] = nz(clients[col], lo=0.0, hi=1.0)
    for col in ["ead_baseline","e_opt","ingreso_base","ingreso_opt","EL_base","EL_opt"]:
        if col in clients.columns:
            clients[col] = nz(clients[col], lo=0.0)

    for col in ["ead_base","ead_opt","ingreso_base","ingreso_opt","EL_base","EL_opt"]:
        if col in segments.columns:
            segments[col] = nz(segments[col], lo=0.0)

    return clients, segments, bdir, diags

def kpi_totals(clients: pd.DataFrame):
    """
    KPIs agregados para las 4 aristas:
    - Yield: Δ ingreso financiero neto
    - Default (EL): Δ pérdida esperada
    - Incentivos: (placeholder si hay costo/beneficio incremental específico)
    - Capital: Δ capital económico (RWA*ratio) aprox con EAD
    """
    df = clients.copy()

    # Yield
    delta_yield = float(df["ingreso_opt"].sum() - df["ingreso_base"].sum())

    # Default (EL)
    delta_EL = float(df["EL_opt"].sum() - df["EL_base"].sum())

    # Incentivos (si tienes columna de costo de incentivos incremental, úsala; aquí placeholder = 0)
    # Puedes agregar 'inc_cost_opt' en bundle para reflejar costos
    delta_inc = float(df.get("inc_benefit_opt", pd.Series([0]*len(df))).sum()
                      - df.get("inc_benefit_base", pd.Series([0]*len(df))).sum())

    # Capital: aproximación con EAD * RW * ratio (no depende de PD/LGD aquí, pero puedes refinar)
    cap_base = float(nz(df["ead_baseline"]).sum() * RISK_WEIGHT * CAPITAL_RATIO)
    cap_opt  = float(nz(df["e_opt"]).sum()         * RISK_WEIGHT * CAPITAL_RATIO)
    delta_cap = cap_opt - cap_base
    cap_liberado = -delta_cap if delta_cap < 0 else 0.0

    tot = {
        "Δ Yield": delta_yield,
        "Δ EL": delta_EL,
        "Δ Inc": delta_inc,
        "Δ Cap": delta_cap,
        "Lib Cap": cap_liberado,
    }
    return tot

def show_kpi_row(totals: dict):
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("Δ Yield",   fmt_clp(totals.get("Δ Yield",0)))
    col2.metric("Δ EL",      fmt_clp(totals.get("Δ EL",0)))
    col3.metric("Δ Inc",     fmt_clp(totals.get("Δ Inc",0)))
    col4.metric("Δ Capital", fmt_clp(totals.get("Δ Cap",0)))
    col5.metric("Liberación de Capital", fmt_clp(totals.get("Lib Cap",0)))

# --------------------------------- Layout UI ---------------------------------
st.set_page_config(page_title="MVP Bancario – WDOF", layout="wide")
st.title("MVP Bancario – Motor WDOF (Tarjetas)")

clients, segments, bundle_dir, diags = load_bundle()

# Sidebar: diagnóstico claro
with st.sidebar:
    st.header("Diagnóstico")
    st.write("BUNDLE_DIR (resuelto):", diags["bundle_dir"])
    st.write("clients.csv existe:", "✅" if diags["clients_exists"] else "❌", diags["clients_path"])
    st.write("segments.csv existe:", "✅" if diags["segments_exists"] else "❌", diags["segments_path"])
    st.write("Reglas capital: RW", RISK_WEIGHT, " | Ratio", CAPITAL_RATIO)

# Si no hay data, avisar y salir temprano para evitar errores
if clients.empty or segments.empty:
    st.warning("No se encontró el bundle con ambos CSV. Sube/Sync a out/dashboard_bundle o define BUNDLE_DIR en la configuración del despliegue.")
    st.stop()

# ---------------------------- Resumen Ejecutivo ------------------------------
st.subheader("Resumen Ejecutivo (4 Aristas)")
tot = kpi_totals(clients)
show_kpi_row(tot)

# ---------------------------- Arista 1 — Default -----------------------------
st.markdown("### Arista 1 — Default (Pérdida Esperada)")
colA, colB = st.columns([1,2])
with colA:
    st.write("*Story*: Ajustes reducen/alteran la EL mediante cambios en exposición (EAD) y tasas (indirectamente PD/LGD si el bundle lo incorpora).")
    st.write("*EL Base*:", fmt_clp(float(clients["EL_base"].sum())))
    st.write("*EL Opt* :", fmt_clp(float(clients["EL_opt"].sum())))
    st.write("*Δ EL*   :", fmt_clp(float(clients["EL_opt"].sum() - clients["EL_base"].sum())))
with colB:
    seg_d = segments[["segmento","EL_base","EL_opt"]].copy()
    seg_d["Δ EL"] = seg_d["EL_opt"] - seg_d["EL_base"]
    st.dataframe(seg_d, use_container_width=True)

# ----------------------------- Arista 2 — Yield ------------------------------
st.markdown("### Arista 2 — Yield (Ingreso Financiero Neto)")
colA, colB = st.columns([1,2])
with colA:
    st.write("*Story*: Variaciones de tasa efectiva y EAD modifican el ingreso financiero neto.")
    base_y = float(clients["ingreso_base"].sum())
    opt_y  = float(clients["ingreso_opt"].sum())
    st.write("*Yield Base*:", fmt_clp(base_y))
    st.write("*Yield Opt* :", fmt_clp(opt_y))
    st.write("*Δ Yield*   :", fmt_clp(opt_y - base_y))
with colB:
    seg_y = segments[["segmento","ingreso_base","ingreso_opt"]].copy()
    seg_y["Δ ingreso"] = seg_y["ingreso_opt"] - seg_y["ingreso_base"]
    st.dataframe(seg_y, use_container_width=True)

# -------------------------- Arista 3 — Incentivos ----------------------------
st.markdown("### Arista 3 — Incentivos")
st.write("*Story*: Si tu bundle incluye columnas de beneficio/costo incremental de incentivos, se mostrarán aquí. Por defecto, Δ Inc = 0.")
inc_cols = [c for c in clients.columns if c.startswith("inc_")]
if inc_cols:
    st.write("Columnas detectadas en clientes:", ", ".join(inc_cols))
else:
    st.info("No se detectaron columnas de incentivos en el bundle (p.ej., inc_benefit_base / inc_benefit_opt).")

# --------------------------- Arista 4 — Capital ------------------------------
st.markdown("### Arista 4 — Capital Económico")
colA, colB = st.columns([1,2])
with colA:
    cap_base = float(nz(clients["ead_baseline"]).sum() * RISK_WEIGHT * CAPITAL_RATIO)
    cap_opt  = float(nz(clients["e_opt"]).sum()         * RISK_WEIGHT * CAPITAL_RATIO)
    st.write("*Cap. Base*:", fmt_clp(cap_base))
    st.write("*Cap. Opt* :", fmt_clp(cap_opt))
    st.write("*Δ Cap*    :", fmt_clp(cap_opt - cap_base))
with colB:
    seg_cap = segments[["segmento","ead_base","ead_opt"]].copy()
    seg_cap["Cap_base"] = seg_cap["ead_base"] * RISK_WEIGHT * CAPITAL_RATIO
    seg_cap["Cap_opt"]  = seg_cap["ead_opt"]  * RISK_WEIGHT * CAPITAL_RATIO
    seg_cap["Δ Cap"]    = seg_cap["Cap_opt"] - seg_cap["Cap_base"]
    st.dataframe(seg_cap[["segmento","Cap_base","Cap_opt","Δ Cap"]], use_container_width=True)

# -------------------------- Drill-down de clientes ---------------------------
st.markdown("### Drill-down de clientes")
with st.expander("Ver tabla de clientes"):
    cols_show = ["id_cliente","apr_efectiva","r_opt","ead_baseline","e_opt","pd_score","lgd_pred","ingreso_base","ingreso_opt","EL_base","EL_opt"]
    for c in cols_show:
        if c not in clients.columns:
            clients[c] = np.nan
    df_show = clients[cols_show].copy()
    st.dataframe(df_show, use_container_width=True)

st.success("Dashboard cargado.")
