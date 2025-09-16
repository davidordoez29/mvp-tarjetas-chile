# app/app_dashboard.py
# =============================================================================
# Dashboard WDOF – MVP Tarjetas (Default, Yield, Incentivos, Capital)
# Auto-repara: si faltan bundle CSVs, intenta construirlos desde master/data_raw
# y, como último recurso, simula un bundle pequeño para no dejar de operar.
# =============================================================================

from pathlib import Path
import os
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------- Config & Constantes ----------------------------
DEFAULT_FUNDING_COST = 0.03     # costo financiero base
RISK_WEIGHT          = 0.75     # retail ilustrativo
CAPITAL_RATIO        = 0.105    # 10.5%

# ------------------------------- Utilidades ---------------------------------
def nz(s, lo=None, hi=None, fill=0.0):
    x = pd.to_numeric(s, errors="coerce").fillna(fill)
    if lo is not None or hi is not None:
        x = x.clip(lower=lo, upper=hi)
    return x

def fmt_clp(v):
    try:
        return f"$ {v:,.0f}".replace(",", ".")
    except Exception:
        return str(v)

def rename_any(df, mapping):
    out = df.copy()
    for dst, candidates in mapping.items():
        for c in candidates:
            if c in out.columns:
                out = out.rename(columns={c: dst})
                break
    return out

# ----------------------- Resolve bundle & self-heal --------------------------
@st.cache_data(show_spinner=False)
def _resolve_repo_root() -> Path:
    try:
        return Path(_file_).resolve().parents[1]
    except Exception:
        return Path("/content/mvp-tarjetas-chile")

@st.cache_data(show_spinner=False)
def _resolve_bundle_dir() -> Path:
    env_dir = os.environ.get("BUNDLE_DIR", "").strip()
    cands = []
    if env_dir:
        cands.append(Path(env_dir))
    repo_root = _resolve_repo_root()
    cands += [
        repo_root / "out" / "dashboard_bundle",
        repo_root / "out",
        Path("/content") / "mvp-tarjetas-chile" / "out" / "dashboard_bundle",
        Path("/content") / "out" / "dashboard_bundle",
    ]
    required = {"dashboard_bundle_clients.csv", "dashboard_bundle_segments.csv"}
    for c in cands:
        try:
            if c.exists():
                names = {p.name for p in c.glob("*.csv")}
                if required.issubset(names):
                    return c
        except Exception:
            pass
    # si no encontró, preferimos dentro del repo
    return ( _resolve_repo_root() / "out" / "dashboard_bundle" )

def _build_from_master_or_raw(bundle_dir: Path) -> bool:
    """
    Intenta crear el bundle:
    1) desde out/master_piso3.csv + out/opt_output_piso3.csv (si existe)
    2) desde data/raw (clientes, labels, ead, cuentas, saldos)
    return True si pudo crear ambos CSV.
    """
    repo = _resolve_repo_root()
    out_dir   = repo / "out"
    data_raw  = repo / "data" / "raw"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    master_p = out_dir / "master_piso3.csv"
    opt_p    = out_dir / "opt_output_piso3.csv"

    def ensure_clients_segments(base_df: pd.DataFrame, opt_df: pd.DataFrame | None):
        # opt plano si no existe
        if opt_df is None or "id_cliente" not in opt_df.columns:
            opt_df = base_df[["id_cliente","apr_efectiva","ead_baseline"]].rename(
                columns={"apr_efectiva":"r_opt","ead_baseline":"e_opt"}
            )
        else:
            opt_df = rename_any(opt_df, {
                "id_cliente": ["id_cliente","customer_id","id"],
                "r_opt":      ["r_opt","apr_opt","rate_opt","tasa_opt","r_star","apr_efectiva"],
                "e_opt":      ["e_opt","ead_opt","EAD_opt","exposure_opt","e_star","ead_baseline"],
            })
            if "r_opt" not in opt_df.columns:
                opt_df["r_opt"] = base_df["apr_efectiva"]
            if "e_opt" not in opt_df.columns:
                opt_df["e_opt"] = base_df["ead_baseline"]

        clients = base_df.merge(opt_df[["id_cliente","r_opt","e_opt"]], on="id_cliente", how="left")
        clients["r_opt"] = nz(clients["r_opt"], lo=0.0, hi=1.0)
        clients["e_opt"] = nz(clients["e_opt"], lo=0.0)
        clients["ingreso_base"] = (clients["apr_efectiva"] - DEFAULT_FUNDING_COST) * clients["ead_baseline"]
        clients["ingreso_opt"]  = (clients["r_opt"]       - DEFAULT_FUNDING_COST) * clients["e_opt"]
        clients["EL_base"]      = clients["pd_score"] * clients["lgd_pred"] * clients["ead_baseline"]
        clients["EL_opt"]       = clients["pd_score"] * clients["lgd_pred"] * clients["e_opt"]

        keep = ["id_cliente","segmento","apr_efectiva","r_opt","ead_baseline","e_opt",
                "pd_score","lgd_pred","ingreso_base","ingreso_opt","EL_base","EL_opt"]
        for c in keep:
            if c not in clients.columns:
                clients[c] = np.nan
        clients = clients[keep].copy()

        segs = (
            clients.groupby("segmento", as_index=False)
                   .agg(n=("id_cliente","count"),
                        ead_base=("ead_baseline","sum"),
                        ead_opt=("e_opt","sum"),
                        ingreso_base=("ingreso_base","sum"),
                        ingreso_opt=("ingreso_opt","sum"),
                        EL_base=("EL_base","sum"),
                        EL_opt=("EL_opt","sum"))
        )
        return clients, segs

    # 1) Intentar desde master
    try:
        if master_p.exists():
            base = pd.read_csv(master_p)
            base = rename_any(base, {
                "id_cliente":   ["id_cliente","customer_id","id"],
                "apr_efectiva": ["apr_efectiva","apr","tasa","rate","apr_base"],
                "ead_baseline": ["ead_baseline","ead","EAD","exposure","saldo"],
                "pd_score":     ["pd_score","pd","pd_hat","prob_default"],
                "lgd_pred":     ["lgd_pred","lgd","lgd_hat","loss_given_default"],
                "segmento":     ["segmento","segment","seg","bucket"],
            })
            # defaults si faltan
            for col, d in [("apr_efectiva",0.35), ("ead_baseline",0.0), ("pd_score",0.05), ("lgd_pred",0.45)]:
                if col not in base.columns: base[col] = d
            base["apr_efectiva"] = nz(base["apr_efectiva"], lo=0.0, hi=1.0, fill=0.35)
            base["ead_baseline"] = nz(base["ead_baseline"], lo=0.0, fill=0.0)
            base["pd_score"]     = nz(base["pd_score"],     lo=1e-6, hi=1.0, fill=0.05)
            base["lgd_pred"]     = nz(base["lgd_pred"],     lo=1e-6, hi=1.0, fill=0.45)

            if "segmento" not in base.columns:
                q = pd.qcut(base["pd_score"], q=4, labels=["Q1_bajo","Q2","Q3","Q4_alto"])
                base["segmento"] = q.astype(str)

            opt_df = pd.read_csv(opt_p) if opt_p.exists() else None
            clients, segs = ensure_clients_segments(base, opt_df)

            (bundle_dir/"dashboard_bundle_clients.csv").parent.mkdir(parents=True, exist_ok=True)
            clients.to_csv(bundle_dir/"dashboard_bundle_clients.csv", index=False)
            segs.to_csv(bundle_dir/"dashboard_bundle_segments.csv", index=False)
            return True
    except Exception:
        pass

    # 2) Construir master desde data/raw
    try:
        clientes_p = data_raw / "clientes.csv"
        labels_p   = data_raw / "labels_riesgo.csv"
        saldos_p   = data_raw / "saldos_mensuales.csv"
        cuentas_p  = data_raw / "cuentas.csv"
        ead_p      = data_raw / "ead_baseline.csv"

        # mínimos
        if not (clientes_p.exists() and labels_p.exists()):
            return False

        clientes = pd.read_csv(clientes_p)
        clientes = rename_any(clientes, {"id_cliente":["id_cliente","customer_id","id"]})
        if "id_cliente" not in clientes.columns:
            clientes["id_cliente"] = np.arange(1, len(clientes)+1)

        labels = pd.read_csv(labels_p)
        labels = rename_any(labels, {
            "id_cliente":["id_cliente","customer_id","id"],
            "pd_score":  ["pd_score","pd","pd_hat","prob_default"],
            "lgd_pred":  ["lgd_pred","lgd","lgd_hat","loss_given_default"],
        })
        if "id_cliente" not in labels.columns:
            return False
        if "pd_score" not in labels.columns: labels["pd_score"] = 0.05
        if "lgd_pred" not in labels.columns: labels["lgd_pred"] = 0.45

        # EAD
        if ead_p.exists():
            ead_df = pd.read_csv(ead_p)
            ead_df = rename_any(ead_df, {"id_cliente":["id_cliente","customer_id","id"],
                                         "ead_baseline":["ead_baseline","ead","EAD","exposure","saldo"]})
            ead_df = ead_df[["id_cliente","ead_baseline"]].copy()
        else:
            if saldos_p.exists():
                sal = pd.read_csv(saldos_p)
                sal = rename_any(sal, {"id_cliente":["id_cliente","customer_id","id"],
                                       "fecha":["fecha","periodo","mes"],
                                       "saldo":["saldo","balance","ead","EAD","exposure"]})
                if all(c in sal.columns for c in ["id_cliente","fecha","saldo"]):
                    sal["fecha"] = pd.to_datetime(sal["fecha"], errors="coerce")
                    last = sal.sort_values("fecha").groupby("id_cliente").tail(1)
                    ead_df = last[["id_cliente","saldo"]].rename(columns={"saldo":"ead_baseline"})
                else:
                    ead_df = clientes[["id_cliente"]].assign(ead_baseline=0.0)
            else:
                ead_df = clientes[["id_cliente"]].assign(ead_baseline=0.0)

        # APR
        apr_df = None
        if cuentas_p.exists():
            ctas = pd.read_csv(cuentas_p)
            ctas = rename_any(ctas, {"id_cliente":["id_cliente","customer_id","id"],
                                     "apr_efectiva":["apr_efectiva","apr","tasa","rate","apr_base"]})
            if "id_cliente" in ctas.columns and "apr_efectiva" in ctas.columns:
                apr_df = ctas[["id_cliente","apr_efectiva"]].copy()

        base = clientes[["id_cliente"]].drop_duplicates().copy()
        if apr_df is not None:
            base = base.merge(apr_df, on="id_cliente", how="left")
        else:
            base["apr_efectiva"] = 0.35
        base = base.merge(ead_df, on="id_cliente", how="left")
        base = base.merge(labels[["id_cliente","pd_score","lgd_pred"]], on="id_cliente", how="left")

        # defaults + saneo
        for col, d in [("apr_efectiva",0.35), ("ead_baseline",0.0), ("pd_score",0.05), ("lgd_pred",0.45)]:
            if col not in base.columns: base[col] = d
        base["apr_efectiva"] = nz(base["apr_efectiva"], lo=0.0, hi=1.0, fill=0.35)
        base["ead_baseline"] = nz(base["ead_baseline"], lo=0.0, fill=0.0)
        base["pd_score"]     = nz(base["pd_score"],     lo=1e-6, hi=1.0, fill=0.05)
        base["lgd_pred"]     = nz(base["lgd_pred"],     lo=1e-6, hi=1.0, fill=0.45)
        if "segmento" not in base.columns:
            q = pd.qcut(base["pd_score"], q=4, labels=["Q1_bajo","Q2","Q3","Q4_alto"])
            base["segmento"] = q.astype(str)

        # guardar master para trazabilidad
        out_dir.mkdir(parents=True, exist_ok=True)
        base.to_csv(out_dir/"master_piso3.csv", index=False)

        clients, segs = ensure_clients_segments(base, opt_df=None)
        (bundle_dir/"dashboard_bundle_clients.csv").parent.mkdir(parents=True, exist_ok=True)
        clients.to_csv(bundle_dir/"dashboard_bundle_clients.csv", index=False)
        segs.to_csv(bundle_dir/"dashboard_bundle_segments.csv", index=False)
        return True
    except Exception:
        return False

def _simulate_bundle(bundle_dir: Path) -> None:
    """Último recurso: simula un bundle pequeño pero válido."""
    n = 2000
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "id_cliente":   np.arange(1, n+1),
        "apr_efectiva": np.clip(rng.normal(0.35, 0.05, n), 0.05, 0.60),
        "ead_baseline": np.clip(rng.lognormal(mean=11, sigma=0.5, size=n), 0, None),
        "pd_score":     np.clip(rng.normal(0.05, 0.02, n), 0.001, 0.25),
        "lgd_pred":     np.clip(rng.normal(0.45, 0.10, n), 0.05, 0.95),
    })
    q = pd.qcut(df["pd_score"], q=4, labels=["Q1_bajo","Q2","Q3","Q4_alto"])
    df["segmento"] = q.astype(str)
    df["r_opt"] = np.clip(df["apr_efectiva"] - 0.01, 0.03, 0.60)
    df["e_opt"] = df["ead_baseline"] * 1.02
    df["ingreso_base"] = (df["apr_efectiva"] - DEFAULT_FUNDING_COST) * df["ead_baseline"]
    df["ingreso_opt"]  = (df["r_opt"]       - DEFAULT_FUNDING_COST) * df["e_opt"]
    df["EL_base"]      = df["pd_score"] * df["lgd_pred"] * df["ead_baseline"]
    df["EL_opt"]       = df["pd_score"] * df["lgd_pred"] * df["e_opt"]
    clients = df[["id_cliente","segmento","apr_efectiva","r_opt","ead_baseline","e_opt",
                  "pd_score","lgd_pred","ingreso_base","ingreso_opt","EL_base","EL_opt"]].copy()
    segs = (
        clients.groupby("segmento", as_index=False)
               .agg(n=("id_cliente","count"),
                    ead_base=("ead_baseline","sum"),
                    ead_opt=("e_opt","sum"),
                    ingreso_base=("ingreso_base","sum"),
                    ingreso_opt=("ingreso_opt","sum"),
                    EL_base=("EL_base","sum"),
                    EL_opt=("EL_opt","sum"))
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    clients.to_csv(bundle_dir/"dashboard_bundle_clients.csv", index=False)
    segs.to_csv(bundle_dir/"dashboard_bundle_segments.csv", index=False)

@st.cache_data(show_spinner=False)
def ensure_bundle():
    """
    Garantiza que existan los 2 CSV del bundle; si no, intenta construirlos;
    si falla, simula.
    """
    bdir = _resolve_bundle_dir()
    cpath = bdir/"dashboard_bundle_clients.csv"
    spath = bdir/"dashboard_bundle_segments.csv"
    if cpath.exists() and spath.exists():
        return bdir, True, "ok (preexistente)"
    # intentar construir
    ok = _build_from_master_or_raw(bdir)
    if ok:
        return bdir, True, "construido"
    # simular
    _simulate_bundle(bdir)
    return bdir, True, "simulado"

@st.cache_data(show_spinner=False)
def load_bundle():
    """Carga el bundle ya garantizado por ensure_bundle()."""
    bdir, ok, how = ensure_bundle()
    cpath = bdir / "dashboard_bundle_clients.csv"
    spath = bdir / "dashboard_bundle_segments.csv"

    diags = {
        "bundle_dir": str(bdir),
        "status": how,
        "clients_exists": cpath.exists(),
        "segments_exists": spath.exists(),
        "clients_path": str(cpath),
        "segments_path": str(spath),
    }

    clients = pd.read_csv(cpath)
    segments = pd.read_csv(spath)

    # Tipos y límites
    for col in ["apr_efectiva","r_opt","pd_score","lgd_pred"]:
        if col in clients.columns: clients[col] = nz(clients[col], lo=0.0, hi=1.0)
    for col in ["ead_baseline","e_opt","ingreso_base","ingreso_opt","EL_base","EL_opt"]:
        if col in clients.columns: clients[col] = nz(clients[col], lo=0.0)
    for col in ["ead_base","ead_opt","ingreso_base","ingreso_opt","EL_base","EL_opt"]:
        if col in segments.columns: segments[col] = nz(segments[col], lo=0.0)

    return clients, segments, bdir, diags

def kpi_totals(clients: pd.DataFrame):
    df = clients.copy()
    delta_yield = float(df["ingreso_opt"].sum() - df["ingreso_base"].sum())
    delta_EL    = float(df["EL_opt"].sum()      - df["EL_base"].sum())
    delta_inc   = float(df.get("inc_benefit_opt", pd.Series([0]*len(df))).sum()
                        - df.get("inc_benefit_base", pd.Series([0]*len(df))).sum())
    cap_base = float(nz(df["ead_baseline"]).sum() * RISK_WEIGHT * CAPITAL_RATIO)
    cap_opt  = float(nz(df["e_opt"]).sum()        * RISK_WEIGHT * CAPITAL_RATIO)
    delta_cap = cap_opt - cap_base
    cap_liberado = -delta_cap if delta_cap < 0 else 0.0
    return {"Δ Yield": delta_yield, "Δ EL": delta_EL, "Δ Inc": delta_inc,
            "Δ Cap": delta_cap, "Lib Cap": cap_liberado}

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
    st.header("Diagnóstico del Bundle")
    st.write("Estado:", diags.get("status"))
    st.write("BUNDLE_DIR:", diags["bundle_dir"])
    st.write("clients_csv:", "✅" if diags["clients_exists"] else "❌", diags["clients_path"])
    st.write("segments_csv:", "✅" if diags["segments_exists"] else "❌", diags["segments_path"])
    st.write("Reglas capital: RW", RISK_WEIGHT, " | Ratio", CAPITAL_RATIO)

# ---------------------------- Resumen Ejecutivo ------------------------------
st.subheader("Resumen Ejecutivo (4 Aristas)")
tot = kpi_totals(clients)
show_kpi_row(tot)

# ---------------------------- Arista 1 — Default -----------------------------
st.markdown("### Arista 1 — Default (Pérdida Esperada)")
colA, colB = st.columns([1,2])
with colA:
    st.write("*Story*: Ajustes de exposición/tasa impactan la EL.")
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
st.write("*Story*: Si el bundle incluye métricas de incentivos, se reflejan aquí. Por defecto Δ Inc = 0.")
inc_cols = [c for c in clients.columns if c.startswith("inc_")]
if inc_cols:
    st.write("Columnas detectadas:", ", ".join(inc_cols))
else:
    st.info("No se detectaron columnas de incentivos (p.ej., inc_benefit_base/inc_benefit_opt).")

# --------------------------- Arista 4 — Capital ------------------------------
st.markdown("### Arista 4 — Capital Económico")
colA, colB = st.columns([1,2])
with colA:
    cap_base = float(nz(clients["ead_baseline"]).sum() * RISK_WEIGHT * CAPITAL_RATIO)
    cap_opt  = float(nz(clients["e_opt"]).sum()        * RISK_WEIGHT * CAPITAL_RATIO)
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
st.markdown("### Drill-do
