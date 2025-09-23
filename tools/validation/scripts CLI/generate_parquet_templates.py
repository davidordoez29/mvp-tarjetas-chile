#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WDOF - Generador de Plantillas Parquet
Uso:
  python generate_parquet_templates.py --minimal /ruta/template_minimal_headers.csv --full /ruta/template_full_headers.csv --outdir /ruta/out

Requisitos:
  pip install pyarrow pandas
"""
import argparse, sys
from pathlib import Path
import pandas as pd

def to_parquet(headers_csv, out_path):
    headers = pd.read_csv(headers_csv).columns.tolist()
    df = pd.DataFrame(columns=headers)
    try:
        df.to_parquet(out_path, index=False)
        print(f"✅ Parquet generado → {out_path}")
    except Exception as e:
        print(f"❌ No se pudo generar Parquet ({out_path.name}). Instala 'pyarrow'. Error: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimal", required=True, help="CSV de encabezados mínimos")
    ap.add_argument("--full", required=True, help="CSV de encabezados FULL")
    ap.add_argument("--outdir", required=True, help="Directorio de salida")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    to_parquet(args.minimal, outdir / "template_minimal_headers.parquet")
    to_parquet(args.full, outdir / "template_full_headers.parquet")

if __name__ == "__main__":
    main()
