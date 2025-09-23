#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WDOF - Validador de Inputs
Uso:
  python validate_inputs.py --schema /ruta/schema.json --input /ruta/datos.csv --outdir /ruta/salidas

Requisitos:
  pip install jsonschema pandas

Salida:
  - validation_report.json : resumen + errores por fila/columna
  - validation_errors.csv  : tabla plana de errores (fila, columna, detalle)
  - validation_summary.txt : conteos y top issues
"""
import argparse, json, sys, os
from pathlib import Path
import pandas as pd

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except Exception as e:
    print("❌ Falta dependencia 'jsonschema'. Instala con: pip install jsonschema")
    sys.exit(1)

def load_schema(path):
    with open(path, "r") as f:
        return json.load(f)

def validate_row(obj, validator):
    errs = []
    for error in validator.iter_errors(obj):
        # ruta al campo
        path = ".".join([str(p) for p in error.path]) if error.path else "(objeto)"
        errs.append({
            "field": path,
            "message": error.message,
            "validator": error.validator,
            "validator_value": error.validator_value
        })
    return errs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True, help="Ruta a JSON Schema")
    ap.add_argument("--input", required=True, help="CSV de entrada a validar")
    ap.add_argument("--outdir", required=True, help="Directorio de salida")
    args = ap.parse_args()

    schema = load_schema(args.schema)
    validator = Draft202012Validator(schema)

    df = pd.read_csv(args.input)
    # convertir cada fila a dict
    errors = []
    for idx, row in df.iterrows():
        obj = row.dropna().to_dict()
        # Convertir NaN a None (para que no rompa constraints de tipo)
        for k,v in obj.items():
            if pd.isna(v):
                obj[k] = None
        row_errs = validate_row(obj, validator)
        for e in row_errs:
            errors.append({"row": int(idx), **e})

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # JSON report
    report = {
        "schema": str(Path(args.schema).name),
        "input": str(Path(args.input).name),
        "rows": int(len(df)),
        "errors": errors,
        "error_count": int(len(errors))
    }
    with open(outdir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # CSV plano de errores
    if errors:
        pd.DataFrame(errors).to_csv(outdir / "validation_errors.csv", index=False)
    else:
        pd.DataFrame(columns=["row","field","message","validator","validator_value"]).to_csv(outdir / "validation_errors.csv", index=False)

    # Summary txt
    lines = []
    lines.append(f"Schema    : {Path(args.schema).name}")
    lines.append(f"Input     : {Path(args.input).name}")
    lines.append(f"Rows      : {len(df)}")
    lines.append(f"Err count : {len(errors)}")
    if errors:
        top_fields = pd.Series([e["field"] for e in errors]).value_counts().head(10)
        lines.append("Top fields with errors:")
        for f, c in top_fields.items():
            lines.append(f"  - {f}: {int(c)}")
    with open(outdir / "validation_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("✅ Validación terminada.")
    print(f" - {outdir/'validation_report.json'}")
    print(f" - {outdir/'validation_errors.csv'}")
    print(f" - {outdir/'validation_summary.txt'}")

if __name__ == "__main__":
    main()
