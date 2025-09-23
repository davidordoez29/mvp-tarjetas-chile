# WDOF – Validación de Entradas (Esquemas + Plantillas)

## Archivos
- JSON Schema (mínimo): schema_minimal_operativo.json
- JSON Schema (full):   schema_full_enterprise.json
- Template CSV (mínimo): template_minimal_headers.csv
- Template CSV (full):   template_full_headers.csv
- Script validador:      validate_inputs.py
- Script parquet:        generate_parquet_templates.py
- Parquet generados ahora: (pyarrow no disponible; generar en Colab con el script) 

## Uso rápido

### 1) Validar un CSV contra un schema
```bash
pip install jsonschema pandas
python validate_inputs.py --schema schema_minimal_operativo.json --input datos_banco.csv --outdir ./val_out
```

### 2) Generar plantillas Parquet (en un entorno con pyarrow)
```bash
pip install pyarrow pandas
python generate_parquet_templates.py --minimal template_minimal_headers.csv --full template_full_headers.csv --outdir ./parquet_out
```
