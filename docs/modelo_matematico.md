# Modelo Matemático — MVP Bancario (4 Aristas)

- *Escenario*: base
- *Periodo base*: ANUAL

## Supuestos principales

- PD anual, LGD de cartera retail sin colateral, APR efectiva anual dentro de bandas.
- Elasticidad precio-volumen calibrada; sensibilidad PD a tasa (BETA_PD_RATE).
- Capital proxy: Capital = RW × K × EAD con RW=0.75, K=10.5% (ajustable).
- Provisiones ~ EL (proxy regulatoria).

## Fórmulas clave

- *EL* = PD × LGD × EAD
- *Ingreso* ≈ (APR − funding_cost) × EAD − ops_cost × EAD
- *Capital* ≈ RW × K × EAD
- *Incentivos ROI* = Ingreso incremental / Costo incentivos

## Guardrails

| arista   | regla                                  |    valor_base |    valor_opt |   objetivo | OK    |
|:---------|:---------------------------------------|--------------:|-------------:|-----------:|:------|
| A1/A2    | EL_opt ≤ EL_base×EL_OBJ_FACTOR         |   7.74469e+07 |  9.68782e+07 |       0.9  | False |
| A3       | Costo_incentivos ≤ ingreso_base×budget |   1.15022e+09 |  2.40859e+07 |       0.06 | True  |
| A3       | ROI_port ≥ ROI_min                     | nan           | 20.0954      |       0.15 | True  |
| A4       | Capital_opt ≤ Capital_base             |   3.64764e+08 |  4.74191e+08 |     nan    | False |

## Limitaciones y validación

- Datos simulados; se recalibra con datos reales del banco.
- Stress testing disponible vía escenarios (base/estrés/optimista).
- Conversión temporal con fórmulas consistentes.
