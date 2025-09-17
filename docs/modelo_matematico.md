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

| Métrica            | Base        | Optimizado   | Cambio Absoluto   | Cambio Relativo   |
|:-------------------|:------------|:-------------|:------------------|:------------------|
| Yield (Ingreso)    | $0          | $0           | $0                | -                 |
| Expected Loss (EL) | $77.446.930 | $96.878.232  | $19.431.302       | 25.09%            |
| Capital Requerido  | $0          | $474.190.847 | $474.190.847      | -                 |

## Limitaciones y validación

- Datos simulados; se recalibra con datos reales del banco.
- Stress testing disponible vía escenarios (base/estrés/optimista).
- Conversión temporal con fórmulas consistentes.
