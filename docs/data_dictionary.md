# Diccionario de Datos (v0)

## clientes.csv
- id_cliente (str)
- fecha_alta (date)
- ingreso_mensual (float)
- region (str)
- segmento (str)

## cuentas.csv
- id_cliente (str)
- id_cuenta (str)
- limite (float)
- apr (float)

## saldos_mensuales.csv
- id_cliente (str)
- mes (yyyy-mm)
- saldo (float)
- pago (float)
- dpd (int)  # días de atraso

## transacciones.csv
- id_cliente (str)
- fecha (date)
- monto (float)
- mcc (str)  # categoría comercio
- canal (str)

## labels_riesgo.csv
- id_cliente (str)
- default_12m (0/1)
- lgd_real (0..1)
- stage_ifrs9 (1/2)

## interchange.csv
- mcc (str)
- margen (0..1)
