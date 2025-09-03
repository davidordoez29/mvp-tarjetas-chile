# 📊 MVP Bancario — Optimización de Rentabilidad Ajustada por Riesgo

## 🚀 Introducción
Este proyecto desarrolla un *MVP (Minimum Viable Product = Producto Mínimo Viable)* bancario para *optimizar la rentabilidad ajustada por riesgo* en tarjetas de crédito y préstamos.  

Integra:
- *Simulación de datos bancarios sintéticos* (clientes, cuentas, transacciones).  
- *Modelos de riesgo:* PD (Probability of Default), LGD (Loss Given Default), EAD (Exposure At Default).  
- *Optimización matemática (Linear Programming con PuLP):* maximiza utilidad neta esperada considerando ingresos financieros, pérdidas y costos.  

---

## 📂 Estructura del Proyecto
. ├── 01_simulacion_datos.ipynb       # Piso 1: generación de datos sintéticos ├── 02_modelos_riesgo.ipynb         # Piso 2: modelos de riesgo + Piso 3: optimización ├── requirements.txt                # Librerías necesarias ├── docs/                           # Manuales, informes ejecutivos, figuras └── data/                           # Archivos CSV generados (simulados + optimizados)
---

## 🏗️ Metodología por Pisos
- *Piso 0 — Preparación del entorno:* GitHub + Colab + dependencias ✅  
- *Piso 1 — Simulación:* datasets sintéticos de clientes, cuentas, saldos, transacciones ✅  
- *Piso 2 — Modelos de Riesgo:* entrenamiento de PD, LGD, EAD + cálculo de EL (Expected Loss) ✅  
- *Piso 3 — Optimización:* maximización de rentabilidad ajustada al riesgo con programación lineal ✅  
- *Piso 4 — Dashboard Ejecutivo (en desarrollo):* visualización ejecutiva de KPIs 📊  

---

## ⚙️ Orden de Ejecución
1. Abrir *01_simulacion_datos.ipynb* → correr todas las celdas → genera clientes.csv, cuentas.csv, saldos_mensuales.csv, etc.  
2. Abrir *02_modelos_riesgo.ipynb* y ejecutar:  
   - Celda 18 → preparar dataset de optimización (opt_input_piso3.csv).  
   - Celda 19 → optimización básica con PuLP.  
   - Celda 20 → escenarios y restricciones avanzadas.  
   - Celda 21 → visualizaciones (histogramas, scatter PD vs tasa, barras por segmento).  
   - Celda 22 → informe ejecutivo en Markdown + CSV de KPIs.  
   - Celda 23 → consolidado final (CSV + Parquet).  

---

## 📊 Outputs Principales
- *Datos de simulación:* clientes.csv, cuentas.csv, saldos_mensuales.csv, transacciones.csv  
- *Resultados optimizados:*  
  - opt_results_*.csv → decisiones óptimas por escenario  
  - opt_kpis_*.csv → KPIs comparativos  
  - fig_*.png → gráficos de tasas, PD, utilidades  
  - informe_ejecutivo_*.md → resumen ejecutivo  
  - consolidado_piso3_*.csv → consolidado final  

---

## 📈 Ejemplo de KPI (Escenario BASE_PD12_R45)
- *EAD total asignado:* 1,200,000  
- *Utilidad total:* 85,000  
- *PD ponderado (EAD):* 0.095  
- *APR promedio ponderado:* 0.31  

---

## 🛠️ Tecnologías
- Python 3.12  
- Pandas, Numpy  
- LightGBM, Scikit-learn  
- PuLP (Linear Programming)  
- Matplotlib  

---

## 📌 Estado del Proyecto
- Piso 0: ✅  
- Piso 1: ✅  
- Piso 2: ✅  
- Piso 3: ✅  
- Piso 4 (Dashboard Ejecutivo): 🔄 en desarrollo  

---

## 👥 Autores
Desarrollado por *William Ordóñez & Socio IA* 🤝  
Proyecto de *Propiedad Intelectual (IP)* con potencial de comercialización bancaria.
