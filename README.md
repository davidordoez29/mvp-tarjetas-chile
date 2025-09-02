# MVP – Optimización de Rentabilidad de Tarjetas de Crédito (Chile)

*Piso 0 – Configuración y Gobierno*

Este repositorio es la base del MVP para optimizar la rentabilidad de tarjetas en 4 aristas:
1) Riesgo de impago, 2) Yield por cliente, 3) Incentivos eficientes, 4) Capital/Provisiones.

## Estructura esperada (iremos creando en los siguientes pasos)
.
├── data/
│   ├── raw/              # datos simulados/entrada
│   └── processed/        # datasets limpios
├── docs/                 # documentación (diccionario de datos, notas)
├── notebooks/            # notebooks (Colab/Jupyter)
├── src/
│   ├── dashboard/        # Streamlit
│   ├── models/           # PD/LGD/EAD
│   ├── optimizer/        # OR-Tools / PuLP / CVXPY
│   └── utils/            # utilidades: io, métricas, validación
└── requirements.txt

## Cómo ejecutar gratis en Google Colab
1. Abre Colab → File > Open notebook > GitHub y pega la URL de este repo.
2. Crea un notebook y ejecuta: !pip install -r requirements.txt.

## Licencia
MIT.
