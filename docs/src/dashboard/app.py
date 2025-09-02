import streamlit as st

st.set_page_config(page_title='MVP Tarjetas – Dashboard', layout='wide')
st.title('MVP – Optimización de Rentabilidad de Tarjetas (Chile)')

st.markdown('''
*Piso 7 – Dashboard (placeholder)*  
Este panel mostrará KPIs por arista: Riesgo, Yield, Incentivos y Capital.
''')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric('ROE Portafolio', '—', '—')
with col2:
    st.metric('ECL (12m)', '—', '—')
with col3:
    st.metric('Capital Consumido', '—', '—')

st.info('Cargar resultados simulados más adelante.')
