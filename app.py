import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io

# 1. CONFIGURACIÓN Y ESTÉTICA "ECO-DARK"
st.set_page_config(page_title="Expreso Diemar - Carbon Tracker", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.9)), 
                    url("https://raw.githubusercontent.com/nicolascolonna23/DetectorDesvios/main/IMG_3101.jpg"); 
        background-size: cover; background-attachment: fixed;
    }
    [data-testid="stSidebar"] { background-color: rgba(10, 15, 10, 0.98); }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px; border-radius: 10px; border-top: 4px solid #2e7d32;
    }
    h1, h2, h3, h4, span, p { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DESDE GOOGLE SHEETS
@st.cache_data(ttl=60) 
def get_sheets_data():
    # URL de publicación para CSV
    base_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR35NkYPtJrOrdYHLGUH7GIW93s5cPAqQ0zEk5fP1c3gvErwbUW7HJ2OeWBYaBVsYKVmCf0yhLvs6eG/pub?output=csv"
    gid_telemetria = "1044040871"
    gid_unidades = "882343299"
    
    def download_sheet(gid):
        url = f"{base_url}&gid={gid}"
        response = requests.get(url)
        return pd.read_csv(io.StringIO(response.text))

    df_tel = download_sheet(gid_telemetria)
    df_con = download_sheet(gid_unidades)
    
    # --- RENOMBRADO INTELIGENTE ---
    # Limpiamos nombres de columnas (mayúsculas, sin tildes, sin espacios extras)
    for d in [df_tel, df_con]:
        d.columns = d.columns.str.strip().str.replace('í', 'i').str.replace('á', 'a').str.upper()

    # Mapeo de nombres para asegurar que el código los encuentre
    map_tel = {'DISTANCIA RECORRIDA TELEMETRIA': 'KMS', 'RALENTI (LTS)': 'LTS_RALENTI'}
    map_con = {'EMISIONES (KG CO2)': 'CO2'}
    
    df_tel = df_tel.rename(columns=map_tel)
    df_con = df_con.rename(columns=map_con)

    # Limpieza de Dominio y Fechas
    for d in [df_tel, df_con]:
        if 'DOMINIO' in d.columns:
            d['DOMINIO'] = d['DOMINIO'].astype(str).str.replace(' ', '').str.upper()
        if 'FECHA' in d.columns:
            d['FECHA_DT'] = pd.to_datetime(d['FECHA'], errors='coerce')
            d['KEY_TIEMPO'] = d['FECHA_DT'].dt.strftime('%Y-%m')

    # UNIÓN: Pegamos las dos hojas
    df = pd.merge(df_tel, df_con, on=["DOMINIO", "KEY_TIEMPO"], suffixes=('', '_DROP'))
    return df.loc[:,~df.columns.str.contains('_DROP')]

try:
    df_master = get_sheets_data()
except Exception as e:
    st.error(f"❌ Error al leer los datos: {e}")
    st.stop()

# 3. SIDEBAR Y FILTROS
with st.sidebar:
    st.image("https://raw.githubusercontent.com/nicolascolonna23/DetectorDesvios/main/logo_diemar4.png", width=200)
    st.divider()
    meses_cruzados = sorted(df_master["KEY_TIEMPO"].unique().tolist(), reverse=True)
    mes_sel = st.selectbox("📅 Período de Análisis", meses_cruzados)
    marcas = ["Todas"] + sorted(df_master["MARCA"].unique().tolist())
    marca_sel = st.selectbox("🏭 Filtrar Marca", marcas)

df_actual = df_master[df_master["KEY_TIEMPO"] == mes_sel]
if marca_sel != "Todas":
    df_actual = df_actual[df_actual["MARCA"] == marca_sel]

# 4. DASHBOARD
st.title(f"🌿 Centro de Sustentabilidad — {mes_sel}")

if df_actual.empty:
    st.warning("No hay datos para mostrar.")
else:
    # MÉTRICAS (Usando los nuevos nombres cortos)
    c1, c2, c3, c4 = st.columns(4)
    
    co2_total = df_actual['CO2'].sum()
    kms_total = df_actual['KMS'].sum()
    lts_ralenti = df_actual['LTS_RALENTI'].sum() if 'LTS_RALENTI' in df_actual.columns else 0
    
    c1.metric("CO₂ EMITIDO", f"{co2_total:,.0f} kg")
    c2.metric("KM RECORRIDOS", f"{kms_total:,.0f} km")
    
    intensidad = (co2_total / kms_total * 1000) if kms_total > 0 else 0
    c3.metric("INTENSIDAD CO₂", f"{intensidad:.1f} g/km")
    
    arboles = int(co2_total / 20)
    c4.metric("COMPENSACIÓN", f"{arboles} Árboles")

    st.divider()

    # GRÁFICOS
    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        st.subheader("📊 Emisiones por Patente")
        fig_bar = px.bar(df_actual.sort_values("CO2", ascending=False), 
                         x="DOMINIO", y="CO2", color="CO2",
                         color_continuous_scale="Greens", template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.subheader("📉 CO₂ por Marca")
        fig_pie = px.pie(df_actual, values='CO2', names='MARCA', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("📋 Auditoría de Unidades")
    st.dataframe(df_actual[['DOMINIO', 'MARCA', 'FECHA', 'KMS', 'CO2']], use_container_width=True)

st.caption("Sincronizado con Google Sheets | Actualización cada 60s")
