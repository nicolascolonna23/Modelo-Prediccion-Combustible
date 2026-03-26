import pandas as pd
import streamlit as st
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import io

# Configuración
st.set_page_config(page_title="Expreso Diemar - Analítica", layout="wide")
st.title("🚛 Dashboard de Consumo de Flota")

# 1. FUENTES DE DATOS
base_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR35NkYPtJrOrdYHLGUH7GIW93s5cPAqQ0zEk5fP1c3gvErwbUW7HJ2OeWBYaBVsYKVmCf0yhLvs6eG/pub?output=csv"
gid_telemetria = "1044040871"
gid_unidades = "882343299" 
url1 = f"{base_url}&gid={gid_telemetria}"
url2 = f"{base_url}&gid={gid_unidades}"

@st.cache_data(ttl=60)
def cargar_datos():
    try:
        df1 = pd.read_csv(url1)
        df2 = pd.read_csv(url2)

        # Normalización de nombres
        for d in [df1, df2]:
            d.columns = d.columns.str.strip().str.upper()

        # Cruce por FECHA y DOMINIO
        df_merged = pd.merge(df1, df2, on=["FECHA", "DOMINIO"], how="inner")
        
        # --- MAPEO SEGÚN TU CAPTURA ---
        # Renombramos las columnas para que el código las entienda
        rename_dict = {
            'KILOMETRAJE SATELITAL': 'KM',
            'LITROS_CONSUMIDOS_X': 'LITROS_CONSUMIDOS', # Usamos la versión de telemetría
            'EMISIONES (KG CO2)': 'EMISIONES',
            'TIEMPO FUNCIONAMIENTO MOTOR (H:M)': 'TIEMPO_MOTOR'
        }
        df_merged = df_merged.rename(columns=rename_dict)
        
        # Filtramos solo lo que necesitamos
        cols_finales = ["FECHA", "DOMINIO", "KM", "LITROS_CONSUMIDOS", "EMISIONES"]
        return df_merged[cols_finales].dropna()
        
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

df = cargar_datos()

if not df.empty:
    # --- MÉTRICAS SUPERIORES ---
    lts_tot = df["LITROS_CONSUMIDOS"].sum()
    km_tot = df["KM"].sum()
    co2_tot = df["EMISIONES"].sum()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("⛽ Litros Totales", f"{lts_tot:,.0f} L")
    c2.metric("🛣️ Km Totales", f"{km_tot:,.0f} km")
    c3.metric("🌿 CO2 Total", f"{co2_tot:,.0f} kg")
    
    st.divider()
    
    # --- GRÁFICOS ---
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader("📊 Consumo por Patente")
        st.bar_chart(df.groupby("DOMINIO")["LITROS_CONSUMIDOS"].sum())
        
    with col_r:
        st.subheader("🌍 Emisiones por Patente")
        st.bar_chart(df.groupby("DOMINIO")["EMISIONES"].sum())
    
    st.divider()
    
    # --- TABLA DE DATOS ---
    st.subheader("📋 Detalle de Operación Validado")
    st.dataframe(df, use_container_width=True)
else:
    st.info("💡 Esperando coincidencia de datos entre las hojas de Google Sheets.")
