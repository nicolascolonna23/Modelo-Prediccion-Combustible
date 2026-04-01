import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io

# 1. ESTÉTICA
st.set_page_config(page_title="Expreso Diemar - Carbon Tracker", layout="wide")
st.title("🌿 Reporte de Operación - Expreso Diemar")

# 2. CARGA DE DATOS CON LIMPIEZA DE TIPOS
@st.cache_data(ttl=60)
def get_clean_data():
    base_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR35NkYPtJrOrdYHLGUH7GIW93s5cPAqQ0zEk5fP1c3gvErwbUW7HJ2OeWBYaBVsYKVmCf0yhLvs6eG/pub?output=csv"
    df_tel = pd.read_csv(f"{base_url}&gid=1044040871")
    df_uni = pd.read_csv(f"{base_url}&gid=882343299")
    
    def clean_columns(df):
        # Normalizar nombres
        df.columns = df.columns.str.strip().str.upper().str.replace('Í', 'I').str.replace('Á', 'A')
        
        # Buscador y Limpiador de números
        mapeo = {
            "DOMINIO": "DOMINIO", "FECHA": "FECHA", "MARCA": "MARCA",
            "DISTANCIA": "KMS", "KM": "KMS",
            "EMISIONES": "CO2", "CO2": "CO2",
            "RALENTI": "RALENTI"
        }
        
        # Renombrar según palabras clave
        for col in df.columns:
            for clave, nuevo_nombre in mapeo.items():
                if clave in col:
                    df = df.rename(columns={col: nuevo_nombre})
        
        # --- EL TRUCO PARA EVITAR EL TYPEERROR ---
        # Forzamos que KMS, CO2 y RALENTI sean números. 
        # Si hay un texto, lo convierte en 0 (NaN -> 0)
        for col_num in ["KMS", "CO2", "RALENTI"]:
            if col_num in df.columns:
                df[col_num] = pd.to_numeric(df[col_num].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        
        return df

    df_tel = clean_columns(df_tel)
    df_uni = clean_columns(df_uni)

    # Preparar fechas y merge
    for d in [df_tel, df_uni]:
        if 'FECHA' in d.columns:
            d['FECHA_DT'] = pd.to_datetime(d['FECHA'], errors='coerce')
            d['MES'] = d['FECHA_DT'].dt.strftime('%Y-%m')
        if 'DOMINIO' in d.columns:
            d['DOMINIO'] = d['DOMINIO'].astype(str).str.replace(' ', '').str.upper()

    df_merged = pd.merge(df_tel, df_uni, on=["DOMINIO", "MES"], suffixes=('', '_DROP'))
    return df_merged.loc[:, ~df_merged.columns.str.contains('_DROP')]

try:
    df_master = get_clean_data()
except Exception as e:
    st.error(f"Error crítico: {e}")
    st.stop()

# 3. FILTROS Y DASHBOARD
meses = sorted(df_master["MES"].unique().tolist(), reverse=True)
mes_sel = st.sidebar.selectbox("Seleccionar Mes", meses)
df_filtrado = df_master[df_master["MES"] == mes_sel]

# 4. MÉTRICAS (Ahora seguras de sumar)
if not df_filtrado.empty:
    c1, c2, c3 = st.columns(3)
    
    # El sum() ahora siempre funcionará porque forzamos float arriba
    v_co2 = df_filtrado['CO2'].sum()
    v_kms = df_filtrado['KMS'].sum()
    v_ral = df_filtrado['RALENTI'].sum() if 'RALENTI' in df_filtrado.columns else 0

    c1.metric("CO₂ TOTAL", f"{v_co2:,.0f} kg")
    c2.metric("KM TOTALES", f"{v_kms:,.0f} km")
    c3.metric("RALENTÍ", f"{v_ral:,.1f} L")

    st.divider()
    st.subheader("📋 Detalle Unidades")
    st.dataframe(df_filtrado[['DOMINIO', 'MARCA', 'KMS', 'CO2', 'RALENTI']], use_container_width=True)
