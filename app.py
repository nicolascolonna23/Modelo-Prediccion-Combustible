import pandas as pd
import streamlit as st
import requests
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
        # Carga cruda
        df1 = pd.read_csv(url1)
        df2 = pd.read_csv(url2)

        # --- LIMPIEZA INDIVIDUAL ---
        def limpiar_tabla(temp_df):
            # Columnas a mayúsculas
            temp_df.columns = temp_df.columns.str.strip().str.upper()
            # Normalizar Patente
            if 'DOMINIO' in temp_df.columns:
                temp_df['DOMINIO'] = temp_df['DOMINIO'].astype(str).str.replace(' ', '').str.upper()
            # Normalizar Fecha
            if 'FECHA' in temp_df.columns:
                temp_df['FECHA'] = pd.to_datetime(temp_df['FECHA'], errors='coerce').dt.date
            # Renombrar columnas clave antes del cruce para evitar _X e _Y
            reemplazos = {
                'KILOMETRAJE SATELITAL': 'KM',
                'LITROS CONSUMIDOS TELEMETRIA': 'LITROS_FINAL',
                'LITROS_CONSUMIDOS': 'LITROS_FINAL',
                'EMISIONES (KG CO2)': 'CO2_FINAL'
            }
            return temp_df.rename(columns=reemplazos)

        df1 = limpiar_tabla(df1)
        df2 = limpiar_tabla(df2)

        # --- CRUCE (Merge) ---
        # Solo nos quedamos con las columnas que queremos de df2 para no duplicar 'LITROS'
        cols_interes_df2 = [c for c in df2.columns if c in ['FECHA', 'DOMINIO', 'CO2_FINAL', 'MARCA', 'CHOFER']]
        df2_filtrado = df2[cols_interes_df2]

        df_merged = pd.merge(df1, df2_filtrado, on=["FECHA", "DOMINIO"], how="inner")
        
        # Si el cruce está vacío, devolvemos error para diagnosticar
        if df_merged.empty:
            return "VACIO", df1, df2

        return "OK", df_merged, None
        
    except Exception as e:
        return f"ERROR: {e}", None, None

# Ejecución
estado, df, debug = cargar_datos()

if estado == "OK":
    # --- MÉTRICAS ---
    # Usamos los nombres que forzamos arriba
    litros = df['LITROS_FINAL'].sum() if 'LITROS_FINAL' in df.columns else 0
    kms = df['KM'].sum() if 'KM' in df.columns else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("⛽ Litros Totales", f"{litros:,.0f} L")
    c2.metric("🛣️ Km Totales", f"{kms:,.0f} km")
    if 'CO2_FINAL' in df.columns:
        c3.metric("🌿 CO2 Total", f"{df['CO2_FINAL'].sum():,.0f} kg")
    
    st.divider()

    # --- GRÁFICOS ---
    st.subheader("📊 Consumo por Patente")
    chart_data = df.groupby("DOMINIO")["LITROS_FINAL"].sum().sort_values(ascending=False)
    st.bar_chart(chart_data)
    
    st.subheader("📋 Datos Detallados")
    st.dataframe(df, use_container_width=True)

elif estado == "VACIO":
    st.error("❌ No hay coincidencia exacta de FECHA y DOMINIO entre las hojas.")
    st.info("Revisá que ambas hojas tengan datos del mismo día y las patentes escritas igual.")
else:
    st.error(f"Hubo un problema: {estado}")
