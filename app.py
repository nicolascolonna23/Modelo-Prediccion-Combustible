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
        # Carga
        df1 = pd.read_csv(url1)
        df2 = pd.read_csv(url2)

        # Limpieza de nombres (Mayúsculas y sin espacios)
        df1.columns = df1.columns.str.strip().str.upper()
        df2.columns = df2.columns.str.strip().str.upper()

        # Cruce
        df_merged = pd.merge(df1, df2, on=["FECHA", "DOMINIO"], how="inner")
        
        # --- BUSCADOR INTELIGENTE DE COLUMNAS ---
        # Buscamos en el resultado del merge qué nombres quedaron
        final_cols = {}
        for col in df_merged.columns:
            if "KILOMETRAJE" in col or "KM" in col: final_cols["KM"] = col
            if "LITROS_CONSUMIDOS" in col: final_cols["LITROS"] = col # Pesca _X o _Y
            if "EMISIONES" in col: final_cols["CO2"] = col
            if "CHOFER" in col: final_cols["CHOFER"] = col
            if "MARCA" in col: final_cols["MARCA"] = col

        # Verificamos que tengamos lo básico
        if "KM" in final_cols and "LITROS" in final_cols:
            df_final = df_merged.rename(columns={
                final_cols["KM"]: "KM",
                final_cols["LITROS"]: "LITROS_CONSUMIDOS",
                final_cols.get("CO2", "EMISIONES"): "EMISIONES"
            })
            
            # Solo nos quedamos con una columna de Litros (por si quedaron las dos)
            if isinstance(df_final["LITROS_CONSUMIDOS"], pd.DataFrame):
                df_final["LITROS_CONSUMIDOS"] = df_final["LITROS_CONSUMIDOS"].iloc[:, 0]

            return df_final.dropna(subset=["KM", "LITROS_CONSUMIDOS"])
        
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")
        return pd.DataFrame()

df = cargar_datos()

if not df.empty:
    # --- MÉTRICAS ---
    lts = df["LITROS_CONSUMIDOS"].sum()
    kms = df["KM"].sum()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("⛽ Litros Totales", f"{lts:,.0f} L")
    c2.metric("🛣️ Km Totales", f"{kms:,.0f} km")
    if "EMISIONES" in df.columns:
        c3.metric("🌿 CO2 Total", f"{df['EMISIONES'].sum():,.0f} kg")
    
    st.divider()
    
    # --- GRÁFICOS ---
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📊 Consumo por Patente")
        st.bar_chart(df.groupby("DOMINIO")["LITROS_CONSUMIDOS"].sum())
    with col_r:
        st.subheader("🚛 Km por Unidad")
        st.bar_chart(df.groupby("DOMINIO")["KM"].sum())
    
    st.divider()
    
    # --- TABLA ---
    st.subheader("📋 Detalle Validado")
    st.dataframe(df, use_container_width=True)
else:
    st.info("💡 No hay coincidencias. Asegurate que la 'FECHA' y el 'DOMINIO' estén escritos igual en ambas hojas de Google Sheets.")
