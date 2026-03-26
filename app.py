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

        # Normalización extrema de nombres
        for d in [df1, df2]:
            d.columns = d.columns.str.strip().str.upper()

        def normalizar(df_temp):
            rename_map = {}
            for col in df_temp.columns:
                if "DISTANCIA" in col or "KM" in col: rename_map[col] = "KM"
                # AMPLIAMOS LA BÚSQUEDA DE LITROS:
                if "LITROS" in col or "LTS" in col or "CONSUMO" in col and "100" not in col: 
                    rename_map[col] = "LITROS_CONSUMIDOS"
                if "RALENTI" in col: rename_map[col] = "RALENTI_LTS"
                if "100KM" in col: rename_map[col] = "L100KM"
            return df_temp.rename(columns=rename_map)

        df1 = normalizar(df1)
        df2 = normalizar(df2)

        # Cruce
        df_merged = pd.merge(df1, df2, on=["FECHA", "DOMINIO"], how="inner")
        
        # SI FALLA, MOSTRAMOS LAS COLUMNAS DISPONIBLES PARA DEBUGGEAR
        cols_necesarias = ["KM", "LITROS_CONSUMIDOS", "RALENTI_LTS"]
        presentes = [c for c in cols_necesarias if c in df_merged.columns]
        
        if len(presentes) < len(cols_necesarias):
            st.error(f"⚠️ Error de columnas. Encontradas: {presentes}")
            st.write("Columnas disponibles en el archivo:", list(df_merged.columns))
            return pd.DataFrame()

        df_merged = df_merged.dropna(subset=cols_necesarias)
        return df_merged
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

df = cargar_datos()

if not df.empty:
    # --- MÉTRICAS ---
    df["COSTO"] = df["LITROS_CONSUMIDOS"] * 1250 # Precio fijo para probar
    
    c1, c2, c3 = st.columns(3)
    c1.metric("⛽ Litros Totales", f"{df['LITROS_CONSUMIDOS'].sum():,.0f} L")
    c2.metric("💰 Costo Total", f"$ {df['COSTO'].sum():,.0f}")
    c3.metric("🛣️ Km Totales", f"{df['KM'].sum():,.0f} km")
    
    st.divider()
    st.subheader("📊 Consumo por Patente")
    st.bar_chart(df.groupby("DOMINIO")["LITROS_CONSUMIDOS"].sum())
    
    st.subheader("📋 Datos Cruzados")
    st.dataframe(df, use_container_width=True)
else:
    st.info("💡 Revisá arriba la lista de columnas disponibles para identificar cómo se llaman los litros en tu Excel.")
