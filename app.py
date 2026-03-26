import pandas as pd
import streamlit as st
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

        # Normalizamos nombres de columnas a MAYÚSCULAS y sacamos espacios
        df1.columns = df1.columns.str.strip().str.upper()
        df2.columns = df2.columns.str.strip().str.upper()

        # Limpiamos patentes antes de unir
        for d in [df1, df2]:
            if 'DOMINIO' in d.columns:
                d['DOMINIO'] = d['DOMINIO'].astype(str).str.replace(' ', '').str.upper()

        # UNIÓN (Merge) por DOMINIO (más flexible que por fecha)
        # Agrupamos df1 para tener totales por patente
        df1_resumen = df1.groupby('DOMINIO').size().reset_index(name='VIAJES') # Solo para inicializar
        
        # Buscamos KM y LITROS en df1
        col_km = [c for c in df1.columns if 'KILOMETRAJE' in c or 'KM' in c][0]
        col_lts = [c for c in df1.columns if 'LITROS' in c][0]
        
        df1_data = df1.groupby('DOMINIO').agg({col_km: 'sum', col_lts: 'sum'}).reset_index()
        df1_data.columns = ['DOMINIO', 'KM', 'LITROS']

        # Buscamos CO2 y MARCA en df2
        cols_df2 = ['DOMINIO']
        if any('EMISIONES' in c or 'CO2' in c for c in df2.columns):
            col_co2 = [c for c in df2.columns if 'EMISIONES' in c or 'CO2' in c][0]
            cols_df2.append(col_co2)
        if 'MARCA' in df2.columns:
            cols_df2.append('MARCA')

        df2_data = df2[cols_df2].groupby('DOMINIO').first().reset_index()
        if 'CO2' in locals() or 'col_co2' in locals():
            df2_data = df2_data.rename(columns={col_co2: 'CO2'})

        # Unimos todo
        df_final = pd.merge(df1_data, df2_data, on="DOMINIO", how="left")

        # Limpiamos los números al final (sacamos puntos y comas de texto)
        for c in ['KM', 'LITROS', 'CO2']:
            if c in df_final.columns:
                df_final[c] = pd.to_numeric(df_final[c].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce').fillna(0)

        return df_final
    except Exception as e:
        st.error(f"Error procesando datos: {e}")
        return pd.DataFrame()

df = cargar_datos()

if not df.empty:
    # --- MÉTRICAS ---
    c1, c2, c3 = st.columns(3)
    c1.metric("⛽ Litros Totales", f"{df['LITROS'].sum():,.0f} L")
    c2.metric("🛣️ Km Totales", f"{df['KM'].sum():,.0f} km")
    if 'CO2' in df.columns:
        c3.metric("🌿 CO2 Total", f"{df['CO2'].sum():,.0f} kg")

    st.divider()

    # --- GRÁFICOS ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📊 Consumo por Patente")
        st.bar_chart(df.set_index("DOMINIO")["LITROS"])
    with col_b:
        if 'MARCA' in df.columns:
            st.subheader("🚛 Consumo por Marca")
            st.bar_chart(df.groupby("MARCA")["LITROS"].sum())

    # --- TABLA ---
    st.subheader("📋 Resumen Consolidado")
    st.dataframe(df, use_container_width=True)
else:
    st.warning("⚠️ No se pudieron cruzar las patentes. Revisá que la columna 'DOMINIO' exista en ambas hojas.")
