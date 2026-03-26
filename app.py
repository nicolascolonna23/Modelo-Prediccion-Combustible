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

        # --- LIMPIEZA PROFUNDA ANTES DEL CRUCE ---
        for d in [df1, df2]:
            # Nombres de columnas en mayúsculas y sin espacios
            d.columns = d.columns.str.strip().str.upper()
            
            # Limpieza de DOMINIO: sacamos espacios y pasamos a mayúsculas
            if 'DOMINIO' in d.columns:
                d['DOMINIO'] = d['DOMINIO'].astype(str).str.replace(' ', '').str.upper()
            
            # Limpieza de FECHA: forzamos formato fecha pura (sin horas)
            if 'FECHA' in d.columns:
                d['FECHA'] = pd.to_datetime(d['FECHA'], errors='coerce').dt.date

        # CRUCE (Merge)
        df_merged = pd.merge(df1, df2, on=["FECHA", "DOMINIO"], how="inner")
        
        # Si el cruce da vacío, mostramos qué hay en cada una para diagnosticar
        if df_merged.empty:
            return "ERROR_VACIO", df1, df2

        # Buscador de columnas para gráficos
        final_cols = {}
        for col in df_merged.columns:
            if "KILOMETRAJE" in col or "KM" in col: final_cols["KM"] = col
            if "LITROS_CONSUMIDOS" in col: final_cols["LITROS"] = col
            if "EMISIONES" in col: final_cols["CO2"] = col

        df_final = df_merged.rename(columns={
            final_cols.get("KM"): "KM",
            final_cols.get("LITROS"): "LITROS_CONSUMIDOS",
            final_cols.get("CO2"): "EMISIONES"
        })

        return "OK", df_final, None
        
    except Exception as e:
        return f"ERROR_SISTEMA: {e}", None, None

# Ejecución
estado, resultado, debug = cargar_datos()

if estado == "OK":
    df = resultado
    # Métricas
    c1, c2, c3 = st.columns(3)
    c1.metric("⛽ Litros Totales", f"{df['LITROS_CONSUMIDOS'].sum():,.0f} L")
    c2.metric("🛣️ Km Totales", f"{df['KM'].sum():,.0f} km")
    if "EMISIONES" in df.columns:
        c3.metric("🌿 CO2 Total", f"{df['EMISIONES'].sum():,.0f} kg")
    
    st.divider()
    st.subheader("📊 Consumo por Patente")
    st.bar_chart(df.groupby("DOMINIO")["LITROS_CONSUMIDOS"].sum())
    
    st.subheader("📋 Detalle de Datos Cruzados")
    st.dataframe(df, use_container_width=True)

elif estado == "ERROR_VACIO":
    st.error("❌ No hay coincidencias entre las dos hojas.")
    st.write("### 🔍 Diagnóstico de Datos:")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Datos Telemetría (Hoja 1):**")
        st.write(debug[['FECHA', 'DOMINIO']].head())
    with col_b:
        st.write("**Datos Unidades/CO2 (Hoja 2):**")
        st.write(resultado[['FECHA', 'DOMINIO']].head())
    st.info("💡 Asegurate que las fechas y patentes de arriba se vean EXACTAMENTE iguales.")
else:
    st.error(estado)
