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
        df1 = pd.read_csv(url1)
        df2 = pd.read_csv(url2)

        def limpiar_tabla(temp_df):
            # Limpiar nombres de columnas
            temp_df.columns = temp_df.columns.str.strip().str.upper()
            
            # Limpiar Patentes (Sacar espacios)
            if 'DOMINIO' in temp_df.columns:
                temp_df['DOMINIO'] = temp_df['DOMINIO'].astype(str).str.replace(' ', '').str.upper()
            
            # Limpiar Números (sacar puntos de miles y comas decimales)
            for col in temp_df.columns:
                if any(x in col for x in ['KILOMETRAJE', 'LITROS', 'EMISIONES']):
                    temp_df[col] = pd.to_numeric(temp_df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce').fillna(0)
            
            # Renombrado estándar
            reemplazos = {
                'KILOMETRAJE SATELITAL': 'KM',
                'LITROS CONSUMIDOS TELEMETRIA': 'LITROS',
                'LITROS_CONSUMIDOS': 'LITROS',
                'EMISIONES (KG CO2)': 'CO2'
            }
            return temp_df.rename(columns=reemplazos)

        df1 = limpiar_tabla(df1)
        df2 = limpiar_tabla(df2)

        # CRUCE FLEXIBLE: Si el cruce por Fecha y Dominio falla, usamos solo Dominio
        # Agrupamos df1 para tener totales por patente
        df1_agrupado = df1.groupby('DOMINIO').agg({'KM': 'sum', 'LITROS': 'sum'}).reset_index()
        
        # Agrupamos df2 para tener info de la unidad
        columnas_df2 = [c for c in ['DOMINIO', 'MARCA', 'CO2'] if c in df2.columns]
        df2_agrupado = df2.groupby('DOMINIO').agg({c: 'first' if c == 'MARCA' else 'sum' for c in columnas_df2 if c != 'DOMINIO'}).reset_index()

        df_final = pd.merge(df1_agrupado, df2_agrupado, on="DOMINIO", how="inner")
        
        return df_final
    except Exception as e:
        st.error(f"Error técnico: {e}")
        return pd.DataFrame()

df = cargar_datos()

if not df.empty:
    # --- MÉTRICAS ---
    lts_val = float(df['LITROS'].sum())
    kms_val = float(df['KM'].sum())
    
    c1, c2, c3 = st.columns(3)
    c1.metric("⛽ Litros Totales", f"{lts_val:,.0f} L")
    c2.metric("🛣️ Km Totales", f"{kms_val:,.0f} km")
    if 'CO2' in df.columns:
        c3.metric("🌿 CO2 Total", f"{float(df['CO2'].sum()):,.0f} kg")
    
    st.divider()

    # --- GRÁFICOS ---
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader("📊 Consumo por Patente")
        st.bar_chart(df.set_index("DOMINIO")["LITROS"])
        
    with col_r:
        if 'MARCA' in df.columns:
            st.subheader("🚛 Consumo por Marca")
            resumen_marca = df.groupby("MARCA")["LITROS"].sum()
            st.bar_chart(resumen_marca)
    
    st.divider()
    st.subheader("📋 Resumen Consolidado")
    st.dataframe(df, use_container_width=True)
else:
    st.warning("⚠️ No se encontraron coincidencias de Patentes (DOMINIO) entre ambas hojas.")
    st.info("Asegúrate de que los Dominios estén escritos igual (ej: AC078XC).")
