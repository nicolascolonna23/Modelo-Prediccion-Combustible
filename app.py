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
            temp_df.columns = temp_df.columns.str.strip().str.upper()
            if 'DOMINIO' in temp_df.columns:
                temp_df['DOMINIO'] = temp_df['DOMINIO'].astype(str).str.replace(' ', '').str.upper()
            if 'FECHA' in temp_df.columns:
                temp_df['FECHA'] = pd.to_datetime(temp_df['FECHA'], errors='coerce').dt.date
            
            # Limpieza de números (sacar puntos de miles y comas decimales si vienen como texto)
            cols_numericas = ['KILOMETRAJE SATELITAL', 'LITROS CONSUMIDOS TELEMETRIA', 'LITROS_CONSUMIDOS', 'EMISIONES (KG CO2)']
            for col in cols_numericas:
                if col in temp_df.columns:
                    temp_df[col] = pd.to_numeric(temp_df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce').fillna(0)
            
            reemplazos = {
                'KILOMETRAJE SATELITAL': 'KM',
                'LITROS CONSUMIDOS TELEMETRIA': 'LITROS_FINAL',
                'LITROS_CONSUMIDOS': 'LITROS_FINAL',
                'EMISIONES (KG CO2)': 'CO2_FINAL'
            }
            return temp_df.rename(columns=reemplazos)

        df1 = limpiar_tabla(df1)
        df2 = limpiar_tabla(df2)

        # Cruce simplificado: solo por DOMINIO si la fecha te está dando problemas, 
        # pero probemos con ambos primero
        df_merged = pd.merge(df1, df2[['FECHA', 'DOMINIO', 'CO2_FINAL', 'MARCA']], on=["FECHA", "DOMINIO"], how="inner")
        
        return df_merged
    except Exception as e:
        st.error(f"Error técnico: {e}")
        return pd.DataFrame()

df = cargar_datos()

if not df.empty:
    # --- MÉTRICAS ---
    # Usamos sum() directo y aseguramos que sean floats
    lts_val = float(df['LITROS_FINAL'].sum())
    kms_val = float(df['KM'].sum())
    co2_val = float(df['CO2_FINAL'].sum()) if 'CO2_FINAL' in df.columns else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("⛽ Litros Totales", f"{lts_val:,.0f} L")
    c2.metric("🛣️ Km Totales", f"{kms_val:,.0f} km")
    c3.metric("🌿 CO2 Total", f"{co2_val:,.0f} kg")
    
    st.divider()

    # --- GRÁFICOS ---
    st.subheader("📊 Consumo por Patente")
    chart_data = df.groupby("DOMINIO")["LITROS_FINAL"].sum().sort_values(ascending=False)
    st.bar_chart(chart_data)
    
    st.subheader("📋 Auditoría de Datos")
    st.dataframe(df, use_container_width=True)
else:
    st.warning("⚠️ No hay datos cruzados. Revisá que las fechas y patentes coincidan en ambas pestañas.")
    # Debug para que veas qué está pasando
    st.write("Si ves esto, es porque la FECHA o el DOMINIO no hacen 'match'.")
