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

        def limpiar_y_mapear(temp_df):
            # Normalizar nombres de columnas a mayúsculas
            temp_df.columns = temp_df.columns.str.strip().str.upper()
            
            # Buscador inteligente de columnas
            mapping = {}
            for col in temp_df.columns:
                if "DOMINIO" in col: mapping[col] = "DOMINIO"
                if "FECHA" in col: mapping[col] = "FECHA"
                if "LITROS" in col: mapping[col] = "LITROS"
                if "KM" in col or "KILOMETRAJE" in col: mapping[col] = "KM"
                if "EMISIONES" in col or "CO2" in col: mapping[col] = "CO2"
                if "MARCA" in col: mapping[col] = "MARCA"
            
            temp_df = temp_df.rename(columns=mapping)
            
            # Limpiar Dominios
            if "DOMINIO" in temp_df.columns:
                temp_df["DOMINIO"] = temp_df["DOMINIO"].astype(str).str.replace(" ", "").str.upper()
            
            # Limpiar Números
            for c in ["LITROS", "KM", "CO2"]:
                if c in temp_df.columns:
                    temp_df[c] = pd.to_numeric(temp_df[c].astype(str).str.replace(".", "").str.replace(",", "."), errors="coerce").fillna(0)
            
            return temp_df

        df1 = limpiar_y_mapear(df1)
        df2 = limpiar_y_mapear(df2)

        # CRUCE POR DOMINIO (Unión total de lo que haya en ambas hojas)
        # Agrupamos df1 para evitar duplicados si hay varias fechas
        df1_resumen = df1.groupby("DOMINIO").agg({
            "KM": "sum", 
            "LITROS": "sum"
        }).reset_index()

        # Seleccionamos info extra de df2
        cols_df2 = [c for c in ["DOMINIO", "MARCA", "CO2"] if c in df2.columns]
        df2_resumen = df2[cols_df2].groupby("DOMINIO").first().reset_index()

        df_final = pd.merge(df1_resumen, df2_resumen, on="DOMINIO", how="left")
        
        return df_final
    except Exception as e:
        st.error(f"Error técnico: {e}")
        return pd.DataFrame()

df = cargar_datos()

if not df.empty:
    # Verificamos si las columnas existen antes de operar
    lts_tot = df["LITROS"].sum() if "LITROS" in df.columns else 0
    km_tot = df["KM"].sum() if "KM" in df.columns else 0
    
    # --- MÉTRICAS ---
    c1, c2, c3 = st.columns(3)
    c1.metric("⛽ Litros Totales", f"{lts_tot:,.0f} L")
    c2.metric("🛣️ Km Totales", f"{km_tot:,.0f} km")
    if "CO2" in df.columns:
        c3.metric("🌿 CO2 Total", f"{df['CO2'].sum():,.0f} kg")

    st.divider()

    # --- GRÁFICOS ---
    col_a, col_b = st.columns(2)
    
    with col_a:
        if "LITROS" in df.columns:
            st.subheader("📊 Litros por Patente")
            st.bar_chart(df.set_index("DOMINIO")["LITROS"])
            
    with col_b:
        if "KM" in df.columns:
            st.subheader("🛣️ Km por Patente")
            st.bar_chart(df.set_index("DOMINIO")["KM"])

    # --- TABLA ---
    st.subheader("📋 Resumen de Datos")
    st.dataframe(df, use_container_width=True)
else:
    st.warning("⚠️ No se pudieron procesar los datos. Revisá que las columnas de Google Sheets tengan los nombres correctos.")
