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
            temp_df.columns = temp_df.columns.str.strip().str.upper()
            
            # Buscador inteligente de columnas
            new_cols = {}
            for col in temp_df.columns:
                if "DOMINIO" in col: new_cols[col] = "DOMINIO"
                if "LITROS" in col: new_cols[col] = "LITROS"
                if "KM" in col or "KILOMETR" in col: new_cols[col] = "KM"
                if "CO2" in col or "EMISION" in col: new_cols[col] = "CO2"
                if "MARCA" in col: new_cols[col] = "MARCA"
            
            temp_df = temp_df.rename(columns=new_cols)
            
            # Limpieza de Patentes
            if "DOMINIO" in temp_df.columns:
                temp_df["DOMINIO"] = temp_df["DOMINIO"].astype(str).str.replace(" ", "").str.upper()
            
            # Limpieza de Números (IMPORTANTE: sacamos puntos de miles que bloquean el cálculo)
            for c in ["LITROS", "KM", "CO2"]:
                if c in temp_df.columns:
                    # Convertimos a string, quitamos puntos, cambiamos comas por puntos y a numero
                    temp_df[c] = temp_df[c].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                    temp_df[c] = pd.to_numeric(temp_df[c], errors="coerce").fillna(0)
            
            return temp_df

        df1 = limpiar_y_mapear(df1)
        df2 = limpiar_y_mapear(df2)

        # Unimos los datos
        # Agrupamos df1 para tener totales por patente de Litros y KM
        df1_resumen = df1.groupby("DOMINIO").agg({"KM": "sum", "LITROS": "sum"}).reset_index()

        # Tomamos el CO2 y Marca de la otra hoja
        cols_df2 = [c for c in ["DOMINIO", "MARCA", "CO2"] if c in df2.columns]
        df2_resumen = df2[cols_df2].groupby("DOMINIO").first().reset_index()

        df_final = pd.merge(df1_resumen, df2_resumen, on="DOMINIO", how="left")
        
        return df_final
    except Exception as e:
        st.error(f"Error técnico: {e}")
        return pd.DataFrame()

df = cargar_datos()

if not df.empty:
    # --- MÉTRICAS ---
    lts_tot = df["LITROS"].sum()
    km_tot = df["KM"].sum()
    co2_tot = df["CO2"].sum() if "CO2" in df.columns else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("⛽ Litros Totales", f"{lts_tot:,.0f} L")
    c2.metric("🛣️ Km Totales", f"{km_tot:,.0f} km")
    c3.metric("🌿 CO2 Total", f"{co2_tot:,.0f} kg")

    st.divider()

    # --- GRÁFICOS ---
    st.subheader("📊 Consumo por Patente")
    if lts_tot > 0:
        st.bar_chart(df.set_index("DOMINIO")["LITROS"])
    else:
        st.info("Aún no hay datos de litros para graficar. Revisá los nombres en la Hoja 1.")

    # --- TABLA ---
    st.subheader("📋 Resumen Consolidado")
    st.dataframe(df, use_container_width=True)
else:
    st.warning("⚠️ No se pudieron cargar los datos.")
