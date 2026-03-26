import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import io

# Configuración
st.set_page_config(page_title="Expreso Diemar - Analítica", layout="wide")
st.title("🚛 Dashboard de Consumo de Flota")

# ==============================
# 1. FUENTES DE DATOS (GOOGLE SHEETS)
# ==============================
base_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR35NkYPtJrOrdYHLGUH7GIW93s5cPAqQ0zEk5fP1c3gvErwbUW7HJ2OeWBYaBVsYKVmCf0yhLvs6eG/pub?output=csv"

# Asegurate que estos GIDs correspondan a las pestañas correctas
gid_telemetria = "1044040871"
gid_unidades = "882343299" 

url1 = f"{base_url}&gid={gid_telemetria}"
url2 = f"{base_url}&gid={gid_unidades}"

# ==============================
# 2. CARGAR Y LIMPIAR DATOS (VERSION ROBUSTA)
# ==============================
@st.cache_data(ttl=300)
def cargar_datos():
    try:
        # Carga cruda
        df1 = pd.read_csv(url1)
        df2 = pd.read_csv(url2)

        # Limpieza extrema de nombres de columnas
        for d in [df1, df2]:
            d.columns = d.columns.str.strip().str.replace('í', 'i').str.replace('á', 'a').str.upper()

        # Mapeo manual para asegurar que existan KM, LITROS y RALENTI
        # Buscamos coincidencias aunque el nombre varíe un poco
        def normalizar(df_temp):
            rename_map = {}
            for col in df_temp.columns:
                if "DISTANCIA" in col or "KM" in col: rename_map[col] = "KM"
                if "LITROS CONSUMIDOS" in col: rename_map[col] = "LITROS_CONSUMIDOS"
                if "RALENTI" in col: rename_map[col] = "RALENTI_LTS"
                if "CONSUMO C/ 100KM" in col: rename_map[col] = "L100KM"
            return df_temp.rename(columns=rename_map)

        df1 = normalizar(df1)
        df2 = normalizar(df2)

        # Cruce de tablas
        df_merged = pd.merge(df1, df2, on=["FECHA", "DOMINIO"], how="inner")
        
        # Verificación de columnas necesarias post-merge
        cols_necesarias = ["KM", "LITROS_CONSUMIDOS", "RALENTI_LTS"]
        cols_presentes = [c for c in cols_necesarias if c in df_merged.columns]
        
        if len(cols_presentes) < len(cols_necesarias):
            st.error(f"Faltan columnas tras el cruce. Encontradas: {cols_presentes}")
            st.info("Revisá que 'FECHA' y 'DOMINIO' coincidan exactos en ambas hojas.")
            return pd.DataFrame()

        # Limpieza final
        df_merged = df_merged.dropna(subset=cols_necesarias)
        df_merged = df_merged[(df_merged["KM"] > 0) & (df_merged["LITROS_CONSUMIDOS"] > 0)]
        
        return df_merged
    except Exception as e:
        st.error(f"Error en procesamiento: {e}")
        return pd.DataFrame()

df = cargar_datos()

if df.empty:
    st.info("Esperando datos válidos de Google Sheets...")
    st.stop()

# ==============================
# 3. PROCESAMIENTO
# ==============================
df["FECHA"] = pd.to_datetime(df["FECHA"], errors='coerce')
df["MES"] = df["FECHA"].dt.month
df["AÑO"] = df["FECHA"].dt.year

# Precio Gasoil (Default por si falla el scraping)
precio_fijo = 1250 

# ==============================
# 4. METRICAS Y COSTOS
# ==============================
df["COSTO"] = df["LITROS_CONSUMIDOS"] * precio_fijo
df["COSTO_RALENTI"] = df["RALENTI_LTS"] * precio_fijo

lts_tot = df["LITROS_CONSUMIDOS"].sum()
costo_tot = df["COSTO"].sum()
costo_ral = df["COSTO_RALENTI"].sum()

# ==============================
# 5. DASHBOARD
# ==============================
c1, c2, c3 = st.columns(3)
c1.metric("⛽ Litros Totales", f"{lts_tot:,.0f} L")
c2.metric("💰 Costo Total", f"$ {costo_tot:,.0f}")
c3.metric("🛑 Costo Ralenti", f"$ {costo_ral:,.0f}")

st.divider()

# Gráfico de Consumo
st.subheader("📊 Consumo por Patente")
chart_data = df.groupby("DOMINIO")["LITROS_CONSUMIDOS"].sum().sort_values(ascending=False)
st.bar_chart(chart_data)

# Tabla Detallada
st.subheader("📋 Detalle de Operación")
st.dataframe(df[["FECHA", "DOMINIO", "KM", "LITROS_CONSUMIDOS", "RALENTI_LTS", "COSTO"]], use_container_width=True)

# ==============================
# 6. MODELO IA (PREDICCION)
# ==============================
try:
    X = df[["KM", "L100KM", "RALENTI_LTS"]]
    y = df["LITROS_CONSUMIDOS"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    st.sidebar.success("Modelo de IA entrenado")
except:
    st.sidebar.warning("No hay datos suficientes para la IA")
