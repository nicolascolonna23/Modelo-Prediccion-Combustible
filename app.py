import pandas as pd
import streamlit as st
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore')

# 1. Configuración de página
st.set_page_config(
    page_title="Expreso Diemar — Predicción Combustible",
    page_icon="🚛",
    layout="wide",
)

# --- CARGA DE DATOS ---
@st.cache_data(ttl=300)
def cargar_datos():
    archivos = os.listdir('.')
    file_tel = next((f for f in archivos if "ANALISIS" in f.upper() and f.endswith('.xlsx')), None)
    file_unid = next((f for f in archivos if "UNIDADES" in f.upper() and f.endswith('.xlsx')), None)

    if not file_tel or not file_unid:
        st.error(f"❌ No se detectan los archivos. Archivos en repo: {archivos}")
        st.stop()

    try:
        # Cargamos los Excel
        df1 = pd.read_excel(file_tel, engine="openpyxl")
        df2 = pd.read_excel(file_unid, engine="openpyxl")

        def limpiar(df):
            # Normalizar columnas
            df.columns = [str(c).strip().upper() for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Mapeo de nombres
            cm = {}
            for c in df.columns:
                if any(x in c for x in ["DOMINIO", "PATENTE", "UNIDAD"]): cm[c] = "DOMINIO"
                elif any(x in c for x in ["LITROS", "CONSUMID"]): cm[c] = "LITROS"
                elif any(x in c for x in ["KM", "DISTANCIA", "KILOMETR"]): cm[c] = "KM"
                elif "MARCA" in c: cm[c] = "MARCA"
                elif "FECHA" in c: cm[c] = "FECHA"
                elif "EMPRESA" in c: cm[c] = "EMPRESA"
            df = df.rename(columns=cm)

            # --- CORRECCIÓN CRÍTICA DE FECHAS ---
            if "FECHA" in df.columns:
                # Convertimos la columna a una lista simple para evitar el error de Pandas
                fechas_puras = df["FECHA"].tolist()
                df["FECHA"] = pd.to_datetime(fechas_puras, errors='coerce')
                
                # Eliminamos lo que no sea fecha y FILTRAMOS SOLO 2026
                df = df[df["FECHA"].notna()]
                df = df[df["FECHA"].dt.year == 2026]

            # --- CORRECCIÓN DE NÚMEROS ---
            for col in ["LITROS", "KM"]:
                if col in df.columns:
                    # Forzamos conversión a lista antes de to_numeric para evitar ambigüedad
                    valores = df[col].tolist()
                    df[col] = pd.to_numeric(valores, errors="coerce").fillna(0)
            
            return df

        df1 = limpiar(df1)
        df2 = limpiar(df2)

        # Validación post-limpieza
        if df1.empty:
            st.warning("⚠️ Se leyeron los archivos pero no se encontraron datos del año 2026.")
            st.stop()

        if "EMPRESA" in df1.columns:
            df1 = df1[df1["EMPRESA"].astype(str).str.upper().str.contains("LAD|DIEMAR", na=False)]

        # Consumo L/100
        if "KM" in df1.columns and "LITROS" in df1.columns:
            df1["L100KM"] = (df1["LITROS"] / df1["KM"].replace(0, np.nan) * 100).round(2).fillna(0)

        return df1, df2

    except Exception as e:
        st.error(f"❌ Error procesando los datos: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- PRECIO GASOIL ---
@st.cache_data(ttl=3600)
def obtener_precio():
    try:
        r = requests.get("https://surtidores.com.ar/precios/", timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        import re
        precios = [int(n) for n in re.findall(r'\b(\d{3,4})\b', soup.get_text()) if 800 < int(n) < 2500]
        return float(precios[-1]), "Surtidores.com.ar"
    except:
        return 2150.0, "Referencia"

# --- EJECUCIÓN ---
df_raw, df_unid = cargar_datos()
precio, fuente = obtener_precio()

# UI
st.sidebar.title("Expreso Diemar")
pg = st.sidebar.radio("Navegación", ["Dashboard Principal", "Modelo Predictivo"])

if pg == "Dashboard Principal":
    st.header("🚛 Dashboard LAD 2026")
    st.write(f"Precio Combustible: ${precio:,.0f} ({fuente})")
    
    # KPIs Rápidos
    lts = df_raw["LITROS"].sum()
    kms = df_raw["KM"].sum()
    prom = (lts/kms*100) if kms > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Litros Totales", f"{lts:,.0f}")
    c2.metric("KM Totales", f"{kms:,.0f}")
    c3.metric("L/100km Promedio", f"{prom:.2f}")

    st.divider()
    st.dataframe(df_raw, use_container_width=True)
else:
    st.header("🧠 Modelo Predictivo")
    st.info("Datos de 2026 listos.")
