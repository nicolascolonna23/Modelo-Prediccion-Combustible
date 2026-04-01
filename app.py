import pandas as pd
import streamlit as st
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Expreso Diemar 2026", layout="wide")

# --- CARGA DE DATOS ---
@st.cache_data(ttl=300)
def cargar_datos():
    archivos = os.listdir('.')
    file_tel = next((f for f in archivos if "ANALISIS" in f.upper() and f.endswith('.xlsx')), None)
    file_unid = next((f for f in archivos if "UNIDADES" in f.upper() and f.endswith('.xlsx')), None)

    if not file_tel:
        st.error(f"❌ No se encuentra el archivo de consumos. Archivos en repo: {archivos}")
        st.stop()

    try:
        df1 = pd.read_excel(file_tel, engine="openpyxl")
        
        # 1. Limpieza inicial de columnas
        df1.columns = [str(c).strip().upper() for c in df1.columns]
        
        # 2. Mapeo Manual Forzado (Detecta tus nombres reales)
        mapeo = {}
        for c in df1.columns:
            if any(x in c for x in ["DOMINIO", "PATENTE", "UNIDAD"]): mapeo[c] = "DOMINIO"
            elif any(x in c for x in ["LITROS", "CONSUMO", "CONSUMID", "LTS"]): mapeo[c] = "LITROS"
            elif any(x in c for x in ["KM", "DISTANCIA", "KILOMETR"]): mapeo[c] = "KM"
            elif "FECHA" in c: mapeo[c] = "FECHA"
            elif "EMPRESA" in c: mapeo[c] = "EMPRESA"
        
        df1 = df1.rename(columns=mapeo)

        # 3. Validar si después del mapeo existen las columnas
        columnas_faltantes = [col for col in ["LITROS", "KM", "FECHA"] if col not in df1.columns]
        if columnas_faltantes:
            st.error(f"❌ No encontré las columnas: {columnas_faltantes}")
            st.write("Columnas detectadas en tu Excel:", list(df1.columns))
            st.stop()

        # 4. Filtro de Fecha (Ignorar 2024/2025)
        df1["FECHA"] = pd.to_datetime(df1["FECHA"], errors='coerce')
        df1 = df1.dropna(subset=["FECHA"])
        df1 = df1[df1["FECHA"].dt.year == 2026]

        # 5. Limpieza de Números
        for col in ["LITROS", "KM"]:
            df1[col] = pd.to_numeric(df1[col], errors="coerce").fillna(0)

        return df1
    except Exception as e:
        st.error(f"❌ Error crítico: {e}")
        st.stop()

# --- EJECUCIÓN ---
df_raw = cargar_datos()

# Scraper de precio simple
try:
    r = requests.get("https://surtidores.com.ar/precios/", timeout=5)
    soup = BeautifulSoup(r.text, "html.parser")
    import re
    precios = [int(n) for n in re.findall(r'\b(\d{3,4})\b', soup.get_text()) if 800 < int(n) < 2500]
    precio_actual = float(precios[-1])
except:
    precio_actual = 2100.0

# --- DASHBOARD ---
st.sidebar.title("Expreso Diemar")
st.header("🚛 Dashboard LAD 2026")

if df_raw.empty:
    st.warning("No hay datos cargados para el año 2026.")
else:
    # Metricas calculadas sobre el DF filtrado
    lts = df_raw["LITROS"].sum()
    kms = df_raw["KM"].sum()
    prom = (lts/kms*100) if kms > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Litros Totales", f"{lts:,.0f}")
    c2.metric("KM Totales", f"{kms:,.0f}")
    c3.metric("Promedio L/100km", f"{prom:.2f}")

    st.divider()
    st.dataframe(df_raw, use_container_width=True)
