import pandas as pd
import streamlit as st
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Expreso Diemar 2026", layout="wide")

@st.cache_data(ttl=300)
def cargar_datos():
    archivos = os.listdir('.')
    # Buscamos el archivo que contenga "ANALISIS"
    file_tel = next((f for f in archivos if "ANALISIS" in f.upper() and f.endswith('.xlsx')), None)

    if not file_tel:
        st.error(f"❌ No se encuentra el archivo ANALISIS CONSUMO en el repo. Archivos: {archivos}")
        st.stop()

    try:
        # 1. Carga inicial
        df = pd.read_excel(file_tel, engine="openpyxl")
        
        # 2. Normalizar nombres de columnas (Mayúsculas y sin espacios)
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        # 3. Mapeo Manual de Columnas
        mapeo = {}
        for c in df.columns:
            if any(x in c for x in ["DOMINIO", "PATENTE", "UNIDAD"]): mapeo[c] = "DOMINIO"
            elif any(x in c for x in ["LITROS", "CONSUMO", "CONSUMID", "LTS"]): mapeo[c] = "LITROS"
            elif any(x in c for x in ["KM", "DISTANCIA", "KILOMETR"]): mapeo[c] = "KM"
            elif "FECHA" in c: mapeo[c] = "FECHA"
            elif "EMPRESA" in c: mapeo[c] = "EMPRESA"
        
        df = df.rename(columns=mapeo)

        # --- SOLUCIÓN AL ERROR: arg must be a list ---
        # 4. Procesar FECHAS (Ignorar 2024 y 2025)
        if "FECHA" in df.columns:
            # Convertimos a lista de Python pura para evitar errores de Series
            lista_fechas = df["FECHA"].tolist()
            df["FECHA"] = pd.to_datetime(lista_fechas, errors='coerce')
            df = df.dropna(subset=["FECHA"])
            # Filtro estricto 2026
            df = df[df["FECHA"].dt.year == 2026]

        # 5. Procesar NÚMEROS (LITROS y KM)
        for col in ["LITROS", "KM"]:
            if col in df.columns:
                lista_num = df[col].tolist()
                df[col] = pd.to_numeric(lista_num, errors="coerce").fillna(0)

        return df

    except Exception as e:
        st.error(f"❌ Error en el procesamiento: {e}")
        st.stop()

# --- LÓGICA DE EJECUCIÓN ---
df_raw = cargar_datos()

# Scraper de precio (Opcional, con respaldo)
try:
    r = requests.get("https://surtidores.com.ar/precios/", timeout=5)
    soup = BeautifulSoup(r.text, "html.parser")
    import re
    precios = [int(n) for n in re.findall(r'\b(\d{3,4})\b', soup.get_text()) if 800 < int(n) < 3000]
    precio_actual = float(precios[-1])
except:
    precio_actual = 2026.0

# --- INTERFAZ ---
st.sidebar.title("Expreso Diemar")
st.header("🚛 Dashboard LAD 2026")

if df_raw.empty:
    st.warning("⚠️ El archivo se cargó pero no hay datos del año 2026.")
else:
    # Cálculos sobre el Dashboard
    lts = df_raw["LITROS"].sum()
    kms = df_raw["KM"].sum()
    prom = (lts / kms * 100) if kms > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Litros 2026", f"{lts:,.0f}")
    c2.metric("KM 2026", f"{kms:,.0f}")
    c3.metric("L/100km Promedio", f"{prom:.2f}")

    st.divider()
    st.subheader("Registros Detallados (Solo 2026)")
    st.dataframe(df_raw, use_container_width=True, hide_index=True)
