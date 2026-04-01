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
    # Buscamos el archivo que contenga "ANALISIS"
    file_tel = next((f for f in archivos if "ANALISIS" in f.upper() and f.endswith('.xlsx')), None)

    if not file_tel:
        st.error(f"❌ No se encuentra el archivo ANALISIS CONSUMO en el repo. Archivos detectados: {archivos}")
        st.stop()

    try:
        # Carga del Excel
        df = pd.read_excel(file_tel, engine="openpyxl")
        
        # Normalizar nombres de columnas
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        # Mapeo de Columnas
        mapeo = {}
        for c in df.columns:
            if any(x in c for x in ["DOMINIO", "PATENTE", "UNIDAD"]): mapeo[c] = "DOMINIO"
            elif any(x in c for x in ["LITROS", "CONSUMO", "CONSUMID", "LTS"]): mapeo[c] = "LITROS"
            elif any(x in c for x in ["KM", "DISTANCIA", "KILOMETR"]): mapeo[c] = "KM"
            elif "FECHA" in c: mapeo[c] = "FECHA"
            elif "EMPRESA" in c: mapeo[c] = "EMPRESA"
        
        df = df.rename(columns=mapeo)

        # --- PROCESAMIENTO SEGURO ---
        
        # 4. Procesar FECHAS (Filtro 2026)
        if "FECHA" in df.columns:
            # Usamos pd.to_datetime directamente sobre la serie
            df["FECHA"] = pd.to_datetime(df["FECHA"], errors='coerce')
            df = df.dropna(subset=["FECHA"])
            # Filtrar solo el año 2026
            df = df[df["FECHA"].dt.year == 2026]

        # 5. Procesar NÚMEROS (LITROS y KM)
        columnas_num = [c for c in ["LITROS", "KM"] if c in df.columns]
        for col in columnas_num:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    except Exception as e:
        st.error(f"❌ Error en el procesamiento de datos: {e}")
        st.stop()

# --- SCRAPER PRECIO ---
@st.cache_data(ttl=3600)
def obtener_precio():
    try:
        r = requests.get("https://surtidores.com.ar/precios/", timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        import re
        precios = [int(n) for n in re.findall(r'\b(\d{3,4})\b', soup.get_text()) if 800 < int(n) < 3000]
        return float(precios[-1]), "Surtidores.com.ar"
    except:
        return 2025.0, "Referencia"

# --- EJECUCIÓN ---
df_raw = cargar_datos()
precio_actual, fuente_p = obtener_precio()

# Interfaz
st.sidebar.title("Expreso Diemar")
st.sidebar.markdown("---")
opcion = st.sidebar.radio("Navegación", ["Dashboard Principal", "Modelo Predictivo"])

if opcion == "Dashboard Principal":
    st.header("🚛 Dashboard LAD 2026")
    st.write(f"**Precio Combustible:** ${precio_actual:,.0f}/L ({fuente_p})")

    if df_raw.empty:
        st.warning("⚠️ No se encontraron datos del año 2026 en el archivo subido.")
    else:
        # Métricas
        lts_tot = df_raw["LITROS"].sum()
        km_tot = df_raw["KM"].sum()
        cons_prom = (lts_tot / km_tot * 100) if km_tot > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Litros 2026", f"{lts_tot:,.0f}")
        c2.metric("KM 2026", f"{km_tot:,.0f}")
        c3.metric("L/100km Promedio", f"{cons_prom:.2f}")

        st.divider()
        st.subheader("Registros 2026")
        st.dataframe(df_raw, use_container_width=True, hide_index=True)
else:
    st.header("🧠 Modelo Predictivo")
    st.info("Datos de telemetría de 2026 listos para análisis.")
