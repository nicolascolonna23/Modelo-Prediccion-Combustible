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
    page_title="Expreso Diemar — Dashboard 2026",
    page_icon="🚛",
    layout="wide",
)

# --- FUNCIÓN DE CARGA AJUSTADA A TUS CAPTURAS ---
@st.cache_data(ttl=300)
def cargar_datos():
    archivos = os.listdir('.')
    file_tel = next((f for f in archivos if "ANALISIS" in f.upper() and f.endswith('.xlsx')), None)

    if not file_tel:
        st.error(f"❌ No se encontró el archivo Excel. Archivos en repo: {archivos}")
        st.stop()

    try:
        # Cargamos el Excel
        df = pd.read_excel(file_tel, engine="openpyxl")
        
        # Limpieza básica: quitar espacios extras en los nombres de columnas
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        # --- MAPEO EXACTO SEGÚN TUS FOTOS ---
        mapeo = {}
        for c in df.columns:
            # Captura 'DOMINIO'
            if "DOMINIO" in c: mapeo[c] = "DOMINIO"
            # Captura 'LITROS CONSUMIDOS' o 'LITROS CARGADOS'
            elif "LITROS" in c: mapeo[c] = "LITROS"
            # Captura 'DISTANCIA RECORRIDA' o 'KILOMETRAJE'
            elif any(x in c for x in ["DISTANCIA", "KM", "KILOMETR"]): mapeo[c] = "KM"
            # Captura 'FECHA'
            elif "FECHA" in c: mapeo[c] = "FECHA"
            # Captura 'TAG' o 'EMPRESA'
            elif any(x in c for x in ["TAG", "EMPRESA"]): mapeo[c] = "EMPRESA"
        
        df = df.rename(columns=mapeo)

        # --- PROCESAMIENTO SEGURO DE DATOS ---
        
        # 1. FECHAS (Filtrar solo 2026)
        if "FECHA" in df.columns:
            # Convertimos a fecha forzando errores a NaT
            df["FECHA"] = pd.to_datetime(df["FECHA"], errors='coerce')
            df = df.dropna(subset=["FECHA"])
            # FILTRO SOLO 2026 (Ignora 2024 y 2025 de tus fotos)
            df = df[df["FECHA"].dt.year == 2026]

        # 2. NÚMEROS (LITROS y KM)
        for col in ["LITROS", "KM"]:
            if col in df.columns:
                # Quitamos posibles caracteres extraños y pasamos a número
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    except Exception as e:
        st.error(f"❌ Error procesando los datos: {e}")
        st.stop()

# --- PRECIO GASOIL ---
@st.cache_data(ttl=3600)
def obtener_precio():
    try:
        r = requests.get("https://surtidores.com.ar/precios/", timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        import re
        precios = [int(n) for n in re.findall(r'\b(\d{3,4})\b', soup.get_text()) if 800 < int(n) < 3000]
        return float(precios[-1]), "Surtidores.com.ar"
    except:
        return 2100.0, "Referencia"

# --- LÓGICA DE INTERFAZ ---
df_raw = cargar_datos()
precio_act, fuente_p = obtener_precio()

st.sidebar.title("Expreso Diemar")
pg = st.sidebar.radio("Navegación", ["Dashboard Principal", "Modelo Predictivo"])

if pg == "Dashboard Principal":
    st.header("🚛 Dashboard LAD 2026")
    st.write(f"**Precio Combustible:** ${precio_act:,.0f}/L ({fuente_p})")

    if df_raw.empty:
        st.warning("⚠️ No hay datos del año 2026. (Se están ignorando los registros de 2024 y 2025 que vi en tus fotos).")
    else:
        # Métricas
        lts = df_raw["LITROS"].sum()
        kms = df_raw["KM"].sum()
        prom = (lts/kms*100) if kms > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Litros 2026", f"{lts:,.0f}")
        c2.metric("KM 2026", f"{kms:,.0f}")
        c3.metric("L/100km Promedio", f"{prom:.2f}")

        st.divider()
        st.subheader("Registros Filtrados (Solo 2026)")
        st.dataframe(df_raw, use_container_width=True, hide_index=True)
else:
    st.header("🧠 Modelo Predictivo")
    st.info("Datos de 2026 listos para el entrenamiento.")
