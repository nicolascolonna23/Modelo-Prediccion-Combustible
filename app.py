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

# --- FUNCIÓN DE CARGA REFORZADA ---
@st.cache_data(ttl=300)
def cargar_datos():
    archivos = os.listdir('.')
    # Buscamos el archivo de consumo (insensible a mayúsculas/minúsculas)
    file_tel = next((f for f in archivos if "ANALISIS" in f.upper() and f.endswith('.xlsx')), None)

    if not file_tel:
        st.error(f"❌ No se encontró el archivo de Excel en el repositorio. Archivos detectados: {archivos}")
        st.stop()

    try:
        # Carga directa del Excel
        df = pd.read_excel(file_tel, engine="openpyxl")
        
        # Limpieza de nombres de columnas
        df.columns = [str(c).strip().upper() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Mapeo manual para asegurar compatibilidad
        mapeo = {}
        for c in df.columns:
            if any(x in c for x in ["DOMINIO", "PATENTE", "UNIDAD"]): mapeo[c] = "DOMINIO"
            elif any(x in c for x in ["LITROS", "CONSUMO", "CONSUMID", "LTS"]): mapeo[c] = "LITROS"
            elif any(x in c for x in ["KM", "DISTANCIA", "KILOMETR"]): mapeo[c] = "KM"
            elif "FECHA" in c: mapeo[c] = "FECHA"
            elif "EMPRESA" in c: mapeo[c] = "EMPRESA"
        
        df = df.rename(columns=mapeo)

        # --- PROCESAMIENTO DE FECHAS (FILTRO 2026) ---
        if "FECHA" in df.columns:
            # Forzamos la conversión a fecha ignorando errores
            df["FECHA"] = pd.to_datetime(df["FECHA"], errors='coerce')
            # Eliminamos filas que no tengan una fecha válida (NaT)
            df = df[df["FECHA"].notna()]
            # Mantenemos EXCLUSIVAMENTE el año 2026
            df = df[df["FECHA"].dt.year == 2026]

        # --- PROCESAMIENTO DE NÚMEROS ---
        for col in ["LITROS", "KM"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    except Exception as e:
        st.error(f"❌ Error en el procesamiento de datos: {e}")
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

# --- EJECUCIÓN Y DASHBOARD ---
df_raw = cargar_datos()
precio_actual, fuente_p = obtener_precio()

st.sidebar.title("Expreso Diemar")
st.sidebar.markdown("---")
opcion = st.sidebar.radio("Navegación", ["Dashboard Principal", "Modelo Predictivo"])

if opcion == "Dashboard Principal":
    st.header("🚛 Dashboard LAD 2026")
    st.write(f"**Precio Combustible:** ${precio_actual:,.0f}/L ({fuente_p})")

    if df_raw.empty:
        st.warning("⚠️ Se leyó el archivo, pero no hay datos correspondientes al año 2026. (Los datos de 2024/2025 fueron ignorados).")
    else:
        # Métricas principales
        lts_tot = df_raw["LITROS"].sum()
        km_tot = df_raw["KM"].sum()
        cons_prom = (lts_tot / km_tot * 100) if km_tot > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Litros 2026", f"{lts_tot:,.0f}")
        c2.metric("KM 2026", f"{km_tot:,.0f}")
        c3.metric("L/100km Promedio", f"{cons_prom:.2f}")

        st.divider()
        st.subheader("Registros Telemetría 2026")
        st.dataframe(df_raw, use_container_width=True, hide_index=True)
else:
    st.header("🧠 Modelo Predictivo")
    st.info("Los datos de 2026 están procesados y listos para el análisis de tendencias.")
