import pandas as pd
import streamlit as st
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Expreso Diemar — Dashboard", layout="wide")

@st.cache_data(ttl=300)
def cargar_datos():
    archivos = os.listdir('.')
    file_tel = next((f for f in archivos if "ANALISIS" in f.upper() and f.endswith('.xlsx')), None)

    if not file_tel:
        st.error(f"❌ No se encontró el archivo Excel. Archivos en repo: {archivos}")
        st.stop()

    try:
        df = pd.read_excel(file_tel, engine="openpyxl")
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        # Mapeo flexible basado en tus capturas
        mapeo = {}
        for c in df.columns:
            if "DOMINIO" in c: mapeo[c] = "DOMINIO"
            elif "LITROS" in c: mapeo[c] = "LITROS"
            elif any(x in c for x in ["DISTANCIA", "KM", "KILOMETR"]): mapeo[c] = "KM"
            elif "FECHA" in c: mapeo[c] = "FECHA"
            elif any(x in c for x in ["TAG", "EMPRESA"]): mapeo[c] = "EMPRESA"
        
        df = df.rename(columns=mapeo)

        # --- PROCESAMIENTO DE FECHAS (Solución al error de Series) ---
        if "FECHA" in df.columns:
            # Convertimos a lista y luego a fecha para evitar errores de Pandas
            fechas_lista = list(df["FECHA"])
            df["FECHA"] = pd.to_datetime(fechas_lista, errors='coerce')
            df = df.dropna(subset=["FECHA"])
            
            # FILTRO: Intentar 2026, si está vacío, mostrar 2025
            df_2026 = df[df["FECHA"].dt.year == 2026]
            if not df_2026.empty:
                df = df_2026
            else:
                st.info("ℹ️ Mostrando datos de 2025 (No se detectaron registros de 2026 aún).")
                df = df[df["FECHA"].dt.year == 2025]

        # Limpieza de Números
        for col in ["LITROS", "KM"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    except Exception as e:
        st.error(f"❌ Error en el procesamiento: {e}")
        st.stop()

# --- EJECUCIÓN ---
df_raw = cargar_datos()

# Interfaz Simple
st.sidebar.title("Expreso Diemar")
st.header("🚛 Dashboard de Telemetría")

if df_raw.empty:
    st.warning("⚠️ El archivo no contiene datos válidos de 2025 o 2026.")
else:
    lts = df_raw["LITROS"].sum()
    kms = df_raw["KM"].sum()
    prom = (lts/kms*100) if kms > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Litros Totales", f"{lts:,.0f}")
    c2.metric("KM Totales", f"{kms:,.0f}")
    c3.metric("L/100km Promedio", f"{prom:.2f}")

    st.divider()
    st.dataframe(df_raw, use_container_width=True, hide_index=True)
