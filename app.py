import pandas as pd
import streamlit as st
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Expreso Diemar — Dashboard", layout="wide")

@st.cache_data(ttl=300)
def cargar_datos():
    # Busca el archivo en la carpeta actual Y en subcarpeta COMBUSTIBLE
    carpetas = ['.', 'COMBUSTIBLE']
    file_tel = None
    for carpeta in carpetas:
        if os.path.exists(carpeta):
            archivos = os.listdir(carpeta)
            match = next((f for f in archivos if "ANALISIS" in f.upper() and f.endswith('.xlsx')), None)
            if match:
                file_tel = os.path.join(carpeta, match)
                break

    if not file_tel:
        st.error("❌ No se encontró el archivo Excel (debe contener 'ANALISIS' en el nombre).")
        st.stop()

    try:
        df = pd.read_excel(file_tel, engine="openpyxl")
        df.columns = [str(c).strip().upper() for c in df.columns]

        # Mapeo flexible de columnas
        mapeo = {}
        for c in df.columns:
            if "DOMINIO" in c:
                mapeo[c] = "DOMINIO"
            elif "LITROS" in c:
                mapeo[c] = "LITROS"
            elif any(x in c for x in ["DISTANCIA", "KM", "KILOMETR"]):
                mapeo[c] = "KM"
            elif "FECHA" in c:
                mapeo[c] = "FECHA"
            elif any(x in c for x in ["TAG", "EMPRESA"]):
                mapeo[c] = "EMPRESA"

        df = df.rename(columns=mapeo)

        # --- PROCESAMIENTO DE FECHAS (corregido) ---
        if "FECHA" in df.columns:
            # Pasamos la Serie directamente, sin convertir a lista
            df["FECHA"] = pd.to_datetime(df["FECHA"], errors='coerce')
            df = df.dropna(subset=["FECHA"])

            # FILTRO: Mostrar 2026 si existe, sino 2025
            df_2026 = df[df["FECHA"].dt.year == 2026]
            if not df_2026.empty:
                df = df_2026
            else:
                st.info("ℹ️ Mostrando datos de 2025 (No se detectaron registros de 2026 aún).")
                df = df[df["FECHA"].dt.year == 2025]

        # Limpieza de números
        for col in ["LITROS", "KM"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    except Exception as e:
        st.error(f"❌ Error en el procesamiento: {e}")
        st.stop()


# --- EJECUCIÓN ---
df_raw = cargar_datos()

# Interfaz
st.sidebar.title("Expreso Diemar")
st.header("🚛 Dashboard de Telemetría")

if df_raw.empty:
    st.warning("⚠️ El archivo no contiene datos válidos de 2025 o 2026.")
else:
    lts = df_raw["LITROS"].sum()
    kms = df_raw["KM"].sum()
    prom = (lts / kms * 100) if kms > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Litros Totales", f"{lts:,.0f}")
    c2.metric("KM Totales", f"{kms:,.0f}")
    c3.metric("L/100km Promedio", f"{prom:.2f}")

    st.divider()
    st.dataframe(df_raw, use_container_width=True, hide_index=True)
