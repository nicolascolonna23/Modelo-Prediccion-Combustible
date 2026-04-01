import pandas as pd
import streamlit as st
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Expreso Diemar — Dashboard", layout="wide")

@st.cache_data(ttl=300)
def cargar_datos():
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
        st.error("No se encontro el archivo Excel.")
        st.stop()

    try:
        df = pd.read_excel(file_tel, engine="openpyxl")
        df.columns = [str(c).strip().upper() for c in df.columns]

        mapeo = {}
        for c in df.columns:
            if "DOMINIO" in c:
                mapeo[c] = "DOMINIO"
            elif "LITROS" in c:
                mapeo[c] = "LITROS"
            elif any(x in c for x in ["DISTANCIA", "KILOMETR"]) or c == "KM":
                mapeo[c] = "KM"
            elif "FECHA" in c:
                mapeo[c] = "FECHA"
            elif any(x in c for x in ["TAG", "EMPRESA"]):
                mapeo[c] = "EMPRESA"

        df = df.rename(columns=mapeo)
        df = df.loc[:, ~df.columns.duplicated()]

        if "FECHA" in df.columns:
            df["FECHA"] = pd.to_datetime(df["FECHA"], errors='coerce')
            df = df.dropna(subset=["FECHA"])
            df_2026 = df[df["FECHA"].dt.year == 2026]
            if not df_2026.empty:
                df = df_2026
            else:
                st.info("Mostrando datos de 2025.")
                df = df[df["FECHA"].dt.year == 2025]

        for col in ["LITROS", "KM"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    except Exception as e:
        import traceback
        st.error(f"Error: {e}")
        st.code(traceback.format_exc())
        st.stop()


df_raw = cargar_datos()

st.sidebar.title("Expreso Diemar")
st.header("🚛 Dashboard de Telemetria")

if df_raw.empty:
    st.warning("Sin datos validos de 2025 o 2026.")
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
