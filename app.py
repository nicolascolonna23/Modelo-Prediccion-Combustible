import pandas as pd
import streamlit as st
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# Configuración inicial
st.set_page_config(
    page_title="Expreso Diemar — Predicción Combustible",
    page_icon="🚛",
    layout="wide",
)

# --- RECURSOS VISUALES ---
LOGO_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/logo_diemar4.png"
IVECO_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/S-Way-6x2-1.webp"
SCANIA_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/2016p.png"

# --- CARGA DE DATOS (LOCAL DESDE GITHUB) ---
@st.cache_data(ttl=300)
def cargar_datos():
    FILE_TEL = "telemetria.xlsx"
    FILE_UNID = "unidades.xlsx"

    try:
        if not os.path.exists(FILE_TEL) or not os.path.exists(FILE_UNID):
            return pd.DataFrame(), pd.DataFrame()

        df1 = pd.read_excel(FILE_TEL, engine="openpyxl")
        df2 = pd.read_excel(FILE_UNID, engine="openpyxl")

        def limpiar(df):
            df.columns = [str(c).strip().upper() for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]
            
            rename_dict = {}
            for c in df.columns:
                if any(x in c for x in ["DOMINIO", "PATENTE", "UNIDAD"]): rename_dict[c] = "DOMINIO"
                elif any(x in c for x in ["LITROS", "CONSUMID"]): rename_dict[c] = "LITROS"
                elif any(x in c for x in ["KM", "DISTANCIA", "KILOMETR"]): rename_dict[c] = "KM"
                elif "MARCA" in c: rename_dict[c] = "MARCA"
                elif "FECHA" in c: rename_dict[c] = "FECHA"
                elif "L/100" in c: rename_dict[c] = "L100KM"
                elif "RALENT" in c: rename_dict[c] = "RALENTI"
                elif "EMPRESA" in c: rename_dict[c] = "EMPRESA"
            
            df = df.rename(columns=rename_dict)
            
            if "DOMINIO" in df.columns:
                df["DOMINIO"] = df["DOMINIO"].astype(str).str.strip().str.upper()
            
            if "FECHA" in df.columns:
                df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
            
            for col in ["LITROS", "KM", "L100KM", "RALENTI"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            return df

        df1 = limpiar(df1)
        df2 = limpiar(df2)

        # Lógica de negocio LAD/Diemar
        if "EMPRESA" in df1.columns:
            df1 = df1[df1["EMPRESA"].str.upper().str.contains("LAD|DIEMAR", na=False)]
        
        if "FECHA" in df1.columns:
            df1 = df1[df1["FECHA"].dt.year == 2026]

        # Calcular L/100 si falta
        if "L100KM" not in df1.columns or (df1["L100KM"] == 0).all():
            df1["L100KM"] = (df1["LITROS"] / df1["KM"].replace(0, np.nan) * 100).round(2).fillna(0)

        return df1, df2
    except Exception as e:
        st.error(f"Error en carga local: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- SCRAPER PRECIO ---
@st.cache_data(ttl=3600)
def obtener_precio_gasoil():
    try:
        import re
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get("https://surtidores.com.ar/precios/", headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        texto = soup.get_text()
        numeros = re.findall(r'\b(\d{3,4})\b', texto)
        numeros = [int(n) for n in numeros if 800 <= int(n) <= 2500]
        if numeros: return float(numeros[-1]), "surtidores.com.ar"
    except: pass
    return 2100.0, "referencia estimada"

# --- LÓGICA PRINCIPAL ---
df_raw, df_unid = cargar_datos()

if df_raw.empty:
    st.error("⚠️ No se detectan los archivos Excel. Asegúrate de que telemetria.xlsx y unidades.xlsx estén en el repositorio.")
    st.stop()

# Filtros Sidebar
st.sidebar.image(LOGO_URL, width=160)
pg = st.sidebar.radio("Navegación", ["Dashboard Principal", "Modelo Predictivo"])

df = df_raw.copy()
precio_gasoil, fuente = obtener_precio_gasoil()

# UI Principal
if pg == "Dashboard Principal":
    st.title("🚛 Expreso Diemar — Dashboard 2026")
    st.markdown(f"**Precio actual:** ${precio_gasoil:,.0f}/L | Fuente: {fuente}")
    
    # KPIs
    lts = df["LITROS"].sum()
    kms = df["KM"].sum()
    l100 = (lts / kms * 100) if kms > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Litros Totales", f"{lts:,.0f}")
    c2.metric("KM Totales", f"{kms:,.0f}")
    c3.metric("Promedio L/100km", f"{l100:.2f}")

    st.divider()
    st.subheader("Datos de la Flota")
    st.dataframe(df, use_container_width=True)

else:
    st.title("🧠 Modelo Predictivo")
    # Aquí puedes re-pegar tu lógica de regresión polinomial que usaba 'df'
    st.info("Modelo basado en los datos locales de telemetria.xlsx")
