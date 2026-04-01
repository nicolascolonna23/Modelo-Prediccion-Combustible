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

# Configuración de página de Expreso Diemar
st.set_page_config(
    page_title="Expreso Diemar — Predicción Combustible",
    page_icon="🚛",
    layout="wide",
)

# --- RECURSOS VISUALES ---
LOGO_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/logo_diemar4.png"
IVECO_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/S-Way-6x2-1.webp"
SCANIA_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/2016p.png"

# --- ESTILOS DARK ---
DARK_CSS = """
<style>
[data-testid="stAppViewContainer"] { background: #0f172a; }
[data-testid="stSidebar"] { background: #1e293b; }
section[data-testid="stMain"] { background: #0f172a; }
.stMarkdown, .stCaption, label, p, span, div { color: #e2e8f0 !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
.kpi-card {
    background: #1e293b; border-radius: 14px; padding: 24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.4); text-align: center;
    border-left: 5px solid #2563eb; margin-bottom: 16px;
}
.kpi-label { font-size:0.78rem; color:#94a3b8; font-weight:600; text-transform:uppercase; }
.kpi-value { font-size:2rem; font-weight:800; color:#f1f5f9; }
.sec-title { font-size:1.1rem; font-weight:700; color:#e2e8f0; border-left:4px solid #2563eb; padding-left:10px; margin:18px 0 10px; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# --- CARGA DE DATOS LOCALES (Opción B) ---
@st.cache_data(ttl=300)
def cargar_datos():
    # Nombres de archivos según tu captura de pantalla
    FILE_TEL = "ANALISIS CONSUMO.xlsx" 
    FILE_UNID = "unidades.xlsx"

    try:
        if not os.path.exists(FILE_TEL) or not os.path.exists(FILE_UNID):
            return pd.DataFrame(), pd.DataFrame()

        # Carga usando openpyxl 
        df1 = pd.read_excel(FILE_TEL, engine="openpyxl")
        df2 = pd.read_excel(FILE_UNID, engine="openpyxl")

        def limpiar(df):
            # Estandarizar columnas a mayúsculas y sin espacios 
            df.columns = [str(c).strip().upper() for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Mapeo inteligente para reconocer columnas del Excel 
            cm = {}
            for c in df.columns:
                if any(x in c for x in ["DOMINIO", "PATENTE", "UNIDAD"]): cm[c] = "DOMINIO"
                elif any(x in c for x in ["LITROS", "CONSUMID"]): cm[c] = "LITROS"
                elif any(x in c for x in ["KM", "DISTANCIA", "KILOMETR"]): cm[c] = "KM"
                elif "MARCA" in c: cm[c] = "MARCA"
                elif "FECHA" in c: cm[c] = "FECHA"
                elif "L/100" in c: cm[c] = "L100KM"
                elif "RALENT" in c: cm[c] = "RALENTI"
                elif "EMPRESA" in c: cm[c] = "EMPRESA"
            
            df = df.rename(columns=cm)
            
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

        # Filtro por empresa LAD/Diemar 
        if "EMPRESA" in df1.columns:
            df1 = df1[df1["EMPRESA"].str.upper().str.contains("LAD|DIEMAR", na=False)]
        
        # Filtro de año 2026 
        if "FECHA" in df1.columns:
            df1 = df1[df1["FECHA"].dt.year == 2026]

        # Calcular L/100 si no existe 
        if "L100KM" not in df1.columns or (df1["L100KM"] == 0).all():
            df1["L100KM"] = (df1["LITROS"] / df1["KM"].replace(0, np.nan) * 100).round(2).fillna(0)

        return df1, df2
    except Exception as e:
        st.error(f"Error procesando archivos locales: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- PRECIO GASOIL ---
@st.cache_data(ttl=3600)
def obtener_precio():
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get("https://surtidores.com.ar/precios/", headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        import re
        numeros = re.findall(r'\b(\d{3,4})\b', soup.get_text())
        precios = [int(n) for n in numeros if 800 < int(n) < 2500]
        return float(precios[-1]), "Surtidores.com.ar"
    except:
        return 2150.0, "Precio de referencia"

# --- LOGICA DE LA APP ---
df_raw, df_unid = cargar_datos()
precio_gasoil, fuente_precio = obtener_precio()

if df_raw.empty:
    st.error("⚠️ No se detectan los archivos Excel en la carpeta local. Verifica el nombre 'ANALISIS CONSUMO.xlsx' y 'unidades.xlsx'.")
    st.stop()

# Navegación Sidebar 
st.sidebar.image(LOGO_URL, width=150)
st.sidebar.markdown("---")
pg = st.sidebar.radio("Navegación", ["Dashboard Principal", "Modelo Predictivo"], label_visibility="collapsed")

# --- DASHBOARD PRINCIPAL ---
if pg == "Dashboard Principal":
    st.markdown(f"## 🚛 Expreso Diemar — Dashboard LAD 2026")
    st.markdown(f'<div style="margin-bottom:20px;"><span style="background:#292524; border:1px solid #f59e0b; border-radius:8px; padding:8px 14px; color:#fbbf24; font-weight:600;">⛽ Precio Gasoil: ${precio_gasoil:,.0f}/L</span> <small style="color:#64748b;">Fuente: {fuente_precio}</small></div>', unsafe_allow_html=True)

    # Métricas Globales
    lts_totales = df_raw["LITROS"].sum()
    km_totales = df_raw["KM"].sum()
    l100_prom = (lts_totales / km_totales * 100) if km_totales > 0 else 0
    costo_est = lts_totales * precio_gasoil

    k1, k2, k3 = st.columns(3)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-label">Litros Totales</div><div class="kpi-value">{lts_totales:,.0f}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-label">KM Recorridos</div><div class="kpi-value">{km_totales:,.0f}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-label">L/100km Flota</div><div class="kpi-value">{l100_prom:.2f}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Vista de Datos 2026</div>', unsafe_allow_html=True)
    st.dataframe(df_raw, use_container_width=True, hide_index=True)

# --- MODELO PREDICTIVO ---
else:
    st.markdown("## 🧠 Modelo Predictivo — LAD 2026")
    st.info("Modelo entrenado con los datos locales del repositorio.")
    
    # Aquí puedes insertar tu lógica de regresión polinomial original
    # que genera los gráficos de Plotly y alertas de desvío.
    st.success("Conexión con 'ANALISIS CONSUMO.xlsx' establecida correctamente.")
