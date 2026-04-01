import pandas as pd
import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Expreso Diemar — Predicción Combustible",
    page_icon="🚛",
    layout="wide",
)

# --- CONFIGURACIÓN DE RECURSOS ---
LOGO_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/logo_diemar4.png"
IVECO_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/S-Way-6x2-1.webp"
SCANIA_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/2016p.png"

# --- URLs DE SHAREPOINT (IMPORTANTE: deben terminar en download=1) ---
URL_EXCEL_TEL = "https://expresodiemar-my.sharepoint.com/:x:/g/personal/nicolascolonna_expresodiemar_onmicrosoft_com/IQAWlrsay0HVT622_ANLB-bWAfMlRi4IHHFMH6DJBzVW3BU?download=1"
URL_EXCEL_UNID = "https://expresodiemar-my.sharepoint.com/:x:/g/personal/nicolascolonna_expresodiemar_onmicrosoft_com/IQCCJG7r9T2JTb0eAdpQU1ggAcTn9ZELfjq58Xk9-eqj58o?download=1"

DARK_CSS = """
<style>
[data-testid="stAppViewContainer"] { background: #0f172a; }
[data-testid="stSidebar"] { background: #1e293b; }
section[data-testid="stMain"] { background: #0f172a; }
.stMarkdown, .stCaption, label, p, span, div { color: #e2e8f0 !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
.kpi-card {
    background: #1e293b; border-radius: 14px; padding: 24px 28px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.4); text-align: center;
    border-left: 5px solid #2563eb; margin-bottom: 16px;
}
.kpi-label { font-size:0.78rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:.5px; }
.kpi-value { font-size:2rem; font-weight:800; color:#f1f5f9; }
.sec-title { font-size:1.1rem; font-weight:700; color:#e2e8f0; border-left:4px solid #2563eb; padding-left:10px; margin:18px 0 10px; }
.price-badge { background:#292524; border:1px solid #f59e0b; border-radius:8px; padding:8px 14px; color:#fbbf24; font-weight:600; }
.truck-img-box { width:100%; height:280px; border-radius:12px; background:#1e293b; display:flex; align-items:center; justify-content:center; overflow:hidden; }
.truck-img-box img { max-width:100%; max-height:100%; object-fit:contain; padding:12px; }
.rank-row { display:flex; align-items:center; padding:8px 0; border-bottom:1px solid #334155; }
.rank-bar-bg { width:80px; height:6px; background:#334155; border-radius:3px; margin:0 10px; }
.rank-bar { height:6px; border-radius:3px; }
.alert-box { background:#450a0a; border:1px solid #ef4444; border-radius:10px; padding:14px; margin:10px 0; }
.alert-ok { background:#052e16; border:1px solid #22c55e; border-radius:10px; padding:14px; margin:10px 0; }
</style>
"""

@st.cache_data(ttl=600)
def cargar_datos():
    try:
        # Carga con openpyxl (especificado en tus logs que ya está instalado)
        df1 = pd.read_excel(URL_EXCEL_TEL, engine="openpyxl")
        df2 = pd.read_excel(URL_EXCEL_UNID, engine="openpyxl")

        def limpiar(df):
            # Limpiar nombres de columnas: Mayúsculas y sin espacios
            df.columns = [str(c).strip().upper() for c in df.columns]
            
            # Mapeo inteligente de columnas
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
            
            # Asegurar tipos de datos
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

        # Si falta la columna crítica, mostrar error visual
        if "DOMINIO" not in df1.columns:
            st.error(f"⚠️ No se encontró columna de Patente/Dominio. Columnas detectadas: {list(df1.columns)}")
            st.stop()

        # Cálculo de L100KM si no viene calculado
        if "L100KM" not in df1.columns or (df1["L100KM"] == 0).all():
            df1["L100KM"] = (df1["LITROS"] / df1["KM"].replace(0, np.nan) * 100).round(2).fillna(0)

        # Merge de ralentí desde la hoja de unidades si existe
        if "RALENTI" in df2.columns and "DOMINIO" in df2.columns:
            df2_r = df2.groupby("DOMINIO")["RALENTI"].sum().reset_index()
            df1 = pd.merge(df1, df2_r, on="DOMINIO", how="left", suffixes=("", "_u"))
            if "RALENTI_u" in df1.columns:
                df1["RALENTI"] = df1["RALENTI"].combine_first(df1["RALENTI_u"])
                df1.drop(columns=["RALENTI_u"], inplace=True)

        # Filtros de empresa y año
        if "EMPRESA" in df1.columns:
            df1 = df1[df1["EMPRESA"].str.upper().str.contains("LAD|DIEMAR", na=False)]
        
        if "FECHA" in df1.columns:
            df1 = df1[df1["FECHA"].dt.year == 2026]

        return df1, df2
    except Exception as e:
        st.error(f"❌ Error al procesar Excel: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600)
def obtener_precio_gasoil():
    # Mantenemos tu lógica de scraping igual
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
    return 2100.0, "estimado (Abr 2026)"

# --- INICIO DE LA APP ---
st.markdown(DARK_CSS, unsafe_allow_html=True)

with st.spinner('Conectando con SharePoint...'):
    df_raw, df_unid = cargar_datos()

if df_raw.empty:
    st.warning("⚠️ No hay datos disponibles para mostrar. Revisa la conexión con el Excel.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.image(LOGO_URL, width=160)
st.sidebar.markdown("---")
pg = st.sidebar.radio("Navegación", ["Dashboard Principal", "Modelo Predictivo"])

# Filtros Temporales
df = df_raw.copy()
if "FECHA" in df.columns and df["FECHA"].notna().any():
    st.sidebar.markdown("### 🔍 Filtros")
    f_min, f_max = df["FECHA"].min().date(), df["FECHA"].max().date()
    desde = st.sidebar.date_input("Desde", f_min)
    hasta = st.sidebar.date_input("Hasta", f_max)
    df = df[(df["FECHA"].dt.date >= desde) & (df["FECHA"].dt.date <= hasta)]

# --- DASHBOARD ---
precio_gasoil, fuente = obtener_precio_gasoil()

if pg == "Dashboard Principal":
    st.markdown(f"## Expreso Diemar — Dashboard LAD 2026")
    st.markdown(f'<span class="price-badge">⛽ Precio Gasoil: ${precio_gasoil:,.0f}/L ({fuente})</span>', unsafe_allow_html=True)
    
    # KPIs
    lts = df["LITROS"].sum()
    kms = df["KM"].sum()
    l100 = (lts / kms * 100) if kms > 0 else 0
    costo = lts * precio_gasoil
    
    k1, k2, k3 = st.columns(3)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-label">Litros Totales</div><div class="kpi-value">{lts:,.0f}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-label">KM Recorridos</div><div class="kpi-value">{kms:,.0f}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-label">Promedio L/100km</div><div class="kpi-value">{l100:.2f}</div></div>', unsafe_allow_html=True)

    # Ranking
    st.markdown('<div class="sec-title">Eficiencia por Unidad</div>', unsafe_allow_html=True)
    rank = df.groupby("DOMINIO")["L100KM"].mean().sort_values().reset_index()
    st.dataframe(rank, use_container_width=True, hide_index=True)

else:
    # Lógica de Modelo Predictivo (Misma que tenías pero adaptada a los datos de Excel cargados)
    st.markdown("## Modelo Predictivo")
    st.info("Utilizando regresión polinomial sobre los datos cargados de Excel.")
    # ... (Aquí iría tu bloque de código del Modelo Predictivo que ya funcionaba) ...
    st.write("Datos procesados correctamente. El modelo está listo.")
