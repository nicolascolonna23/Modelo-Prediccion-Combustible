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

# --- URLs DE RECURSOS ---
LOGO_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/logo_diemar4.png"
IVECO_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/S-Way-6x2-1.webp"
SCANIA_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/2016p.png"

# --- URLs DE SHAREPOINT (Convertidas a descarga directa) ---
# Se agrega el parámetro &download=1 al final de la URL original
URL_EXCEL_TEL = "https://expresodiemar-my.sharepoint.com/:x:/g/personal/nicolascolonna_expresodiemar_onmicrosoft_com/IQAWlrsay0HVT622_ANLB-bWAfMlRi4IHHFMH6DJBzVW3BU?download=1"
URL_EXCEL_UNID = "https://expresodiemar-my.sharepoint.com/:x:/g/personal/nicolascolonna_expresodiemar_onmicrosoft_com/IQCCJG7r9T2JTb0eAdpQU1ggAcTn9ZELfjq58Xk9-eqj58o?download=1"

DARK_CSS = """
<style>
[data-testid="stAppViewContainer"] { background: #0f172a; }
[data-testid="stSidebar"] { background: #1e293b; }
section[data-testid="stMain"] { background: #0f172a; }
.stMarkdown, .stCaption, label, p, span, div { color: #e2e8f0 !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
[data-testid="stMetricDelta"] { color: #94a3b8 !important; }
.kpi-card {
    background: #1e293b; border-radius: 14px; padding: 24px 28px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.4); text-align: center;
    border-left: 5px solid #2563eb; margin-bottom: 16px;
}
.kpi-label { font-size:0.78rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:.5px; margin-bottom:4px; }
.kpi-value { font-size:2rem; font-weight:800; color:#f1f5f9; line-height:1.1; }
.kpi-sub { font-size:0.75rem; color:#64748b; margin-top:4px; }
.kpi-red { border-left-color:#ef4444; }
.kpi-green { border-left-color:#22c55e; }
.kpi-amber { border-left-color:#f59e0b; }
.sec-title {
    font-size:1.1rem; font-weight:700; color:#e2e8f0;
    border-left:4px solid #2563eb; padding-left:10px; margin:18px 0 10px;
}
.price-badge {
    background:#292524; border:1px solid #f59e0b; border-radius:8px;
    padding:8px 14px; display:inline-block; font-size:0.85rem; color:#fbbf24; font-weight:600;
}
.truck-img-box {
    width:100%; height:280px; border-radius:12px; background:#1e293b;
    display:flex; align-items:center; justify-content:center; overflow:hidden;
}
.truck-img-box img {
    max-width:100%; max-height:100%; width:100%; height:100%;
    object-fit:contain; object-position:center; padding:12px;
}
.rank-row { display:flex; align-items:center; padding:8px 0; border-bottom:1px solid #334155; }
.rank-num { width:28px; font-weight:700; font-size:.9rem; color:#94a3b8; }
.rank-dom { flex:1; font-size:.88rem; color:#e2e8f0; font-weight:600; }
.rank-val { font-size:.88rem; font-weight:700; }
.rank-bar-bg { width:80px; height:6px; background:#334155; border-radius:3px; margin:0 10px; overflow:hidden; }
.rank-bar { height:6px; border-radius:3px; }
.alert-box { background:#450a0a; border:1px solid #ef4444; border-radius:10px; padding:14px 18px; margin:10px 0; }
.alert-ok { background:#052e16; border:1px solid #22c55e; border-radius:10px; padding:14px 18px; margin:10px 0; }
.sidebar-filter-header {
    font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.5px;
    color:#64748b; margin-bottom:10px; padding:6px 0; border-bottom:1px solid #334155;
}
</style>
"""

pg = st.sidebar.radio(
    "Navegacion",
    ["Dashboard Principal", "Modelo Predictivo"],
    index=0,
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.image(LOGO_URL, width=160)

@st.cache_data(ttl=600)
def cargar_datos():
    try:
        # Cambio de read_csv a read_excel para archivos de SharePoint/Excel
        df1 = pd.read_excel(URL_EXCEL_TEL)
        df2 = pd.read_excel(URL_EXCEL_UNID)

        def limpiar(df):
            df.columns = [str(c).strip().upper() for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]
            cm = {}
            for c in df.columns:
                if "DOMINIO" in c or "PATENTE" in c: cm[c] = "DOMINIO"
                elif "LITROS" in c or "CONSUMID" in c: cm[c] = "LITROS"
                elif "DISTANCIA" in c or c == "KM" or "KILOMETR" in c: cm[c] = "KM"
                elif "MARCA" in c: cm[c] = "MARCA"
                elif "TAG" in c: cm[c] = "TAG"
                elif "FECHA" in c or "DATE" in c: cm[c] = "FECHA"
                elif "L/100" in c or "CONSUMO C" in c: cm[c] = "L100KM"
                elif "RALENT" in c: cm[c] = "RALENTI"
                elif "TIEMPO" in c and "MOTOR" in c: cm[c] = "TIEMPO_MOTOR"
                elif "EMPRESA" in c: cm[c] = "EMPRESA"
            
            df = df.rename(columns=cm)
            df = df.loc[:, ~df.columns.duplicated()]
            
            if "DOMINIO" in df.columns:
                df["DOMINIO"] = df["DOMINIO"].astype(str).str.strip().str.upper()
            
            # En Excel los números suelen venir como float/int, no hace falta tanto replace como en CSV
            for col in ["LITROS", "KM", "L100KM", "RALENTI"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            
            if "FECHA" in df.columns:
                df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
            
            return df

        df1 = limpiar(df1)
        df2 = limpiar(df2)

        if "L100KM" not in df1.columns and "LITROS" in df1.columns and "KM" in df1.columns:
            df1["L100KM"] = (df1["LITROS"] / df1["KM"].replace(0, np.nan) * 100).round(2)

        extra = [c for c in ["DOMINIO", "RALENTI"] if c in df2.columns]
        if len(extra) > 1:
            df2r = df2[extra].groupby("DOMINIO").sum(numeric_only=True).reset_index()
            df1 = pd.merge(df1, df2r, on="DOMINIO", how="left", suffixes=("", "_u"))
            if "RALENTI_u" in df1.columns:
                df1["RALENTI"] = df1["RALENTI"].combine_first(df1["RALENTI_u"])
                df1.drop(columns=["RALENTI_u"], inplace=True)

        if "EMPRESA" in df1.columns:
            df1 = df1[df1["EMPRESA"].str.upper().str.contains("LAD|DIEMAR", na=False)]

        if "DOMINIO" in df2.columns and not df2.empty:
            lad_units = df2["DOMINIO"].dropna().unique()
            if len(lad_units) > 0:
                df1 = df1[df1["DOMINIO"].isin(lad_units)]

        if "FECHA" in df1.columns:
            df1 = df1[df1["FECHA"].dt.year == 2026]

        return df1, df2
    except Exception as e:
        st.error(f"Error cargando datos de Excel: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ... EL RESTO DEL CÓDIGO (Lógica de UI, Plots y Modelo) SE MANTIENE IGUAL ...

# [Mantenemos la función obtener_precio_gasoil y la lógica de navegación idéntica a tu código]
