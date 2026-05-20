# ═══════════════════════════════════════════════════════════════════════════════
#  EXPRESO DIEMAR — Dashboard de Monitoreo de Flota v5 (DM Light Theme)
#  IER v6: Z-Score + Tanh
# ═══════════════════════════════════════════════════════════════════════════════
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
LOGO_URL    = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/logo_dm.png"
LOGO_SIDEBAR_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/logo_diemar4.png"
IVECO_URL   = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/S-Way-6x2-1.webp"
SCANIA_URL  = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/2016p.png"
STRALIS_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/image.png"
SWAY_PATENTES   = ['AH522SI', 'AH862UB', 'AH938VO', 'AH842GQ']
SCANIA_PATENTES = ['AD247MQ', 'AE423IW']
LIMITE_VELOCIDAD = 88
BASE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR35NkYPtJrOrdYHLGUH7GIW93s5cPAqQ0zEk5fP1c3gvErwbUW7HJ2OeWBYaBVsYKVmCf0yhLvs6eG/pub?output=csv"
GID_TEL  = "0"
GID_UNID = "882343299"
GID_VEL  = "1563993963"
URL_TEL  = f"{BASE_URL}&gid={GID_TEL}"
URL_UNID = f"{BASE_URL}&gid={GID_UNID}"
URL_VEL  = f"{BASE_URL}&gid={GID_VEL}"
CARGA_URL = "http://bi.sistemaexpreso.com.ar/reporte_hojas.xlsx"

# ═══════════════════════════════════════════════════════════════════════════════
#  PALETA DM — modo claro corporativo
# ═══════════════════════════════════════════════════════════════════════════════
DARK_CSS = """
<style>
[data-testid="stAppViewContainer"] { background: #f5f7fa; }
[data-testid="stSidebar"] { background: #1e3a5f; }
[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
[data-testid="stSidebar"] hr { border-color: #334155 !important; }

/* Inputs del sidebar - fondo blanco, texto oscuro para que se vean */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] input[type="text"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-color: #cbd5e1 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="popover"] * {
    color: #0f172a !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #0f172a !important; }

/* Dropdowns desplegados (popover) - fondo blanco */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] ul,
[data-baseweb="popover"] li {
    background-color: #ffffff !important;
    color: #0f172a !important;
}
[data-baseweb="popover"] li:hover { background-color: #eff6ff !important; }

/* Tags de multiselect (patentes seleccionadas) */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: #2563eb !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span { color: #ffffff !important; }

/* Radio buttons del navegador */
[data-testid="stSidebar"] [role="radiogroup"] label { color: #f1f5f9 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label p { color: #f1f5f9 !important; }

/* Logo del sidebar - sin fondo (transparente) */
[data-testid="stSidebar"] [data-testid="stImage"] {
    margin: 8px 0;
}

section[data-testid="stMain"] { background: #f5f7fa; }
.stMarkdown, .stCaption, label, p, span, div { color: #1e293b; }
[data-testid="stMetricValue"] { color: #0f172a !important; }
[data-testid="stMetricDelta"] { color: #475569 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; }

.kpi-card {
    background: #ffffff; border-radius: 10px; padding: 22px 26px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.08), 0 1px 2px rgba(15,23,42,0.04);
    text-align: center; border-left: 4px solid #1e3a5f; margin-bottom: 14px;
}
.kpi-label  { font-size:0.75rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:.5px; margin-bottom:4px; }
.kpi-value  { font-size:1.9rem; font-weight:800; color:#0f172a; line-height:1.1; }
.kpi-sub    { font-size:0.74rem; color:#94a3b8; margin-top:4px; }
.kpi-red    { border-left-color:#dc2626; }
.kpi-green  { border-left-color:#16a34a; }
.kpi-amber  { border-left-color:#d97706; }
.kpi-purple { border-left-color:#7c3aed; }

.sec-title {
    font-size:1.05rem; font-weight:700; color:#1e3a5f;
    border-left:4px solid #1e3a5f; padding-left:10px; margin:20px 0 12px;
}
.price-badge {
    background:#fef3c7; border:1px solid #d97706; border-radius:6px;
    padding:6px 12px; display:inline-block; font-size:0.82rem; color:#92400e; font-weight:600;
}
.truck-img-box {
    width:100%; height:280px; border-radius:10px; background:#ffffff;
    display:flex; align-items:center; justify-content:center; overflow:hidden;
    border: 1px solid #e2e8f0;
}
.truck-img-box img {
    max-width:100%; max-height:100%; width:100%; height:100%;
    object-fit:contain; object-position:center; padding:12px;
}
.rank-row    { display:flex; align-items:center; padding:8px 0; border-bottom:1px solid #e2e8f0; }
.rank-num    { width:28px; font-weight:700; font-size:.9rem; color:#64748b; }
.rank-dom    { flex:1; font-size:.88rem; color:#1e293b; font-weight:600; }
.rank-val    { font-size:.88rem; font-weight:700; }
.rank-bar-bg { width:80px; height:6px; background:#e2e8f0; border-radius:3px; margin:0 10px; overflow:hidden; }
.rank-bar    { height:6px; border-radius:3px; }
.alert-box   { background:#fef2f2; border:1px solid #dc2626; border-radius:8px; padding:14px 18px; margin:10px 0; color:#7f1d1d; }
.alert-box b { color:#7f1d1d; }
.alert-ok    { background:#f0fdf4; border:1px solid #16a34a; border-radius:8px; padding:14px 18px; margin:10px 0; color:#14532d; }
.alert-ok b  { color:#14532d; }
.highlight-max { background:#fef2f2; border:1px solid #dc2626; border-radius:8px; padding:14px 18px; margin:6px 0; color:#7f1d1d; }
.highlight-max b { color:#7f1d1d; }
.highlight-min { background:#f0fdf4; border:1px solid #16a34a; border-radius:8px; padding:14px 18px; margin:6px 0; color:#14532d; }
.highlight-min b { color:#14532d; }
.training-badge {
    background:#eff6ff; border:1px solid #1e3a5f; border-radius:6px;
    padding:6px 12px; display:inline-block; font-size:0.78rem; color:#1e3a5f; font-weight:600;
    margin-bottom: 12px;
}
.sidebar-filter-header {
    font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.5px;
    color:#94a3b8 !important; margin-bottom:10px; padding:6px 0; border-bottom:1px solid #334155;
}
[data-testid="stSidebar"] [data-testid="stDateInput"] label,
[data-testid="stSidebar"] [data-testid="stMultiSelect"] label,
[data-testid="stSidebar"] [data-testid="stSelectbox"] label {
    font-size:.78rem !important; color:#cbd5e1 !important; font-weight:600 !important;
}
.ier-info-box {
    background:#eff6ff; border:1px solid #1e3a5f; border-radius:8px;
    padding:14px 18px; margin:10px 0; font-size:.85rem; color:#1e3a5f; line-height:1.6;
}
.ier-info-box b { color:#0f172a; }
.ier-method-box {
    background:#f0fdf4; border:1px solid #16a34a; border-radius:8px;
    padding:14px 18px; margin:10px 0; font-size:.82rem; color:#14532d; line-height:1.7;
}
.ier-method-box b { color:#052e16; }
.ier-gauge-wrap {
    background:#ffffff; border-radius:10px; padding:18px 22px;
    border-left:4px solid #1e3a5f; margin-bottom:12px; text-align:center;
    box-shadow: 0 1px 3px rgba(15,23,42,0.08);
}
.ier-score-big { font-size:2.4rem; font-weight:900; line-height:1; }
.ier-clasif    { font-size:.85rem; font-weight:700; margin-top:4px; color:#1e293b; }
.ier-comp-row  {
    display:flex; align-items:center; justify-content:space-between;
    background:#f8fafc; border-radius:6px; padding:8px 14px; margin:4px 0;
    font-size:.82rem; border:1px solid #e2e8f0;
}
.ier-comp-label { color:#475569; flex:1; }
.ier-comp-val   { font-weight:700; color:#0f172a; }
.ier-comp-bar-bg { width:90px; height:6px; background:#e2e8f0; border-radius:3px; margin:0 10px; overflow:hidden; }
.ier-comp-bar    { height:6px; border-radius:3px; }
.vel-badge {
    background:#fff7ed; border:1px solid #ea580c; border-radius:6px;
    padding:3px 10px; display:inline-block; font-size:.78rem; color:#9a3412; font-weight:700;
}
.zscore-badge {
    background:#eef2ff; border:1px solid #4f46e5; border-radius:5px;
    padding:2px 8px; display:inline-block; font-size:.72rem; color:#3730a3; font-weight:600;
}
.dm-header {
    background: #ffffff;
    border-radius: 10px; padding: 18px 24px; margin-bottom: 18px;
    box-shadow: 0 2px 6px rgba(30,58,95,0.08);
    border-left: 5px solid #1e3a5f;
}
.dm-header-title { font-size:1.5rem; font-weight:800; color:#1e3a5f; }
.dm-header-sub { font-size:.85rem; color:#64748b; margin-top:4px; }
</style>
"""
pg = st.sidebar.radio(
    "Navegacion",
    ["Dashboard Principal", "Modelo Predictivo", "Análisis por Patente", "Datos Operativos"],
    index=0,
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.image(LOGO_SIDEBAR_URL, width=160)
@st.cache_data(ttl=600)
def cargar_datos():
    try:
        df1 = pd.read_csv(URL_TEL)
        df2 = pd.read_csv(URL_UNID)
        def limpiar(df):
            df.columns = [str(c).strip().upper() for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]
            cm = {}
            for c in df.columns:
                if   "DOMINIO"   in c or "PATENTE"  in c:              cm[c] = "DOMINIO"
                elif "LITROS"    in c or "CONSUMID" in c:              cm[c] = "LITROS"
                elif "DISTANCIA" in c or c == "KM" or "KILOMETR" in c: cm[c] = "KM"
                elif "MARCA"     in c:                                  cm[c] = "MARCA"
                elif "TAG"       in c:                                  cm[c] = "TAG"
                elif "FECHA"     in c or "DATE"     in c:              cm[c] = "FECHA"
                elif "L/100"     in c or "CONSUMO C" in c:             cm[c] = "L100KM"
                elif "RALENT"    in c:                                  cm[c] = "RALENTI"
                elif "TIEMPO"    in c and "MOTOR"   in c:              cm[c] = "TIEMPO_MOTOR"
                elif "EMPRESA"   in c:                                  cm[c] = "EMPRESA"
            df = df.rename(columns=cm).loc[:, ~df.rename(columns=cm).columns.duplicated()]
            if "DOMINIO" in df.columns:
                df["DOMINIO"] = df["DOMINIO"].astype(str).str.strip().str.upper()
            for col in ["LITROS", "KM", "L100KM", "RALENTI"]:
                if col in df.columns:
                    serie = df[col]
                    if isinstance(serie, pd.DataFrame):
                        serie = serie.iloc[:, 0]
                    serie = serie.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
                    df[col] = pd.to_numeric(serie, errors="coerce").fillna(0)
            if "FECHA" in df.columns:
                df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce", dayfirst=True)
            return df
        df1 = limpiar(df1)
        df2 = limpiar(df2)
        if "L100KM" not in df1.columns and "LITROS" in df1.columns and "KM" in df1.columns:
            df1["L100KM"] = (df1["LITROS"] / df1["KM"].replace(0, np.nan) * 100).round(2)
        if 'RALENTI' in df2.columns and 'DOMINIO' in df2.columns:
            if 'FECHA' in df2.columns and 'FECHA' in df1.columns:
                df2_ral = df2[['DOMINIO','FECHA','RALENTI']].copy()
                df2_ral = df2_ral[df2_ral['RALENTI'] > 0]
                df2_ral['_MES'] = df2_ral['FECHA'].dt.to_period('M')
                df1['_MES']     = df1['FECHA'].dt.to_period('M')
                df1 = df1.merge(
                    df2_ral[['DOMINIO','_MES','RALENTI']].rename(columns={'RALENTI':'_RAL_u'}),
                    on=['DOMINIO','_MES'], how='left'
                )
                if 'RALENTI' not in df1.columns:
                    df1['RALENTI'] = df1['_RAL_u'].fillna(0)
                else:
                    df1['RALENTI'] = df1['RALENTI'].combine_first(df1['_RAL_u']).fillna(0)
                df1.drop(columns=['_RAL_u','_MES'], inplace=True, errors='ignore')
        if "EMPRESA" in df1.columns:
            df1 = df1[df1["EMPRESA"].str.upper().str.contains("LAD|DIEMAR", na=False)]
        if "DOMINIO" in df2.columns and not df2.empty:
            lad_units = df2["DOMINIO"].dropna().unique()
            if len(lad_units) > 0:
                df1 = df1[df1["DOMINIO"].isin(lad_units)]
        return df1, df2
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame(), pd.DataFrame()
@st.cache_data(ttl=600)
def cargar_velocidad():
    try:
        df = pd.read_csv(URL_VEL)
        df.columns = [str(c).strip() for c in df.columns]
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if   "movil" in cl or "patente" in cl or "dominio" in cl: col_map[c] = "DOMINIO"
            elif "fecha"    in cl:                                      col_map[c] = "FECHA"
            elif "veloc"    in cl:                                      col_map[c] = "VELOCIDAD"
            elif "gravedad" in cl:                                      col_map[c] = "GRAVEDAD"
            elif "tipo"     in cl:                                      col_map[c] = "TIPO"
        df = df.rename(columns=col_map)
        if "DOMINIO" in df.columns:
            df["DOMINIO"] = df["DOMINIO"].astype(str).str.strip().str.upper()
        if "FECHA" in df.columns:
            df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce", dayfirst=True)
        if "VELOCIDAD" not in df.columns:
            for c in df.columns:
                try:
                    serie = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
                    if serie.dropna().between(50, 200).mean() > 0.5:
                        df = df.rename(columns={c: "VELOCIDAD"})
                        break
                except Exception:
                    continue
        if "VELOCIDAD" not in df.columns:
            return pd.DataFrame(columns=["DOMINIO","FECHA","VELOCIDAD","EXCESO_KMH"])
        df["VELOCIDAD"] = pd.to_numeric(
            df["VELOCIDAD"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
        df = df[df["VELOCIDAD"] > LIMITE_VELOCIDAD].copy()
        df["EXCESO_KMH"] = (df["VELOCIDAD"] - LIMITE_VELOCIDAD).round(1)
        keep = [c for c in ["DOMINIO","FECHA","VELOCIDAD","EXCESO_KMH","GRAVEDAD","TIPO"] if c in df.columns]
        return df[keep].dropna(subset=["DOMINIO","FECHA"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["DOMINIO","FECHA","VELOCIDAD","EXCESO_KMH"])
@st.cache_data(ttl=3600)
def cargar_carga():
    try:
        import re
        df = pd.read_excel(CARGA_URL)
        df.columns = [str(c).strip() for c in df.columns]
        col_unid   = next((c for c in df.columns if 'UNID'    in c.upper()), None)
        col_peso   = next((c for c in df.columns if 'PESO'    in c.upper() and 'ENTREGAD' in c.upper()), None)
        col_fecha  = next((c for c in df.columns if 'FECHA'   in c.upper()), None)
        col_estado = next((c for c in df.columns if 'ESTADO'  in c.upper()), None)
        if not all([col_unid, col_peso, col_fecha]):
            return pd.DataFrame()
        if col_estado:
            df = df[df[col_estado].astype(str).str.upper() == 'FINALIZADA']
        df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
        df[col_peso]  = pd.to_numeric(df[col_peso],  errors='coerce').fillna(0)
        df = df[(df[col_peso] > 0) & df[col_fecha].notna()].copy()
        def norm_pat(p):
            return re.sub(r'\s+', '', str(p).strip().upper())
        df['_pats'] = df[col_unid].astype(str).str.split(',')
        df = df.explode('_pats')
        df['DOMINIO']  = df['_pats'].apply(norm_pat)
        df['MES']      = df[col_fecha].dt.to_period('M')
        df['PESO_TON'] = df[col_peso] / 1000.0
        return (df.groupby(['DOMINIO','MES'])
                  .agg(PESO_TON=('PESO_TON','sum'))
                  .reset_index())
    except Exception:
        return pd.DataFrame()
@st.cache_data(ttl=3600)
def cargar_viajes_todos():
    """
    Carga TODOS los viajes finalizados incluyendo los sin peso registrado.
    Permite calcular % viajes sin carga por patente (proxy de retornos vacíos).
    """
    try:
        import re
        df = pd.read_excel(CARGA_URL)
        df.columns = [str(c).strip() for c in df.columns]
        col_unid   = next((c for c in df.columns if 'UNID'    in c.upper()), None)
        col_peso   = next((c for c in df.columns if 'PESO'    in c.upper() and 'ENTREGAD' in c.upper()), None)
        col_fecha  = next((c for c in df.columns if 'FECHA'   in c.upper()), None)
        col_estado = next((c for c in df.columns if 'ESTADO'  in c.upper()), None)
        if not all([col_unid, col_peso, col_fecha]):
            return pd.DataFrame()
        if col_estado:
            df = df[df[col_estado].astype(str).str.upper() == 'FINALIZADA']
        df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
        df[col_peso]  = pd.to_numeric(df[col_peso], errors='coerce').fillna(0)
        df = df[df[col_fecha].notna()].copy()
        def norm_pat(p):
            return re.sub(r'\s+', '', str(p).strip().upper())
        df['_pats']     = df[col_unid].astype(str).str.split(',')
        df              = df.explode('_pats')
        df['DOMINIO']   = df['_pats'].apply(norm_pat)
        df['MES']       = df[col_fecha].dt.to_period('M')
        df['PESO_TON']  = df[col_peso] / 1000.0
        df['CON_CARGA'] = (df['PESO_TON'] > 0).astype(int)
        return df[['DOMINIO','MES','PESO_TON','CON_CARGA']].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()
@st.cache_data(ttl=3600)
def obtener_precio_gasoil():
    try:
        CKAN_URL = (
            "https://datos.energia.gob.ar/api/3/action/datastore_search"
            "?resource_id=80ac25de-a44a-4445-9215-090cf55cfda5"
            "&limit=1000"
        )
        r = requests.get(CKAN_URL, timeout=12)
        data = r.json()
        records = data.get('result', {}).get('records', [])
        if records:
            import re as _re
            gasoil_rows = [
                rec for rec in records
                if any(
                    kw in str(rec.get('producto', '')).upper()
                    for kw in ['GASOIL', 'DIESEL', 'GAS OIL']
                )
            ]
            if not gasoil_rows:
                gasoil_rows = records
            precio_fields = ['precio_ars', 'precio', 'precio_venta', 'importe', 'valor']
            precios = []
            for rec in gasoil_rows:
                for f in precio_fields:
                    v = rec.get(f)
                    if v is not None:
                        try:
                            p = float(str(v).replace(',', '.'))
                            if 500 < p < 10000:
                                precios.append(p)
                                break
                        except (ValueError, TypeError):
                            pass
            if precios:
                precio_med = float(sorted(precios)[len(precios)//2])
                return precio_med, "datos.energia.gob.ar (oficial)"
    except Exception:
        pass
    try:
        import re
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get("https://surtidores.com.ar/precios/", headers=headers, timeout=10)
        soup  = BeautifulSoup(r.text, "html.parser")
        texto = soup.get_text(separator=" ")
        idx_2026 = texto.find("2026")
        if idx_2026 != -1:
            segmento   = texto[idx_2026:idx_2026 + 600]
            gasoil_idx = segmento.lower().find("gasoil")
            if gasoil_idx != -1:
                linea   = segmento[gasoil_idx:gasoil_idx + 120]
                numeros = re.findall(r'\b(\d{3,4})\b', linea)
                numeros = [int(n) for n in numeros if 500 <= int(n) <= 5000]
                if numeros:
                    return float(numeros[-1]), "surtidores.com.ar (2026)"
        matches = re.findall(r'[Gg]as[oi][il][^\d]*(\d{3,4})', texto[:8000])
        if matches:
            precio = float(matches[0])
            if 500 < precio < 5000:
                return precio, "surtidores.com.ar"
    except Exception:
        pass
    return 2025.0, "referencia estimada"
def asignar_modelo(dominio):
    d = str(dominio).strip().upper()
    if d in SWAY_PATENTES:   return 'S-Way'
    if d in SCANIA_PATENTES: return 'Scania'
    return 'Stralis'
def calcular_score_zscore(series, higher_is_better=True, k=0.4, min_sigma_pct=0.05):
    series = pd.to_numeric(series, errors='coerce')
    if series.dropna().count() <= 1:
        return pd.Series(1.0, index=series.index)
    mu    = series.mean()
    sigma = series.std(ddof=0)
    if sigma < 1e-9 and abs(mu) < 1e-9:
        return pd.Series(1.0, index=series.index)
    sigma_floor = min_sigma_pct * abs(mu) if abs(mu) > 1e-9 else 0.0
    sigma_eff   = max(sigma, sigma_floor)
    if sigma_eff < 1e-9:
        return pd.Series(1.0, index=series.index)
    z = (series - mu) / sigma_eff
    if not higher_is_better:
        z = -z
    scores = 1.0 + 1.5 * np.tanh(k * z)
    return scores.clip(0.4, 2.5).fillna(1.0)
def calcular_ier(df, df_vel=None, df_carga=None):
    if 'DOMINIO' not in df.columns or df.empty:
        return pd.DataFrame()
    df_c = df[df['L100KM'] > 0].copy()
    if df_c.empty:
        return pd.DataFrame()
    agg_dict = {'L100KM': ('L100KM','mean'), 'KM': ('KM','sum'), 'LITROS': ('LITROS','sum')}
    if 'MES_PERIODO' in df_c.columns:
        agg_dict['MESES'] = ('MES_PERIODO','nunique')
    agg = df_c.groupby('DOMINIO').agg(**agg_dict).reset_index()
    if 'RALENTI' in df_c.columns:
        ral = df_c.groupby('DOMINIO').agg(_RAL=('RALENTI','sum'), _LTS=('LITROS','sum')).reset_index()
        ral['_RATIO'] = ral['_RAL'] / ral['_LTS'].replace(0, np.nan)
        ral['RALENTI_PCT'] = np.where(ral['_RATIO'].between(0,1.0), (ral['_RATIO']*100).clip(0,100).round(2), 0.0)
        ral['RALENTI_PCT'] = ral['RALENTI_PCT'].fillna(0)
        agg = agg.merge(ral[['DOMINIO','RALENTI_PCT']], on='DOMINIO', how='left')
    else:
        agg['RALENTI_PCT'] = 0.0
    agg['RALENTI_PCT'] = agg['RALENTI_PCT'].fillna(0.0)
    if df_vel is not None and not df_vel.empty and 'DOMINIO' in df_vel.columns:
        vel_counts = df_vel.groupby('DOMINIO').agg(
            EXCESOS=('DOMINIO','count'),
            VEL_MAX=('VELOCIDAD','max'),
            SEVERIDAD=('EXCESO_KMH','sum')
        ).reset_index()
        agg = agg.merge(vel_counts, on='DOMINIO', how='left')
        agg['EXCESOS']   = agg['EXCESOS'].fillna(0).astype(int)
        agg['VEL_MAX']   = agg['VEL_MAX'].fillna(0)
        agg['SEVERIDAD'] = agg['SEVERIDAD'].fillna(0)
    else:
        agg['EXCESOS'] = 0; agg['VEL_MAX'] = 0; agg['SEVERIDAD'] = 0.0
    agg['MODELO'] = agg['DOMINIO'].apply(asignar_modelo)
    if df_carga is not None and not df_carga.empty and 'MES_PERIODO' in df.columns:
        meses_activos = df['MES_PERIODO'].dropna().unique()
        carga_periodo = df_carga[df_carga['MES'].isin(meses_activos)]
        if not carga_periodo.empty:
            carga_agg = carga_periodo.groupby('DOMINIO')['PESO_TON'].sum().reset_index()
            agg = agg.merge(carga_agg, on='DOMINIO', how='left')
            agg['PESO_TON'] = agg['PESO_TON'].fillna(0)
        else:
            agg['PESO_TON'] = 0.0
    else:
        agg['PESO_TON'] = 0.0
    agg['TONKML'] = np.where(
        (agg['PESO_TON']>0)&(agg['LITROS']>0),
        (agg['PESO_TON']*agg['KM'])/agg['LITROS'], np.nan)
    tiene_carga = agg['PESO_TON'].sum() > 0
    def _safe_mean(x):
        v = x.dropna(); return v.mean() if len(v)>0 else np.nan
    modelo_avgs = agg.groupby('MODELO').agg(
        L100KM_MOD=('L100KM','mean'), KM_MOD=('KM','mean'),
        RAL_MOD=('RALENTI_PCT','mean'), EXCESOS_MOD=('EXCESOS','mean'),
        SEVERIDAD_MOD=('SEVERIDAD','mean'),
        TONKML_MOD=('TONKML',_safe_mean)).reset_index()
    agg = agg.merge(modelo_avgs, on='MODELO', how='left')
    for col in ['SCORE_CONSUMO','SCORE_KM','SCORE_VEL','SCORE_RAL']:
        agg[col] = 1.0
    for modelo in agg['MODELO'].unique():
        mask = agg['MODELO']==modelo
        idx  = agg.index[mask]
        if mask.sum()==0: continue
        agg.loc[idx,'SCORE_CONSUMO'] = calcular_score_zscore(agg.loc[idx,'L100KM'], higher_is_better=False, k=0.4, min_sigma_pct=0.05).values
        agg.loc[idx,'SCORE_KM'] = calcular_score_zscore(agg.loc[idx,'KM'], higher_is_better=True, k=0.4, min_sigma_pct=0.05).values
        sev_log = np.log1p(agg.loc[idx,'SEVERIDAD'].astype(float))
        agg.loc[idx,'SCORE_VEL'] = calcular_score_zscore(sev_log, higher_is_better=False, k=0.4, min_sigma_pct=0.30).values
        ral_log = np.log1p(agg.loc[idx,'RALENTI_PCT'].astype(float))
        agg.loc[idx,'SCORE_RAL'] = calcular_score_zscore(ral_log, higher_is_better=False, k=0.4, min_sigma_pct=0.30).values
    agg['IER'] = (
        0.60 * agg['SCORE_CONSUMO'] +
        0.20 * agg['SCORE_KM']      +
        0.10 * agg['SCORE_RAL']     +
        0.10 * agg['SCORE_VEL']
    ).mul(100).round(1)
    agg['IER'] = agg['IER'].fillna(100.0)
    for modelo in agg['MODELO'].unique():
        mask     = agg['MODELO']==modelo
        grp_mean = agg.loc[mask,'IER'].mean()
        if not np.isnan(grp_mean) and grp_mean>0 and mask.sum()>1:
            agg.loc[mask,'IER'] = (agg.loc[mask,'IER']-grp_mean+100).round(1)
    def clasif(v):
        if   v>=105: return '🟢 Eficiente'
        elif v>= 95: return '🟡 Normal'
        elif v>= 85: return '🟠 Atención'
        else:        return '🔴 Crítico'
    agg['CLASIFICACION'] = agg['IER'].apply(clasif)
    agg['TONKML']     = agg['TONKML'].fillna(0).round(2)
    agg['TONKML_MOD'] = agg['TONKML_MOD'].fillna(0).round(2)
    keep = ['DOMINIO','MODELO','IER','CLASIFICACION',
            'L100KM','L100KM_MOD','RALENTI_PCT','RAL_MOD',
            'KM','KM_MOD','LITROS','EXCESOS','SEVERIDAD','SEVERIDAD_MOD','VEL_MAX','EXCESOS_MOD',
            'PESO_TON','TONKML','TONKML_MOD',
            'SCORE_CONSUMO','SCORE_KM','SCORE_VEL','SCORE_RAL']
    if 'MESES' in agg.columns: keep.append('MESES')
    return agg[keep].sort_values('IER', ascending=False).reset_index(drop=True)
with st.spinner('Cargando telemetría, velocidades y datos de carga...'):
    df_raw, df_unid = cargar_datos()
    df_vel_raw      = cargar_velocidad()
    df_carga_raw    = cargar_carga()
    df_viajes_raw   = cargar_viajes_todos()
if df_raw.empty:
    st.warning('No se pudieron cargar datos.')
    st.stop()
precio_gasoil, precio_fuente = obtener_precio_gasoil()
st.markdown(DARK_CSS, unsafe_allow_html=True)
if 'DOMINIO' in df_raw.columns:
    df_raw['MODELO'] = df_raw['DOMINIO'].apply(asignar_modelo)
df_full = df_raw.copy()
anios_disponibles = (sorted(df_full['FECHA'].dt.year.dropna().unique().tolist(), reverse=True)
                     if 'FECHA' in df_full.columns else [2025])
anio_sel = st.sidebar.selectbox('Año de visualización', anios_disponibles, index=0)
df = (df_full[df_full['FECHA'].dt.year==anio_sel].copy()
      if 'FECHA' in df_full.columns else df_full.copy())
if not df_vel_raw.empty and 'FECHA' in df_vel_raw.columns:
    df_vel_anio = df_vel_raw[df_vel_raw['FECHA'].dt.year==anio_sel].copy()
else:
    df_vel_anio = df_vel_raw.copy()
if 'FECHA' in df.columns and df['FECHA'].notna().any():
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-filter-header">Filtros</div>', unsafe_allow_html=True)
    periodos_disponibles = sorted(df['FECHA'].dt.to_period('M').dropna().unique().tolist())
    periodos_str = [str(p) for p in periodos_disponibles]
    if periodos_str:
        desde_idx = st.sidebar.selectbox('Desde (mes/año)', options=periodos_str, index=0)
        hasta_idx = st.sidebar.selectbox('Hasta (mes/año)', options=periodos_str, index=len(periodos_str)-1)
        desde_periodo = pd.Period(desde_idx, 'M')
        hasta_periodo = pd.Period(hasta_idx, 'M')
        df = df[(df['FECHA'].dt.to_period('M')>=desde_periodo)&(df['FECHA'].dt.to_period('M')<=hasta_periodo)]
    marcas_disp   = sorted(df['MARCA'].dropna().unique().tolist())   if 'MARCA'   in df.columns else []
    marcas_sel    = st.sidebar.multiselect('Marca', marcas_disp, default=marcas_disp)
    patentes_disp = sorted(df['DOMINIO'].dropna().unique().tolist()) if 'DOMINIO' in df.columns else []
    patentes_sel  = st.sidebar.multiselect('Patente', patentes_disp, default=[], placeholder="Todas las patentes")
    if marcas_sel   and 'MARCA'   in df.columns: df = df[df['MARCA'].isin(marcas_sel)]
    if patentes_sel and 'DOMINIO' in df.columns: df = df[df['DOMINIO'].isin(patentes_sel)]
if df.empty:
    st.warning(f'Sin datos para {anio_sel} con los filtros seleccionados.')
    st.stop()
df['MES_PERIODO'] = df['FECHA'].dt.to_period('M')
df['MES_NUM']     = df['FECHA'].dt.month
meses_df = df.groupby('MES_PERIODO').agg(LITROS=('LITROS','sum'),KM=('KM','sum')).reset_index().sort_values('MES_PERIODO')
meses_df['L100'] = (meses_df['LITROS']/meses_df['KM'].replace(0,np.nan)*100).round(2)
ralenti_total = df['RALENTI'].sum() if 'RALENTI' in df.columns else 0
ralenti_delta_txt = ''
if 'RALENTI' in df.columns and 'MES_PERIODO' in df.columns:
    _mg = df.groupby('MES_PERIODO').agg(_RAL=('RALENTI','sum'),_LTS=('LITROS','sum')).reset_index().sort_values('MES_PERIODO')
    if len(_mg)>=2:
        _curr=_mg.iloc[-1]; _prev=_mg.iloc[-2]
        _pct_curr=_curr['_RAL']/_curr['_LTS']*100 if _curr['_LTS']>0 else 0
        _pct_prev=_prev['_RAL']/_prev['_LTS']*100 if _prev['_LTS']>0 else 0
        _dr=_pct_curr-_pct_prev
        ralenti_delta_txt=f"{'▲' if _dr>0 else '▼'} {abs(_dr):.1f}pp vs mes ant."
df_full_clean = df_full[df_full['FECHA'].notna()&(df_full['KM']>0)].copy()
df_full_clean['MES_PERIODO'] = df_full_clean['FECHA'].dt.to_period('M')
meses_hist_full = df_full_clean.groupby('MES_PERIODO').agg(LITROS=('LITROS','sum'),KM=('KM','sum')).reset_index().sort_values('MES_PERIODO')
meses_hist_full['L100'] = (meses_hist_full['LITROS']/meses_hist_full['KM'].replace(0,np.nan)*100).round(2)
meses_hist_full = meses_hist_full[meses_hist_full['KM']>0].copy()
n_meses_entrenamiento = len(meses_hist_full)
if not df.empty and not df_vel_anio.empty and 'FECHA' in df_vel_anio.columns:
    _mes_min = df['FECHA'].dropna().dt.to_period('M').min()
    _mes_max = df['FECHA'].dropna().dt.to_period('M').max()
    _vel_periodos = df_vel_anio['FECHA'].dropna().dt.to_period('M')
    df_vel_filtrado = df_vel_anio[(_vel_periodos>=_mes_min)&(_vel_periodos<=_mes_max)&(df_vel_anio['DOMINIO'].isin(df['DOMINIO'].unique()))].copy()
else:
    df_vel_filtrado = df_vel_anio.copy()
df_ier = calcular_ier(df, df_vel_filtrado, df_carga=df_carga_raw)
total_excesos  = len(df_vel_filtrado) if not df_vel_filtrado.empty else 0
vel_max_global = (df_vel_filtrado['VELOCIDAD'].max() if not df_vel_filtrado.empty and 'VELOCIDAD' in df_vel_filtrado.columns else 0)

# Paleta Plotly clara
PLOT_BG   = 'rgba(255,255,255,0.6)'
PAPER_BG  = 'rgba(0,0,0,0)'
FONT_COL  = '#1e293b'
GRID_COL  = '#e2e8f0'
AXIS_COL  = '#64748b'
LEGEND_BG = 'rgba(248,250,252,0.95)'

# ═══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 1 — DASHBOARD PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
if pg == "Dashboard Principal":
    col_logo, col_title = st.columns([1,5])
    with col_logo: st.image(LOGO_URL, width=130)
    with col_title:
        st.markdown(f"""<div class='dm-header'>
        <div class='dm-header-title'>Expreso Diemar — Dashboard LAD {anio_sel}</div>
        <div class='dm-header-sub'>Telemetría flota LAD · Año {anio_sel} · Actualización automática</div>
        </div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">Métricas Globales — {anio_sel}</div>', unsafe_allow_html=True)
    lts_total  = df['LITROS'].sum() if 'LITROS' in df.columns else 0
    kms_total  = df['KM'].sum()     if 'KM'     in df.columns else 0
    l100_prom  = round(lts_total/kms_total*100,2) if kms_total>0 else 0
    costo_est  = lts_total*precio_gasoil
    n_unidades = df['DOMINIO'].nunique() if 'DOMINIO' in df.columns else 0
    ralenti_pct = round(ralenti_total/lts_total*100,1) if lts_total>0 else 0
    if len(meses_df)>=2:
        delta_l100 = meses_df['L100'].iloc[-1]-meses_df['L100'].iloc[-2]
        delta_txt  = f"{'▲' if delta_l100>0 else '▼'} {abs(delta_l100):.2f} vs mes anterior"
        delta_col  = 'kpi-red' if delta_l100>0 else 'kpi-green'
    else:
        delta_txt, delta_col = '', ''
    def kpi(cont, color, label, value, sub=''):
        cont.markdown(f'<div class="kpi-card {color}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)
    k1,k2,k3 = st.columns(3)
    kpi(k1,'','Litros totales',f'{lts_total:,.0f}',f'litros {anio_sel}')
    kpi(k2,'','KM recorridos',f'{kms_total:,.0f}',f'kilómetros {anio_sel}')
    kpi(k3,delta_col,'L/100km flota',f'{l100_prom:.2f}',delta_txt)
    k4,k5,k6 = st.columns(3)
    kpi(k4,'kpi-amber','Costo estimado',f'${costo_est/1e6:.1f}M',f'@ ${precio_gasoil:,.0f}/L')
    kpi(k5,'kpi-green','Unidades activas',f'{n_unidades}','dominios únicos')
    _ral_sub = (f'{ralenti_total:,.0f} L · {ralenti_delta_txt}' if ralenti_delta_txt else f'{ralenti_total:,.0f} L en ralentí')
    kpi(k6,'kpi-amber','% Ralentí',f'{ralenti_pct:.1f}%',_ral_sub)
    st.divider()
    st.markdown(f'<div class="sec-title">Rendimiento por Modelo — {anio_sel}</div>', unsafe_allow_html=True)
    def stats_modelo(patentes_lista):
        if 'DOMINIO' not in df.columns: return {'l100':0,'lts':0,'kms':0,'n':0,'total':len(patentes_lista)}
        sub = df[df['DOMINIO'].isin(patentes_lista)]
        if sub.empty: return {'l100':0,'lts':0,'kms':0,'n':0,'total':len(patentes_lista)}
        lts=sub['LITROS'].sum(); kms=sub['KM'].sum()
        return {'l100':round(lts/kms*100,2) if kms>0 else 0,'lts':lts,'kms':kms,'n':sub['DOMINIO'].nunique(),'total':len(patentes_lista)}
    todas_patentes   = df['DOMINIO'].dropna().unique().tolist() if 'DOMINIO' in df.columns else []
    stralis_patentes = [p for p in todas_patentes if p not in SWAY_PATENTES and p not in SCANIA_PATENTES]
    s_sway=stats_modelo(SWAY_PATENTES); s_scania=stats_modelo(SCANIA_PATENTES); s_stralis=stats_modelo(stralis_patentes)
    tc1,tc2,tc3 = st.columns(3)
    for col_t,modelo,img_url,s,pats_label in [
        (tc1,'S-Way',IVECO_URL,s_sway,'AH522SI · AH862UB · AH938VO · AH842GQ'),
        (tc2,'Scania',SCANIA_URL,s_scania,'AD247MQ · AE423IW'),
        (tc3,'Stralis',STRALIS_URL,s_stralis,'Resto de la flota')]:
        with col_t:
            st.markdown(f'<div class="truck-img-box"><img src="{img_url}" alt="{modelo}" /></div>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            sc1,sc2,sc3=st.columns(3)
            sc1.metric('L/100km', f"{s['l100']:.1f}" if s['l100']>0 else '—')
            sc2.metric('Unidades', f"{s['total']}")
            lts_fmt = f"{s['lts']/1000:.1f}k" if s['lts']>=1000 else f"{s['lts']:.0f}"
            sc3.metric(f'Lts {anio_sel}', lts_fmt if s['lts']>0 else '—')
            st.caption(f"Patentes: {pats_label} | {s['kms']:,.0f} km")
    st.divider()
    st.markdown(f'<div class="sec-title">Ranking de Eficiencia — {anio_sel}</div>', unsafe_allow_html=True)
    rcol1,rcol2 = st.columns(2)
    def render_ranking(col,titulo,df_rank,color_fn):
        with col:
            st.markdown(f'**{titulo}**')
            if df_rank.empty: st.info('Sin datos.'); return
            vmin,vmax=df_rank['L100KM'].min(),df_rank['L100KM'].max()
            rh='<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;">'
            rh+='<div style="font-size:.72rem;display:flex;justify-content:space-between;color:#64748b;margin-bottom:6px;"><span>Unidad</span><span>L/100km</span></div>'
            for i,(_,r) in enumerate(df_rank.iterrows(),1):
                v=r['L100KM']; pct=int((v-vmin)/(vmax-vmin)*100) if vmax!=vmin else 50; cb=color_fn(i)
                rh+=(f'<div class="rank-row"><div class="rank-num">#{i}</div><div class="rank-dom">{r["DOMINIO"]}</div>'
                     f'<div class="rank-bar-bg"><div class="rank-bar" style="width:{pct}%;background:{cb}"></div></div>'
                     f'<div class="rank-val" style="color:{cb}">{v:.2f}</div></div>')
            rh+='</div>'
            st.markdown(rh, unsafe_allow_html=True)
    if 'DOMINIO' in df.columns and 'L100KM' in df.columns:
        base=df[df['L100KM']>0].groupby('DOMINIO')['L100KM'].mean().round(2).reset_index()
        render_ranking(rcol1,'TOP 10 más eficientes (menor L/100km)',base.sort_values('L100KM').head(10),
                       lambda i:'#16a34a' if i<=3 else ('#d97706' if i<=6 else '#dc2626'))
        render_ranking(rcol2,'TOP 10 menos eficientes (mayor L/100km)',base.sort_values('L100KM',ascending=False).head(10),
                       lambda i:'#dc2626' if i<=3 else ('#d97706' if i<=6 else '#16a34a'))
    st.divider()
    st.markdown(f'<div class="sec-title">Índice de Eficiencia Relativa (IER v6) — {anio_sel}</div>', unsafe_allow_html=True)
    tiene_vel = total_excesos>0
    pond_txt = (
        f"<b>60%</b> L/100km &nbsp;·&nbsp; "
        f"<b>20%</b> KM totales &nbsp;·&nbsp; "
        f"<b>10%</b> % Ralentí &nbsp;·&nbsp; "
        f"<b>10%</b> Severidad vel. {'OK' if tiene_vel else 'sin datos'} "
    )
    st.markdown(f"""<div class="ier-info-box">
    <b>¿Qué es el IER v6?</b> Métrica estadísticamente justa: cada camión se compara <b>solo contra el promedio de su propio modelo</b> — Stralis vs Stralis, S‑Way vs S‑Way, Scania vs Scania.<br>
    <b>Scoring:</b> Z-Score + Tanh. El promedio del grupo obtiene IER ≈ 100. Mayor IER = mejor rendimiento relativo.<br>
    <b>Velocidad:</b> mide <b>severidad acumulada</b> (km/h sobre el límite) — no cantidad de eventos.<br>
    <b>Ponderación:</b>&nbsp;{pond_txt}<br>
    <b>Escala:</b>&nbsp;🟢 Eficiente ≥105 &nbsp;·&nbsp; 🟡 Normal 95–105 &nbsp;·&nbsp; 🟠 Atención 85–95 &nbsp;·&nbsp; 🔴 Crítico &lt;85
    </div>""", unsafe_allow_html=True)
    with st.expander('Metodología — ¿Por qué Z-Score + Tanh?'):
        st.markdown("""<div class="ier-method-box">
        <b>Problema del ratio simple:</b><br>
        · Un camión 50% mejor: ratio=2.0 | Un camión 50% peor: ratio=0.67 — asimetría injusta<br><br>
        <b>Solución — Z-Score + Tanh:</b><br>
        Paso 1 — Z = (valor − promedio_modelo) / desv.std_modelo<br>
        Paso 2 — Ajuste de dirección (consumo bajo=bueno → invertir z)<br>
        Paso 3 — score = 1.0 + 1.5 × tanh(0.4 × z)<br>
        &nbsp;&nbsp;→ z=0 → score=1.0 → IER=100 | z=+1 → score≈1.57 | z=−1 → score≈0.43
        </div>""", unsafe_allow_html=True)
    if not df_ier.empty:
        cats=df_ier['CLASIFICACION'].value_counts()
        ic1,ic2,ic3,ic4=st.columns(4)
        ic1.metric('🟢 Eficiente',int(cats.get('🟢 Eficiente',0)),'IER ≥ 105')
        ic2.metric('🟡 Normal',int(cats.get('🟡 Normal',0)),'IER 95–105')
        ic3.metric('🟠 Atención',int(cats.get('🟠 Atención',0)),'IER 85–95')
        ic4.metric('🔴 Crítico',int(cats.get('🔴 Crítico',0)),'IER < 85')
        st.markdown('<br>', unsafe_allow_html=True)
        def ier_bar_color(v):
            if v>=105: return '#16a34a'
            elif v>=95: return '#d97706'
            elif v>=85: return '#ea580c'
            else: return '#dc2626'
        df_ier_sorted = df_ier.sort_values(['MODELO','IER'],ascending=[True,False])
        fig_ier=go.Figure()
        MODELO_COLOR={'S-Way':'#2563eb','Scania':'#ea580c','Stralis':'#7c3aed'}
        for modelo in MODELO_COLOR:
            subset=df_ier_sorted[df_ier_sorted['MODELO']==modelo]
            if subset.empty: continue
            hover=[]
            for _,row in subset.iterrows():
                tkml_txt=(f"Carga: {row['TONKML']:.1f} ton·km/L (prom: {row['TONKML_MOD']:.1f})" if row['TONKML']>0 else "Carga: sin datos")
                ral_txt = f"dato no confiable (S-Way)" if row['MODELO']=='S-Way' else f"{row['RALENTI_PCT']:.1f}%"
                severidad = row.get('SEVERIDAD', 0)
                hover.append(f"<b>{row['DOMINIO']}</b> ({row['MODELO']})<br>IER: <b>{row['IER']:.1f}</b> — {row['CLASIFICACION']}<br>"
                             f"L/100km: {row['L100KM']:.2f} (prom {row['MODELO']}: {row['L100KM_MOD']:.2f}) score: {row['SCORE_CONSUMO']:.2f}<br>"
                             f"% Ralentí: {ral_txt}  score: {row['SCORE_RAL']:.2f}<br>"
                             f"Vel. severidad: {severidad:.0f} km/h acum. · {int(row['EXCESOS'])} eventos · score: {row['SCORE_VEL']:.2f}<br>"
                             f"Vel. máx: {row['VEL_MAX']:.0f} km/h<br>KM total: {row['KM']:,.0f}  score: {row['SCORE_KM']:.2f}<br>"
                             f"{tkml_txt} (referencia, no entra en IER)")
            fig_ier.add_trace(go.Bar(y=subset['DOMINIO'],x=subset['IER'],name=modelo,orientation='h',
                marker=dict(color=[ier_bar_color(v) for v in subset['IER']],line=dict(color='rgba(15,23,42,0.15)',width=1)),
                text=[f"{v:.1f}" for v in subset['IER']],textposition='outside',textfont=dict(color=FONT_COL,size=10),
                hovertemplate='%{customdata}<extra></extra>',customdata=hover))
        ier_min=max(50,df_ier_sorted['IER'].min()-10); ier_max=min(175,df_ier_sorted['IER'].max()+25)
        fig_ier.add_vline(x=100,line_dash='solid',line_color='#d97706',line_width=2.5,
                          annotation_text='Base 100',annotation_position='top',annotation_font_color='#92400e',annotation_font_size=11)
        fig_ier.update_layout(paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font=dict(color=FONT_COL),barmode='overlay',
            xaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL),title=dict(text='IER  (100 = promedio de su modelo)',font=dict(color=AXIS_COL)),range=[ier_min,ier_max]),
            yaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL,size=10),categoryorder='array',categoryarray=df_ier_sorted['DOMINIO'].tolist()),
            height=max(380,len(df_ier_sorted)*44),margin=dict(l=10,r=130,t=60,b=30),showlegend=False)
        st.plotly_chart(fig_ier, use_container_width=True)
        st.caption('Verde = mejor que su modelo · Rojo = peor · Línea naranja = base 100 · Hover para detalle completo')
        with st.expander('Tabla detallada IER (todos los componentes)'):
            show_cols=['DOMINIO','MODELO','IER','CLASIFICACION','L100KM','L100KM_MOD','RALENTI_PCT','RAL_MOD',
                       'EXCESOS','SEVERIDAD','SEVERIDAD_MOD','VEL_MAX','KM',
                       'SCORE_CONSUMO','SCORE_KM','SCORE_VEL','SCORE_RAL']
            ier_show=df_ier[[c for c in show_cols if c in df_ier.columns]].copy()
            col_rename={'DOMINIO':'Patente','MODELO':'Modelo','IER':'IER','CLASIFICACION':'Clasificación',
                'L100KM':'L/100km','L100KM_MOD':'Prom L/100km',
                'RALENTI_PCT':'% Ralentí','RAL_MOD':'% Ral Prom mod.',
                'EXCESOS':f'Cant. Excesos >{LIMITE_VELOCIDAD}km/h',
                'SEVERIDAD':'Severidad total (km/h acum.)','SEVERIDAD_MOD':'Sev. total prom mod.',
                'VEL_MAX':'Vel. Máx (km/h)', 'KM':'KM total',
                'SCORE_CONSUMO':'S.Consumo (60%)','SCORE_KM':'S.KM (20%)','SCORE_VEL':'S.Vel (10%)','SCORE_RAL':'S.Ralentí (10%)'}
            ier_show=ier_show.rename(columns=col_rename)
            for c in ['IER','L/100km','Prom L/100km','Severidad total (km/h acum.)','Sev. total prom mod.']:
                if c in ier_show.columns: ier_show[c]=ier_show[c].round(1)
            for c in ['S.Consumo (60%)','S.KM (20%)','S.Vel (10%)','S.Ralentí (10%)']:
                if c in ier_show.columns: ier_show[c]=ier_show[c].round(3)
            if 'KM total' in ier_show.columns: ier_show['KM total']=ier_show['KM total'].apply(lambda x:f'{x:,.0f}')
            st.dataframe(ier_show, use_container_width=True, hide_index=True)
    else:
        st.info('Sin datos suficientes para calcular el IER.')
    if not df_vel_filtrado.empty and 'DOMINIO' in df_vel_filtrado.columns:
        st.divider()
        st.markdown(f'<div class="sec-title">Ranking Severidad Velocidad >{LIMITE_VELOCIDAD} km/h — {anio_sel}</div>', unsafe_allow_html=True)
        st.caption(f'Métrica: suma total de km/h sobre el límite (frecuencia × magnitud). 5 eventos a 95 km/h (sum=35) es más grave que 10 eventos a 89 km/h (sum=10).')
        vel_rank=(df_vel_filtrado.groupby('DOMINIO').agg(
            CANTIDAD=('DOMINIO','count'),
            VEL_MAX=('VELOCIDAD','max'),
            VEL_PROM=('VELOCIDAD','mean'),
            SEVERIDAD=('EXCESO_KMH','sum')
        ).reset_index().sort_values('SEVERIDAD',ascending=False))
        vel_rank['MODELO']=vel_rank['DOMINIO'].apply(asignar_modelo)
        fig_vel=go.Figure([go.Bar(x=vel_rank['DOMINIO'],y=vel_rank['SEVERIDAD'].round(0),
            marker_color=['#dc2626' if v==vel_rank['SEVERIDAD'].max() else '#ea580c' for v in vel_rank['SEVERIDAD']],
            text=vel_rank['SEVERIDAD'].round(0).astype(int),textposition='outside',textfont=dict(color=FONT_COL,size=10),
            hovertemplate='<b>%{x}</b><br>Severidad total: +%{y:.0f} km/h acumulados sobre límite<extra></extra>')])
        fig_vel.update_layout(paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font=dict(color=FONT_COL),
            xaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL,size=10),tickangle=-45),
            yaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL),title=dict(text=f'km/h acumulados sobre {LIMITE_VELOCIDAD} km/h',font=dict(color=AXIS_COL))),
            height=380,margin=dict(l=10,r=10,t=20,b=80),showlegend=False)
        st.plotly_chart(fig_vel, use_container_width=True)
        with st.expander('Tabla de excesos por unidad'):
            vel_show=vel_rank[['DOMINIO','MODELO','SEVERIDAD','CANTIDAD','VEL_MAX','VEL_PROM']].copy()
            vel_show.columns=['Patente','Modelo',f'Severidad total (km/h acum.)',f'Cant. eventos >{LIMITE_VELOCIDAD}km/h','Vel. Máx (km/h)','Vel. Prom (km/h)']
            vel_show['Vel. Máx (km/h)']=vel_show['Vel. Máx (km/h)'].round(1)
            vel_show['Vel. Prom (km/h)']=vel_show['Vel. Prom (km/h)'].round(1)
            vel_show['Severidad total (km/h acum.)']=vel_show['Severidad total (km/h acum.)'].round(1)
            st.dataframe(vel_show, use_container_width=True, hide_index=True)
    st.divider()
    with st.expander(f'Datos completos {anio_sel}'):
        cols_s=[c for c in ['DOMINIO','MARCA','MODELO','FECHA','KM','LITROS','L100KM','RALENTI'] if c in df.columns]
        st.dataframe(df[cols_s], use_container_width=True, height=380)
    st.caption(f'Datos {anio_sel}: Google Sheets Expreso Diemar | Precio: {precio_fuente} | Excesos: satelital >{LIMITE_VELOCIDAD} km/h | Actualización cada 10 min')
# ═══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 2 — MODELO PREDICTIVO
# ═══════════════════════════════════════════════════════════════════════════════
elif pg == "Modelo Predictivo":
    col_logo2,col_title2=st.columns([1,5])
    with col_logo2: st.image(LOGO_URL, width=130)
    with col_title2:
        st.markdown("""<div class='dm-header'>
        <div class='dm-header-title'>Modelo Predictivo — LAD</div>
        <div class='dm-header-sub'>Entrenado con todo el histórico · Regresión polinomial · Simulador What-If</div>
        </div>""", unsafe_allow_html=True)
    anos_en_hist=(sorted(df_full_clean['FECHA'].dt.year.unique().tolist()) if 'FECHA' in df_full_clean.columns else [])
    anos_str=" · ".join(str(a) for a in anos_en_hist)
    st.markdown(f'<div class="training-badge">Modelo entrenado con {n_meses_entrenamiento} meses históricos ({anos_str})</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    hist=meses_hist_full.copy(); hist['T']=range(len(hist))
    if len(hist)>=2:
        X=hist['T'].values.reshape(-1,1); y_l100=hist['L100'].values; y_lts=hist['LITROS'].values
        degree=min(2,len(hist)-1)
        poly=PolynomialFeatures(degree=degree); Xp=poly.fit_transform(X)
        model_l100=LinearRegression().fit(Xp,y_l100); model_lts=LinearRegression().fit(Xp,y_lts)
        r2_l100=model_l100.score(Xp,y_l100)
        residuals=y_l100-model_l100.predict(Xp); std_res=np.std(residuals)
        t_max=hist['T'].max(); ultimo=hist['MES_PERIODO'].iloc[-1]
        n_pred=max(3,12-ultimo.month)
        t_fut=np.array(range(t_max+1,t_max+1+n_pred)).reshape(-1,1)
        Xf=poly.transform(t_fut)
        pred_l100=np.clip(model_l100.predict(Xf),0,100); pred_lts=np.clip(model_lts.predict(Xf),0,None)
        meses_fut=[(ultimo+i+1).strftime('%b %Y') for i in range(n_pred)]
        st.markdown(f'<div class="sec-title">Predicción meses restantes {ultimo.year} ({n_pred} meses)</div>', unsafe_allow_html=True)
        def render_pred_card(mes, l100_p, lts_p, precio_gasoil):
            costo_p = lts_p * precio_gasoil
            return (
                f'<div class="kpi-card" style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:14px 10px;text-align:center;border-left:4px solid #ea580c;">'
                f'<div style="font-size:.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;">{mes}</div>'
                f'<div style="font-size:1.55rem;font-weight:700;color:#ea580c;line-height:1.1;">{l100_p:.2f}</div>'
                f'<div style="font-size:.78rem;color:#64748b;margin-bottom:6px;">L/100km</div>'
                f'<div style="border-top:1px solid #e2e8f0;padding-top:6px;margin-top:2px;">'
                f'<div style="font-size:.82rem;color:#1e293b;font-weight:600;">{lts_p:,.0f} L</div>'
                f'<div style="font-size:.75rem;color:#94a3b8;">${costo_p/1e6:.2f}M est.</div>'
                f'</div></div>'
            )
        n_kpi=min(4,n_pred); kpi_cols=st.columns(n_kpi)
        for c,mes,l100_p,lts_p in zip(kpi_cols,meses_fut[:n_kpi],pred_l100[:n_kpi],pred_lts[:n_kpi]):
            with c: st.markdown(render_pred_card(mes,l100_p,lts_p,precio_gasoil), unsafe_allow_html=True)
        if n_pred>4:
            kpi_cols2=st.columns(min(4,n_pred-4))
            for c,mes,l100_p,lts_p in zip(kpi_cols2,meses_fut[4:],pred_l100[4:],pred_lts[4:]):
                with c: st.markdown(render_pred_card(mes,l100_p,lts_p,precio_gasoil), unsafe_allow_html=True)
        st.divider()
        st.markdown('<div class="sec-title">Evolución histórica completa con Proyección</div>', unsafe_allow_html=True)
        all_labels=[str(p) for p in hist['MES_PERIODO']]+meses_fut
        all_hist=hist['L100'].tolist()+[None]*n_pred
        all_pred=[None]*(len(hist)-1)+[float(hist['L100'].iloc[-1])]+[float(v) for v in pred_l100]
        upper_vals=([None]*(len(hist)-1)+[float(hist['L100'].iloc[-1])+1.5*std_res]+[float(v)+1.5*std_res for v in pred_l100])
        lower_vals=([None]*(len(hist)-1)+[float(hist['L100'].iloc[-1])-1.5*std_res]+[float(v)-1.5*std_res for v in pred_l100])
        pred_start=len(hist)-1; pred_labels=all_labels[pred_start:]
        upper_clean=[upper_vals[i] for i in range(pred_start,len(all_labels))]
        lower_clean=[lower_vals[i] for i in range(pred_start,len(all_labels))]
        unique_years=sorted(set(p.year for p in hist['MES_PERIODO']))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=pred_labels+pred_labels[::-1],y=upper_clean+lower_clean[::-1],fill='toself',fillcolor='rgba(37,99,235,0.12)',line=dict(color='rgba(0,0,0,0)'),name='Intervalo ±1.5σ',hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=pred_labels,y=upper_clean,mode='lines',line=dict(color='#2563eb',width=1,dash='dot'),name='CI sup',hovertemplate='CI sup: %{y:.2f} L/100km<extra></extra>'))
        fig.add_trace(go.Scatter(x=pred_labels,y=lower_clean,mode='lines',line=dict(color='#2563eb',width=1,dash='dot'),name='CI inf',hovertemplate='CI inf: %{y:.2f} L/100km<extra></extra>'))
        hist_x=[all_labels[i] for i,v in enumerate(all_hist) if v is not None]; hist_y=[v for v in all_hist if v is not None]
        fig.add_trace(go.Scatter(x=hist_x,y=hist_y,mode='lines+markers',line=dict(color='#dc2626',width=2.5),marker=dict(size=7,color='#dc2626',line=dict(color='#fff',width=1.5)),name='Histórico',hovertemplate='%{x}<br>Real: <b>%{y:.2f} L/100km</b><extra></extra>'))
        pred_x=[all_labels[i] for i,v in enumerate(all_pred) if v is not None]; pred_y=[v for v in all_pred if v is not None]
        fig.add_trace(go.Scatter(x=pred_x,y=pred_y,mode='lines+markers',line=dict(color='#d97706',width=4,dash='dash'),marker=dict(size=11,color='#d97706',symbol='diamond',line=dict(color='#ffffff',width=1.5)),name='Predicción',hovertemplate='%{x}<br>Pred: <b>%{y:.2f} L/100km</b><extra></extra>'))
        for yr in unique_years[1:]:
            yr_label=f'Ene {yr}'
            if yr_label in all_labels:
                fig.add_vline(x=yr_label,line_width=1,line_dash='dot',line_color='#cbd5e1',annotation_text=str(yr),annotation_position='top',annotation_font_color='#64748b',annotation_font_size=10)
        fig.update_layout(paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font=dict(color=FONT_COL),
            legend=dict(bgcolor=LEGEND_BG,bordercolor='#cbd5e1',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
            xaxis=dict(gridcolor=GRID_COL,linecolor='#cbd5e1',tickfont=dict(color=AXIS_COL,size=10),title=dict(text='Período',font=dict(color=AXIS_COL)),tickangle=-45),
            yaxis=dict(gridcolor=GRID_COL,linecolor='#cbd5e1',tickfont=dict(color=AXIS_COL,size=11),title=dict(text='L/100km',font=dict(color=AXIS_COL))),
            height=450,margin=dict(l=10,r=10,t=50,b=60),hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f'±1.5σ intervalo de confianza | Línea roja = histórico ({n_meses_entrenamiento} meses) | Línea naranja = predicción')

        # ─────────────────────────────────────────────────────────────────────
        #  ALERTA DE DESVÍO — Selectbox para elegir mes
        # ─────────────────────────────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="sec-title">Alerta de Desvío — Predicción vs. Real</div>', unsafe_allow_html=True)
        if len(hist)>=3:
            # Lista de meses disponibles para comparar (necesitan al menos 2 meses previos para entrenar)
            meses_comparables = hist['MES_PERIODO'].iloc[2:].tolist()
            meses_comp_str = [str(m) for m in meses_comparables]

            col_sel, col_info = st.columns([2,3])
            with col_sel:
                mes_elegido_str = st.selectbox(
                    'Mes a comparar contra predicción',
                    options=meses_comp_str,
                    index=len(meses_comp_str)-1,
                    help='Elegí el mes real que querés comparar contra la predicción generada con los datos previos a ese mes'
                )
            with col_info:
                st.caption(f'El modelo se reentrena con los meses anteriores al elegido y predice qué consumo esperaba para ese mes. Luego compara contra el valor real.')

            # Encontrar índice del mes elegido en hist
            mes_elegido = pd.Period(mes_elegido_str, 'M')
            idx_elegido = hist[hist['MES_PERIODO']==mes_elegido].index[0]
            valor_real = float(hist.loc[idx_elegido,'L100'])

            # Reentrenar con meses ANTERIORES al elegido
            hist_prev=hist.iloc[:idx_elegido].copy(); hist_prev['T']=range(len(hist_prev))
            if len(hist_prev)>=2:
                degree_prev=min(2,len(hist_prev)-1)
                poly_prev=PolynomialFeatures(degree=degree_prev)
                Xprev=poly_prev.fit_transform(hist_prev['T'].values.reshape(-1,1))
                m_prev=LinearRegression().fit(Xprev,hist_prev['L100'].values)
                X_pred_prev=poly_prev.transform(np.array([[len(hist_prev)]]).reshape(-1,1))
                pred_mes=float(np.clip(m_prev.predict(X_pred_prev),0,100)[0])
                residuals_prev = hist_prev['L100'].values - m_prev.predict(Xprev)
                std_res_prev = np.std(residuals_prev) if len(residuals_prev)>1 else std_res

                desvio = valor_real - pred_mes
                desvio_pct = (desvio/pred_mes*100) if pred_mes>0 else 0
                umbral = 1.5*std_res_prev

                if abs(desvio)>umbral:
                    dir_txt='SUPERIOR' if desvio>0 else 'INFERIOR'
                    st.markdown(f'<div class="alert-box"><b>DESVÍO DETECTADO — {mes_elegido_str}</b><br>Consumo real: <b>{valor_real:.2f} L/100km</b> &nbsp;|&nbsp; Predicción: <b>{pred_mes:.2f} L/100km</b><br>Desvío: <b>{desvio:+.2f} L/100km ({desvio_pct:+.1f}%)</b> — {dir_txt} al intervalo esperado (±{umbral:.2f})</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-ok"><b>Sin desvío — {mes_elegido_str}</b><br>Consumo real: <b>{valor_real:.2f} L/100km</b> &nbsp;|&nbsp; Predicción: <b>{pred_mes:.2f} L/100km</b><br>Desvío: <b>{desvio:+.2f} L/100km ({desvio_pct:+.1f}%)</b> — dentro del intervalo esperado (±{umbral:.2f})</div>', unsafe_allow_html=True)
            else:
                st.info('No hay suficientes meses previos para reentrenar el modelo.')

        st.divider()
        st.markdown('<div class="sec-title">Simulador What-If</div>', unsafe_allow_html=True)
        delta_precio_pct=st.slider('Variación precio combustible (%)',min_value=-30,max_value=50,value=0,step=1)
        precio_sim=precio_gasoil*(1+delta_precio_pct/100)
        wf1,wf2=st.columns(2)
        with wf1:
            st.markdown(f'<div class="kpi-card kpi-amber"><div class="kpi-label">Precio Simulado</div><div class="kpi-value">${precio_sim:,.0f}/L</div><div class="kpi-sub">{delta_precio_pct:+.1f}% vs hoy</div></div>', unsafe_allow_html=True)
        with wf2:
            costo_sim_m1=pred_lts[0]*precio_sim/1e6; costo_base_m1=pred_lts[0]*precio_gasoil/1e6; diff_costo=costo_sim_m1-costo_base_m1
            color_wf2='kpi-red' if diff_costo>0 else 'kpi-green'
            st.markdown(f'<div class="kpi-card {color_wf2}"><div class="kpi-label">Costo {meses_fut[0]}</div><div class="kpi-value">${costo_sim_m1:.2f}M</div><div class="kpi-sub">{diff_costo:+.2f}M vs base</div></div>', unsafe_allow_html=True)
        cost_df=pd.DataFrame({'Mes':meses_fut,'L/100km pred.':[round(v,2) for v in pred_l100],'Litros est.':[round(v,0) for v in pred_lts],
            'Costo base M$':[round(v*precio_gasoil/1e6,2) for v in pred_lts],'Costo simulado M$':[round(v*precio_sim/1e6,2) for v in pred_lts],
            'Dif. M$':[round(v*(precio_sim-precio_gasoil)/1e6,2) for v in pred_lts]})
        st.dataframe(cost_df, use_container_width=True, hide_index=True)
    else:
        st.info('Se necesitan al menos 2 meses de datos históricos para el modelo predictivo.')
    st.caption(f'Modelo entrenado con {n_meses_entrenamiento} meses | Precio: {precio_fuente} | Actualización cada 10 min')
# ═══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 3 — ANÁLISIS POR PATENTE
# ═══════════════════════════════════════════════════════════════════════════════
elif pg == "Análisis por Patente":
    col_logo3,col_title3=st.columns([1,5])
    with col_logo3: st.image(LOGO_URL, width=130)
    with col_title3:
        st.markdown(f"""<div class='dm-header'>
        <div class='dm-header-title'>Análisis por Patente — {anio_sel}</div>
        <div class='dm-header-sub'>Consumo · IER v6 · Excesos velocidad · Promedios</div>
        </div>""", unsafe_allow_html=True)
    if df.empty or 'DOMINIO' not in df.columns: st.warning('Sin datos disponibles.'); st.stop()
    resumen=df.groupby('DOMINIO').agg(LITROS_TOTAL=('LITROS','sum'),KM_TOTAL=('KM','sum'),MESES=('MES_PERIODO','nunique')).reset_index()
    resumen['L100KM_PROM']=(resumen['LITROS_TOTAL']/resumen['KM_TOTAL'].replace(0,np.nan)*100).round(2)
    resumen['LITROS_PROM_MES']=(resumen['LITROS_TOTAL']/resumen['MESES'].replace(0,np.nan)).round(0)
    resumen=resumen[resumen['KM_TOTAL']>0].sort_values('L100KM_PROM',ascending=False)
    resumen['MODELO']=resumen['DOMINIO'].apply(asignar_modelo)
    if not df_ier.empty:
        resumen=resumen.merge(df_ier[['DOMINIO','IER','CLASIFICACION','EXCESOS','VEL_MAX']],on='DOMINIO',how='left')
    else:
        resumen['IER']='—'; resumen['CLASIFICACION']='—'; resumen['EXCESOS']=0; resumen['VEL_MAX']=0
    if resumen.empty: st.warning('Sin datos suficientes.'); st.stop()
    patente_max=resumen.iloc[0]; patente_min=resumen.iloc[-1]
    st.markdown(f'<div class="sec-title">Destacados {anio_sel}</div>', unsafe_allow_html=True)
    hc1,hc2=st.columns(2)
    with hc1:
        st.markdown(f'<div class="highlight-max"><b>Mayor consumo — {patente_max["DOMINIO"]}</b> <span style="color:#64748b;font-size:.8rem;">({patente_max["MODELO"]})</span><br>Promedio: <b>{patente_max["L100KM_PROM"]:.2f} L/100km</b> &nbsp;|&nbsp; Total: <b>{patente_max["LITROS_TOTAL"]:,.0f} L</b> &nbsp;|&nbsp; {patente_max["KM_TOTAL"]:,.0f} km &nbsp;|&nbsp; {int(patente_max["MESES"])} meses activa</div>', unsafe_allow_html=True)
    with hc2:
        st.markdown(f'<div class="highlight-min"><b>Menor consumo — {patente_min["DOMINIO"]}</b> <span style="color:#64748b;font-size:.8rem;">({patente_min["MODELO"]})</span><br>Promedio: <b>{patente_min["L100KM_PROM"]:.2f} L/100km</b> &nbsp;|&nbsp; Total: <b>{patente_min["LITROS_TOTAL"]:,.0f} L</b> &nbsp;|&nbsp; {patente_min["KM_TOTAL"]:,.0f} km &nbsp;|&nbsp; {int(patente_min["MESES"])} meses activa</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown(f'<div class="sec-title">Promedio L/100km por Patente — {anio_sel}</div>', unsafe_allow_html=True)
    colors_bar=[('#dc2626' if r['DOMINIO']==patente_max['DOMINIO'] else ('#16a34a' if r['DOMINIO']==patente_min['DOMINIO'] else '#2563eb')) for _,r in resumen.iterrows()]
    fig_bar=go.Figure([go.Bar(x=resumen['DOMINIO'],y=resumen['L100KM_PROM'],marker_color=colors_bar,
        text=resumen['L100KM_PROM'].apply(lambda v:f'{v:.1f}'),textposition='outside',textfont=dict(color=FONT_COL,size=10),
        hovertemplate='<b>%{x}</b><br>L/100km: %{y:.2f}<extra></extra>')])
    promedio_flota=resumen['L100KM_PROM'].mean()
    fig_bar.add_hline(y=promedio_flota,line_dash='dot',line_color='#d97706',line_width=2,annotation_text=f'Promedio flota: {promedio_flota:.2f}',annotation_position='top right',annotation_font_color='#92400e',annotation_font_size=11)
    fig_bar.update_layout(paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font=dict(color=FONT_COL),
        xaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL,size=10),tickangle=-45),
        yaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL),title=dict(text='L/100km',font=dict(color=AXIS_COL))),
        height=420,margin=dict(l=10,r=10,t=30,b=80),showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption('Rojo = mayor consumo · Verde = menor consumo · Azul = resto · Línea naranja = promedio flota')
    st.divider()
    st.markdown(f'<div class="sec-title">Consumo Mensual por Patente (L/100km) — {anio_sel}</div>', unsafe_allow_html=True)
    if 'MES_PERIODO' in df.columns:
        pivot=df[df['L100KM']>0].groupby(['DOMINIO','MES_PERIODO'])['L100KM'].mean().round(2).reset_index()
        pivot['MES_STR']=pivot['MES_PERIODO'].astype(str)
        pivot_wide=pivot.pivot(index='DOMINIO',columns='MES_STR',values='L100KM')
        pivot_wide=pivot_wide.reindex(index=resumen['DOMINIO'].tolist()).dropna(how='all')
        if not pivot_wide.empty:
            z_vals=pivot_wide.values.tolist(); x_vals=list(pivot_wide.columns); y_vals=list(pivot_wide.index)
            text_vals=[]
            for row_data in z_vals:
                row_text=[]
                for v in row_data:
                    try: row_text.append(f'{float(v):.1f}' if v is not None and not np.isnan(float(v)) else '')
                    except: row_text.append('')
                text_vals.append(row_text)
            fig_heat=go.Figure(go.Heatmap(z=z_vals,x=x_vals,y=y_vals,text=text_vals,texttemplate='%{text}',
                colorscale=[[0.0,'#dcfce7'],[0.35,'#16a34a'],[0.65,'#d97706'],[1.0,'#dc2626']],
                colorbar=dict(title=dict(text='L/100km',font=dict(color=AXIS_COL)),tickfont=dict(color=AXIS_COL),bgcolor='rgba(0,0,0,0)'),
                hovertemplate='Patente: <b>%{y}</b><br>Mes: %{x}<br>L/100km: <b>%{z:.2f}</b><extra></extra>'))
            fig_heat.update_layout(paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font=dict(color=FONT_COL),
                xaxis=dict(tickfont=dict(color=AXIS_COL,size=10),tickangle=-45,side='bottom'),
                yaxis=dict(tickfont=dict(color=AXIS_COL,size=10)),height=max(300,len(y_vals)*40),margin=dict(l=10,r=10,t=20,b=60))
            st.plotly_chart(fig_heat, use_container_width=True)
    st.divider()
    st.markdown('<div class="sec-title">Detalle Individual por Patente</div>', unsafe_allow_html=True)
    pat_sel=st.selectbox('Seleccioná una patente para ver su evolución',resumen['DOMINIO'].tolist())
    if pat_sel:
        df_pat=df[df['DOMINIO']==pat_sel].copy()
        if 'MES_PERIODO' in df_pat.columns:
            df_pat_mes=df_pat.groupby('MES_PERIODO').agg(LITROS=('LITROS','sum'),KM=('KM','sum')).reset_index().sort_values('MES_PERIODO')
            df_pat_mes['L100']=(df_pat_mes['LITROS']/df_pat_mes['KM'].replace(0,np.nan)*100).round(2)
            df_pat_mes['MES_STR']=df_pat_mes['MES_PERIODO'].astype(str)
            l100_prom_pat=df_pat_mes['L100'].mean(); lts_total_pat=df_pat_mes['LITROS'].sum(); kms_total_pat=df_pat_mes['KM'].sum()
            marca_pat=df_pat['MARCA'].iloc[0] if 'MARCA' in df_pat.columns else '—'
            modelo_pat=df_pat['MODELO'].iloc[0] if 'MODELO' in df_pat.columns else '—'
            pk1,pk2,pk3,pk4,pk5=st.columns(5)
            pk1.metric('Patente',pat_sel); pk2.metric('Marca',marca_pat); pk3.metric('Modelo',modelo_pat)
            pk4.metric('L/100km promedio',f'{l100_prom_pat:.2f}'); pk5.metric('Litros totales',f'{lts_total_pat:,.0f}')
            if not df_ier.empty and pat_sel in df_ier['DOMINIO'].values:
                ier_row=df_ier[df_ier['DOMINIO']==pat_sel].iloc[0]
                st.markdown('<div class="sec-title">Índice de Eficiencia Relativa (IER v6)</div>', unsafe_allow_html=True)
                ier_v=ier_row['IER']
                sc_color=('#16a34a' if ier_v>=105 else ('#d97706' if ier_v>=95 else ('#ea580c' if ier_v>=85 else '#dc2626')))
                ia1,ia2,ia3=st.columns([1,2,2])
                with ia1:
                    st.markdown(f'<div class="ier-gauge-wrap"><div class="kpi-label">IER v6</div><div class="ier-score-big" style="color:{sc_color};">{ier_v:.1f}</div><div class="ier-clasif">{ier_row["CLASIFICACION"]}</div><div style="font-size:.72rem;color:#94a3b8;margin-top:6px;">base 100 = prom. {modelo_pat}</div></div>', unsafe_allow_html=True)
                with ia2:
                    st.markdown('<div style="font-size:.8rem;color:#475569;font-weight:600;margin-bottom:6px;">Componentes del IER (60/20/10/10)</div>', unsafe_allow_html=True)
                    def comp_bar(label,score,peso):
                        pct=min(int(score*50),100); bc='#16a34a' if score>=1 else '#dc2626'
                        st.markdown(f'<div class="ier-comp-row"><div class="ier-comp-label">{label} <span style="color:#94a3b8;">({peso}%)</span></div><div class="ier-comp-bar-bg"><div class="ier-comp-bar" style="width:{pct}%;background:{bc}"></div></div><div class="ier-comp-val" style="color:{bc};">{score*100:.0f}</div></div>', unsafe_allow_html=True)
                    comp_bar('L/100km', ier_row['SCORE_CONSUMO'], 60)
                    comp_bar('KM totales', ier_row['SCORE_KM'], 20)
                    comp_bar('% Ralentí', ier_row['SCORE_RAL'], 10)
                    comp_bar(f'Severidad vel.', ier_row['SCORE_VEL'], 10)
                with ia3:
                    st.markdown(f'<div style="font-size:.8rem;color:#475569;font-weight:600;margin-bottom:6px;">Esta unidad vs. promedio {modelo_pat}</div>', unsafe_allow_html=True)
                    delta_l100=ier_row['L100KM']-ier_row['L100KM_MOD']
                    severidad_u = ier_row.get('SEVERIDAD', 0)
                    severidad_m = ier_row.get('SEVERIDAD_MOD', 0)
                    delta_sev = severidad_u - severidad_m
                    st.metric('L/100km',f"{ier_row['L100KM']:.2f}",f"{delta_l100:+.2f} vs prom. {modelo_pat} ({ier_row['L100KM_MOD']:.2f})",delta_color='inverse')
                    st.metric(f'Severidad vel. (km/h acum. sobre {LIMITE_VELOCIDAD})',f"{severidad_u:.0f}",f"{delta_sev:+.0f} vs prom. {modelo_pat} ({severidad_m:.0f})",delta_color='inverse')
                    st.metric(f'Eventos >{LIMITE_VELOCIDAD} km/h',f"{int(ier_row['EXCESOS'])} eventos",'ref. — la severidad usa km/h acumulados')
                    if ier_row.get('PESO_TON',0)>0:
                        delta_tkml=ier_row['TONKML']-ier_row['TONKML_MOD']
                        st.metric('ton·km/L',f"{ier_row['TONKML']:.1f}",f"{delta_tkml:+.1f} vs prom. {modelo_pat} ({ier_row['TONKML_MOD']:.1f})",delta_color='normal')
                    else:
                        st.metric('ton·km/L','sin datos','score neutral (1.0)')
            df_vel_pat=(df_vel_filtrado[df_vel_filtrado['DOMINIO']==pat_sel] if not df_vel_filtrado.empty else pd.DataFrame())
            if not df_vel_pat.empty:
                st.markdown(f'<div class="sec-title">Excesos de Velocidad >{LIMITE_VELOCIDAD} km/h — {pat_sel}</div>', unsafe_allow_html=True)
                severidad_pat = df_vel_pat['EXCESO_KMH'].sum() if 'EXCESO_KMH' in df_vel_pat.columns else 0
                exceso_prom_pat = df_vel_pat['EXCESO_KMH'].mean() if 'EXCESO_KMH' in df_vel_pat.columns else 0
                vp1,vp2,vp3,vp4=st.columns(4)
                vp1.metric('Eventos totales',len(df_vel_pat))
                vp2.metric('Severidad total',f"{severidad_pat:.0f} km/h acum.",f"promedio por evento: +{exceso_prom_pat:.1f} km/h")
                vp3.metric('Vel. máxima',f"{df_vel_pat['VELOCIDAD'].max():.0f} km/h",f"+{df_vel_pat['VELOCIDAD'].max()-LIMITE_VELOCIDAD:.0f} km/h sobre límite")
                vp4.metric('Vel. promedio en exceso',f"{df_vel_pat['VELOCIDAD'].mean():.1f} km/h")
            st.divider()
            fig_pat=go.Figure()
            fig_pat.add_trace(go.Bar(x=df_pat_mes['MES_STR'],y=df_pat_mes['LITROS'],name='Litros',marker_color='rgba(37,99,235,0.5)',yaxis='y2',hovertemplate='%{x}<br>Litros: <b>%{y:,.0f}</b><extra></extra>'))
            fig_pat.add_trace(go.Scatter(x=df_pat_mes['MES_STR'],y=df_pat_mes['L100'],name='L/100km',mode='lines+markers',line=dict(color='#dc2626',width=2.5),marker=dict(size=8,color='#dc2626',line=dict(color='#fff',width=1.5)),hovertemplate='%{x}<br>L/100km: <b>%{y:.2f}</b><extra></extra>'))
            fig_pat.add_hline(y=l100_prom_pat,line_dash='dot',line_color='#d97706',annotation_text=f'Prom: {l100_prom_pat:.2f}',annotation_font_color='#92400e')
            fig_pat.update_layout(paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font=dict(color=FONT_COL),
                xaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL,size=10),tickangle=-30),
                yaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL),title=dict(text='L/100km',font=dict(color='#dc2626'))),
                yaxis2=dict(overlaying='y',side='right',tickfont=dict(color='#2563eb'),title=dict(text='Litros',font=dict(color='#2563eb')),showgrid=False),
                legend=dict(bgcolor=LEGEND_BG,bordercolor='#cbd5e1',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
                height=370,margin=dict(l=10,r=50,t=40,b=50))
            st.plotly_chart(fig_pat, use_container_width=True)
            with st.expander(f'Tabla mensual — {pat_sel}'):
                df_show=df_pat_mes[['MES_STR','LITROS','KM','L100']].rename(columns={'MES_STR':'Mes','LITROS':'Litros','KM':'KM','L100':'L/100km'})
                df_show['Litros']=df_show['Litros'].apply(lambda x:f'{x:,.0f}')
                df_show['KM']=df_show['KM'].apply(lambda x:f'{x:,.0f}')
                st.dataframe(df_show, use_container_width=True, hide_index=True)
    st.divider()
    st.markdown(f'<div class="sec-title">Tabla Resumen — Todas las Patentes {anio_sel}</div>', unsafe_allow_html=True)
    cols_show=['DOMINIO','MODELO','LITROS_TOTAL','KM_TOTAL','L100KM_PROM','LITROS_PROM_MES','MESES']
    col_names=['Patente','Modelo','Litros Total','KM Total','L/100km Prom','Litros/Mes Prom','Meses Activa']
    for c,n in [('IER','IER'),('CLASIFICACION','Clasificación IER'),(f'EXCESOS',f'Excesos >{LIMITE_VELOCIDAD}km/h'),('VEL_MAX','Vel. Máx (km/h)')]:
        if c in resumen.columns: cols_show.append(c); col_names.append(n)
    resumen_show=resumen[cols_show].copy(); resumen_show.columns=col_names
    resumen_show['Litros Total']=resumen_show['Litros Total'].apply(lambda x:f'{x:,.0f}')
    resumen_show['KM Total']=resumen_show['KM Total'].apply(lambda x:f'{x:,.0f}')
    resumen_show['Litros/Mes Prom']=resumen_show['Litros/Mes Prom'].apply(lambda x:f'{x:,.0f}')
    st.dataframe(resumen_show, use_container_width=True, hide_index=True)
    st.caption(f'Datos {anio_sel} · Google Sheets Expreso Diemar · Actualización cada 10 min')
# ═══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 4 — DATOS OPERATIVOS
# ═══════════════════════════════════════════════════════════════════════════════
elif pg == "Datos Operativos":
    col_logo4,col_title4=st.columns([1,5])
    with col_logo4: st.image(LOGO_URL, width=130)
    with col_title4:
        st.markdown(f"""<div class='dm-header'>
        <div class='dm-header-title'>Datos Operativos — {anio_sel}</div>
        <div class='dm-header-sub'>Peso entregado por patente · Ton·km/L · Productividad de carga</div>
        </div>""", unsafe_allow_html=True)
    if df_carga_raw is None or df_carga_raw.empty:
        st.warning('No hay datos de carga disponibles. Verificá la conexión al sistema BI (reporte_hojas.xlsx).')
        st.stop()
    _patentes_ld = df['DOMINIO'].dropna().unique()
    df_carga_anio=df_carga_raw[
        (df_carga_raw['MES'].apply(lambda p:p.year)==anio_sel) &
        (df_carga_raw['DOMINIO'].isin(_patentes_ld))
    ].copy() if not df_carga_raw.empty else pd.DataFrame()
    if df_carga_anio.empty: st.warning(f'Sin datos de carga para {anio_sel}.'); st.stop()
    df_carga_anio['MES_STR']=df_carga_anio['MES'].astype(str)
    df_carga_anio['MODELO']=df_carga_anio['DOMINIO'].apply(asignar_modelo)
    st.markdown(f'<div class="sec-title">Resumen de Carga — {anio_sel}</div>', unsafe_allow_html=True)
    peso_total=df_carga_anio['PESO_TON'].sum(); n_pat_con_carga=df_carga_anio['DOMINIO'].nunique()
    peso_prom_pat=peso_total/n_pat_con_carga if n_pat_con_carga>0 else 0; meses_con_carga=df_carga_anio['MES'].nunique()
    def kpi2(cont,color,label,value,sub=''):
        cont.markdown(f'<div class="kpi-card {color}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)
    ck1,ck2,ck3,ck4=st.columns(4)
    kpi2(ck1,'kpi-purple','Peso Total Entregado',f'{peso_total:,.1f}',f'toneladas {anio_sel}')
    kpi2(ck2,'','Patentes con Carga',f'{n_pat_con_carga}',f'de {df["DOMINIO"].nunique()} activas')
    kpi2(ck3,'kpi-green','Prom. por Patente',f'{peso_prom_pat:,.1f}','toneladas anuales')
    kpi2(ck4,'kpi-amber','Meses con datos',f'{meses_con_carga}',f'de {anio_sel}')
    st.divider()
    st.markdown(f'<div class="sec-title">Diagnóstico de Carga — Matriz L/100km vs kg/km — {anio_sel}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ier-info-box">
    <b>¿Cómo leer esta matriz?</b>
    Cada punto es una patente. Los ejes separan consumo (L/100km) y densidad de carga (kg transportados por km recorrido).<br>
    La línea divisoria es la <b>mediana de la flota</b> en cada eje.<br>
    <b>% viajes sin peso:</b> viajes finalizados en el sistema BI con Peso Entregado = 0 — proxy de retornos en vacío o sin registro.
    </div>""", unsafe_allow_html=True)
    _km_pat   = df[df['KM']>0].groupby('DOMINIO')['KM'].sum().reset_index()
    _l100_pat = df[df['L100KM']>0].groupby('DOMINIO')['L100KM'].mean().reset_index()
    _tons_pat  = df_carga_anio.groupby('DOMINIO')['PESO_TON'].sum().reset_index()
    _mat = _km_pat.merge(_l100_pat, on='DOMINIO').merge(_tons_pat, on='DOMINIO', how='inner')
    _mat['KG_KM']  = (_mat['PESO_TON'] * 1000 / _mat['KM']).round(2)
    _mat['MODELO'] = _mat['DOMINIO'].apply(asignar_modelo)
    if not df_viajes_raw.empty:
        _vj = df_viajes_raw[
            (df_viajes_raw['MES'].apply(lambda p: p.year) == anio_sel) &
            (df_viajes_raw['DOMINIO'].isin(_patentes_ld))
        ]
        _vj_stats = _vj.groupby('DOMINIO').agg(
            N_TOTAL    =('CON_CARGA','count'),
            N_CARGADOS =('CON_CARGA','sum')
        ).reset_index()
        _vj_stats['N_VACIOS']   = _vj_stats['N_TOTAL'] - _vj_stats['N_CARGADOS']
        _vj_stats['PCT_VACIOS'] = (_vj_stats['N_VACIOS'] / _vj_stats['N_TOTAL'] * 100).round(1)
        _mat = _mat.merge(_vj_stats[['DOMINIO','N_TOTAL','N_CARGADOS','N_VACIOS','PCT_VACIOS']], on='DOMINIO', how='left')
    else:
        _mat['N_TOTAL'] = 0; _mat['N_VACIOS'] = 0; _mat['PCT_VACIOS'] = 0.0
    if not _mat.empty and len(_mat) >= 2:
        _l100_med = _mat['L100KM'].median()
        _kgkm_med = _mat['KG_KM'].median()
        def _cuadrante(row):
            bajo = row['L100KM'] <= _l100_med
            alto  = row['KG_KM'] >= _kgkm_med
            if   bajo and alto:  return '🟢 Ideal',         '#16a34a', 'Eficiente y bien cargado. Rendimiento óptimo.'
            elif bajo and not alto: return '🟡 Subutilizado','#d97706', 'Consumo eficiente pero baja densidad de carga. Revisar asignación de rutas o retornos en vacío.'
            elif not bajo and alto: return '🟠 Consumo alto','#ea580c', 'Bien cargado pero consume en exceso para el peso transportado. Revisar mecánica o conducción.'
            else:                   return '🔴 Crítico',     '#dc2626', 'Consumo alto y baja carga simultáneamente. Intervención urgente en mecánica y operación.'
        _mat[['CUAD_LABEL','CUAD_COLOR','CUAD_DESC']] = pd.DataFrame(
            _mat.apply(_cuadrante, axis=1).tolist(), index=_mat.index)
        _x_min = _mat['L100KM'].min() * 0.95; _x_max = _mat['L100KM'].max() * 1.05
        _y_min = _mat['KG_KM'].min()  * 0.90; _y_max = _mat['KG_KM'].max()  * 1.10
        fig_mat = go.Figure()
        _quad_cfg = [
            ([_x_min, _l100_med], [_kgkm_med, _y_max], 'rgba(22,163,74,0.10)',  '#16a34a', '🟢 IDEAL',
             'Eficiente + bien cargado', _x_min, _y_max, 'top left'),
            ([_l100_med, _x_max], [_kgkm_med, _y_max], 'rgba(234,88,12,0.10)', '#ea580c', '🟠 CONSUMO ALTO',
             'Carga OK · consumo excesivo', _x_max, _y_max, 'top right'),
            ([_x_min, _l100_med], [_y_min, _kgkm_med], 'rgba(217,119,6,0.10)', '#d97706', '🟡 SUBUTILIZADO',
             'Eficiente · poca carga/retornos vacíos', _x_min, _y_min, 'bottom left'),
            ([_l100_med, _x_max], [_y_min, _kgkm_med], 'rgba(220,38,38,0.10)',  '#dc2626', '🔴 CRÍTICO',
             'Consumo alto + baja carga', _x_max, _y_min, 'bottom right'),
        ]
        for _xr, _yr, _fc, _ec, _title, _sub, _ax, _ay, _apos in _quad_cfg:
            fig_mat.add_shape(type='rect', x0=_xr[0], x1=_xr[1], y0=_yr[0], y1=_yr[1],
                fillcolor=_fc, line=dict(color=_ec, width=0.5, dash='dot'), layer='below')
            _xanchor = 'left' if 'left' in _apos else 'right'
            _yanchor = 'top'  if 'top'  in _apos else 'bottom'
            _pad_x   = (_x_max - _x_min) * 0.015 * (1 if _xanchor=='left' else -1)
            _pad_y   = (_y_max - _y_min) * 0.025 * (-1 if _yanchor=='top' else 1)
            fig_mat.add_annotation(
                x=_ax + _pad_x, y=_ay + _pad_y,
                text=f'<b>{_title}</b><br><span style="font-size:9px;">{_sub}</span>',
                showarrow=False, xanchor=_xanchor, yanchor=_yanchor,
                font=dict(size=11, color=_ec),
                bgcolor='rgba(255,255,255,0.85)', borderpad=4
            )
        fig_mat.add_vline(x=_l100_med, line_dash='dash', line_color='#64748b', line_width=1.5)
        fig_mat.add_hline(y=_kgkm_med, line_dash='dash', line_color='#64748b', line_width=1.5)
        fig_mat.add_annotation(x=_l100_med, y=_y_max, text=f'mediana L/100km = {_l100_med:.1f}',
            showarrow=False, yanchor='bottom', font=dict(size=9, color='#64748b'),
            bgcolor='rgba(255,255,255,0.85)', borderpad=3)
        fig_mat.add_annotation(x=_x_min, y=_kgkm_med, text=f'mediana kg/km = {_kgkm_med:.0f}',
            showarrow=False, xanchor='left', font=dict(size=9, color='#64748b'),
            bgcolor='rgba(255,255,255,0.85)', borderpad=3)
        fig_mat.add_trace(go.Scatter(
            x=_mat['L100KM'], y=_mat['KG_KM'],
            mode='markers+text',
            text=_mat['DOMINIO'],
            textposition='top center',
            textfont=dict(size=10, color='#0f172a', family='monospace'),
            marker=dict(size=18, color=_mat['CUAD_COLOR'],
                        line=dict(color='white', width=2),
                        symbol='circle'),
            customdata=_mat[['CUAD_LABEL','CUAD_DESC','PCT_VACIOS','N_TOTAL','N_VACIOS','MODELO','KG_KM','PESO_TON']].values,
            hovertemplate=(
                '<b>%{text}</b>  <i>%{customdata[5]}</i><br>'
                '─────────────────────<br>'
                'L/100km: <b>%{x:.2f}</b>  (mediana flota: ' + f'{_l100_med:.1f})<br>' +
                'kg/km: <b>%{y:.1f}</b>  (mediana flota: ' + f'{_kgkm_med:.0f})<br>' +
                'Peso total: <b>%{customdata[7]:.1f} ton</b><br>'
                'Viajes sin peso: <b>%{customdata[4]:.0f} / %{customdata[3]:.0f} (%{customdata[2]:.1f}%)</b><br>'
                '─────────────────────<br>'
                '<b>%{customdata[0]}</b><br>'
                '<i>%{customdata[1]}</i><extra></extra>'
            ),
            showlegend=False
        ))
        fig_mat.update_layout(
            paper_bgcolor=PAPER_BG, plot_bgcolor='rgba(255,255,255,0.7)',
            font=dict(color=FONT_COL),
            xaxis=dict(
                gridcolor=GRID_COL, tickfont=dict(color=AXIS_COL, size=11),
                title=dict(
                    text='L / 100 km   —   litros consumidos por cada 100 km recorridos   (← menor = más eficiente)',
                    font=dict(color=AXIS_COL, size=11)
                ),
                range=[_x_min, _x_max]
            ),
            yaxis=dict(
                gridcolor=GRID_COL, tickfont=dict(color=AXIS_COL, size=11),
                title=dict(
                    text='kg / km   —   kg de carga entregados por km recorrido   (↑ mayor = más productivo)',
                    font=dict(color=AXIS_COL, size=11)
                ),
                range=[_y_min, _y_max]
            ),
            height=560, margin=dict(l=70, r=50, t=40, b=70)
        )
        st.plotly_chart(fig_mat, use_container_width=True)
        st.caption(f'Cada punto = una patente · Hover para diagnóstico completo · Líneas punteadas = mediana de la flota · L/100km: dato de telemetría · kg/km: carga del BI ÷ km telemetría')
        st.markdown('<div class="sec-title">Diagnóstico Individual por Patente</div>', unsafe_allow_html=True)
        _mat_sorted = _mat.sort_values('CUAD_COLOR', key=lambda s: s.map({'#dc2626':0,'#ea580c':1,'#d97706':2,'#16a34a':3}))
        _diag_cols = st.columns(min(4, len(_mat_sorted)))
        for _i, (_, _r) in enumerate(_mat_sorted.iterrows()):
            with _diag_cols[_i % len(_diag_cols)]:
                _pct_v = _r.get('PCT_VACIOS', 0)
                _n_tot = int(_r.get('N_TOTAL', 0))
                _n_vac = int(_r.get('N_VACIOS', 0))
                _vacios_txt = f"{_n_vac}/{_n_tot} viajes sin peso ({_pct_v:.0f}%)" if _n_tot > 0 else "sin datos de viajes"
                _bc = _r['CUAD_COLOR']
                st.markdown(f"""
                <div style="background:#ffffff;border-radius:10px;padding:16px;border-left:4px solid {_bc};margin-bottom:12px;box-shadow:0 1px 3px rgba(15,23,42,0.08);">
                  <div style="font-size:.95rem;font-weight:800;color:#0f172a;">{_r['DOMINIO']}</div>
                  <div style="font-size:.72rem;color:#94a3b8;margin-bottom:8px;">{_r['MODELO']}</div>
                  <div style="font-size:1.2rem;font-weight:700;color:{_bc};margin-bottom:6px;">{_r['CUAD_LABEL']}</div>
                  <div style="font-size:.75rem;color:#475569;line-height:1.5;">
                    L/100km: <b style="color:#0f172a;">{_r['L100KM']:.2f}</b><br>
                    kg/km: <b style="color:#0f172a;">{_r['KG_KM']:.1f}</b><br>
                    Viajes sin peso: <b style="color:#92400e;">{_vacios_txt}</b>
                  </div>
                  <div style="font-size:.72rem;color:#64748b;margin-top:8px;font-style:italic;">{_r['CUAD_DESC']}</div>
                </div>""", unsafe_allow_html=True)
        with st.expander('Tabla completa diagnóstico'):
            _tbl = _mat[['DOMINIO','MODELO','L100KM','KG_KM','PESO_TON','N_TOTAL','N_CARGADOS','N_VACIOS','PCT_VACIOS','CUAD_LABEL']].copy()
            _tbl.columns = ['Patente','Modelo','L/100km','kg/km','Peso total (ton)','Viajes total','Con carga','Sin carga','% sin carga','Cuadrante']
            _tbl['L/100km']         = _tbl['L/100km'].round(2)
            _tbl['kg/km']           = _tbl['kg/km'].round(1)
            _tbl['Peso total (ton)'] = _tbl['Peso total (ton)'].round(1)
            st.dataframe(_tbl.sort_values('% sin carga', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info('Sin datos suficientes para armar la matriz (se necesitan datos de telemetría y carga simultáneos).')
    st.divider()
    st.markdown(f'<div class="sec-title">Peso Entregado por Mes y Patente (toneladas) — {anio_sel}</div>', unsafe_allow_html=True)
    pivot_carga=(df_carga_anio.pivot_table(index='DOMINIO',columns='MES_STR',values='PESO_TON',aggfunc='sum',fill_value=0).reset_index())
    pivot_carga['TOTAL']=pivot_carga.drop(columns='DOMINIO').sum(axis=1)
    pivot_carga=pivot_carga.sort_values('TOTAL',ascending=False)
    meses_cols=[c for c in pivot_carga.columns if c not in ['DOMINIO','TOTAL']]
    if meses_cols:
        z_vals=pivot_carga[meses_cols].values.tolist(); y_vals=pivot_carga['DOMINIO'].tolist()
        txt_vals=[[f'{v:,.1f}' if v>0 else '' for v in row] for row in z_vals]
        fig_heat_c=go.Figure(go.Heatmap(z=z_vals,x=meses_cols,y=y_vals,text=txt_vals,texttemplate='%{text}',
            colorscale=[[0.0,'#f1f5f9'],[0.3,'#3b82f6'],[0.65,'#7c3aed'],[1.0,'#be185d']],
            colorbar=dict(title=dict(text='Ton',font=dict(color=AXIS_COL)),tickfont=dict(color=AXIS_COL),bgcolor='rgba(0,0,0,0)'),
            hovertemplate='Patente: <b>%{y}</b><br>Mes: %{x}<br>Peso: <b>%{z:,.1f} ton</b><extra></extra>'))
        fig_heat_c.update_layout(paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font=dict(color=FONT_COL),
            xaxis=dict(tickfont=dict(color=AXIS_COL,size=10),tickangle=-45,side='bottom'),
            yaxis=dict(tickfont=dict(color=AXIS_COL,size=10)),height=max(300,len(y_vals)*40),margin=dict(l=10,r=10,t=20,b=60))
        st.plotly_chart(fig_heat_c, use_container_width=True)
    with st.expander('Tabla completa de peso entregado (ton)'):
        pivot_show=pivot_carga.copy()
        for c in meses_cols+['TOTAL']: pivot_show[c]=pivot_show[c].apply(lambda x:f'{x:,.1f}' if x>0 else '—')
        pivot_show=pivot_show.rename(columns={'DOMINIO':'Patente','TOTAL':'TOTAL año'})
        st.dataframe(pivot_show, use_container_width=True, hide_index=True)
    st.markdown(f'<div class="sec-title">Evolución Mensual de Peso Entregado por Patente — {anio_sel}</div>', unsafe_allow_html=True)
    COLORES_PAT=['#2563eb','#ea580c','#16a34a','#7c3aed','#dc2626','#0891b2','#ca8a04','#65a30d']
    fig_bar_c=go.Figure()
    for i,(_,row) in enumerate(pivot_carga.iterrows()):
        dom=row['DOMINIO']; vals=[row.get(m,0) for m in meses_cols]
        fig_bar_c.add_trace(go.Bar(name=dom,x=meses_cols,y=vals,marker_color=COLORES_PAT[i%len(COLORES_PAT)],hovertemplate=f'<b>{dom}</b><br>%{{x}}: <b>%{{y:,.1f}} ton</b><extra></extra>'))
    fig_bar_c.update_layout(barmode='stack',paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font=dict(color=FONT_COL),
        xaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL,size=10),tickangle=-45),
        yaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL),title=dict(text='Toneladas entregadas',font=dict(color=AXIS_COL))),
        legend=dict(bgcolor=LEGEND_BG,bordercolor='#cbd5e1',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
        height=420,margin=dict(l=10,r=10,t=50,b=70))
    st.plotly_chart(fig_bar_c, use_container_width=True)
    st.divider()
    st.markdown(f'<div class="sec-title">Detalle ton·km/L (Productividad de Carga) — {anio_sel}</div>', unsafe_allow_html=True)
    st.markdown("""<div class="ier-info-box"><b>¿Qué es ton·km/L?</b> Mide cuántas toneladas·kilómetro se transportan por cada litro de combustible.<br><b>Fórmula:</b> ton·km/L = Peso entregado (ton) × KM recorridos / Litros consumidos</div>""", unsafe_allow_html=True)
    df_op=df[df['KM']>0].copy(); df_op['MES_STR']=df_op['FECHA'].dt.to_period('M').astype(str)
    km_lts_mes=df_op.groupby(['DOMINIO','MES_STR']).agg(KM=('KM','sum'),LITROS=('LITROS','sum')).reset_index()
    df_carga_str=df_carga_anio[['DOMINIO','MES_STR','PESO_TON']].copy()
    tonkml_mes=km_lts_mes.merge(df_carga_str,on=['DOMINIO','MES_STR'],how='left')
    tonkml_mes['PESO_TON']=tonkml_mes['PESO_TON'].fillna(0)
    tonkml_mes['TONKML']=np.where((tonkml_mes['PESO_TON']>0)&(tonkml_mes['LITROS']>0),(tonkml_mes['PESO_TON']*tonkml_mes['KM'])/tonkml_mes['LITROS'],np.nan).round(2)
    tonkml_mes['MODELO']=tonkml_mes['DOMINIO'].apply(asignar_modelo)
    tkml_valid=tonkml_mes['TONKML'].dropna()
    if not tkml_valid.empty:
        t1,t2,t3,t4=st.columns(4)
        tkml_prom=tkml_valid.mean(); tkml_max=tkml_valid.max(); tkml_min=tkml_valid.min()
        dom_max=tonkml_mes.loc[tonkml_mes['TONKML'].idxmax(),'DOMINIO']; dom_min=tonkml_mes.loc[tonkml_mes['TONKML'].idxmin(),'DOMINIO']
        kpi2(t1,'','Promedio ton·km/L',f'{tkml_prom:.2f}','toda la flota')
        kpi2(t2,'kpi-green',f'Mejor — {dom_max}',f'{tkml_max:.2f}','mayor productividad')
        kpi2(t3,'kpi-red',f'Peor — {dom_min}',f'{tkml_min:.2f}','menor productividad')
        kpi2(t4,'kpi-purple','Período cubierto',f'{tonkml_mes["MES_STR"].nunique()} meses',f'{anio_sel}')
    fig_tkml=go.Figure()
    for i,dom in enumerate(tonkml_mes['DOMINIO'].unique()):
        sub=tonkml_mes[tonkml_mes['DOMINIO']==dom].sort_values('MES_STR')
        sub_valid=sub[sub['TONKML'].notna()]
        if sub_valid.empty: continue
        fig_tkml.add_trace(go.Scatter(x=sub_valid['MES_STR'],y=sub_valid['TONKML'],name=dom,mode='lines+markers',line=dict(color=COLORES_PAT[i%len(COLORES_PAT)],width=2.5),marker=dict(size=8,line=dict(color='#fff',width=1.5)),hovertemplate=f'<b>{dom}</b><br>%{{x}}: <b>%{{y:.2f}} ton·km/L</b><extra></extra>'))
    if not tkml_valid.empty:
        fig_tkml.add_hline(y=tkml_valid.mean(),line_dash='dot',line_color='#d97706',line_width=2,annotation_text=f'Promedio: {tkml_valid.mean():.2f}',annotation_position='top right',annotation_font_color='#92400e',annotation_font_size=11)
    fig_tkml.update_layout(paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font=dict(color=FONT_COL),
        xaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL,size=10),tickangle=-30),
        yaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL),title=dict(text='ton·km/L',font=dict(color=AXIS_COL))),
        legend=dict(bgcolor=LEGEND_BG,bordercolor='#cbd5e1',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
        height=400,margin=dict(l=10,r=10,t=50,b=50))
    st.plotly_chart(fig_tkml, use_container_width=True)
    rank_tkml=tonkml_mes.groupby('DOMINIO').agg(PESO_TON=('PESO_TON','sum'),KM=('KM','sum'),LITROS=('LITROS','sum'),MODELO=('MODELO','first')).reset_index()
    rank_tkml['TONKML_ANUAL']=np.where((rank_tkml['PESO_TON']>0)&(rank_tkml['LITROS']>0),(rank_tkml['PESO_TON']*rank_tkml['KM'])/rank_tkml['LITROS'],np.nan).round(2)
    rank_tkml=rank_tkml[rank_tkml['TONKML_ANUAL'].notna()].sort_values('TONKML_ANUAL',ascending=True)
    if not rank_tkml.empty:
        fig_rank=go.Figure([go.Bar(y=rank_tkml['DOMINIO'],x=rank_tkml['TONKML_ANUAL'],orientation='h',
            marker_color=['#16a34a' if v==rank_tkml['TONKML_ANUAL'].max() else ('#dc2626' if v==rank_tkml['TONKML_ANUAL'].min() else '#2563eb') for v in rank_tkml['TONKML_ANUAL']],
            text=[f'{v:.2f}' for v in rank_tkml['TONKML_ANUAL']],textposition='outside',textfont=dict(color=FONT_COL,size=10),
            hovertemplate='<b>%{y}</b><br>ton·km/L: <b>%{x:.2f}</b><extra></extra>')])
        prom_r=rank_tkml['TONKML_ANUAL'].mean()
        fig_rank.add_vline(x=prom_r,line_dash='dot',line_color='#d97706',line_width=2,annotation_text=f'Prom: {prom_r:.2f}',annotation_position='top',annotation_font_color='#92400e',annotation_font_size=11)
        fig_rank.update_layout(paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font=dict(color=FONT_COL),
            xaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL),title=dict(text='ton·km/L acumulado año',font=dict(color=AXIS_COL))),
            yaxis=dict(gridcolor=GRID_COL,tickfont=dict(color=AXIS_COL,size=10)),
            height=max(300,len(rank_tkml)*50+80),margin=dict(l=10,r=120,t=30,b=30),showlegend=False)
        st.plotly_chart(fig_rank, use_container_width=True)
    st.caption(f'Fuente: reporte_hojas.xlsx (BI Expreso) · Telemetría Google Sheets · Año {anio_sel}')
