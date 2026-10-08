# ═══════════════════════════════════════════════════════════════════════════════
#  EXPRESO DIEMAR — Dashboard de Monitoreo de Flota v4
#  IER v8: consumo real vs. esperado (modelo, mes, carga, ruta) + conducta del chofer
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
LOGO_URL    = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/logo_diemar4.png"
IVECO_URL   = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/S-Way-6x2-1.webp"
SCANIA_URL  = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/2016p.png"
STRALIS_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/image.png"
SWAY_PATENTES   = ['AH522SI', 'AH861UB', 'AH938VO', 'AH842GQ']
SCANIA_PATENTES = ['AD247MQ', 'AE423IW']
LIMITE_VELOCIDAD = 88
BASE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR35NkYPtJrOrdYHLGUH7GIW93s5cPAqQ0zEk5fP1c3gvErwbUW7HJ2OeWBYaBVsYKVmCf0yhLvs6eG/pub?output=csv"
GID_TEL  = "0"
GID_UNID = "882343299"   # hoja DATOS UNIDADES (col D = Ralentí %)
GID_VEL  = "1563993963"
URL_TEL  = f"{BASE_URL}&gid={GID_TEL}"
URL_UNID = f"{BASE_URL}&gid={GID_UNID}"
# Hoja "EXCESOS DE VELOCIDAD" en spreadsheet propio: usar gviz (funciona con compartido por link)
VEL_SHEET_ID = "1u7cckay0IJ60bfoKk2OZo-TjCvTbH9O1wKxNFdSKDCQ"
URL_VEL  = f"https://docs.google.com/spreadsheets/d/{VEL_SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID_VEL}"
# export devuelve el texto tal cual se ve; gviz infiere un tipo por columna y deja
# vacías las celdas que no coinciden (ej. fechas pegadas como texto en un mes nuevo)
URL_VEL_EXPORT = f"https://docs.google.com/spreadsheets/d/{VEL_SHEET_ID}/export?format=csv&gid={GID_VEL}"
CARGA_URL = "http://bi.sistemaexpreso.com.ar/reporte_hojas.xlsx"
# ── DATOS MANEJO (Score Conducción) ───────────────────────────────────────
MANEJO_SHEET_ID = "1teVcN0ejyvGbjWwWOHTmZ8I-17xyGZ0d8hxJ7dwSKm0"
MANEJO_SHEETS   = [
    {"gid": "0",          "modelo": "Stralis"},
    {"gid": "738544003",  "modelo": "S-Way"},
    {"gid": "2022308308", "modelo": "Scania"},
]
# ── DATOS ARREGLOS / REPARACIONES (Gasto por patente) ──────────────────────
ARREGLOS_SHEET_ID = "1u7cckay0IJ60bfoKk2OZo-TjCvTbH9O1wKxNFdSKDCQ"
ARREGLOS_GID      = "33208473"
# ── GASTO COMBUSTIBLE ACTUAL (misma planilla, otra hoja) ──────────────────
GASTO_COMB_SHEET_ID = "1u7cckay0IJ60bfoKk2OZo-TjCvTbH9O1wKxNFdSKDCQ"
GASTO_COMB_GID      = "1071419143"
GASTO_COMB_TIPO     = "X10"   # filtro columna F (c tipo)
DARK_CSS = """
<style>
/* Tema oscuro base en .streamlit/config.toml; acá solo ajustes finos.
   Sin override global de color: los colores de etiquetas/valores se respetan. */
[data-testid="stAppViewContainer"] { background: #0f172a; }
section[data-testid="stMain"] { background: #0f172a; }
[data-testid="stHeader"] { background: rgba(15,23,42,0.85); backdrop-filter: blur(6px); }
[data-testid="stSidebar"] { background: #1e293b; border-right: 1px solid #334155; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: #0f172a; border-color: #334155;
}
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background: #1e293b !important; }
[data-baseweb="popover"] li { color: #e2e8f0 !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: #334155 !important; }
[data-testid="stSidebar"] label p { color: #cbd5e1; font-weight: 600; }
[data-testid="stCaptionContainer"], .stCaption { color: #94a3b8; }
[data-testid="stMetricValue"] { color: #f1f5f9; }
[data-testid="stMetricLabel"] p { color: #94a3b8; }
[data-testid="stExpander"] details { border-color: #334155; background: #172033; }
[data-testid="stDataFrame"] { border: 1px solid #334155; border-radius: 8px; }
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #334155; }
.kpi-card {
    background: #1e293b; border-radius: 14px; padding: 24px 28px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.4); text-align: center;
    border-left: 5px solid #2563eb; margin-bottom: 16px;
}
.kpi-label  { font-size:0.78rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:.5px; margin-bottom:4px; }
.kpi-value  { font-size:2rem; font-weight:800; color:#f1f5f9; line-height:1.1; }
.kpi-sub    { font-size:0.78rem; color:#94a3b8; margin-top:6px; }
.kpi-card   { border: 1px solid #334155; border-left: 5px solid #2563eb; }
.kpi-red    { border-left-color:#ef4444; }
.kpi-green  { border-left-color:#22c55e; }
.kpi-amber  { border-left-color:#f59e0b; }
.kpi-purple { border-left-color:#a855f7; }
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
.rank-row    { display:flex; align-items:center; padding:8px 0; border-bottom:1px solid #334155; }
.rank-num    { width:28px; font-weight:700; font-size:.9rem; color:#94a3b8; }
.rank-dom    { flex:1; font-size:.88rem; color:#e2e8f0; font-weight:600; }
.rank-val    { font-size:.88rem; font-weight:700; }
.rank-bar-bg { width:80px; height:6px; background:#334155; border-radius:3px; margin:0 10px; overflow:hidden; }
.rank-bar    { height:6px; border-radius:3px; }
.alert-box   { background:#450a0a; border:1px solid #ef4444; border-radius:10px; padding:14px 18px; margin:10px 0; }
.alert-ok    { background:#052e16; border:1px solid #22c55e; border-radius:10px; padding:14px 18px; margin:10px 0; }
.highlight-max { background:#450a0a; border:1px solid #ef4444; border-radius:10px; padding:14px 18px; margin:6px 0; }
.highlight-min { background:#052e16; border:1px solid #22c55e; border-radius:10px; padding:14px 18px; margin:6px 0; }
.training-badge {
    background:#1e1b4b; border:1px solid #6366f1; border-radius:8px;
    padding:6px 12px; display:inline-block; font-size:0.8rem; color:#a5b4fc; font-weight:600;
    margin-bottom: 12px;
}
.sidebar-filter-header {
    font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.5px;
    color:#94a3b8; margin-bottom:10px; padding:6px 0; border-bottom:1px solid #334155;
}
[data-testid="stSidebar"] [data-testid="stDateInput"] label,
[data-testid="stSidebar"] [data-testid="stMultiSelect"] label {
    font-size:.78rem !important; color:#94a3b8 !important; font-weight:600 !important;
}
.ier-info-box {
    background:#0f2744; border:1px solid #2563eb; border-radius:10px;
    padding:14px 18px; margin:10px 0; font-size:.85rem; color:#93c5fd; line-height:1.6;
}
.ier-method-box {
    background:#0d1f0d; border:1px solid #16a34a; border-radius:10px;
    padding:14px 18px; margin:10px 0; font-size:.82rem; color:#86efac; line-height:1.7;
}
.ier-gauge-wrap {
    background:#1e293b; border-radius:14px; padding:18px 22px;
    border-left:5px solid #6366f1; margin-bottom:12px; text-align:center;
}
.ier-score-big { font-size:2.4rem; font-weight:900; line-height:1; }
.ier-clasif    { font-size:.85rem; font-weight:700; margin-top:4px; }
.ier-comp-row  {
    display:flex; align-items:center; justify-content:space-between;
    background:#0f172a; border-radius:8px; padding:8px 14px; margin:4px 0;
    font-size:.82rem;
}
.ier-comp-label { color:#94a3b8; flex:1; }
.ier-comp-val   { font-weight:700; color:#e2e8f0; }
.ier-comp-bar-bg { width:90px; height:6px; background:#334155; border-radius:3px; margin:0 10px; overflow:hidden; }
.ier-comp-bar    { height:6px; border-radius:3px; }
.vel-badge {
    background:#2d1b00; border:1px solid #f97316; border-radius:6px;
    padding:3px 10px; display:inline-block; font-size:.78rem; color:#fb923c; font-weight:700;
}
.zscore-badge {
    background:#1e1b4b; border:1px solid #818cf8; border-radius:5px;
    padding:2px 8px; display:inline-block; font-size:.72rem; color:#a5b4fc; font-weight:600;
}
</style>
"""
pg = st.sidebar.radio(
    "Navegacion",
    ["Dashboard Principal", "Modelo Predictivo", "Análisis por Patente", "Datos Operativos", "🗺️ Mapa Excesos", "🔧 Diagnóstico"],
    index=0,
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.image(LOGO_URL, width=160)
def normalizar_patente(valor):
    """Normaliza dominios: upper + solo alfanuméricos.
       Elimina espacios normales, \\xa0, \\u200b y cualquier
       caracter invisible que sobreviva a un str.strip()/\\s+."""
    import re
    s = str(valor).strip().upper()
    return re.sub(r'[^A-Z0-9]', '', s)
def parse_fecha_mixta(serie):
    """Fechas día-primero con formatos mezclados en la misma columna
       ('1/02/2026 3:31:48', '01/09/2026 03:31', '2026-09-01 03:31:48',
       serial de Sheets 46266.14...). pd.to_datetime a secas toma el formato
       de la primera fila y descarta (NaT) las que vienen distinto."""
    s = serie.astype(str).str.strip().str.replace('\xa0', ' ', regex=False)
    s = s.replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NaT': np.nan})
    # ISO (año primero) no debe leerse con dayfirst
    es_iso = s.str.match(r'^\d{4}-\d{1,2}-\d{1,2}', na=False)
    try:
        out = pd.to_datetime(s.where(~es_iso), errors='coerce', dayfirst=True, format='mixed')
    except (TypeError, ValueError):
        out = s.where(~es_iso).apply(lambda v: pd.to_datetime(v, errors='coerce', dayfirst=True))
    out = out.where(~es_iso, pd.to_datetime(s.where(es_iso), errors='coerce'))
    # números de serie de Google Sheets / Excel (días desde 1899-12-30)
    num = pd.to_numeric(s.str.replace(',', '.', regex=False), errors='coerce')
    es_serial = out.isna() & num.between(30000, 80000)
    if es_serial.any():
        out = out.where(~es_serial, pd.Timestamp('1899-12-30') + pd.to_timedelta(num.where(es_serial), unit='D'))
    return pd.to_datetime(out, errors='coerce')
@st.cache_data(ttl=600)
def cargar_datos():
    try:
        df1 = pd.read_csv(URL_TEL)
        try:
            df2 = pd.read_csv(URL_UNID)
        except Exception:
            df2 = pd.DataFrame()
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
                elif "RALENT"    in c:                                  cm[c] = "RALENTI_PCT"
                elif "TIEMPO"    in c and "MOTOR"   in c:              cm[c] = "TIEMPO_MOTOR"
                elif "EMPRESA"   in c:                                  cm[c] = "EMPRESA"
            df = df.rename(columns=cm).loc[:, ~df.rename(columns=cm).columns.duplicated()]
            if "DOMINIO" in df.columns:
                df["DOMINIO"] = df["DOMINIO"].apply(normalizar_patente)
            if "RALENTI_PCT" in df.columns:
                # Porcentaje: nunca lleva separador de miles, así que '.' y ',' son
                # decimales. Puede venir como '36,8', '36.8', '36,8%' o fracción '0.368'.
                serie = df["RALENTI_PCT"]
                if isinstance(serie, pd.DataFrame):
                    serie = serie.iloc[:, 0]
                serie = (serie.astype(str).str.replace("%", "", regex=False).str.strip()
                              .str.replace(",", ".", regex=False))
                pct = pd.to_numeric(serie, errors="coerce")
                if pct.dropna().gt(0).any() and pct[pct > 0].max() <= 1:
                    pct = pct * 100
                df["RALENTI_PCT"] = pct.where(pct.between(0, 100)).fillna(0)
            for col in ["LITROS", "KM", "L100KM"]:
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
        if not df2.empty:
            df2 = limpiar(df2)
        if "L100KM" not in df1.columns and "LITROS" in df1.columns and "KM" in df1.columns:
            df1["L100KM"] = (df1["LITROS"] / df1["KM"].replace(0, np.nan) * 100).round(2)
        # ── RALENTÍ ────────────────────────────────────────────────────────────
        # La hoja DATOS UNIDADES (col D) trae el ralentí como PORCENTAJE sobre los
        # litros de ese mes. Se cruza por DOMINIO+MES con la telemetría y:
        #   · RALENTI_PCT  → porcentaje directo de la hoja (para los KPIs)
        #   · RALENTI (L)  → % × litros del período = litros gastados en ralentí
        if ('RALENTI_PCT' in df2.columns and 'DOMINIO' in df2.columns
                and 'FECHA' in df2.columns and 'FECHA' in df1.columns and 'LITROS' in df1.columns):
            df2_ral = df2[['DOMINIO', 'FECHA', 'RALENTI_PCT']].copy()
            df2_ral = df2_ral[df2_ral['RALENTI_PCT'] > 0]
            df2_ral['_MES'] = df2_ral['FECHA'].dt.to_period('M')
            # un % por DOMINIO+MES (si hubiera filas repetidas, promedio)
            df2_ral = (df2_ral.groupby(['DOMINIO', '_MES'], as_index=False)['RALENTI_PCT']
                              .mean())
            df1['_MES'] = df1['FECHA'].dt.to_period('M')
            df1 = df1.merge(
                df2_ral.rename(columns={'RALENTI_PCT': '_RAL_PCT_u'}),
                on=['DOMINIO', '_MES'], how='left'
            )
            if 'RALENTI_PCT' in df1.columns:
                df1['RALENTI_PCT'] = (df1['RALENTI_PCT'].where(df1['RALENTI_PCT'] > 0)
                                      .combine_first(df1['_RAL_PCT_u']).fillna(0))
            else:
                df1['RALENTI_PCT'] = df1['_RAL_PCT_u'].fillna(0)
            df1['RALENTI'] = (df1['RALENTI_PCT'] / 100.0) * df1['LITROS']
            df1.drop(columns=['_RAL_PCT_u', '_MES'], inplace=True, errors='ignore')
        else:
            df1['RALENTI_PCT'] = 0.0
            df1['RALENTI'] = 0.0
        if "EMPRESA" in df1.columns:
            df1 = df1[df1["EMPRESA"].str.upper().str.contains("LAD|DIEMAR", na=False)]
        return df1, df2
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame(), pd.DataFrame()
@st.cache_data(ttl=600)
def cargar_velocidad():
    """Lee hoja EXCESOS DE VELOCIDAD. Devuelve (df, diag).
       Estructura confirmada: A:Movil B:Fecha del evento C:Latitud D:Longitud
       E:Ubicacion F:Tipo de evento G:Gravedad H:Observacion I:velocidad."""
    import io
    diag = {"url": URL_VEL, "status": None, "err": "", "raw_rows": 0,
            "raw_cols": [], "mapped_cols": [], "tras_filtros": 0,
            "tras_fecha": 0, "tras_velocidad_gt_limite": 0,
            "muestra_raw": None, "muestra_proc": None,
            "n_lat_validas": 0, "n_vel_validas": 0}
    try:
        r = None
        for _url in (URL_VEL_EXPORT, URL_VEL):
            try:
                _r = requests.get(_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            except Exception:
                continue
            r = _r
            if _r.status_code == 200 and not _r.text.lstrip().lower().startswith(("<!doctype","<html")):
                diag["url"] = _url
                break
        if r is None:
            diag["err"] = "Sin respuesta de Google Sheets"
            return pd.DataFrame(columns=["DOMINIO","FECHA","VELOCIDAD","EXCESO_KMH","LAT","LON","UBICACION"]), diag
        r.encoding = "utf-8"
        diag["status"] = r.status_code
        if r.status_code != 200:
            diag["err"] = f"HTTP {r.status_code}"
            return pd.DataFrame(columns=["DOMINIO","FECHA","VELOCIDAD","EXCESO_KMH","LAT","LON","UBICACION"]), diag
        # Detectar si devuelve HTML (login wall)
        if r.text.lstrip().lower().startswith(("<!doctype","<html")):
            diag["err"] = "Respuesta HTML (sheet no público o sin permisos)"
            return pd.DataFrame(columns=["DOMINIO","FECHA","VELOCIDAD","EXCESO_KMH","LAT","LON","UBICACION"]), diag
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [str(c).strip() for c in df.columns]
        diag["raw_rows"] = len(df)
        diag["raw_cols"] = list(df.columns)
        diag["muestra_raw"] = df.head(5).copy()
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if   cl == "movil" or "patente" in cl or "dominio" in cl:    col_map[c] = "DOMINIO"
            elif cl == "tipo de evento" or cl == "tipo":                 col_map[c] = "TIPO"
            elif "fecha"    in cl:                                       col_map[c] = "FECHA"
            elif cl == "velocidad" or cl.startswith("veloc"):            col_map[c] = "VELOCIDAD"
            elif "gravedad" in cl:                                       col_map[c] = "GRAVEDAD"
            elif cl == "latitud" or cl == "lat":                         col_map[c] = "LAT"
            elif cl == "longitud" or cl in ("lon","lng","long"):         col_map[c] = "LON"
            elif cl == "ubicacion" or "direcc" in cl:                    col_map[c] = "UBICACION"
        df = df.rename(columns=col_map)
        diag["mapped_cols"] = list(df.columns)
        if "DOMINIO" in df.columns:
            df["DOMINIO"] = df["DOMINIO"].apply(normalizar_patente)
        if "FECHA" in df.columns:
            _fecha_txt = df["FECHA"].copy()
            df["FECHA"] = parse_fecha_mixta(df["FECHA"])
            _malas = _fecha_txt[df["FECHA"].isna() & _fecha_txt.notna()]
            diag["n_fechas_invalidas"] = int(len(_malas))
            diag["muestra_fechas_invalidas"] = _malas.astype(str).head(10).tolist()
        # Parser formato AR (coma decimal) para LAT/LON/VELOCIDAD
        def parse_ar(serie):
            s = serie.astype(str).str.strip()
            tiene_coma = s.str.contains(",", na=False)
            s_coma = s.where(~tiene_coma, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
            return pd.to_numeric(s_coma, errors="coerce")
        for c in ["LAT","LON","VELOCIDAD"]:
            if c in df.columns:
                df[c] = parse_ar(df[c])
        if "LAT" in df.columns:
            mask = df["LAT"].abs() > 90
            df.loc[mask, "LAT"] = df.loc[mask, "LAT"] / 100
            diag["n_lat_validas"] = int(df["LAT"].between(-55,-21).sum())
        if "LON" in df.columns:
            mask = df["LON"].abs() > 180
            df.loc[mask, "LON"] = df.loc[mask, "LON"] / 100
        # Coordenada exacta desde html_LatLng ("...@-33.8983,-59.4740"): en algunos
        # meses Latitud/Longitud vienen redondeadas a enteros (-34 / -59).
        _col_ll = next((c for c in df.columns if "latlng" in str(c).lower()), None)
        if _col_ll is not None:
            _ll = df[_col_ll].astype(str).str.extract(r'@\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)')
            _lat_p = pd.to_numeric(_ll[0], errors="coerce")
            _lon_p = pd.to_numeric(_ll[1], errors="coerce")
            _ok = _lat_p.between(-55, -21) & _lon_p.between(-74, -53)
            if "LAT" in df.columns and "LON" in df.columns:
                df["LAT"] = _lat_p.where(_ok, df["LAT"])
                df["LON"] = _lon_p.where(_ok, df["LON"])
            else:
                df["LAT"], df["LON"] = _lat_p.where(_ok), _lon_p.where(_ok)
            diag["n_coords_precisas"] = int(_ok.sum())
            if "LAT" in df.columns:
                diag["n_lat_validas"] = int(df["LAT"].between(-55,-21).sum())
        # Fallback: detectar VELOCIDAD por heurística
        if "VELOCIDAD" not in df.columns:
            for c in df.columns:
                if c in ("DOMINIO","FECHA","LAT","LON","UBICACION","GRAVEDAD","TIPO"): continue
                try:
                    serie = parse_ar(df[c])
                    if serie.dropna().between(50, 200).mean() > 0.5:
                        df["VELOCIDAD"] = serie
                        break
                except Exception:
                    continue
        if "VELOCIDAD" not in df.columns:
            diag["err"] = "No se encontró columna VELOCIDAD"
            return pd.DataFrame(columns=["DOMINIO","FECHA","VELOCIDAD","EXCESO_KMH","LAT","LON","UBICACION"]), diag
        diag["n_vel_validas"] = int(df["VELOCIDAD"].notna().sum())
        diag["tras_fecha"] = int(df["FECHA"].notna().sum()) if "FECHA" in df.columns else 0
        df_post = df[df["VELOCIDAD"] > LIMITE_VELOCIDAD].copy()
        diag["tras_velocidad_gt_limite"] = len(df_post)
        df_post["EXCESO_KMH"] = (df_post["VELOCIDAD"] - LIMITE_VELOCIDAD).round(1)
        keep = [c for c in ["DOMINIO","FECHA","VELOCIDAD","EXCESO_KMH","GRAVEDAD","TIPO","LAT","LON","UBICACION"] if c in df_post.columns]
        df_post = df_post[keep].dropna(subset=["DOMINIO","FECHA"]).reset_index(drop=True)
        diag["tras_filtros"] = len(df_post)
        diag["muestra_proc"] = df_post.head(5).copy() if not df_post.empty else None
        diag["err"] = "OK" if not df_post.empty else "Sin filas tras filtros"
        return df_post, diag
    except Exception as e:
        diag["err"] = f"Excepción: {type(e).__name__}: {e}"
        return pd.DataFrame(columns=["DOMINIO","FECHA","VELOCIDAD","EXCESO_KMH","LAT","LON","UBICACION"]), diag
@st.cache_data(ttl=3600)
def cargar_carga(tractores_validos=None):
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
            return normalizar_patente(p)
        tractores_set = set()
        if tractores_validos is not None:
            tractores_set = {norm_pat(t) for t in tractores_validos if pd.notna(t)}
        def elegir_tractor(celda):
            pats = [norm_pat(p) for p in str(celda).split(',') if str(p).strip()]
            if not pats:
                return ''
            if tractores_set:
                for p in pats:
                    if p in tractores_set:
                        return p
            return pats[0]
        df['DOMINIO'] = df[col_unid].apply(elegir_tractor)
        df['MES']      = df[col_fecha].dt.to_period('M')
        df['PESO_TON'] = df[col_peso] / 1000.0
        return (df.groupby(['DOMINIO','MES'])
                  .agg(PESO_TON=('PESO_TON','sum'))
                  .reset_index())
    except Exception:
        return pd.DataFrame()
@st.cache_data(ttl=3600)
def cargar_viajes_todos():
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
            return normalizar_patente(p)
        df['DOMINIO']   = df[col_unid].astype(str).str.split(',').str[0].apply(norm_pat)
        df['MES']       = df[col_fecha].dt.to_period('M')
        df['PESO_TON']  = df[col_peso] / 1000.0
        df['CON_CARGA'] = (df['PESO_TON'] > 0).astype(int)
        return df[['DOMINIO','MES','PESO_TON','CON_CARGA']].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()
# Límites de plausibilidad del peso por viaje (en toneladas). Un viaje con 0 t es
# un viaje vacío (válido). Valores entre 0 y el mínimo, o por encima del máximo,
# se consideran error de carga humana y se descartan del cálculo del IER.
CARGA_MIN_TON_VIAJE = 3.0
CARGA_MAX_TON_VIAJE = 35.0
@st.cache_data(ttl=3600)
def cargar_viajes_ier(tractores_validos=None):
    """Viajes por DOMINIO+MES para el IER: peso promedio por viaje (solo pesos
    plausibles) y mezcla de rutas (origen → destino)."""
    vacio = pd.DataFrame(columns=['DOMINIO','MES','PESO_TON','RUTA'])
    diag = {'ruta': False, 'n_viajes': 0, 'n_descartados': 0}
    try:
        df = pd.read_excel(CARGA_URL)
        df.columns = [str(c).strip() for c in df.columns]
        cols_up = {c: c.upper() for c in df.columns}
        col_unid   = next((c for c, u in cols_up.items() if 'UNID' in u), None)
        col_peso   = next((c for c, u in cols_up.items() if 'PESO' in u and 'ENTREGAD' in u), None)
        col_fecha  = next((c for c, u in cols_up.items() if 'FECHA' in u), None)
        col_estado = next((c for c, u in cols_up.items() if 'ESTADO' in u), None)
        col_orig   = next((c for c, u in cols_up.items() if 'ORIGEN' in u), None)
        col_dest   = next((c for c, u in cols_up.items() if 'DESTINO' in u), None)
        if not all([col_unid, col_peso, col_fecha]):
            return vacio, diag
        if col_estado:
            df = df[df[col_estado].astype(str).str.upper() == 'FINALIZADA']
        df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
        df = df[df[col_fecha].notna()].copy()
        peso_ton = pd.to_numeric(df[col_peso], errors='coerce') / 1000.0
        tractores_set = {normalizar_patente(t) for t in (tractores_validos or ()) if pd.notna(t)}
        def elegir_tractor(celda):
            pats = [normalizar_patente(p) for p in str(celda).split(',') if str(p).strip()]
            if not pats:
                return ''
            for p in pats:
                if p in tractores_set:
                    return p
            return pats[0]
        out = pd.DataFrame({
            'DOMINIO': df[col_unid].apply(elegir_tractor),
            'MES': df[col_fecha].dt.to_period('M'),
            'PESO_TON': peso_ton,
        })
        if col_orig and col_dest:
            def norm_lugar(s):
                s = str(s).strip().upper()
                return '' if s in ('', 'NAN', 'NONE') else ' '.join(s.split())
            o = df[col_orig].apply(norm_lugar); d = df[col_dest].apply(norm_lugar)
            out['RUTA'] = np.where((o != '') & (d != ''), o + ' → ' + d, '')
            diag['ruta'] = True
        else:
            out['RUTA'] = ''
        valido = out['PESO_TON'].notna() & (
            (out['PESO_TON'] == 0) |
            out['PESO_TON'].between(CARGA_MIN_TON_VIAJE, CARGA_MAX_TON_VIAJE))
        diag['n_viajes'] = int(len(out)); diag['n_descartados'] = int((~valido).sum())
        # Los viajes con peso imposible siguen sirviendo para la ruta, pero no para el peso.
        out.loc[~valido, 'PESO_TON'] = np.nan
        return out[out['DOMINIO'] != ''].reset_index(drop=True), diag
    except Exception as e:
        diag['err'] = str(e)[:160]
        return vacio, diag
@st.cache_data(ttl=3600)
def obtener_precio_gasoil():
    return 2300.0, "valor base manual"
@st.cache_data(ttl=600)
def cargar_datos_manejo():
    dfs = []
    diag = []
    for sheet in MANEJO_SHEETS:
        url = f"https://docs.google.com/spreadsheets/d/{MANEJO_SHEET_ID}/gviz/tq?tqx=out:csv&gid={sheet['gid']}"
        try:
            r = requests.get(url, timeout=15)
            status = r.status_code
            if status != 200:
                diag.append({'modelo': sheet['modelo'], 'gid': sheet['gid'], 'status': status, 'rows': 0, 'col_score': '—', 'err': f'HTTP {status}'})
                continue
            df = pd.read_csv(url, header=0)
            df.columns = [str(c).strip() for c in df.columns]
            def find_col(keywords):
                for c in df.columns:
                    cu = c.upper()
                    if all(k.upper() in cu for k in keywords):
                        return c
                return None
            col_mes   = find_col(['MES']) or df.columns[0]
            col_dom = find_col(['DOMINIO']) or find_col(['MATRÍCULA']) or find_col(['MATRICULA']) or find_col(['PATENTE']) or find_col(['VEHICULO']) or find_col(['VEHÍCULO']) or find_col(['UNIDAD']) or find_col(['MOVIL']) or find_col(['MÓVIL'])
            col_score = find_col(['SCORE GENERAL']) or find_col(['SCORE_GENERAL']) or find_col(['SCOREGENERAL'])
            if col_dom is None or col_score is None:
                diag.append({'modelo': sheet['modelo'], 'gid': sheet['gid'], 'status': status,
                             'rows': len(df), 'col_score': col_score or '—',
                             'err': f'Falta DOMINIO o SCORE GENERAL. Cols: {list(df.columns)[:10]}'})
                continue
            # Parsear MES hoja por hoja (evita que pandas infiera un único formato
            # de fecha al concatenar hojas con formatos distintos, p.ej. "2026-04"
            # en Stralis/S-Way vs "1/04/2026" en Scania — eso descartaba Scania entero).
            _mes_parsed = pd.to_datetime(df[col_mes], errors='coerce', dayfirst=True)
            if _mes_parsed.notna().sum() < len(df):
                _mes_parsed2 = pd.to_datetime(df[col_mes], errors='coerce', dayfirst=False)
                _mes_parsed = _mes_parsed.fillna(_mes_parsed2)
            _dom_norm     = df[col_dom].apply(normalizar_patente)
            _score_parsed = pd.to_numeric(
                df[col_score].astype(str).str.replace(',', '.').str.replace(r'[^\d.\-]', '', regex=True),
                errors='coerce')
            tmp = pd.DataFrame({
                'MES': _mes_parsed,
                'DOMINIO': df[col_dom],
                'SCORE_CONDUCCION': _score_parsed,
            })
            diag.append({'modelo': sheet['modelo'], 'gid': sheet['gid'], 'status': status,
                         'rows': len(df), 'col_score': col_score, 'err': 'OK',
                         'dominio_raw_sample': [repr(v) for v in df[col_dom].head(10).tolist()],
                         'dominio_raw_lens': [len(str(v)) for v in df[col_dom].head(10).tolist()],
                         'mes_ok': int(_mes_parsed.notna().sum()),
                         'dom_ok': int((_dom_norm.str.len() > 2).sum()),
                         'score_ok': int(_score_parsed.notna().sum()),
                         'n_total': len(df),
                         'mes_raw_sample': [repr(v) for v in df[col_mes].head(10).tolist()],
                         'score_raw_sample': [repr(v) for v in df[col_score].head(10).tolist()]})
            dfs.append(tmp)
        except Exception as e:
            diag.append({'modelo': sheet['modelo'], 'gid': sheet['gid'], 'status': '?', 'rows': 0, 'col_score': '—', 'err': str(e)[:120]})
            continue
    if not dfs:
        return pd.DataFrame(columns=['DOMINIO', 'MES', 'SCORE_CONDUCCION']), diag
    out = pd.concat(dfs, ignore_index=True)
    out['DOMINIO'] = out['DOMINIO'].apply(normalizar_patente)
    out['MES']     = pd.to_datetime(out['MES'], errors='coerce', dayfirst=True)
    out['SCORE_CONDUCCION'] = pd.to_numeric(out['SCORE_CONDUCCION'], errors='coerce')
    out = out[out['MES'].notna() & (out['DOMINIO'].str.len() > 2) & out['SCORE_CONDUCCION'].notna()]
    return out[['DOMINIO', 'MES', 'SCORE_CONDUCCION']].reset_index(drop=True), diag
def cargar_arreglos():
    from io import StringIO
    diag = {'status': '?', 'rows': 0, 'cols': [], 'col_dom': None,
            'col_fecha': None, 'col_monto': None, 'err': ''}
    try:
        url = f"https://docs.google.com/spreadsheets/d/{ARREGLOS_SHEET_ID}/gviz/tq?tqx=out:csv&gid={ARREGLOS_GID}"
        r = requests.get(url, timeout=20)
        diag['status'] = r.status_code
        if r.status_code != 200:
            diag['err'] = f'HTTP {r.status_code}'
            return pd.DataFrame(), diag
        df = pd.read_csv(StringIO(r.text))
        df.columns = [str(c).strip() for c in df.columns]
        diag['cols'] = list(df.columns)
        diag['rows'] = len(df)
        col_dom   = next((c for c in df.columns if any(k in c.upper() for k in ['PATENTE','DOMINIO','MOVIL','MÓVIL','UNIDAD','INTERNO'])), None)
        col_fecha = next((c for c in df.columns if 'FECHA' in c.upper() or 'DATE' in c.upper()), None)
        col_monto = next((c for c in df.columns if any(k in c.upper() for k in ['IMPORTE','MONTO','COSTO','GASTO','TOTAL','PRECIO','VALOR'])), None)
        col_desc  = next((c for c in df.columns if any(k in c.upper() for k in ['DESCRIP','DETALLE','CONCEPTO','TIPO','TRABAJO','ARREGLO','REPARAC','OBSERV'])), None)
        diag['col_dom'] = col_dom; diag['col_fecha'] = col_fecha; diag['col_monto'] = col_monto
        if col_dom is None or col_monto is None:
            diag['err'] = f'No se detectó columna patente/monto. Cols: {list(df.columns)}'
            return pd.DataFrame(), diag
        def parse_monto(s):
            s = str(s).strip()
            if s == '' or s.lower() == 'nan':
                return np.nan
            import re
            s = re.sub(r'[^\d,.\-]', '', s)
            if ',' in s and '.' in s:
                s = s.replace('.', '').replace(',', '.')
            elif ',' in s:
                s = s.replace(',', '.')
            return pd.to_numeric(s, errors='coerce')
        out = pd.DataFrame()
        out['DOMINIO']     = df[col_dom].apply(normalizar_patente)
        out['MONTO']       = df[col_monto].apply(parse_monto)
        out['FECHA']       = pd.to_datetime(df[col_fecha], errors='coerce', dayfirst=True) if col_fecha else pd.NaT
        out['DESCRIPCION'] = df[col_desc].astype(str).str.strip() if col_desc else ''
        out['MONTO']       = pd.to_numeric(out['MONTO'], errors='coerce')
        out = out[(out['DOMINIO'].str.len() > 2) & out['MONTO'].notna() & (out['MONTO'] > 0)].copy()
        out['MES'] = out['FECHA'].dt.to_period('M')
        diag['err'] = 'OK'
        return out.reset_index(drop=True), diag
    except Exception as e:
        diag['err'] = str(e)[:160]
        return pd.DataFrame(), diag
@st.cache_data(ttl=600)
def cargar_gasto_combustible():
    from io import StringIO
    diag = {'status':'?', 'rows':0, 'cols':[], 'mes':None, 'n_mes':0, 'tipo_filter':GASTO_COMB_TIPO, 'err':''}
    try:
        url = f"https://docs.google.com/spreadsheets/d/{GASTO_COMB_SHEET_ID}/gviz/tq?tqx=out:csv&gid={GASTO_COMB_GID}"
        r = requests.get(url, timeout=20)
        diag['status'] = r.status_code
        if r.status_code != 200:
            diag['err'] = f'HTTP {r.status_code}'
            return np.nan, None, 0, diag
        df = pd.read_csv(StringIO(r.text), header=0)
        diag['cols'] = list(df.columns)
        diag['rows'] = len(df)
        if df.shape[1] < 9:
            diag['err'] = f'La hoja tiene solo {df.shape[1]} columnas (se esperan al menos 9 hasta I).'
            return np.nan, None, 0, diag
        col_fecha = df.iloc[:, 0]
        col_tipo  = df.iloc[:, 5]
        col_monto = df.iloc[:, 8]
        def parse_monto(s):
            s = str(s).strip()
            if not s or s.lower() == 'nan': return np.nan
            import re
            s = re.sub(r'[^\d,.\-]', '', s)
            if ',' in s and '.' in s: s = s.replace('.', '').replace(',', '.')
            elif ',' in s:            s = s.replace(',', '.')
            return pd.to_numeric(s, errors='coerce')
        sub = pd.DataFrame({
            'FECHA': pd.to_datetime(col_fecha, errors='coerce', dayfirst=True),
            'TIPO' : col_tipo.astype(str).str.strip().str.upper(),
            'MONTO': col_monto.apply(parse_monto),
        })
        sub = sub[(sub['TIPO']==GASTO_COMB_TIPO.upper()) & sub['FECHA'].notna() & sub['MONTO'].notna() & (sub['MONTO']>0)]
        hoy = pd.Timestamp.now().normalize()
        sub = sub[sub['FECHA'] <= hoy]
        if sub.empty:
            diag['err'] = f'Sin filas tipo "{GASTO_COMB_TIPO}" con fecha (≤ hoy) y monto válidos.'
            return np.nan, None, 0, diag
        sub['MES'] = sub['FECHA'].dt.to_period('M')
        mes_max = sub['MES'].max()
        sub_mes = sub[sub['MES']==mes_max]
        gasto_prom = float(sub_mes['MONTO'].mean())
        diag['mes'] = str(mes_max); diag['n_mes'] = int(len(sub_mes)); diag['err'] = 'OK'
        return gasto_prom, str(mes_max), int(len(sub_mes)), diag
    except Exception as e:
        diag['err'] = str(e)[:160]
        return np.nan, None, 0, diag
def asignar_modelo(dominio):
    d = normalizar_patente(dominio)
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
# ── IER v8: consumo real vs. consumo esperado ──────────────────────────────────
# El chofer se evalúa solo por lo que controla. El consumo se compara contra lo
# que "debería" gastar ese camión dado lo que no controla: modelo, mes (viento,
# temperatura), peso promedio por viaje y rutas recorridas.
IER_PESOS = {'CONSUMO': 0.40, 'MANEJO': 0.25, 'RALENTI': 0.20, 'VEL': 0.15}
IER_KM_CONFIANZA = 3000            # con pocos km el desvío se acerca a 0 (IER≈100)
IER_TOPE_CARGA = 0.10              # el ajuste por peso mueve el esperado como máx ±10 %
IER_TOPE_RUTA = 0.08               # el ajuste por ruta mueve el esperado como máx ±8 %
IER_PENDIENTE_CARGA_DEF = 0.012    # +1,2 % de consumo por tonelada extra por viaje
IER_PENDIENTE_CARGA_RANGO = (0.003, 0.020)
IER_L100_PLAUSIBLE = (10.0, 80.0)  # meses con L/100km fuera de rango no se evalúan
def _pendiente_theil_sen(x, y, max_n=400):
    """Pendiente robusta (mediana de pendientes entre pares): un dato errado no la arrastra."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) > max_n:
        sel = np.random.default_rng(0).choice(len(x), max_n, replace=False)
        x, y = x[sel], y[sel]
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 0.5
    if ok.sum() < 10:
        return np.nan
    return float(np.median((y[j] - y[i])[ok] / dx[ok]))
def calcular_ier(df, df_vel=None, df_viajes=None, df_manejo=None):
    if 'DOMINIO' not in df.columns or df.empty:
        return pd.DataFrame()
    df_c = df[df['L100KM'] > 0].copy()
    if df_c.empty:
        return pd.DataFrame()
    if 'MES_PERIODO' not in df_c.columns:
        df_c['MES_PERIODO'] = df_c['FECHA'].dt.to_period('M') if 'FECHA' in df_c.columns else pd.NaT
    agg_dict = {'KM': ('KM','sum'), 'LITROS': ('LITROS','sum'), 'MESES': ('MES_PERIODO','nunique')}
    agg = df_c.groupby('DOMINIO').agg(**agg_dict).reset_index()
    agg['MODELO'] = agg['DOMINIO'].apply(asignar_modelo)
    # ── 1. Tabla patente × mes ────────────────────────────────────────────────
    mes = (df_c.groupby(['DOMINIO','MES_PERIODO'], dropna=False)
               .agg(KM=('KM','sum'), LITROS=('LITROS','sum'), L100_MEAN=('L100KM','mean'))
               .reset_index())
    mes['L100'] = np.where((mes['KM'] > 0) & (mes['LITROS'] > 0),
                           mes['LITROS'] / mes['KM'].where(mes['KM'] > 0) * 100, mes['L100_MEAN'])
    mes['MODELO'] = mes['DOMINIO'].apply(asignar_modelo)
    mes = mes[mes['L100'].between(*IER_L100_PLAUSIBLE)].copy()
    # ── 2. Peso promedio por viaje (solo pesos plausibles) ───────────────────
    mes['TON_VIAJE'] = np.nan; mes['N_VIAJES'] = 0
    vj = pd.DataFrame()
    if df_viajes is not None and not df_viajes.empty and 'MES' in df_viajes.columns:
        vj = df_viajes[df_viajes['MES'].isin(set(mes['MES_PERIODO'].dropna()))].rename(columns={'MES':'MES_PERIODO'})
        if not vj.empty:
            vt = (vj.groupby(['DOMINIO','MES_PERIODO'])
                    .agg(TON_VIAJE=('PESO_TON','mean'), N_VIAJES=('PESO_TON','size'),
                         N_PESO=('PESO_TON','count'))
                    .reset_index())
            mes = mes.drop(columns=['TON_VIAJE','N_VIAJES']).merge(vt, on=['DOMINIO','MES_PERIODO'], how='left')
            mes['N_VIAJES'] = mes['N_VIAJES'].fillna(0).astype(int)
    if 'N_PESO' not in mes.columns:
        mes['N_PESO'] = 0
    mes['N_PESO'] = mes['N_PESO'].fillna(0)
    # ── 3. Consumo base: mediana del mismo modelo en el mismo mes ─────────────
    #    (el viento, la temperatura y la época del año afectan a todos por igual)
    grp = ['MODELO','MES_PERIODO']
    mes['BASE'] = mes.groupby(grp, dropna=False)['L100'].transform('median')
    ton_ref = mes.groupby(grp, dropna=False)['TON_VIAJE'].transform('median')
    mes['TON_REF'] = ton_ref.fillna(mes.groupby('MODELO')['TON_VIAJE'].transform('median'))
    # Sin peso válido → se asume el peso típico del grupo (ni premio ni castigo).
    dx = (mes['TON_VIAJE'].fillna(mes['TON_REF']) - mes['TON_REF']).fillna(0)
    m_ok = mes['TON_VIAJE'].notna() & mes['TON_REF'].notna() & (mes['BASE'] > 0)
    pend = (_pendiente_theil_sen(dx[m_ok], mes.loc[m_ok,'L100'] / mes.loc[m_ok,'BASE'] - 1)
            if m_ok.sum() >= 15 else np.nan)
    pend_fuente = 'estimada con datos de la flota' if np.isfinite(pend) else 'valor de referencia'
    pend = float(np.clip(pend if np.isfinite(pend) else IER_PENDIENTE_CARGA_DEF, *IER_PENDIENTE_CARGA_RANGO))
    mes['F_CARGA'] = 1 + (pend * dx).clip(-IER_TOPE_CARGA, IER_TOPE_CARGA)
    # ── 4. Ajuste por ruta (origen → destino) ────────────────────────────────
    mes['F_RUTA'] = 1.0; mes['RUTA_PRINCIPAL'] = ''
    n_rutas_aj = 0
    tiene_rutas = (not vj.empty and 'RUTA' in vj.columns and (vj['RUTA'] != '').any())
    if tiene_rutas:
        vr = (vj[vj['RUTA'] != ''].groupby(['DOMINIO','MES_PERIODO','RUTA']).size()
                .rename('N').reset_index())
        res = mes[['DOMINIO','MES_PERIODO']].assign(RES=mes['L100'] / (mes['BASE'] * mes['F_CARGA']) - 1)
        vr = vr.merge(res, on=['DOMINIO','MES_PERIODO'], how='inner')
        if not vr.empty:
            vr['RN'] = vr['RES'] * vr['N']
            st_r = vr.groupby('RUTA').agg(RN=('RN','sum'), N=('N','sum'),
                                          N_UM=('DOMINIO','size'), N_DOM=('DOMINIO','nunique'))
            # Efecto de la ruta: cuánto más (o menos) gastan, en promedio, los camiones
            # que la hacen. Solo se usa si la recorrieron al menos 2 camiones distintos
            # (si no, se confundiría la ruta con el chofer) y se suaviza si hay pocos datos.
            st_r['EF'] = (st_r['RN'] / st_r['N']) * st_r['N_UM'] / (st_r['N_UM'] + 3)
            st_r.loc[st_r['N_DOM'] < 2, 'EF'] = 0.0
            st_r['EF'] = st_r['EF'].clip(-IER_TOPE_RUTA, IER_TOPE_RUTA)
            n_rutas_aj = int((st_r['EF'] != 0).sum())
            vr = vr.merge(st_r[['EF']], left_on='RUTA', right_index=True, how='left')
            vr['EN'] = vr['EF'] * vr['N']
            fr = vr.groupby(['DOMINIO','MES_PERIODO']).agg(EN=('EN','sum'), NT=('N','sum')).reset_index()
            fr['F_RUTA_N'] = (1 + fr['EN'] / fr['NT']).clip(1 - IER_TOPE_RUTA, 1 + IER_TOPE_RUTA)
            princ = (vr.sort_values('N').drop_duplicates(['DOMINIO','MES_PERIODO'], keep='last')
                       [['DOMINIO','MES_PERIODO','RUTA']])
            mes = (mes.merge(fr[['DOMINIO','MES_PERIODO','F_RUTA_N']], on=['DOMINIO','MES_PERIODO'], how='left')
                      .merge(princ, on=['DOMINIO','MES_PERIODO'], how='left'))
            mes['F_RUTA'] = mes['F_RUTA_N'].fillna(1.0)
            mes['RUTA_PRINCIPAL'] = mes['RUTA'].fillna('')
    # ── 5. Consumo esperado y desvío ─────────────────────────────────────────
    mes['L100_ESP'] = mes['BASE'] * mes['F_CARGA'] * mes['F_RUTA']
    mes = mes[mes['L100_ESP'] > 0].copy()
    mes['DESVIO'] = mes['L100'] / mes['L100_ESP'] - 1
    mes['W'] = mes['KM'].clip(lower=1)
    for c in ['L100','L100_ESP','DESVIO','F_CARGA','F_RUTA']:
        mes['W_' + c] = mes['W'] * mes[c]
    mes['W_TON'] = (mes['TON_VIAJE'] * mes['N_PESO']).fillna(0)
    def _ruta_top(s):
        s = s[s != '']
        return s.value_counts().index[0] if len(s) else ''
    per = mes.groupby('DOMINIO').agg(
        W=('W','sum'), W_L100=('W_L100','sum'), W_L100_ESP=('W_L100_ESP','sum'),
        W_DESVIO=('W_DESVIO','sum'), W_F_CARGA=('W_F_CARGA','sum'), W_F_RUTA=('W_F_RUTA','sum'),
        W_TON=('W_TON','sum'), N_PESO=('N_PESO','sum'), N_VIAJES=('N_VIAJES','sum'),
        KM_EVAL=('KM','sum'), RUTA_PRINCIPAL=('RUTA_PRINCIPAL', _ruta_top)).reset_index()
    per['L100KM']       = per['W_L100'] / per['W']
    per['L100KM_ESP']   = per['W_L100_ESP'] / per['W']
    per['DESVIO_PCT']   = per['W_DESVIO'] / per['W'] * 100
    per['AJ_CARGA_PCT'] = (per['W_F_CARGA'] / per['W'] - 1) * 100
    per['AJ_RUTA_PCT']  = (per['W_F_RUTA'] / per['W'] - 1) * 100
    per['TON_VIAJE']    = np.where(per['N_PESO'] > 0, per['W_TON'] / per['N_PESO'].where(per['N_PESO'] > 0), np.nan)
    # Pocos km → el desvío se "encoge" hacia 0 para que un mes corto no defina el ranking.
    per['DESVIO_AJ']    = per['DESVIO_PCT'] / 100 * per['KM_EVAL'] / (per['KM_EVAL'] + IER_KM_CONFIANZA)
    agg = agg.merge(per[['DOMINIO','L100KM','L100KM_ESP','DESVIO_PCT','DESVIO_AJ','AJ_CARGA_PCT',
                         'AJ_RUTA_PCT','TON_VIAJE','N_VIAJES','RUTA_PRINCIPAL']], on='DOMINIO', how='left')
    agg['TIENE_CONSUMO'] = agg['DESVIO_AJ'].notna()
    agg['L100KM'] = agg['L100KM'].fillna(
        (agg['LITROS'] / agg['KM'].where(agg['KM'] > 0) * 100)).fillna(0)
    agg['N_VIAJES'] = agg['N_VIAJES'].fillna(0).astype(int)
    agg['RUTA_PRINCIPAL'] = agg['RUTA_PRINCIPAL'].fillna('')
    # ── 6. Ralentí (% de litros en ralentí) ──────────────────────────────────
    if 'RALENTI' in df_c.columns and 'RALENTI_PCT' in df_c.columns:
        ral = df_c.groupby('DOMINIO').agg(
            _RAL=('RALENTI', 'sum'), _LTS=('LITROS', 'sum'),
            _RAL_PCT_MEAN=('RALENTI_PCT', 'mean')).reset_index()
        ral['RALENTI_PCT'] = np.where(
            ral['_LTS'] > 0, ral['_RAL'] / ral['_LTS'] * 100, ral['_RAL_PCT_MEAN']
        ).clip(0, 100).round(2)
        agg = agg.merge(ral[['DOMINIO', 'RALENTI_PCT']], on='DOMINIO', how='left')
    else:
        agg['RALENTI_PCT'] = 0.0
    agg['RALENTI_PCT'] = agg['RALENTI_PCT'].fillna(0.0)
    agg['TIENE_RALENTI'] = agg['RALENTI_PCT'] > 0
    # ── 7. Velocidad ─────────────────────────────────────────────────────────
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
    # ── 8. Score de conducción ───────────────────────────────────────────────
    if df_manejo is not None and not df_manejo.empty and 'SCORE_CONDUCCION' in df_manejo.columns:
        manejo_agg = df_manejo.groupby('DOMINIO')['SCORE_CONDUCCION'].mean().reset_index()
        agg = agg.merge(manejo_agg, on='DOMINIO', how='left')
    else:
        agg['SCORE_CONDUCCION'] = np.nan
    agg['TIENE_MANEJO'] = agg['SCORE_CONDUCCION'].notna()
    def _safe_mean(x):
        v = x.dropna(); return v.mean() if len(v) > 0 else np.nan
    modelo_avgs = agg.groupby('MODELO').agg(
        L100KM_MOD=('L100KM','mean'), KM_MOD=('KM','mean'),
        EXCESOS_MOD=('EXCESOS','mean'),
        SEVERIDAD_MOD=('SEVERIDAD','mean'),
        RALENTI_MOD=('RALENTI_PCT', lambda x: _safe_mean(x.where(x > 0))),
        SCORE_MANEJO_MOD=('SCORE_CONDUCCION',_safe_mean)).reset_index()
    agg = agg.merge(modelo_avgs, on='MODELO', how='left')
    # ── 9. Scores (z-score + tanh, siempre dentro del mismo modelo) ──────────
    for col in ['SCORE_CONSUMO','SCORE_MANEJO','SCORE_RALENTI','SCORE_VEL']:
        agg[col] = 1.0
    for modelo in agg['MODELO'].unique():
        idx = agg.index[agg['MODELO'] == modelo]
        cons_idx = idx[agg.loc[idx,'TIENE_CONSUMO'].values]
        if len(cons_idx) > 1:
            agg.loc[cons_idx,'SCORE_CONSUMO'] = calcular_score_zscore(
                1 + agg.loc[cons_idx,'DESVIO_AJ'], higher_is_better=False, k=0.4, min_sigma_pct=0.03).values
        ral_idx = idx[agg.loc[idx,'TIENE_RALENTI'].values]
        if len(ral_idx) > 1:
            agg.loc[ral_idx,'SCORE_RALENTI'] = calcular_score_zscore(
                agg.loc[ral_idx,'RALENTI_PCT'], higher_is_better=False, k=0.4, min_sigma_pct=0.10).values
        sev_log = np.log1p(agg.loc[idx,'SEVERIDAD'].astype(float))
        agg.loc[idx,'SCORE_VEL'] = calcular_score_zscore(sev_log, higher_is_better=False, k=0.4, min_sigma_pct=0.30).values
        manejo_idx = idx[agg.loc[idx,'SCORE_CONDUCCION'].notna().values]
        if len(manejo_idx) > 1:
            agg.loc[manejo_idx,'SCORE_MANEJO'] = calcular_score_zscore(
                agg.loc[manejo_idx,'SCORE_CONDUCCION'], higher_is_better=True, k=0.4, min_sigma_pct=0.05).values
    # ── 10. IER: si falta un componente, su peso se reparte entre los otros
    #        componentes de conducta (ralentí, velocidad, manejo), no al consumo.
    def _ier_row(r):
        w = dict(IER_PESOS)
        s = {'CONSUMO': r['SCORE_CONSUMO'], 'MANEJO': r['SCORE_MANEJO'],
             'RALENTI': r['SCORE_RALENTI'], 'VEL': r['SCORE_VEL']}
        faltan = [k for k, ok in (('CONSUMO', r['TIENE_CONSUMO']), ('MANEJO', r['TIENE_MANEJO']),
                                  ('RALENTI', r['TIENE_RALENTI'])) if not ok]
        libre = sum(w[k] for k in faltan)
        for k in faltan:
            w[k] = 0.0
        receptores = [k for k in ('MANEJO','RALENTI','VEL') if w[k] > 0] or [k for k in w if w[k] > 0]
        tot = sum(w[k] for k in receptores)
        for k in receptores:
            w[k] += libre * w[k] / tot
        return sum(w[k] * s[k] for k in w)
    agg['IER'] = (agg.apply(_ier_row, axis=1) * 100).round(1).fillna(100.0)
    def clasif(v):
        if   v>=105: return '🟢 Eficiente'
        elif v>= 95: return '🟡 Normal'
        elif v>= 85: return '🟠 Atención'
        else:        return '🔴 Crítico'
    agg['CLASIFICACION'] = agg['IER'].apply(clasif)
    for c in ['L100KM','L100KM_ESP','L100KM_MOD','DESVIO_PCT','AJ_CARGA_PCT','AJ_RUTA_PCT','TON_VIAJE']:
        agg[c] = agg[c].round(2)
    keep = ['DOMINIO','MODELO','IER','CLASIFICACION',
            'L100KM','L100KM_ESP','DESVIO_PCT','L100KM_MOD','AJ_CARGA_PCT','AJ_RUTA_PCT',
            'TON_VIAJE','N_VIAJES','RUTA_PRINCIPAL','RALENTI_PCT','RALENTI_MOD',
            'KM','KM_MOD','LITROS','EXCESOS','SEVERIDAD','SEVERIDAD_MOD','VEL_MAX','EXCESOS_MOD',
            'SCORE_CONDUCCION','SCORE_MANEJO_MOD','TIENE_MANEJO','TIENE_RALENTI','TIENE_CONSUMO',
            'SCORE_CONSUMO','SCORE_MANEJO','SCORE_RALENTI','SCORE_VEL','MESES']
    out = agg[keep].sort_values('IER', ascending=False).reset_index(drop=True)
    out.attrs['ier_info'] = {
        'pendiente_carga': pend, 'pendiente_fuente': pend_fuente,
        'tiene_rutas': bool(tiene_rutas), 'rutas_ajustadas': n_rutas_aj,
        'tiene_peso': bool(mes['TON_VIAJE'].notna().any()),
    }
    return out
with st.spinner('Cargando telemetría, velocidades y datos de carga...'):
    df_raw, _      = cargar_datos()
    df_vel_raw, vel_diag = cargar_velocidad()
    tractores_flota = tuple(df_raw['DOMINIO'].dropna().unique()) if (df_raw is not None and not df_raw.empty and 'DOMINIO' in df_raw.columns) else ()
    df_carga_raw    = cargar_carga(tractores_flota)
    df_viajes_ier, viajes_ier_diag = cargar_viajes_ier(tractores_flota)
    df_viajes_raw   = cargar_viajes_todos()
    df_manejo_raw, manejo_diag = cargar_datos_manejo()
    df_arreglos_raw, arreglos_diag = cargar_arreglos()
    gasto_comb_prom, gasto_comb_mes, gasto_comb_n, gasto_comb_diag = cargar_gasto_combustible()
if df_raw.empty:
    st.warning('No se pudieron cargar datos.')
    st.stop()
if not (gasto_comb_prom is None or (isinstance(gasto_comb_prom, float) and np.isnan(gasto_comb_prom))):
    precio_gasoil = float(gasto_comb_prom)
    precio_fuente = f"X10 {gasto_comb_mes} (planilla gastos)"
else:
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
    st.sidebar.markdown('<div class="sidebar-filter-header">🔍 Filtros</div>', unsafe_allow_html=True)
    periodos_disponibles = sorted(df['FECHA'].dt.to_period('M').dropna().unique().tolist())
    periodos_str = [str(p) for p in periodos_disponibles]
    if periodos_str:
        desde_idx = st.sidebar.selectbox('Desde (mes/año)', options=periodos_str, index=0)
        hasta_idx = st.sidebar.selectbox('Hasta (mes/año)', options=periodos_str, index=len(periodos_str)-1)
        desde_periodo = pd.Period(desde_idx, 'M')
        hasta_periodo = pd.Period(hasta_idx, 'M')
        st.session_state['desde_periodo'] = desde_periodo
        st.session_state['hasta_periodo'] = hasta_periodo
        df = df[(df['FECHA'].dt.to_period('M')>=desde_periodo)&(df['FECHA'].dt.to_period('M')<=hasta_periodo)]
    marcas_disp   = sorted(df['MARCA'].dropna().unique().tolist())   if 'MARCA'   in df.columns else []
    marcas_sel    = st.sidebar.multiselect('Marca', marcas_disp, default=marcas_disp)
    patentes_disp = sorted(df['DOMINIO'].dropna().unique().tolist()) if 'DOMINIO' in df.columns else []
    patentes_sel  = st.sidebar.multiselect('Patente', patentes_disp, default=[], placeholder="Todas las patentes")
    if marcas_sel   and 'MARCA'   in df.columns: df = df[df['MARCA'].isin(marcas_sel)]
    if patentes_sel and 'DOMINIO' in df.columns: df = df[df['DOMINIO'].isin(patentes_sel)]
    st.session_state['marcas_sel']   = marcas_sel
    st.session_state['patentes_sel'] = patentes_sel
if df.empty:
    st.warning(f'Sin datos para {anio_sel} con los filtros seleccionados.')
    st.stop()
# ── Helpers L/100km mensual (módulo de evolución) ──────────────────────────
MESES_ABBR = {1:'ene',2:'feb',3:'mar',4:'abr',5:'may',6:'jun',
              7:'jul',8:'ago',9:'sep',10:'oct',11:'nov',12:'dic'}
def etiqueta_mes(p):
    """Period('2025-01','M') -> 'ene 2025'"""
    try:
        return f"{MESES_ABBR[p.month]} {p.year}"
    except Exception:
        return str(p)
def serie_l100(dframe, dominios=None):
    """Serie mensual de L/100km (litros totales / km totales * 100).
       dominios=None -> toda la flota del dataframe recibido."""
    if dframe is None or dframe.empty or 'MES_PERIODO' not in dframe.columns:
        return pd.DataFrame(columns=['MES_PERIODO','LITROS','KM','L100','LABEL'])
    d = dframe
    if dominios:
        d = d[d['DOMINIO'].isin(dominios)]
    if d.empty:
        return pd.DataFrame(columns=['MES_PERIODO','LITROS','KM','L100','LABEL'])
    g = (d.groupby('MES_PERIODO')
           .agg(LITROS=('LITROS','sum'), KM=('KM','sum'))
           .reset_index().sort_values('MES_PERIODO'))
    g = g[g['KM'] > 0].copy()
    g['L100']  = (g['LITROS']/g['KM']*100).round(2)
    g['LABEL'] = g['MES_PERIODO'].apply(etiqueta_mes)
    return g
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
    # Los excesos se filtran por lo elegido en la barra lateral, no por los meses
    # o patentes que tenga la telemetría (la planilla de excesos suele ir adelantada).
    _mes_min = st.session_state.get('desde_periodo', None)
    _hasta_sel = st.session_state.get('hasta_periodo', None)
    _ult_tel = df_full.loc[df_full['FECHA'].dt.year==anio_sel, 'FECHA'].dropna().dt.to_period('M').max()
    if _mes_min is None:
        _mes_min = pd.Period(f'{anio_sel}-01', 'M')
    if _hasta_sel is None or pd.isna(_ult_tel) or _hasta_sel >= _ult_tel:
        _mes_max = pd.Period(f'{anio_sel}-12', 'M')   # "Hasta" en el último mes = hasta hoy
    else:
        _mes_max = _hasta_sel
    _vel_periodos = df_vel_anio['FECHA'].dt.to_period('M')
    _mask_vel = (_vel_periodos>=_mes_min)&(_vel_periodos<=_mes_max)
    _mask_vel &= df_vel_anio['DOMINIO'].isin(df_full['DOMINIO'].dropna().unique())   # solo flota con telemetría
    _pats_sel = st.session_state.get('patentes_sel') or []
    _marcas_sel = st.session_state.get('marcas_sel') or []
    if _pats_sel:
        _mask_vel &= df_vel_anio['DOMINIO'].isin(_pats_sel)
    elif 'MARCA' in df_full.columns and _marcas_sel and set(_marcas_sel) != set(df_full['MARCA'].dropna().unique()):
        _mask_vel &= df_vel_anio['DOMINIO'].isin(df_full.loc[df_full['MARCA'].isin(_marcas_sel), 'DOMINIO'].unique())
    df_vel_filtrado = df_vel_anio[_mask_vel].copy()
else:
    df_vel_filtrado = df_vel_anio.copy()
if not df_manejo_raw.empty and 'MES' in df_manejo_raw.columns and not df.empty:
    _mes_min_p = df['FECHA'].dropna().dt.to_period('M').min()
    _mes_max_p = df['FECHA'].dropna().dt.to_period('M').max()
    _man_periodos = df_manejo_raw['MES'].dt.to_period('M')
    df_manejo_filtrado = df_manejo_raw[(_man_periodos>=_mes_min_p)&(_man_periodos<=_mes_max_p)].copy()
else:
    df_manejo_filtrado = df_manejo_raw.copy()
df_ier = calcular_ier(df, df_vel_filtrado, df_viajes=df_viajes_ier, df_manejo=df_manejo_filtrado)
ier_info = df_ier.attrs.get('ier_info', {}) if not df_ier.empty else {}
total_excesos  = len(df_vel_filtrado) if not df_vel_filtrado.empty else 0
vel_max_global = (df_vel_filtrado['VELOCIDAD'].max() if not df_vel_filtrado.empty and 'VELOCIDAD' in df_vel_filtrado.columns else 0)
# ═══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 1 — DASHBOARD PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
if pg == "Dashboard Principal":
    col_logo, col_title = st.columns([1,5])
    with col_logo: st.image(LOGO_URL, width=130)
    with col_title:
        st.markdown(f"""<div style='padding:8px 0;'>
        <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>Expreso Diemar &mdash; Dashboard LAD {anio_sel}</div>
        <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Telemetría flota LAD &middot; Año {anio_sel} &middot; Actualización automática</div>
        </div>""", unsafe_allow_html=True)
    st.markdown(f'<div style="margin-bottom:12px;"><span class="price-badge">&#9981; Precio gasoil: <b>${precio_gasoil:,.0f}/L</b></span>&nbsp;&nbsp;<span style="font-size:.75rem;color:#94a3b8;">Fuente: {precio_fuente}</span></div>', unsafe_allow_html=True)
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
    kpi(k1,'','⛽ Litros totales',f'{lts_total:,.0f}',f'litros {anio_sel}')
    kpi(k2,'','🛣️ KM recorridos',f'{kms_total:,.0f}',f'kilómetros {anio_sel}')
    kpi(k3,delta_col,'📊 L/100km flota',f'{l100_prom:.2f}',delta_txt)
    k4,k5,k6 = st.columns(3)
    kpi(k4,'kpi-amber','💰 Costo estimado',f'${costo_est/1e6:.1f}M',f'@ ${precio_gasoil:,.0f}/L')
    kpi(k5,'kpi-green','🚛 Unidades activas',f'{n_unidades}','dominios únicos')
    _ral_sub = (f'{ralenti_total:,.0f} L · {ralenti_delta_txt}' if ralenti_delta_txt else f'{ralenti_total:,.0f} L en ralentí')
    kpi(k6,'kpi-amber','⏱️ % Ralentí',f'{ralenti_pct:.1f}%',_ral_sub)
    # ═══════════════════════════════════════════════════════════════════════
    #  MÓDULO — LITROS CADA 100 KM DEL MES + EVOLUCIÓN
    # ═══════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown('<div class="sec-title">⛽ Litros cada 100 km — Consumo del mes y evolución</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns([1.1, 2.4, 1.0])
    with fc1:
        alcance_l100 = st.radio('Período a graficar',
                                ['Rango filtrado', 'Histórico completo'],
                                index=0, key='l100_alcance',
                                help='"Rango filtrado" usa el año y los meses elegidos en la barra lateral. '
                                     '"Histórico completo" muestra todos los meses disponibles (todos los años).')
    if alcance_l100 == 'Histórico completo':
        base_l100 = df_full[df_full['FECHA'].notna()].copy()
        _mk = st.session_state.get('marcas_sel', [])
        _pk = st.session_state.get('patentes_sel', [])
        if _mk and 'MARCA'   in base_l100.columns: base_l100 = base_l100[base_l100['MARCA'].isin(_mk)]
        if _pk and 'DOMINIO' in base_l100.columns: base_l100 = base_l100[base_l100['DOMINIO'].isin(_pk)]
        base_l100['MES_PERIODO'] = base_l100['FECHA'].dt.to_period('M')
    else:
        base_l100 = df.copy()
    with fc2:
        pats_l100_disp = sorted(base_l100['DOMINIO'].dropna().unique().tolist()) if 'DOMINIO' in base_l100.columns else []
        pats_l100_sel  = st.multiselect('Separar por patente (opcional)', pats_l100_disp, default=[],
                                        placeholder='Todas juntas (total flota)', key='l100_pats',
                                        help='Elegí una o varias patentes para ver su curva individual de L/100km.')
    with fc3:
        ver_flota = st.checkbox('Línea total flota', value=True, key='l100_ver_flota')
    serie_flota = serie_l100(base_l100)
    serie_kpi   = serie_l100(base_l100, pats_l100_sel) if pats_l100_sel else serie_flota
    kpi_scope   = (pats_l100_sel[0] if len(pats_l100_sel)==1
                   else (f'{len(pats_l100_sel)} patentes seleccionadas' if pats_l100_sel else 'Total flota'))
    if serie_kpi.empty:
        st.info('Sin kilómetros registrados para la selección actual.')
    else:
        _ult      = serie_kpi.iloc[-1]
        _mes_ult  = _ult['LABEL']
        _l100_ult = float(_ult['L100'])
        if len(serie_kpi) >= 2:
            _prev      = serie_kpi.iloc[-2]
            _d_l100    = _l100_ult - float(_prev['L100'])
            _d_pct     = (_d_l100/float(_prev['L100'])*100) if float(_prev['L100'])>0 else 0
            _d_color   = 'kpi-red' if _d_l100 > 0 else 'kpi-green'
            _d_valor   = f"{'▲' if _d_l100>0 else '▼'} {abs(_d_l100):.2f}"
            _d_sub     = f"{_d_pct:+.1f}% vs {_prev['LABEL']} ({float(_prev['L100']):.2f} L/100km)"
        else:
            _d_color, _d_valor, _d_sub = '', '—', 'sin mes anterior para comparar'
        _prom_per = float(serie_kpi['LITROS'].sum()/serie_kpi['KM'].sum()*100) if serie_kpi['KM'].sum()>0 else 0
        _mejor    = serie_kpi.loc[serie_kpi['L100'].idxmin()]
        _peor     = serie_kpi.loc[serie_kpi['L100'].idxmax()]
        _vs_prom  = _l100_ult - _prom_per
        _col_ult  = 'kpi-green' if _vs_prom <= 0 else 'kpi-red'
        m1, m2, m3 = st.columns(3)
        m1.markdown(
            f'<div class="kpi-card {_col_ult}"><div class="kpi-label">⛽ L/100 km — {_mes_ult}</div>'
            f'<div class="kpi-value">{_l100_ult:.2f}</div>'
            f'<div class="kpi-sub">{kpi_scope} · {_ult["LITROS"]:,.0f} L / {_ult["KM"]:,.0f} km</div></div>',
            unsafe_allow_html=True)
        m2.markdown(
            f'<div class="kpi-card {_d_color}"><div class="kpi-label">📉 Variación mensual</div>'
            f'<div class="kpi-value">{_d_valor}</div>'
            f'<div class="kpi-sub">{_d_sub}</div></div>',
            unsafe_allow_html=True)
        m3.markdown(
            f'<div class="kpi-card kpi-purple"><div class="kpi-label">📊 Promedio del período</div>'
            f'<div class="kpi-value">{_prom_per:.2f}</div>'
            f'<div class="kpi-sub">{len(serie_kpi)} meses · mejor {_mejor["LABEL"]} ({float(_mejor["L100"]):.2f}) · '
            f'peor {_peor["LABEL"]} ({float(_peor["L100"]):.2f})</div></div>',
            unsafe_allow_html=True)
    # ── Gráfico de evolución ──────────────────────────────────────────────
    series_plot = []
    if ver_flota and not serie_flota.empty:
        series_plot.append(('Total flota', serie_flota, '#60a5fa', True))
    PALETA_PAT = ['#f97316','#22c55e','#a78bfa','#ec4899','#facc15','#14b8a6',
                  '#ef4444','#38bdf8','#84cc16','#fb7185','#c084fc','#2dd4bf']
    for i, _pat in enumerate(pats_l100_sel):
        _s = serie_l100(base_l100, [_pat])
        if not _s.empty:
            series_plot.append((_pat, _s, PALETA_PAT[i % len(PALETA_PAT)], False))
    if not series_plot:
        st.info('Elegí al menos una patente o activá la línea de total flota para ver la evolución.')
    elif all(len(s) < 2 for _, s, _c, _f in series_plot):
        st.info('Hay un solo mes en la selección. Ampliá el rango de meses (barra lateral) para ver la evolución.')
    else:
        _labels_orden = (serie_flota['LABEL'].tolist() if not serie_flota.empty
                         else series_plot[0][1]['LABEL'].tolist())
        for _n, _s, _c, _f in series_plot:
            for _lb in _s['LABEL'].tolist():
                if _lb not in _labels_orden:
                    _labels_orden.append(_lb)
        _orden_map = {etiqueta_mes(p): p for _, _s, _c, _f in series_plot for p in _s['MES_PERIODO']}
        _labels_orden = sorted(set(_labels_orden), key=lambda l: _orden_map.get(l))
        _all_vals = [v for _n, _s, _c, _f in series_plot for v in _s['L100'].tolist()]
        _y_min = min(_all_vals); _y_max = max(_all_vals)
        _pad   = max((_y_max - _y_min) * 0.28, 1.2)
        _n_series = len(series_plot)
        fig_l100 = go.Figure()
        for _i_s, (_n, _s, _c, _fill) in enumerate(series_plot):
            _n_pts = len(_s)
            # con varias curvas se rotulan solo los puntos clave para no saturar el gráfico
            if _n_series > 3 and _i_s > 0:
                _txt = [''] * _n_pts
            elif _n_pts <= 14 and _n_series == 1:
                _txt = [f'{v:.1f}'.replace('.', ',') for v in _s['L100']]
            else:
                _idx_key = {0, _n_pts-1, int(_s['L100'].values.argmin()), int(_s['L100'].values.argmax())}
                _txt = [f'{v:.1f}'.replace('.', ',') if i in _idx_key else ''
                        for i, v in enumerate(_s['L100'])]
            # el primer/último rótulo se corre hacia adentro para que no lo corte el borde
            _tpos = ['top right' if i == 0 else ('top left' if i == _n_pts-1 else 'top center')
                     for i in range(_n_pts)]
            fig_l100.add_trace(go.Scatter(
                x=_s['LABEL'], y=_s['L100'], name=_n,
                mode='lines+markers+text',
                text=_txt, textposition=_tpos,
                textfont=dict(color='#cbd5e1', size=10),
                line=dict(color=_c, width=3, shape='spline'),
                marker=dict(size=7, color=_c, line=dict(color='#0f172a', width=1.5)),
                fill='tozeroy' if _fill else None,
                fillcolor=('rgba(96,165,250,0.18)' if _n_series == 1 else 'rgba(96,165,250,0.10)') if _fill else None,
                hovertemplate=f'<b>{_n}</b><br>%{{x}}<br>L/100km: <b>%{{y:.2f}}</b><extra></extra>'))
        if not serie_kpi.empty and _n_series == 1:
            fig_l100.add_hline(y=_prom_per, line_dash='dot', line_color='#f59e0b', line_width=1.5,
                               annotation_text=f'Promedio {_prom_per:.2f}', annotation_position='top left',
                               annotation_font_color='#fbbf24', annotation_font_size=10)
        fig_l100.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,41,59,0.6)',
            font=dict(color='#e2e8f0'),
            legend=dict(bgcolor='rgba(15,23,42,0.8)', bordercolor='#334155', borderwidth=1,
                        orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            showlegend=_n_series > 1,
            xaxis=dict(gridcolor='#334155', linecolor='#334155', tickfont=dict(color='#94a3b8', size=10),
                       title=dict(text='Mes', font=dict(color='#94a3b8')), tickangle=-45,
                       categoryorder='array', categoryarray=_labels_orden),
            yaxis=dict(gridcolor='#334155', linecolor='#334155', tickfont=dict(color='#94a3b8', size=11),
                       title=dict(text='L/100 km', font=dict(color='#94a3b8')),
                       range=[max(0, _y_min - _pad), _y_max + _pad]),
            height=430, margin=dict(l=20, r=25, t=50, b=70), hovermode='x unified')
        st.plotly_chart(fig_l100, use_container_width=True)
        st.caption('Promedio ponderado por mes (litros totales ÷ km totales × 100). '
                   'Usá el selector de patentes para comparar unidades y los filtros de la barra lateral para acotar meses, marca o año.')
        with st.expander('📋 Ver tabla mensual de L/100 km'):
            _tabla = pd.DataFrame({'Mes': _labels_orden})
            for _n, _s, _c, _f in series_plot:
                _tabla = _tabla.merge(
                    _s[['LABEL','L100']].rename(columns={'LABEL':'Mes','L100':_n}),
                    on='Mes', how='left')
            if not serie_flota.empty:
                _tabla = _tabla.merge(
                    serie_flota[['LABEL','LITROS','KM']].rename(
                        columns={'LABEL':'Mes','LITROS':'Litros (flota)','KM':'KM (flota)'}),
                    on='Mes', how='left')
                for _c_fmt in ['Litros (flota)','KM (flota)']:
                    _tabla[_c_fmt] = _tabla[_c_fmt].apply(lambda x: f'{x:,.0f}' if pd.notnull(x) else '—')
            st.dataframe(_tabla, use_container_width=True, hide_index=True)
    # ═══════════════════════════════════════════════════════════════════════
    #  MÓDULO — LITROS vs KILÓMETROS POR MES (correlación)
    # ═══════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown('<div class="sec-title">📈 Litros y kilómetros por mes — ¿se mueven juntos?</div>', unsafe_allow_html=True)
    st.caption(f'Usa los mismos controles de arriba: {kpi_scope} · '
               f'{"histórico completo" if alcance_l100 == "Histórico completo" else "rango filtrado"}.')
    if serie_kpi.empty or len(serie_kpi) < 2:
        st.info('Se necesitan al menos 2 meses en la selección para comparar litros contra kilómetros.')
    else:
        _lbl = serie_kpi['LABEL'].tolist()
        _lts = serie_kpi['LITROS'].astype(float).values
        _kms = serie_kpi['KM'].astype(float).values
        _n_m = len(serie_kpi)
        # ── correlación y ajuste lineal litros = a·km + b ──────────────────
        _r = float(np.corrcoef(_kms, _lts)[0,1]) if (_kms.std() > 0 and _lts.std() > 0) else np.nan
        if _n_m >= 3 and _kms.std() > 0:
            _slope, _intercept = np.polyfit(_kms, _lts, 1)
        else:
            _slope, _intercept = np.nan, np.nan
        # meses en que km y litros se movieron en la misma dirección
        _d_k = np.diff(_kms); _d_l = np.diff(_lts)
        _mismo = int(np.sum(np.sign(_d_k) == np.sign(_d_l)))
        _tot_mov = len(_d_k)
        if np.isnan(_r):          _r_txt, _r_col = 'sin variación', ''
        elif _r >= 0.9:           _r_txt, _r_col = 'muy fuerte — los litros siguen a los km', 'kpi-green'
        elif _r >= 0.7:           _r_txt, _r_col = 'fuerte', 'kpi-green'
        elif _r >= 0.4:           _r_txt, _r_col = 'moderada — pesan otros factores', 'kpi-amber'
        elif _r >= 0:             _r_txt, _r_col = 'débil — el consumo no explica los km', 'kpi-red'
        else:                     _r_txt, _r_col = 'inversa — a más km, menos litros', 'kpi-red'
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f'<div class="kpi-card {_r_col}"><div class="kpi-label">🔗 Correlación km ↔ litros</div>'
            f'<div class="kpi-value">{"—" if np.isnan(_r) else f"{_r:.2f}"}</div>'
            f'<div class="kpi-sub">{_r_txt} · R²={0 if np.isnan(_r) else _r**2:.2f} · {_n_m} meses</div></div>',
            unsafe_allow_html=True)
        _marg_txt = '—' if np.isnan(_slope) else f'{_slope*100:.1f}'
        _marg_sub = ('se necesitan 3 meses o más' if np.isnan(_slope)
                     else f'litros por cada 100 km extra · promedio del período {_prom_per:.2f}')
        c2.markdown(
            f'<div class="kpi-card kpi-purple"><div class="kpi-label">📐 Consumo marginal</div>'
            f'<div class="kpi-value">{_marg_txt}</div>'
            f'<div class="kpi-sub">{_marg_sub}</div></div>',
            unsafe_allow_html=True)
        c3.markdown(
            f'<div class="kpi-card"><div class="kpi-label">🔄 Se mueven igual</div>'
            f'<div class="kpi-value">{_mismo}/{_tot_mov}</div>'
            f'<div class="kpi-sub">meses en que km y litros subieron o bajaron juntos</div></div>',
            unsafe_allow_html=True)
        # ── gráfico 1: litros y km por mes (doble eje) ─────────────────────
        def _compacto(v):
            return f'{v/1000:,.1f}k'.replace('.', ',') if v >= 1000 else f'{v:,.0f}'
        def _txt_clave(vals):
            _k = {0, len(vals)-1, int(vals.argmin()), int(vals.argmax())}
            return [_compacto(v) if i in _k else '' for i, v in enumerate(vals)]
        _tp = ['top right' if i == 0 else ('top left' if i == _n_m-1 else 'top center') for i in range(_n_m)]
        fig_lk = go.Figure()
        fig_lk.add_trace(go.Scatter(
            x=_lbl, y=_lts, name='Litros', mode='lines+markers+text',
            text=_txt_clave(_lts), textposition=_tp, textfont=dict(color='#fcd34d', size=10),
            line=dict(color='#f59e0b', width=3, shape='spline'),
            marker=dict(size=7, color='#f59e0b', line=dict(color='#0f172a', width=1.5)),
            fill='tozeroy', fillcolor='rgba(245,158,11,0.15)',
            hovertemplate='%{x}<br>Litros: <b>%{y:,.0f}</b><extra></extra>'))
        fig_lk.add_trace(go.Scatter(
            x=_lbl, y=_kms, name='Kilómetros', mode='lines+markers+text', yaxis='y2',
            text=_txt_clave(_kms), textposition=_tp, textfont=dict(color='#7dd3fc', size=10),
            line=dict(color='#38bdf8', width=3, shape='spline'),
            marker=dict(size=7, color='#38bdf8', line=dict(color='#0f172a', width=1.5)),
            hovertemplate='%{x}<br>KM: <b>%{y:,.0f}</b><extra></extra>'))
        _pad_l = max((_lts.max()-_lts.min())*0.30, _lts.max()*0.05)
        _pad_k = max((_kms.max()-_kms.min())*0.30, _kms.max()*0.05)
        fig_lk.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,41,59,0.6)',
            font=dict(color='#e2e8f0'),
            legend=dict(bgcolor='rgba(15,23,42,0.8)', bordercolor='#334155', borderwidth=1,
                        orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            xaxis=dict(gridcolor='#334155', linecolor='#334155', tickfont=dict(color='#94a3b8', size=10),
                       title=dict(text='Mes', font=dict(color='#94a3b8')), tickangle=-45,
                       categoryorder='array', categoryarray=_lbl),
            yaxis=dict(gridcolor='#334155', linecolor='#334155', tickfont=dict(color='#fbbf24', size=11),
                       title=dict(text='Litros', font=dict(color='#f59e0b')),
                       range=[max(0, _lts.min()-_pad_l), _lts.max()+_pad_l]),
            yaxis2=dict(overlaying='y', side='right', showgrid=False, linecolor='#334155',
                        tickfont=dict(color='#7dd3fc', size=11),
                        title=dict(text='Kilómetros', font=dict(color='#38bdf8')),
                        range=[max(0, _kms.min()-_pad_k), _kms.max()+_pad_k]),
            height=430, margin=dict(l=20, r=25, t=50, b=70), hovermode='x unified')
        st.plotly_chart(fig_lk, use_container_width=True)
        st.caption('Eje izquierdo (ámbar) litros · eje derecho (celeste) kilómetros. '
                   'Si las dos curvas suben y bajan juntas, el consumo acompaña al trabajo hecho; '
                   'si los litros suben más que los km, el rendimiento empeoró.')
        # ── gráfico 2: dispersión km vs litros + recta de ajuste ───────────
        st.markdown('<div class="sec-title">🎯 Relación consumo — cada punto es un mes</div>', unsafe_allow_html=True)
        fig_disp = go.Figure()
        if not np.isnan(_slope):
            _x_fit = np.array([_kms.min(), _kms.max()])
            _y_fit = _slope*_x_fit + _intercept
            fig_disp.add_trace(go.Scatter(
                x=_x_fit, y=_y_fit, mode='lines', name='Tendencia',
                line=dict(color='#f59e0b', width=2, dash='dash'),
                hovertemplate='Tendencia<extra></extra>'))
        fig_disp.add_trace(go.Scatter(
            x=_kms, y=_lts, mode='markers+text', name='Meses',
            text=[l if (i in (0, _n_m-1)) else '' for i, l in enumerate(_lbl)],
            textposition='top center', textfont=dict(color='#94a3b8', size=9),
            marker=dict(size=12, color=list(range(_n_m)),
                        colorscale=[[0,'#1e3a8a'],[0.5,'#3b82f6'],[1,'#7dd3fc']],
                        line=dict(color='#0f172a', width=1.5),
                        colorbar=dict(title=dict(text='Mes', font=dict(color='#94a3b8', size=10)),
                                      tickvals=[0, _n_m-1], ticktext=[_lbl[0], _lbl[-1]],
                                      tickfont=dict(color='#94a3b8', size=9), thickness=12, len=0.7)),
            customdata=[[l, float(v)] for l, v in zip(_lbl, serie_kpi['L100'])],
            hovertemplate='<b>%{customdata[0]}</b><br>KM: %{x:,.0f}<br>Litros: %{y:,.0f}'
                          '<br>L/100km: %{customdata[1]:.2f}<extra></extra>'))
        fig_disp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,41,59,0.6)',
            font=dict(color='#e2e8f0'), showlegend=False,
            xaxis=dict(gridcolor='#334155', linecolor='#334155', tickfont=dict(color='#94a3b8', size=10),
                       title=dict(text='Kilómetros del mes', font=dict(color='#94a3b8'))),
            yaxis=dict(gridcolor='#334155', linecolor='#334155', tickfont=dict(color='#94a3b8', size=10),
                       title=dict(text='Litros del mes', font=dict(color='#94a3b8'))),
            height=430, margin=dict(l=20, r=25, t=40, b=60))
        st.plotly_chart(fig_disp, use_container_width=True)
        if np.isnan(_slope):
            _lectura = 'Con menos de 3 meses no se puede estimar la recta de tendencia.'
        else:
            _base_txt = (f'y un consumo fijo de ~{_intercept:,.0f} L por mes que no depende de los km '
                         f'(ralentí, arranques, maniobras)' if _intercept > 0
                         else 'sin consumo fijo detectable')
            _dif = _slope*100 - _prom_per
            _comp = ('por debajo del promedio del período: los meses de más km rinden mejor'
                     if _dif < -0.5 else
                     ('por encima del promedio: los meses de más km rinden peor' if _dif > 0.5
                      else 'en línea con el promedio del período'))
            _lectura = (f'Cada 100 km adicionales suman ~{_slope*100:.1f} L — {_comp} ({_prom_per:.2f} L/100km) — {_base_txt}. '
                        f'Los puntos por encima de la recta son meses que gastaron más de lo esperado para esos km; '
                        f'los de abajo, menos.')
        st.caption(_lectura)
        with st.expander('📋 Ver tabla mensual de litros, km y L/100 km'):
            _tk = serie_kpi[['LABEL','LITROS','KM','L100']].copy()
            _tk.columns = ['Mes','Litros','KM','L/100 km']
            _tk['Δ Litros %'] = (_tk['Litros'].pct_change()*100).round(1)
            _tk['Δ KM %']     = (_tk['KM'].pct_change()*100).round(1)
            for _c_fmt in ['Litros','KM']:
                _tk[_c_fmt] = _tk[_c_fmt].apply(lambda x: f'{x:,.0f}')
            for _c_fmt in ['Δ Litros %','Δ KM %']:
                _tk[_c_fmt] = _tk[_c_fmt].apply(lambda x: '—' if pd.isna(x) else f'{x:+.1f}%')
            st.dataframe(_tk, use_container_width=True, hide_index=True)
    st.divider()
    st.markdown(f'<div class="sec-title">Rendimiento por Modelo — {anio_sel}</div>', unsafe_allow_html=True)
    def stats_modelo(patentes_lista):
        if 'DOMINIO' not in df.columns: return {'l100':0,'lts':0,'kms':0,'n':0}
        sub = df[df['DOMINIO'].isin(patentes_lista)]
        if sub.empty: return {'l100':0,'lts':0,'kms':0,'n':0}
        lts=sub['LITROS'].sum(); kms=sub['KM'].sum()
        return {'l100':round(lts/kms*100,2) if kms>0 else 0,'lts':lts,'kms':kms,'n':sub['DOMINIO'].nunique()}
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
            sc1.metric(f'{modelo} — L/100km',f"{s['l100']:.1f}" if s['l100']>0 else '—')
            sc2.metric('Unidades',f"{s['n']}")
            sc3.metric(f'Litros {anio_sel}',f"{s['lts']:,.0f}" if s['lts']>0 else '—')
            st.caption(f"Patentes: {pats_label} | {s['kms']:,.0f} km")
    st.divider()
    st.markdown(f'<div class="sec-title">Ranking de Eficiencia — {anio_sel}</div>', unsafe_allow_html=True)
    rcol1,rcol2 = st.columns(2)
    def render_ranking(col,titulo,df_rank,color_fn):
        with col:
            st.markdown(f'**{titulo}**')
            if df_rank.empty: st.info('Sin datos.'); return
            vmin,vmax=df_rank['L100KM'].min(),df_rank['L100KM'].max()
            rh='<div style="background:#1e293b;border-radius:12px;padding:16px;">'
            rh+='<div style="font-size:.72rem;display:flex;justify-content:space-between;color:#94a3b8;margin-bottom:6px;"><span>Unidad</span><span>L/100km</span></div>'
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
                       lambda i:'#22c55e' if i<=3 else ('#f59e0b' if i<=6 else '#ef4444'))
        render_ranking(rcol2,'TOP 10 menos eficientes (mayor L/100km)',base.sort_values('L100KM',ascending=False).head(10),
                       lambda i:'#ef4444' if i<=3 else ('#f59e0b' if i<=6 else '#22c55e'))
    st.divider()
    st.markdown(f'<div class="sec-title">📊 Índice de Eficiencia Relativa (IER v8) — {anio_sel}</div>', unsafe_allow_html=True)
    tiene_vel       = total_excesos>0
    tiene_manejo_d  = (not df_ier.empty and 'TIENE_MANEJO' in df_ier.columns and bool(df_ier['TIENE_MANEJO'].any()))
    tiene_ral_d     = (not df_ier.empty and 'TIENE_RALENTI' in df_ier.columns and bool(df_ier['TIENE_RALENTI'].any()))
    pond_txt = (
        f"<b>40%</b> Consumo real vs. esperado ⛽ &nbsp;·&nbsp; "
        f"<b>25%</b> Score conducción {'🎯' if tiene_manejo_d else '⚠️ sin datos'} &nbsp;·&nbsp; "
        f"<b>20%</b> % Ralentí {'⏱️' if tiene_ral_d else '⚠️ sin datos'} &nbsp;·&nbsp; "
        f"<b>15%</b> Severidad vel. {'✅' if tiene_vel else '⚠️'}"
    )
    _pend = ier_info.get('pendiente_carga', IER_PENDIENTE_CARGA_DEF)
    if ier_info.get('tiene_peso'):
        _carga_txt = (f"+{_pend*100:.1f}% de consumo esperado por cada tonelada extra por viaje ({ier_info.get('pendiente_fuente','')}), "
                      f"tope ±{IER_TOPE_CARGA*100:.0f}%. Pesos fuera de {CARGA_MIN_TON_VIAJE:.0f}–{CARGA_MAX_TON_VIAJE:.0f} t por viaje se descartan "
                      f"({viajes_ier_diag.get('n_descartados',0)} de {viajes_ier_diag.get('n_viajes',0)} viajes).")
    else:
        _carga_txt = "⚠️ sin datos de peso por viaje — no se ajusta por carga."
    if ier_info.get('tiene_rutas'):
        _ruta_txt = f"{ier_info.get('rutas_ajustadas',0)} rutas con ajuste (solo rutas hechas por ≥2 camiones), tope ±{IER_TOPE_RUTA*100:.0f}%."
    else:
        _ruta_txt = "⚠️ la planilla de cargas no trae columnas ORIGEN/DESTINO — no se ajusta por ruta."
    st.markdown(f"""<div class="ier-info-box">
    <b>¿Qué es el IER v8?</b> Mide solo lo que controla el chofer. Cada camión se compara contra los de <b>su mismo modelo</b>.<br>
    <b>Consumo esperado:</b> lo que debería gastar ese camión según lo que el chofer no controla:
    mediana de L/100km de su modelo <b>en el mismo mes</b> (viento, temperatura, época) × ajuste por <b>peso promedio por viaje</b> × ajuste por <b>ruta</b>.
    Se puntúa el desvío del consumo real contra ese esperado.<br>
    <b>Peso:</b> {_carga_txt}<br>
    <b>Rutas:</b> {_ruta_txt}<br>
    <b>Pocos km:</b> con menos de ~{IER_KM_CONFIANZA:,} km el desvío se suaviza hacia 0 (IER cerca de 100).<br>
    <b>Score conducción:</b> SCORE GENERAL del Google Sheet de manejo. <b>Ralentí:</b> % de litros en ralentí (menor = mejor).
    <b>Velocidad:</b> severidad (km/h acumulados sobre el límite).<br>
    <b>Ponderación:</b>&nbsp;{pond_txt}<br>
    <b>Datos faltantes:</b> si falta un componente, su peso se reparte entre los otros componentes de conducta, no al consumo.<br>
    <b>Escala:</b>&nbsp;🟢 Eficiente ≥105 &nbsp;·&nbsp; 🟡 Normal 95–105 &nbsp;·&nbsp; 🟠 Atención 85–95 &nbsp;·&nbsp; 🔴 Crítico &lt;85
    </div>""", unsafe_allow_html=True)
    with st.expander('🧮 ¿Cómo se calcula el IER? (explicado simple)'):
        st.markdown(f"""<div class="ier-method-box">
        <b>La idea:</b> el IER mide solo lo que depende del chofer. Lo que no depende de él
        (el camión, el viento, el peso, la ruta) se descuenta antes de comparar.<br><br>
        <b>Paso 1 — ¿Cuánto debería gastar este camión?</b><br>
        • Se toma lo que gastaron en promedio los camiones <b>del mismo modelo, en el mismo mes</b>.
        Si en julio hubo mucho viento, todos gastaron más y nadie sale perjudicado.<br>
        • Si llevó <b>más peso por viaje</b> que el resto, se le permite gastar un poco más
        (y si llevó menos, un poco menos). Como máximo ±{IER_TOPE_CARGA*100:.0f}%.<br>
        • Si hizo <b>rutas más pesadas</b> (subidas, ripio, ciudad), también se le permite gastar más. Como máximo ±{IER_TOPE_RUTA*100:.0f}%.<br>
        • Un peso cargado con error (por ejemplo 280 t en un viaje) se ignora: ese viaje no cuenta para el peso.<br><br>
        <b>Paso 2 — ¿Cuánto gastó de verdad?</b><br>
        Se compara el consumo real contra el esperado. Ejemplo: esperado 34 L/100km, real 33 → gastó <b>3% menos</b> de lo esperado (bien).<br><br>
        <b>Paso 3 — Se suman los hábitos del chofer</b><br>
        • <b>40%</b> Consumo real vs. esperado (paso 2)<br>
        • <b>25%</b> Score de conducción (aceleraciones, frenadas, uso del motor)<br>
        • <b>20%</b> Ralentí: porcentaje del combustible gastado con el camión parado y el motor en marcha<br>
        • <b>15%</b> Velocidad: cuánto y cuántas veces pasó el límite<br>
        Si falta algún dato (por ejemplo, no hay score de conducción), ese porcentaje se reparte entre los otros hábitos.<br><br>
        <b>Paso 4 — Resultado</b><br>
        Cada camión se compara solo con los de su modelo. <b>100 = igual al promedio</b>. Más de 100 = mejor que el promedio, menos de 100 = peor.<br>
        Si el camión hizo pocos km, su resultado se acerca a 100, porque con pocos datos no se puede afirmar que sea bueno ni malo.<br><br>
        🟢 105 o más: Eficiente &nbsp;·&nbsp; 🟡 95–105: Normal &nbsp;·&nbsp; 🟠 85–95: Atención &nbsp;·&nbsp; 🔴 menos de 85: Crítico
        </div>""", unsafe_allow_html=True)
    with st.expander('ℹ️ ¿Por qué Z-Score + Tanh? (metodología)'):
        st.markdown("""<div class="ier-method-box">
        <b>Problema del ratio simple:</b><br>
        • Un camión 50% mejor: ratio=2.0 | Un camión 50% peor: ratio=0.67 — asimetría injusta<br><br>
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
            if v>=105: return '#22c55e'
            elif v>=95: return '#f59e0b'
            elif v>=85: return '#f97316'
            else: return '#ef4444'
        orden_ier = st.radio(
            'Orden del ranking IER',
            options=['🏆 Flota completa (mayor → menor)', '🚛 Agrupado por modelo'],
            index=0, horizontal=True, key='ier_sort_mode'
        )
        if orden_ier.startswith('🏆'):
            df_ier_sorted = df_ier.sort_values('IER', ascending=True)
        else:
            df_ier_sorted = df_ier.sort_values(['MODELO','IER'],ascending=[True,False])
        fig_ier=go.Figure()
        MODELO_COLOR={'S-Way':'#60a5fa','Scania':'#f97316','Stralis':'#a78bfa'}
        for modelo in MODELO_COLOR:
            subset=df_ier_sorted[df_ier_sorted['MODELO']==modelo]
            if subset.empty: continue
            hover=[]
            for _,row in subset.iterrows():
                if row['TIENE_CONSUMO']:
                    cons_txt=(f"real {row['L100KM']:.1f} vs esperado {row['L100KM_ESP']:.1f} L/100km ({row['DESVIO_PCT']:+.1f}%)")
                else:
                    cons_txt="sin datos válidos"
                ton_txt=(f"{row['TON_VIAJE']:.1f} t/viaje" if pd.notnull(row['TON_VIAJE']) else "peso sin dato válido")
                ral_txt=(f"{row['RALENTI_PCT']:.1f}% · score: {row['SCORE_RALENTI']:.2f}" if row['TIENE_RALENTI'] else "sin datos")
                severidad = row.get('SEVERIDAD', 0)
                sc_man = row.get('SCORE_CONDUCCION', np.nan)
                sc_man_txt = f"{sc_man:.2f}/10 · score: {row['SCORE_MANEJO']:.2f}" if pd.notnull(sc_man) else "sin datos"
                hover.append(f"<b>{row['DOMINIO']}</b> ({row['MODELO']})<br>IER: <b>{row['IER']:.1f}</b> — {row['CLASIFICACION']}<br>"
                             f"Consumo (40%): {cons_txt} · score: {row['SCORE_CONSUMO']:.2f}<br>"
                             f"&nbsp;&nbsp;ajustes: carga {row['AJ_CARGA_PCT']:+.1f}% ({ton_txt}) · ruta {row['AJ_RUTA_PCT']:+.1f}%<br>"
                             f"Conducción (25%): {sc_man_txt}<br>"
                             f"Ralentí (20%): {ral_txt}<br>"
                             f"Velocidad (15%): severidad {severidad:.0f} km/h acum. · {int(row['EXCESOS'])} eventos · score: {row['SCORE_VEL']:.2f}<br>"
                             f"KM: {row['KM']:,.0f}")
            fig_ier.add_trace(go.Bar(y=subset['DOMINIO'],x=subset['IER'],name=modelo,orientation='h',
                marker=dict(color=[ier_bar_color(v) for v in subset['IER']],line=dict(color='rgba(255,255,255,0.15)',width=1)),
                text=[f"{v:.1f}" for v in subset['IER']],textposition='outside',textfont=dict(color='#e2e8f0',size=10),
                hovertemplate='%{customdata}<extra></extra>',customdata=hover))
        ier_min=max(50,df_ier_sorted['IER'].min()-10); ier_max=min(175,df_ier_sorted['IER'].max()+25)
        fig_ier.add_vline(x=100,line_dash='solid',line_color='#f59e0b',line_width=2.5,
                          annotation_text='Base 100',annotation_position='top',annotation_font_color='#fbbf24',annotation_font_size=11)
        fig_ier.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),barmode='overlay',
            xaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8'),title=dict(text='IER  (100 = promedio de su modelo)',font=dict(color='#94a3b8')),range=[ier_min,ier_max]),
            yaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8',size=10),categoryorder='array',categoryarray=df_ier_sorted['DOMINIO'].tolist()),
            height=max(380,len(df_ier_sorted)*44),margin=dict(l=10,r=130,t=60,b=30),showlegend=False)
        st.plotly_chart(fig_ier, use_container_width=True)
        st.caption('Verde = mejor que su modelo · Rojo = peor · Línea amarilla = base 100 · Hover para detalle completo')
        with st.expander('📋 Ver tabla detallada IER (todos los componentes)'):
            show_cols=['DOMINIO','MODELO','IER','CLASIFICACION','L100KM','L100KM_ESP','DESVIO_PCT',
                       'AJ_CARGA_PCT','TON_VIAJE','N_VIAJES','AJ_RUTA_PCT','RUTA_PRINCIPAL',
                       'SCORE_CONDUCCION','RALENTI_PCT','RALENTI_MOD','EXCESOS','SEVERIDAD','SEVERIDAD_MOD','VEL_MAX','KM',
                       'SCORE_CONSUMO','SCORE_MANEJO','SCORE_RALENTI','SCORE_VEL']
            ier_show=df_ier[[c for c in show_cols if c in df_ier.columns]].copy()
            col_rename={'DOMINIO':'Patente','MODELO':'Modelo','IER':'IER','CLASIFICACION':'Clasificación',
                'L100KM':'L/100km real','L100KM_ESP':'L/100km esperado','DESVIO_PCT':'Desvío %',
                'AJ_CARGA_PCT':'Ajuste carga %','TON_VIAJE':'t/viaje','N_VIAJES':'Viajes',
                'AJ_RUTA_PCT':'Ajuste ruta %','RUTA_PRINCIPAL':'Ruta principal',
                'SCORE_CONDUCCION':'Score Conducción (/10)',
                'RALENTI_PCT':'% Ralentí','RALENTI_MOD':'% Ralentí prom mod.',
                'EXCESOS':f'Cant. Excesos >{LIMITE_VELOCIDAD}km/h',
                'SEVERIDAD':'Severidad total (km/h acum.)','SEVERIDAD_MOD':'Sev. total prom mod.',
                'VEL_MAX':'Vel. Máx (km/h)','KM':'KM total',
                'SCORE_CONSUMO':'S.Consumo (40%)','SCORE_MANEJO':'S.Manejo (25%)',
                'SCORE_RALENTI':'S.Ralentí (20%)','SCORE_VEL':'S.Vel (15%)'}
            ier_show=ier_show.rename(columns=col_rename)
            for c in ['IER','L/100km real','L/100km esperado','Desvío %','Ajuste carga %','t/viaje','Ajuste ruta %',
                      '% Ralentí','% Ralentí prom mod.','Severidad total (km/h acum.)','Sev. total prom mod.','Score Conducción (/10)']:
                if c in ier_show.columns: ier_show[c]=ier_show[c].round(2)
            for c in ['S.Consumo (40%)','S.Manejo (25%)','S.Ralentí (20%)','S.Vel (15%)']:
                if c in ier_show.columns: ier_show[c]=ier_show[c].round(3)
            if 'KM total' in ier_show.columns: ier_show['KM total']=ier_show['KM total'].apply(lambda x:f'{x:,.0f}')
            st.dataframe(ier_show, use_container_width=True, hide_index=True)
            st.caption('Desvío % = (real − esperado) / esperado. Negativo = gastó menos de lo esperado (bueno). '
                       'Esperado = mediana del modelo en el mismo mes × ajuste carga × ajuste ruta.')
    else:
        st.info('Sin datos suficientes para calcular el IER.')
    if not df_vel_filtrado.empty and 'DOMINIO' in df_vel_filtrado.columns:
        # ══════════════════════════════════════════════════════════════════════
        # PODIO TOP 5 IER (SCREENSHOT READY)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown(f'<div class="sec-title">📸 Salón de la Fama: TOP 5 IER (Listo para Captura)</div>', unsafe_allow_html=True)

        if not df_ier.empty and len(df_ier) >= 5:
            # Tomamos el top 5 y agregamos medallas
            top5 = df_ier.head(5).copy()
            medallas = ['🥇', '🥈', '🥉', '🏅', '🏅']
            top5['DOMINIO_MEDALLA'] = [f"{medallas[i]}  {dom}" for i, dom in enumerate(top5['DOMINIO'])]

            # Invertimos el dataframe para que el #1 quede arriba del todo en Plotly
            top5 = top5.iloc[::-1]
            # Colores: 5to, 4to, Bronce, Plata, Oro (en ese orden porque está invertido)
            colores_podio = ['#1e293b', '#334155', '#b45309', '#94a3b8', '#fbbf24']
            fig_top5 = go.Figure(go.Bar(
                x=top5['IER'],
                y=top5['DOMINIO_MEDALLA'],
                orientation='h',
                marker=dict(
                    color=colores_podio,
                    line=dict(color='rgba(255,255,255,0.1)', width=1)
                ),
                # Texto grande adentro de la barra
                text=[f"<b>{ier:.1f}</b>" for ier in top5['IER']],
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(color='#ffffff', size=22, family="Arial Black"),
                hoverinfo='none' # Desactivamos hover para que no moleste en la captura
            ))
            # Ajustamos el layout para que quede súper limpio tipo tarjeta
            fig_top5.update_layout(
                title=dict(
                    text=f"🏆 TOP 5 FLOTA LAD - ÍNDICE DE EFICIENCIA ({anio_sel})",
                    font=dict(size=20, color='#f1f5f9', weight="bold"),
                    x=0.5,
                    y=0.9
                ),
                paper_bgcolor='#0f172a',
                plot_bgcolor='#0f172a',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), # Ocultamos eje X
                yaxis=dict(showgrid=False, tickfont=dict(size=18, weight="bold", color='#e2e8f0')),
                margin=dict(l=20, r=20, t=70, b=20),
                height=380,
                bargap=0.25
            )

            # Le agregamos una anotación chiquita a cada barra con el modelo y el L/100km real
            for i, row in top5.reset_index().iterrows():
                fig_top5.add_annotation(
                    x=row['IER'] - 3, # Lo tiramos un poco a la izquierda del borde de la barra
                    y=i,
                    text=f"{row['MODELO']} | {row['L100KM']:.1f} L/100km",
                    showarrow=False,
                    font=dict(size=13, color='rgba(255,255,255,0.7)'),
                    xanchor='right'
                )
            st.plotly_chart(fig_top5, use_container_width=True, config={'displayModeBar': False}) # Ocultamos la barrita superior de plotly
            st.caption("Tip: Usá `Windows + Shift + S` (o `Cmd + Shift + 4` en Mac) para recortar y compartir este podio.")
        else:
            st.info("No hay suficientes datos procesados para armar el Top 5.")
        st.divider()
        st.markdown(f'<div class="sec-title">🚨 Ranking Severidad Velocidad >{LIMITE_VELOCIDAD} km/h — {anio_sel}</div>', unsafe_allow_html=True)
        st.caption(f'Métrica: suma total de km/h sobre el límite (frecuencia × magnitud). 5 eventos a 95 km/h (sum=35) es más grave que 10 eventos a 89 km/h (sum=10).')
        vel_rank=(df_vel_filtrado.groupby('DOMINIO').agg(
            CANTIDAD=('DOMINIO','count'),
            VEL_MAX=('VELOCIDAD','max'),
            VEL_PROM=('VELOCIDAD','mean'),
            SEVERIDAD=('EXCESO_KMH','sum')
        ).reset_index().sort_values('SEVERIDAD',ascending=False))
        vel_rank['MODELO']=vel_rank['DOMINIO'].apply(asignar_modelo)
        fig_vel=go.Figure([go.Bar(x=vel_rank['DOMINIO'],y=vel_rank['SEVERIDAD'].round(0),
            marker_color=['#ef4444' if v==vel_rank['SEVERIDAD'].max() else '#f97316' for v in vel_rank['SEVERIDAD']],
            text=vel_rank['SEVERIDAD'].round(0).astype(int),textposition='outside',textfont=dict(color='#e2e8f0',size=10),
            hovertemplate='<b>%{x}</b><br>Severidad total: +%{y:.0f} km/h acumulados sobre límite<extra></extra>')])
        fig_vel.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8',size=10),tickangle=-45),
            yaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8'),title=dict(text=f'km/h acumulados sobre {LIMITE_VELOCIDAD} km/h',font=dict(color='#94a3b8'))),
            height=380,margin=dict(l=10,r=10,t=20,b=80),showlegend=False)
        st.plotly_chart(fig_vel, use_container_width=True)
        with st.expander('Ver tabla de excesos por unidad'):
            vel_show=vel_rank[['DOMINIO','MODELO','SEVERIDAD','CANTIDAD','VEL_MAX','VEL_PROM']].copy()
            vel_show.columns=['Patente','Modelo',f'Severidad total (km/h acum.)',f'Cant. eventos >{LIMITE_VELOCIDAD}km/h','Vel. Máx (km/h)','Vel. Prom (km/h)']
            vel_show['Vel. Máx (km/h)']=vel_show['Vel. Máx (km/h)'].round(1)
            vel_show['Vel. Prom (km/h)']=vel_show['Vel. Prom (km/h)'].round(1)
            vel_show['Severidad total (km/h acum.)']=vel_show['Severidad total (km/h acum.)'].round(1)
            st.dataframe(vel_show, use_container_width=True, hide_index=True)
    st.divider()
    _d_ti = st.session_state.get('desde_periodo', None); _h_ti = st.session_state.get('hasta_periodo', None)
    _rango_ti = (f'{_d_ti}' if _d_ti==_h_ti else f'{_d_ti} a {_h_ti}') if _d_ti is not None else str(anio_sel)
    st.markdown(f'<div class="sec-title">🎯 Score Conducción vs Consumo (L/100km) — {_rango_ti}</div>', unsafe_allow_html=True)
    if not df_manejo_filtrado.empty and 'SCORE_CONDUCCION' in df_manejo_filtrado.columns and 'L100KM' in df.columns:
        _sc_man  = df_manejo_filtrado.groupby('DOMINIO')['SCORE_CONDUCCION'].mean()
        _sc_l100 = df[df['L100KM']>0].groupby('DOMINIO')['L100KM'].mean()
        _sc = pd.concat([_sc_man.rename('SCORE'), _sc_l100.rename('L100KM')], axis=1).dropna().reset_index()
        if len(_sc)>=2:
            _sc['MODELO']=_sc['DOMINIO'].apply(asignar_modelo)
            _MC={'S-Way':'#60a5fa','Scania':'#f97316','Stralis':'#a78bfa'}
            fig_sc=go.Figure()
            for _m,_c in _MC.items():
                _sub=_sc[_sc['MODELO']==_m]
                if _sub.empty: continue
                fig_sc.add_trace(go.Scatter(x=_sub['SCORE'],y=_sub['L100KM'],mode='markers+text',
                    name=_m,text=_sub['DOMINIO'],textposition='top center',textfont=dict(size=9,color='#cbd5e1'),
                    marker=dict(size=15,color=_c,line=dict(color='white',width=1.5)),
                    hovertemplate='<b>%{text}</b> ('+_m+')<br>Score conducción: %{x:.2f}/10<br>L/100km: %{y:.2f}<extra></extra>'))
            if len(_sc)>=3:
                _z=np.polyfit(_sc['SCORE'],_sc['L100KM'],1)
                _xs=np.linspace(_sc['SCORE'].min(),_sc['SCORE'].max(),50)
                _corr=_sc['SCORE'].corr(_sc['L100KM'])
                fig_sc.add_trace(go.Scatter(x=_xs,y=_z[0]*_xs+_z[1],mode='lines',name=f'Tendencia (r={_corr:.2f})',
                    line=dict(color='#f59e0b',width=2,dash='dash'),hoverinfo='skip'))
            fig_sc.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
                legend=dict(bgcolor='rgba(15,23,42,0.8)',bordercolor='#334155',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
                xaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8'),title=dict(text='Score conducción (/10) — mayor = mejor →',font=dict(color='#94a3b8'))),
                yaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8'),title=dict(text='L/100km — menor = más eficiente ↓',font=dict(color='#94a3b8'))),
                height=460,margin=dict(l=10,r=10,t=50,b=50))
            st.plotly_chart(fig_sc, use_container_width=True)
            st.caption('Cada punto = una patente · Tendencia negativa esperada: mejor conducción → menor consumo')
            # Transparencia: el scatter solo puede mostrar patentes que tengan score de manejo Y consumo.
            _modelos_presentes = set(_sc['MODELO'].unique())
            _modelos_faltantes = [m for m in ('S-Way','Scania','Stralis') if m not in _modelos_presentes]
            if _modelos_faltantes:
                st.caption(
                    f'ℹ️ Solo aparecen modelos con **score de conducción cargado** en el período: '
                    f'{", ".join(sorted(_modelos_presentes))}. '
                    f'No se muestran {", ".join(_modelos_faltantes)} porque no tienen datos de manejo '
                    f'(revisá la carga de las hojas en la pestaña **Diagnóstico**).'
                )
        else:
            st.info('Se necesitan al menos 2 patentes con score de conducción y consumo para el scatter.')
    else:
        st.info('Sin datos de score de conducción para cruzar con el consumo.')
    st.divider()
    st.markdown(f'<div class="sec-title">🔧 Gasto en Arreglos por Patente — {anio_sel}</div>', unsafe_allow_html=True)
    if df_arreglos_raw is not None and not df_arreglos_raw.empty:
        _arr = df_arreglos_raw.copy()
        _d = st.session_state.get('desde_periodo', None)
        _h = st.session_state.get('hasta_periodo', None)
        _n_raw = len(_arr)
        # 1. APLICAR FILTROS DE PATENTE Y MARCA
        if patentes_sel:
            _arr = _arr[_arr['DOMINIO'].isin(patentes_sel)]
        elif marcas_sel and 'MARCA' in df_full.columns:
            # Obtenemos qué dominios de la telemetría corresponden a las marcas seleccionadas
            dominios_validos = df_full[df_full['MARCA'].isin(marcas_sel)]['DOMINIO'].unique()
            _arr = _arr[_arr['DOMINIO'].isin(dominios_validos)]
        _n_pat = len(_arr)
        # 2. APLICAR FILTRO DE FECHAS
        # Solo filtramos por fecha si la planilla realmente trae fechas válidas;
        # si no hay fechas parseables, mostramos todos los arreglos matcheados.
        _tiene_fechas = ('MES' in _arr.columns) and _arr['MES'].notna().any()
        _sin_fechas = False
        if _tiene_fechas and _d is not None and _h is not None:
            _arr = _arr[_arr['MES'].notna() & (_arr['MES'] >= _d) & (_arr['MES'] <= _h)]
        elif _tiene_fechas and 'FECHA' in _arr.columns:
            _arr = _arr[_arr['FECHA'].dt.year == anio_sel]
        else:
            _sin_fechas = True
        _n_fecha = len(_arr)
        if _arr.empty:
            st.info('Sin gastos de arreglos para el período o las unidades seleccionadas.')
            # Embudo de diagnóstico: muestra en qué filtro se pierden las filas
            _mes_raw = df_arreglos_raw['MES'].dropna()
            _rango_txt = f'{_mes_raw.min()} → {_mes_raw.max()}' if not _mes_raw.empty else 'sin fechas válidas en la planilla'
            _periodo_txt = f'{_d} a {_h}' if (_d is not None and _h is not None) else str(anio_sel)
            st.caption(
                f'🔎 Diagnóstico del filtro · registros en planilla: **{_n_raw}** → tras filtro patente/marca: **{_n_pat}** '
                f'→ tras filtro de fechas ({_periodo_txt}): **{_n_fecha}**. '
                f'Rango de fechas cargado en la planilla de arreglos: **{_rango_txt}**. '
                f'Si el rango no cae dentro de {_periodo_txt}, ajustá el filtro *Desde/Hasta* en la barra lateral.'
            )
        else:
            if _sin_fechas:
                st.warning('⚠️ La planilla de arreglos no tiene fechas válidas, así que se muestran **todos** los arreglos de las unidades seleccionadas (sin filtrar por el período Desde/Hasta). Cargá la columna de fecha en la planilla para poder filtrar por mes.')
            _gp = (_arr.groupby('DOMINIO').agg(GASTO=('MONTO','sum'),N=('MONTO','count'))
                       .reset_index().sort_values('GASTO',ascending=False))
            _gp['MODELO']=_gp['DOMINIO'].apply(asignar_modelo)
            _tot=_gp['GASTO'].sum()
            a1,a2,a3=st.columns(3)
            kpi(a1,'kpi-red','💸 Gasto total arreglos',f'${_tot/1e6:.2f}M',f'{int(_gp["N"].sum())} arreglos')
            kpi(a2,'kpi-amber','🔧 Patente que más gastó',f'{_gp.iloc[0]["DOMINIO"]}',f'${_gp.iloc[0]["GASTO"]:,.0f}')
            kpi(a3,'','📊 Promedio por patente',f'${_gp["GASTO"].mean():,.0f}',f'{len(_gp)} patentes')
            fig_arr=go.Figure([go.Bar(x=_gp['DOMINIO'],y=_gp['GASTO'],
                marker_color=['#ef4444' if v==_gp['GASTO'].max() else '#f97316' for v in _gp['GASTO']],
                text=[f'${v:,.0f}' for v in _gp['GASTO']],textposition='outside',textfont=dict(color='#e2e8f0',size=10),
                customdata=_gp['N'],
                hovertemplate='<b>%{x}</b><br>Gasto: <b>$%{y:,.0f}</b><br>Arreglos: %{customdata}<extra></extra>')])
            _prom=_gp['GASTO'].mean()
            fig_arr.add_hline(y=_prom,line_dash='dot',line_color='#f59e0b',line_width=2,annotation_text=f'Prom: ${_prom:,.0f}',annotation_position='top right',annotation_font_color='#fbbf24',annotation_font_size=11)
            fig_arr.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
                xaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8',size=10),tickangle=-45),
                yaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8'),title=dict(text='Gasto en arreglos ($)',font=dict(color='#94a3b8'))),
                height=420,margin=dict(l=10,r=10,t=30,b=80),showlegend=False)
            st.plotly_chart(fig_arr, use_container_width=True)
            st.caption('🔴 Mayor gasto · 🟠 Resto · Línea amarilla = promedio flota · Fuente: Google Sheet de arreglos')
            with st.expander('📋 Ver detalle de arreglos'):
                _cols_det=[c for c in ['FECHA','DOMINIO','MONTO','DESCRIPCION'] if c in _arr.columns]
                _det=_arr.sort_values('FECHA',ascending=False)[_cols_det].copy()
                _det=_det.rename(columns={'FECHA':'Fecha','DOMINIO':'Patente','MONTO':'Monto','DESCRIPCION':'Descripción'})
                if 'Monto' in _det.columns: _det['Monto']=_det['Monto'].apply(lambda x:f'${x:,.0f}')
                st.dataframe(_det, use_container_width=True, hide_index=True)
    else:
        _arr_err = arreglos_diag.get('err','') if isinstance(arreglos_diag,dict) else ''
        st.info(f'⚠️ No hay datos de arreglos disponibles. ({_arr_err}) Revisá la pestaña 🔧 Diagnóstico.')
    st.divider()
    with st.expander(f'Ver datos completos {anio_sel}'):
        cols_s=[c for c in ['DOMINIO','MARCA','MODELO','FECHA','KM','LITROS','L100KM','RALENTI_PCT','RALENTI'] if c in df.columns]
        st.dataframe(df[cols_s], use_container_width=True, height=380)
    st.caption(f'Datos {anio_sel}: Google Sheets Expreso Diemar | Precio: {precio_fuente} | Excesos: satelital >{LIMITE_VELOCIDAD} km/h | Actualización cada 10 min')
# ═══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 2 — MODELO PREDICTIVO
# ═══════════════════════════════════════════════════════════════════════════════
elif pg == "Modelo Predictivo":
    col_logo2,col_title2=st.columns([1,5])
    with col_logo2: st.image(LOGO_URL, width=130)
    with col_title2:
        st.markdown("""<div style='padding:8px 0;'>
        <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>Modelo Predictivo &mdash; LAD</div>
        <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Entrenado con todo el histórico &middot; Regresión polinomial &middot; Simulador What-If</div>
        </div>""", unsafe_allow_html=True)
    st.markdown(f'<span class="price-badge">&#9981; Precio gasoil: <b>${precio_gasoil:,.0f}/L</b></span>&nbsp;&nbsp;<span style="font-size:.75rem;color:#94a3b8;">Fuente: {precio_fuente}</span>', unsafe_allow_html=True)
    anos_en_hist=(sorted(df_full_clean['FECHA'].dt.year.unique().tolist()) if 'FECHA' in df_full_clean.columns else [])
    anos_str=" · ".join(str(a) for a in anos_en_hist)
    st.markdown(f'<div class="training-badge">🧠 Modelo entrenado con {n_meses_entrenamiento} meses históricos ({anos_str})</div>', unsafe_allow_html=True)
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
        def _pred_card(c, mes, l100_p, lts_p, costo_p):
            c.markdown(f'''<div class="kpi-card kpi-purple" style="padding:16px 18px;">
              <div class="kpi-label">Predicción {mes}</div>
              <div style="font-size:1.55rem;font-weight:800;color:#f1f5f9;line-height:1.15;margin-top:2px;">{l100_p:.2f} <span style="font-size:.85rem;color:#94a3b8;font-weight:600;">L/100km</span></div>
              <div style="font-size:.78rem;color:#94a3b8;margin-top:6px;">{lts_p:,.0f} L · ${costo_p/1e6:.2f}M</div>
            </div>''', unsafe_allow_html=True)
        for _row_start in range(0, n_pred, 4):
            _cols = st.columns(4)
            for _c, _idx in zip(_cols, range(_row_start, min(_row_start+4, n_pred))):
                _mes=meses_fut[_idx]; _l100p=pred_l100[_idx]; _ltsp=pred_lts[_idx]
                _pred_card(_c, _mes, _l100p, _ltsp, _ltsp*precio_gasoil)
        st.divider()
        st.markdown(f'<div class="sec-title">🎯 Consumo Esperado Próximo Mes — {meses_fut[0]}</div>', unsafe_allow_html=True)
        _next_l100=float(pred_l100[0]); _next_lts=float(pred_lts[0]); _next_costo=_next_lts*precio_gasoil
        _sigma=float(std_res); _ult_real=float(hist['L100'].iloc[-1])
        _delta_vs_ult=_next_l100-_ult_real
        _low1,_high1=_next_l100-_sigma,_next_l100+_sigma
        st.markdown(f"""<div class="ier-info-box">
        El modelo proyecta para <b>{meses_fut[0]}</b> un consumo de <b>{_next_l100:.2f} L/100km</b>.
        Según el error típico del modelo (σ residuos = {_sigma:.2f}), lo esperable es que el valor real se
        <b>desvíe ±{_sigma:.2f} L/100km</b> respecto de esa proyección — es decir, debería ubicarse entre
        <b>{_low1:.2f}</b> y <b>{_high1:.2f} L/100km</b> (~68% de probabilidad).
        Un desvío mayor a <b>±{1.5*_sigma:.2f}</b> (±1.5σ) se considera anómalo y dispara la alerta.
        </div>""", unsafe_allow_html=True)
        gc1,gc2=st.columns([3,2])
        with gc1:
            _ax_half=max(3*_sigma,1.0); _ax_min=max(0,_next_l100-_ax_half); _ax_max=_next_l100+_ax_half
            fig_g=go.Figure(go.Indicator(
                mode='gauge+number+delta',value=_next_l100,
                number={'suffix':' L/100km','font':{'color':'#f1f5f9','size':32}},
                delta={'reference':_ult_real,'increasing':{'color':'#ef4444'},'decreasing':{'color':'#22c55e'},'suffix':' vs último mes'},
                title={'text':f'Proyección {meses_fut[0]}','font':{'color':'#94a3b8','size':14}},
                gauge={'axis':{'range':[_ax_min,_ax_max],'tickcolor':'#94a3b8','tickfont':{'color':'#94a3b8'}},
                       'bar':{'color':'#60a5fa'},'bgcolor':'rgba(30,41,59,0.6)','borderwidth':0,
                       'steps':[{'range':[_ax_min,_low1],'color':'rgba(34,197,94,0.18)'},
                                {'range':[_low1,_high1],'color':'rgba(59,130,246,0.30)'},
                                {'range':[_high1,_ax_max],'color':'rgba(239,68,68,0.18)'}],
                       'threshold':{'line':{'color':'#f59e0b','width':3},'thickness':0.85,'value':_ult_real}}))
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#e2e8f0'),height=300,margin=dict(l=20,r=20,t=50,b=10))
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption(f'Banda azul = rango esperado ±1σ ({_low1:.2f}–{_high1:.2f}) · Línea amarilla = último mes real ({_ult_real:.2f}) · Verde/rojo = desvío favorable/desfavorable')
        with gc2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Consumo esperado</div><div class="kpi-value">{_next_l100:.2f}</div><div class="kpi-sub">L/100km · {meses_fut[0]} · {_delta_vs_ult:+.2f} vs último mes</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-card kpi-purple"><div class="kpi-label">Desvío esperado (±1σ)</div><div class="kpi-value">±{_sigma:.2f}</div><div class="kpi-sub">rango {_low1:.2f} – {_high1:.2f} L/100km</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-card kpi-amber"><div class="kpi-label">Litros / costo esperado</div><div class="kpi-value">${_next_costo/1e6:.2f}M</div><div class="kpi-sub">{_next_lts:,.0f} L @ ${precio_gasoil:,.0f}/L</div></div>', unsafe_allow_html=True)
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
        fig.add_trace(go.Scatter(x=pred_labels+pred_labels[::-1],y=upper_clean+lower_clean[::-1],fill='toself',fillcolor='rgba(59,130,246,0.15)',line=dict(color='rgba(0,0,0,0)'),name='Intervalo ±1.5σ',hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=pred_labels,y=upper_clean,mode='lines',line=dict(color='#3b82f6',width=1,dash='dot'),name='CI sup',hovertemplate='CI sup: %{y:.2f} L/100km<extra></extra>'))
        fig.add_trace(go.Scatter(x=pred_labels,y=lower_clean,mode='lines',line=dict(color='#3b82f6',width=1,dash='dot'),name='CI inf',hovertemplate='CI inf: %{y:.2f} L/100km<extra></extra>'))
        hist_x=[all_labels[i] for i,v in enumerate(all_hist) if v is not None]; hist_y=[v for v in all_hist if v is not None]
        fig.add_trace(go.Scatter(x=hist_x,y=hist_y,mode='lines+markers',line=dict(color='#ef4444',width=2.5),marker=dict(size=7,color='#ef4444',line=dict(color='#fff',width=1.5)),name='Histórico',hovertemplate='%{x}<br>Real: <b>%{y:.2f} L/100km</b><extra></extra>'))
        pred_x=[all_labels[i] for i,v in enumerate(all_pred) if v is not None]; pred_y=[v for v in all_pred if v is not None]
        fig.add_trace(go.Scatter(x=pred_x,y=pred_y,mode='lines+markers',line=dict(color='#60a5fa',width=2.5,dash='dash'),marker=dict(size=9,color='#60a5fa',symbol='diamond',line=dict(color='#fff',width=1.5)),name='Predicción',hovertemplate='%{x}<br>Pred: <b>%{y:.2f} L/100km</b><extra></extra>'))
        for yr in unique_years[1:]:
            yr_label=f'Ene {yr}'
            if yr_label in all_labels:
                fig.add_vline(x=yr_label,line_width=1,line_dash='dot',line_color='#334155',annotation_text=str(yr),annotation_position='top',annotation_font_color='#64748b',annotation_font_size=10)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
            legend=dict(bgcolor='rgba(15,23,42,0.8)',bordercolor='#334155',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
            xaxis=dict(gridcolor='#334155',linecolor='#334155',tickfont=dict(color='#94a3b8',size=10),title=dict(text='Período',font=dict(color='#94a3b8')),tickangle=-45),
            yaxis=dict(gridcolor='#334155',linecolor='#334155',tickfont=dict(color='#94a3b8',size=11),title=dict(text='L/100km',font=dict(color='#94a3b8'))),
            height=450,margin=dict(l=10,r=10,t=50,b=60),hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f'±1.5σ intervalo de confianza | Línea roja = histórico ({n_meses_entrenamiento} meses) | Línea azul = predicción')
        st.divider()
        st.markdown('<div class="sec-title">🚨 Alerta de Desvío — Predicción vs. Real</div>', unsafe_allow_html=True)
        mes_eval = (meses_df['MES_PERIODO'].iloc[-1] if not meses_df.empty
                    else hist['MES_PERIODO'].iloc[-1])
        real_l100_eval = (float(meses_df['L100'].iloc[-1]) if not meses_df.empty
                          else float(hist['L100'].iloc[-1]))
        hist_prev = meses_hist_full[meses_hist_full['MES_PERIODO']<mes_eval].copy()
        hist_prev['T']=range(len(hist_prev))
        if len(hist_prev)>=3:
            ultimo_real_mes=str(mes_eval); ultimo_real_l100=real_l100_eval
            degree_prev=min(2,len(hist_prev)-1)
            poly_prev=PolynomialFeatures(degree=degree_prev); Xprev=poly_prev.fit_transform(hist_prev['T'].values.reshape(-1,1))
            m_prev=LinearRegression().fit(Xprev,hist_prev['L100'].values)
            X_pred_prev=poly_prev.transform(np.array([[len(hist_prev)]]).reshape(-1,1))
            pred_ultimo=float(np.clip(m_prev.predict(X_pred_prev),0,100)[0])
            desvio=ultimo_real_l100-pred_ultimo; desvio_pct=(desvio/pred_ultimo*100) if pred_ultimo>0 else 0
            umbral=1.5*std_res
            if abs(desvio)>umbral:
                dir_txt='SUPERIOR' if desvio>0 else 'INFERIOR'
                st.markdown(f'<div class="alert-box"><b>🚨 DESVÍO DETECTADO — {ultimo_real_mes}</b><br>Consumo real: <b>{ultimo_real_l100:.2f} L/100km</b> &nbsp;|&nbsp; Predicción: <b>{pred_ultimo:.2f} L/100km</b><br>Desvío: <b>{desvio:+.2f} L/100km ({desvio_pct:+.1f}%)</b> — {dir_txt} al intervalo esperado (±{umbral:.2f})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-ok"><b>✅ Sin desvío — {ultimo_real_mes}</b><br>Consumo real: <b>{ultimo_real_l100:.2f} L/100km</b> &nbsp;|&nbsp; Predicción: <b>{pred_ultimo:.2f} L/100km</b><br>Desvío: <b>{desvio:+.2f} L/100km ({desvio_pct:+.1f}%)</b> — dentro del intervalo esperado (±{umbral:.2f})</div>', unsafe_allow_html=True)
        else:
            st.info(f'No hay suficiente historial previo a {mes_eval} para evaluar el desvío (se necesitan ≥3 meses anteriores).')
        st.divider()
        st.markdown('<div class="sec-title">🎨 Simulador What-If</div>', unsafe_allow_html=True)
        delta_precio_pct=st.slider('💸 Variación precio combustible (%)',min_value=-30,max_value=50,value=0,step=1)
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
        st.markdown(f"""<div style='padding:8px 0;'>
        <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>Análisis por Patente — {anio_sel}</div>
        <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Consumo · IER v4 · Excesos velocidad · Promedios</div>
        </div>""", unsafe_allow_html=True)
    if df.empty or 'DOMINIO' not in df.columns: st.warning('Sin datos disponibles.'); st.stop()
    resumen=df.groupby('DOMINIO').agg(LITROS_TOTAL=('LITROS','sum'),KM_TOTAL=('KM','sum'),MESES=('MES_PERIODO','nunique')).reset_index()
    resumen['L100KM_PROM']=(resumen['LITROS_TOTAL']/resumen['KM_TOTAL'].replace(0,np.nan)*100).round(2)
    resumen['LITROS_PROM_MES']=(resumen['LITROS_TOTAL']/resumen['MESES'].replace(0,np.nan)).round(0)
    def _dias_activos(s):
        s=s.dropna()
        return (s.max()-s.min()).days+1 if len(s)>0 else 0
    _dias_pat=df.groupby('DOMINIO')['FECHA'].apply(_dias_activos).rename('DIAS_ACTIVOS').reset_index()
    resumen=resumen.merge(_dias_pat,on='DOMINIO',how='left')
    resumen['KM_DIA']=(resumen['KM_TOTAL']/resumen['DIAS_ACTIVOS'].replace(0,np.nan)).round(0)
    resumen=resumen[resumen['KM_TOTAL']>0].sort_values('L100KM_PROM',ascending=False)
    resumen['MODELO']=resumen['DOMINIO'].apply(asignar_modelo)
    if not df_ier.empty:
        resumen=resumen.merge(df_ier[['DOMINIO','IER','CLASIFICACION','EXCESOS','VEL_MAX']],on='DOMINIO',how='left')
    else:
        resumen['IER']='—'; resumen['CLASIFICACION']='—'; resumen['EXCESOS']=0; resumen['VEL_MAX']=0
    if resumen.empty: st.warning('Sin datos suficientes.'); st.stop()
    patente_max=resumen.iloc[0]; patente_min=resumen.iloc[-1]
    st.markdown(f'<div class="sec-title">⚡ Destacados {anio_sel}</div>', unsafe_allow_html=True)
    hc1,hc2=st.columns(2)
    with hc1:
        st.markdown(f'<div class="highlight-max"><b>🔴 Mayor consumo — {patente_max["DOMINIO"]}</b> <span style="color:#94a3b8;font-size:.8rem;">({patente_max["MODELO"]})</span><br>Promedio: <b>{patente_max["L100KM_PROM"]:.2f} L/100km</b> &nbsp;|&nbsp; Total: <b>{patente_max["LITROS_TOTAL"]:,.0f} L</b> &nbsp;|&nbsp; {patente_max["KM_TOTAL"]:,.0f} km &nbsp;|&nbsp; {int(patente_max["MESES"])} meses activa</div>', unsafe_allow_html=True)
    with hc2:
        st.markdown(f'<div class="highlight-min"><b>🟢 Menor consumo — {patente_min["DOMINIO"]}</b> <span style="color:#94a3b8;font-size:.8rem;">({patente_min["MODELO"]})</span><br>Promedio: <b>{patente_min["L100KM_PROM"]:.2f} L/100km</b> &nbsp;|&nbsp; Total: <b>{patente_min["LITROS_TOTAL"]:,.0f} L</b> &nbsp;|&nbsp; {patente_min["KM_TOTAL"]:,.0f} km &nbsp;|&nbsp; {int(patente_min["MESES"])} meses activa</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown(f'<div class="sec-title">Promedio L/100km por Patente — {anio_sel}</div>', unsafe_allow_html=True)
    colors_bar=[('#ef4444' if r['DOMINIO']==patente_max['DOMINIO'] else ('#22c55e' if r['DOMINIO']==patente_min['DOMINIO'] else '#3b82f6')) for _,r in resumen.iterrows()]
    fig_bar=go.Figure([go.Bar(x=resumen['DOMINIO'],y=resumen['L100KM_PROM'],marker_color=colors_bar,
        text=resumen['L100KM_PROM'].apply(lambda v:f'{v:.1f}'),textposition='outside',textfont=dict(color='#e2e8f0',size=10),
        hovertemplate='<b>%{x}</b><br>L/100km: %{y:.2f}<extra></extra>')])
    promedio_flota=resumen['L100KM_PROM'].mean()
    fig_bar.add_hline(y=promedio_flota,line_dash='dot',line_color='#f59e0b',line_width=2,annotation_text=f'Promedio flota: {promedio_flota:.2f}',annotation_position='top right',annotation_font_color='#fbbf24',annotation_font_size=11)
    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
        xaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8',size=10),tickangle=-45),
        yaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8'),title=dict(text='L/100km',font=dict(color='#94a3b8'))),
        height=420,margin=dict(l=10,r=10,t=30,b=80),showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption('🔴 Mayor consumo · 🟢 Menor consumo · 🔵 Resto · Línea amarilla = promedio flota')
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
                colorscale=[[0.0,'#052e16'],[0.35,'#16a34a'],[0.65,'#f59e0b'],[1.0,'#7f1d1d']],
                colorbar=dict(title=dict(text='L/100km',font=dict(color='#94a3b8')),tickfont=dict(color='#94a3b8'),bgcolor='rgba(0,0,0,0)'),
                hovertemplate='Patente: <b>%{y}</b><br>Mes: %{x}<br>L/100km: <b>%{z:.2f}</b><extra></extra>'))
            fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
                xaxis=dict(tickfont=dict(color='#94a3b8',size=10),tickangle=-45,side='bottom'),
                yaxis=dict(tickfont=dict(color='#94a3b8',size=10)),height=max(300,len(y_vals)*40),margin=dict(l=10,r=10,t=20,b=60))
            st.plotly_chart(fig_heat, use_container_width=True)
    st.divider()
    st.markdown('<div class="sec-title">🔍 Detalle Individual por Patente</div>', unsafe_allow_html=True)
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
                st.markdown('<div class="sec-title">📊 Índice de Eficiencia Relativa (IER v8)</div>', unsafe_allow_html=True)
                ier_v=ier_row['IER']
                sc_color=('#22c55e' if ier_v>=105 else ('#f59e0b' if ier_v>=95 else ('#f97316' if ier_v>=85 else '#ef4444')))
                ia1,ia2,ia3=st.columns([1,2,2])
                with ia1:
                    st.markdown(f'<div class="ier-gauge-wrap"><div class="kpi-label">IER v8</div><div class="ier-score-big" style="color:{sc_color};">{ier_v:.1f}</div><div class="ier-clasif">{ier_row["CLASIFICACION"]}</div><div style="font-size:.72rem;color:#94a3b8;margin-top:6px;">base 100 = prom. {modelo_pat}</div></div>', unsafe_allow_html=True)
                with ia2:
                    st.markdown('<div style="font-size:.8rem;color:#94a3b8;font-weight:600;margin-bottom:6px;">Componentes del IER (40/25/20/15)</div>', unsafe_allow_html=True)
                    def comp_bar(label,score,peso):
                        pct=min(int(score*50),100); bc='#22c55e' if score>=1 else '#ef4444'
                        st.markdown(f'<div class="ier-comp-row"><div class="ier-comp-label">{label} <span style="color:#94a3b8;">({peso}%)</span></div><div class="ier-comp-bar-bg"><div class="ier-comp-bar" style="width:{pct}%;background:{bc}"></div></div><div class="ier-comp-val" style="color:{bc};">{score*100:.0f}</div></div>', unsafe_allow_html=True)
                    comp_bar('⛽ Consumo vs. esperado',ier_row['SCORE_CONSUMO'],40)
                    comp_bar('🎯 Score conducción',ier_row['SCORE_MANEJO'],25)
                    comp_bar('⏱️ Ralentí',ier_row['SCORE_RALENTI'],20)
                    comp_bar(f'🚨 Severidad vel.',ier_row['SCORE_VEL'],15)
                with ia3:
                    st.markdown(f'<div style="font-size:.8rem;color:#94a3b8;font-weight:600;margin-bottom:6px;">Esta unidad vs. promedio {modelo_pat}</div>', unsafe_allow_html=True)
                    severidad_u = ier_row.get('SEVERIDAD', 0)
                    severidad_m = ier_row.get('SEVERIDAD_MOD', 0)
                    delta_sev = severidad_u - severidad_m
                    if ier_row['TIENE_CONSUMO']:
                        st.metric('⛽ L/100km real vs. esperado',f"{ier_row['L100KM']:.2f}",
                                  f"{ier_row['DESVIO_PCT']:+.1f}% vs esperado ({ier_row['L100KM_ESP']:.2f})",delta_color='inverse')
                        _ton_t = f"{ier_row['TON_VIAJE']:.1f} t/viaje" if pd.notnull(ier_row['TON_VIAJE']) else 'peso sin dato válido'
                        _ruta_t = f" · ruta principal: {ier_row['RUTA_PRINCIPAL']}" if ier_row['RUTA_PRINCIPAL'] else ''
                        st.caption(f"Ajuste por carga {ier_row['AJ_CARGA_PCT']:+.1f}% ({_ton_t}) · ajuste por ruta {ier_row['AJ_RUTA_PCT']:+.1f}%{_ruta_t}")
                    else:
                        st.metric('⛽ L/100km real vs. esperado','sin datos','peso redistribuido a los otros componentes')
                    st.metric(f'Severidad vel. (km/h acum. sobre {LIMITE_VELOCIDAD})',f"{severidad_u:.0f}",f"{delta_sev:+.0f} vs prom. {modelo_pat} ({severidad_m:.0f})",delta_color='inverse')
                    st.metric(f'Eventos >{LIMITE_VELOCIDAD} km/h',f"{int(ier_row['EXCESOS'])} eventos",'ref. — la severidad usa km/h acumulados')
                    sc_man_v = ier_row.get('SCORE_CONDUCCION', np.nan)
                    sc_man_m = ier_row.get('SCORE_MANEJO_MOD', np.nan)
                    if pd.notnull(sc_man_v):
                        delta_man = sc_man_v - (sc_man_m if pd.notnull(sc_man_m) else sc_man_v)
                        st.metric('🎯 Score conducción',f"{sc_man_v:.2f}/10",f"{delta_man:+.2f} vs prom. {modelo_pat} ({sc_man_m:.2f})" if pd.notnull(sc_man_m) else 'sin promedio',delta_color='normal')
                    else:
                        st.metric('🎯 Score conducción','sin datos','peso redistribuido a los otros componentes')
                    if ier_row['TIENE_RALENTI']:
                        _ral_m = ier_row.get('RALENTI_MOD', np.nan)
                        st.metric('⏱️ % Ralentí',f"{ier_row['RALENTI_PCT']:.1f}%",
                                  f"{ier_row['RALENTI_PCT']-_ral_m:+.1f} pts vs prom. {modelo_pat} ({_ral_m:.1f}%)" if pd.notnull(_ral_m) else 'sin promedio',delta_color='inverse')
                    else:
                        st.metric('⏱️ % Ralentí','sin datos','peso redistribuido a los otros componentes')
            df_vel_pat=(df_vel_filtrado[df_vel_filtrado['DOMINIO']==pat_sel] if not df_vel_filtrado.empty else pd.DataFrame())
            if not df_vel_pat.empty:
                st.markdown(f'<div class="sec-title">🚨 Excesos de Velocidad >{LIMITE_VELOCIDAD} km/h — {pat_sel}</div>', unsafe_allow_html=True)
                severidad_pat = df_vel_pat['EXCESO_KMH'].sum() if 'EXCESO_KMH' in df_vel_pat.columns else 0
                exceso_prom_pat = df_vel_pat['EXCESO_KMH'].mean() if 'EXCESO_KMH' in df_vel_pat.columns else 0
                vp1,vp2,vp3,vp4=st.columns(4)
                vp1.metric('Eventos totales',len(df_vel_pat))
                vp2.metric('Severidad total',f"{severidad_pat:.0f} km/h acum.",f"promedio por evento: +{exceso_prom_pat:.1f} km/h")
                vp3.metric('Vel. máxima',f"{df_vel_pat['VELOCIDAD'].max():.0f} km/h",f"+{df_vel_pat['VELOCIDAD'].max()-LIMITE_VELOCIDAD:.0f} km/h sobre límite")
                vp4.metric('Vel. promedio en exceso',f"{df_vel_pat['VELOCIDAD'].mean():.1f} km/h")
            st.divider()
            fig_pat=go.Figure()
            fig_pat.add_trace(go.Bar(x=df_pat_mes['MES_STR'],y=df_pat_mes['LITROS'],name='Litros',marker_color='rgba(59,130,246,0.5)',yaxis='y2',hovertemplate='%{x}<br>Litros: <b>%{y:,.0f}</b><extra></extra>'))
            fig_pat.add_trace(go.Scatter(x=df_pat_mes['MES_STR'],y=df_pat_mes['L100'],name='L/100km',mode='lines+markers',line=dict(color='#ef4444',width=2.5),marker=dict(size=8,color='#ef4444',line=dict(color='#fff',width=1.5)),hovertemplate='%{x}<br>L/100km: <b>%{y:.2f}</b><extra></extra>'))
            fig_pat.add_hline(y=l100_prom_pat,line_dash='dot',line_color='#f59e0b',annotation_text=f'Prom: {l100_prom_pat:.2f}',annotation_font_color='#fbbf24')
            fig_pat.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
                xaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8',size=10),tickangle=-30),
                yaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8'),title=dict(text='L/100km',font=dict(color='#ef4444'))),
                yaxis2=dict(overlaying='y',side='right',tickfont=dict(color='#3b82f6'),title=dict(text='Litros',font=dict(color='#3b82f6')),showgrid=False),
                legend=dict(bgcolor='rgba(15,23,42,0.8)',bordercolor='#334155',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
                height=370,margin=dict(l=10,r=50,t=40,b=50))
            st.plotly_chart(fig_pat, use_container_width=True)
            with st.expander(f'Ver tabla mensual — {pat_sel}'):
                df_show=df_pat_mes[['MES_STR','LITROS','KM','L100']].rename(columns={'MES_STR':'Mes','LITROS':'Litros','KM':'KM','L100':'L/100km'})
                df_show['Litros']=df_show['Litros'].apply(lambda x:f'{x:,.0f}')
                df_show['KM']=df_show['KM'].apply(lambda x:f'{x:,.0f}')
                st.dataframe(df_show, use_container_width=True, hide_index=True)
    st.divider()
    st.markdown(f'<div class="sec-title">Tabla Resumen — Todas las Patentes {anio_sel}</div>', unsafe_allow_html=True)
    cols_show=['DOMINIO','MODELO','LITROS_TOTAL','KM_TOTAL','KM_DIA','L100KM_PROM','LITROS_PROM_MES','MESES']
    col_names=['Patente','Modelo','Litros Total','KM Total','KM/día','L/100km Prom','Litros/Mes Prom','Meses Activa']
    for c,n in [('IER','IER'),('CLASIFICACION','Clasificación IER'),(f'EXCESOS',f'Excesos >{LIMITE_VELOCIDAD}km/h'),('VEL_MAX','Vel. Máx (km/h)')]:
        if c in resumen.columns: cols_show.append(c); col_names.append(n)
    resumen_show=resumen[cols_show].copy(); resumen_show.columns=col_names
    resumen_show['Litros Total']=resumen_show['Litros Total'].apply(lambda x:f'{x:,.0f}')
    resumen_show['KM Total']=resumen_show['KM Total'].apply(lambda x:f'{x:,.0f}')
    resumen_show['KM/día']=resumen_show['KM/día'].apply(lambda x:f'{x:,.0f}' if pd.notna(x) else '—')
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
        st.markdown(f"""<div style='padding:8px 0;'>
        <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>Datos Operativos</div>
        <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Peso entregado por patente &middot; Ton·km/L &middot; Productividad de carga</div>
        </div>""", unsafe_allow_html=True)
    if df_carga_raw is None or df_carga_raw.empty:
        st.warning('⚠️ No hay datos de carga disponibles. Verificá la conexión al sistema BI (reporte_hojas.xlsx).')
        st.stop()
    _patentes_ld = df['DOMINIO'].dropna().unique()
    _desde = st.session_state.get('desde_periodo', None)
    _hasta = st.session_state.get('hasta_periodo', None)
    if _desde is not None and _hasta is not None:
        df_carga_anio = df_carga_raw[
            (df_carga_raw['MES'] >= _desde) &
            (df_carga_raw['MES'] <= _hasta) &
            (df_carga_raw['DOMINIO'].isin(_patentes_ld))
        ].copy() if not df_carga_raw.empty else pd.DataFrame()
        _rango_txt = f'{_desde} a {_hasta}' if _desde != _hasta else f'{_desde}'
    else:
        df_carga_anio = df_carga_raw[
            (df_carga_raw['MES'].apply(lambda p:p.year)==anio_sel) &
            (df_carga_raw['DOMINIO'].isin(_patentes_ld))
        ].copy() if not df_carga_raw.empty else pd.DataFrame()
        _rango_txt = f'{anio_sel}'
    if df_carga_anio.empty: st.warning(f'Sin datos de carga para {_rango_txt}.'); st.stop()
    df_carga_anio['MES_STR']=df_carga_anio['MES'].astype(str)
    df_carga_anio['MODELO']=df_carga_anio['DOMINIO'].apply(asignar_modelo)
    st.markdown(f'<div class="sec-title">Resumen de Carga — {_rango_txt}</div>', unsafe_allow_html=True)
    peso_total=df_carga_anio['PESO_TON'].sum(); n_pat_con_carga=df_carga_anio['DOMINIO'].nunique()
    peso_prom_pat=peso_total/n_pat_con_carga if n_pat_con_carga>0 else 0; meses_con_carga=df_carga_anio['MES'].nunique()
    def kpi2(cont,color,label,value,sub=''):
        cont.markdown(f'<div class="kpi-card {color}" style="padding:14px 16px;"><div class="kpi-label" style="font-size:.7rem;">{label}</div><div class="kpi-value" style="font-size:1.45rem;">{value}</div><div class="kpi-sub" style="font-size:.7rem;">{sub}</div></div>', unsafe_allow_html=True)
    if df_viajes_raw is not None and not df_viajes_raw.empty:
        if _desde is not None and _hasta is not None:
            _vj_g = df_viajes_raw[(df_viajes_raw['MES']>=_desde)&(df_viajes_raw['MES']<=_hasta)&(df_viajes_raw['DOMINIO'].isin(_patentes_ld))]
        else:
            _vj_g = df_viajes_raw[(df_viajes_raw['MES'].apply(lambda p:p.year)==anio_sel)&(df_viajes_raw['DOMINIO'].isin(_patentes_ld))]
        _n_tot_g = len(_vj_g); _n_vac_g = int((_vj_g['CON_CARGA']==0).sum())
        _pct_vac_g = (_n_vac_g/_n_tot_g*100) if _n_tot_g>0 else 0.0
    else:
        _n_tot_g=0; _n_vac_g=0; _pct_vac_g=0.0
    _color_vac = 'kpi-red' if _pct_vac_g>=30 else ('kpi-amber' if _pct_vac_g>=15 else 'kpi-green')
    ck1,ck2,ck3,ck4,ck5=st.columns(5)
    kpi2(ck1,'kpi-purple','📦 Peso Total Entregado',f'{peso_total:,.1f}',f'toneladas {_rango_txt}')
    kpi2(ck2,'','🚛 Patentes con Carga',f'{n_pat_con_carga}',f'de {df["DOMINIO"].nunique()} activas')
    kpi2(ck3,'kpi-green','📊 Prom. por Patente',f'{peso_prom_pat:,.1f}','toneladas período')
    kpi2(ck4,'kpi-amber','📅 Meses con datos',f'{meses_con_carga}',f'{_rango_txt}')
    kpi2(ck5,_color_vac,'🚫 % Retornos Vacíos',f'{_pct_vac_g:.1f}%',f'{_n_vac_g} de {_n_tot_g} viajes sin carga' if _n_tot_g>0 else 'sin datos de viajes')
    st.divider()
    st.markdown(f'<div class="sec-title">🔬 Diagnóstico de Carga — Matriz L/100km vs kg/km — {_rango_txt}</div>', unsafe_allow_html=True)
    st.markdown("""<div class="ier-info-box">
    <b>¿Cómo leer esta matriz?</b> Cada punto es una patente. Los ejes separan consumo (L/100km) y densidad de carga (kg transportados por km recorrido).<br>
    La línea divisoria es la <b>mediana de la flota</b> en cada eje.<br>
    <b>% viajes sin peso:</b> viajes finalizados en el BI con Peso Entregado = 0 — proxy de retornos vacíos.
    </div>""", unsafe_allow_html=True)
    _km_pat   = df[df['KM']>0].groupby('DOMINIO')['KM'].sum().reset_index()
    _l100_pat = df[df['L100KM']>0].groupby('DOMINIO')['L100KM'].mean().reset_index()
    _tons_pat = df_carga_anio.groupby('DOMINIO')['PESO_TON'].sum().reset_index()
    _mat = _km_pat.merge(_l100_pat, on='DOMINIO').merge(_tons_pat, on='DOMINIO', how='inner')
    _mat['KG_KM']  = (_mat['PESO_TON'] * 1000 / _mat['KM']).round(2)
    _mat['MODELO'] = _mat['DOMINIO'].apply(asignar_modelo)
    if df_viajes_raw is not None and not df_viajes_raw.empty:
        if _desde is not None and _hasta is not None:
            _vj = df_viajes_raw[(df_viajes_raw['MES']>=_desde)&(df_viajes_raw['MES']<=_hasta)&(df_viajes_raw['DOMINIO'].isin(_patentes_ld))]
        else:
            _vj = df_viajes_raw[(df_viajes_raw['MES'].apply(lambda p:p.year)==anio_sel)&(df_viajes_raw['DOMINIO'].isin(_patentes_ld))]
        _vj_stats = _vj.groupby('DOMINIO').agg(N_TOTAL=('CON_CARGA','count'),N_CARGADOS=('CON_CARGA','sum')).reset_index()
        _vj_stats['N_VACIOS']   = _vj_stats['N_TOTAL'] - _vj_stats['N_CARGADOS']
        _vj_stats['PCT_VACIOS'] = (_vj_stats['N_VACIOS']/_vj_stats['N_TOTAL']*100).round(1)
        _mat = _mat.merge(_vj_stats[['DOMINIO','N_TOTAL','N_CARGADOS','N_VACIOS','PCT_VACIOS']], on='DOMINIO', how='left')
        for _c in ['N_TOTAL','N_CARGADOS','N_VACIOS','PCT_VACIOS']:
            _mat[_c] = pd.to_numeric(_mat[_c], errors='coerce').fillna(0)
        for _c in ['N_TOTAL','N_CARGADOS','N_VACIOS']:
            _mat[_c] = _mat[_c].astype(int)
    else:
        _mat['N_TOTAL']=0; _mat['N_CARGADOS']=0; _mat['N_VACIOS']=0; _mat['PCT_VACIOS']=0.0
    if not _mat.empty and len(_mat)>=2:
        _l100_med = _mat['L100KM'].median()
        _kgkm_med = _mat['KG_KM'].median()
        def _cuadrante(row):
            bajo = row['L100KM'] <= _l100_med
            alto = row['KG_KM']  >= _kgkm_med
            if   bajo and alto:     return '🟢 Ideal','#22c55e','Eficiente y bien cargado.'
            elif bajo and not alto: return '🟡 Subutilizado','#f59e0b','Consumo eficiente pero baja densidad de carga. Revisar rutas / retornos vacíos.'
            elif not bajo and alto: return '🟠 Consumo alto','#f97316','Bien cargado pero consume en exceso. Revisar mecánica/conducción.'
            else:                   return '🔴 Crítico','#ef4444','Consumo alto y baja carga. Intervención urgente.'
        _mat[['CUAD_LABEL','CUAD_COLOR','CUAD_DESC']] = pd.DataFrame(_mat.apply(_cuadrante,axis=1).tolist(), index=_mat.index)
        _x_min = _mat['L100KM'].min()*0.95; _x_max = _mat['L100KM'].max()*1.05
        _y_min = _mat['KG_KM'].min()*0.90;  _y_max = _mat['KG_KM'].max()*1.10
        fig_mat = go.Figure()
        _quad_cfg = [
            ([_x_min,_l100_med],[_kgkm_med,_y_max],'rgba(34,197,94,0.12)','#22c55e','🟢 IDEAL','Eficiente + bien cargado',_x_min,_y_max,'top left'),
            ([_l100_med,_x_max],[_kgkm_med,_y_max],'rgba(249,115,22,0.12)','#f97316','🟠 CONSUMO ALTO','Carga OK · consumo excesivo',_x_max,_y_max,'top right'),
            ([_x_min,_l100_med],[_y_min,_kgkm_med],'rgba(245,158,11,0.12)','#f59e0b','🟡 SUBUTILIZADO','Eficiente · poca carga',_x_min,_y_min,'bottom left'),
            ([_l100_med,_x_max],[_y_min,_kgkm_med],'rgba(239,68,68,0.12)','#ef4444','🔴 CRÍTICO','Consumo alto + baja carga',_x_max,_y_min,'bottom right'),
        ]
        for _xr,_yr,_fc,_ec,_title,_sub,_ax,_ay,_apos in _quad_cfg:
            fig_mat.add_shape(type='rect', x0=_xr[0], x1=_xr[1], y0=_yr[0], y1=_yr[1],
                fillcolor=_fc, line=dict(color=_ec, width=0.5, dash='dot'), layer='below')
            _xanchor='left' if 'left' in _apos else 'right'
            _yanchor='top'  if 'top'  in _apos else 'bottom'
            _pad_x=(_x_max-_x_min)*0.015*(1 if _xanchor=='left' else -1)
            _pad_y=(_y_max-_y_min)*0.025*(-1 if _yanchor=='top' else 1)
            fig_mat.add_annotation(x=_ax+_pad_x, y=_ay+_pad_y,
                text=f'<b>{_title}</b><br><span style="font-size:9px;">{_sub}</span>',
                showarrow=False, xanchor=_xanchor, yanchor=_yanchor,
                font=dict(size=11, color=_ec), bgcolor='rgba(15,23,42,0.75)', borderpad=4)
        fig_mat.add_vline(x=_l100_med, line_dash='dash', line_color='#64748b', line_width=1.5)
        fig_mat.add_hline(y=_kgkm_med, line_dash='dash', line_color='#64748b', line_width=1.5)
        fig_mat.add_annotation(x=_l100_med,y=_y_max,text=f'mediana L/100km = {_l100_med:.1f}',
            showarrow=False, yanchor='bottom', font=dict(size=9,color='#94a3b8'),
            bgcolor='rgba(15,23,42,0.7)', borderpad=3)
        fig_mat.add_annotation(x=_x_min,y=_kgkm_med,text=f'mediana kg/km = {_kgkm_med:.0f}',
            showarrow=False, xanchor='left', font=dict(size=9,color='#94a3b8'),
            bgcolor='rgba(15,23,42,0.7)', borderpad=3)
        fig_mat.add_trace(go.Scatter(
            x=_mat['L100KM'], y=_mat['KG_KM'],
            mode='markers+text',
            text=_mat['DOMINIO'],
            textposition='top center',
            textfont=dict(size=10, color='#f1f5f9', family='monospace'),
            marker=dict(size=18, color=_mat['CUAD_COLOR'], line=dict(color='white',width=2), symbol='circle'),
            customdata=_mat[['CUAD_LABEL','CUAD_DESC','PCT_VACIOS','N_TOTAL','N_VACIOS','MODELO','KG_KM','PESO_TON']].values,
            hovertemplate=(
                '<b>%{text}</b>  <i>%{customdata[5]}</i><br>'
                '─────────────────────<br>'
                '⛽ L/100km: <b>%{x:.2f}</b>  (mediana flota: ' + f'{_l100_med:.1f})<br>' +
                '📦 kg/km: <b>%{y:.1f}</b>  (mediana flota: ' + f'{_kgkm_med:.0f})<br>' +
                '⚖️ Peso total: <b>%{customdata[7]:.1f} ton</b><br>'
                '🚫 Viajes sin peso: <b>%{customdata[4]:.0f} / %{customdata[3]:.0f} (%{customdata[2]:.1f}%)</b><br>'
                '─────────────────────<br>'
                '<b>%{customdata[0]}</b><br>'
                '<i>%{customdata[1]}</i><extra></extra>'
            ),
            showlegend=False
        ))
        fig_mat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,41,59,0.6)', font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='#334155', tickfont=dict(color='#94a3b8',size=11),
                title=dict(text='L / 100 km   —   litros consumidos por cada 100 km   (← menor = más eficiente)', font=dict(color='#94a3b8',size=11)),
                range=[_x_min,_x_max]),
            yaxis=dict(gridcolor='#334155', tickfont=dict(color='#94a3b8',size=11),
                title=dict(text='kg / km   —   kg de carga entregados por km   (↑ mayor = más productivo)', font=dict(color='#94a3b8',size=11)),
                range=[_y_min,_y_max]),
            height=560, margin=dict(l=70,r=50,t=40,b=70))
        st.plotly_chart(fig_mat, use_container_width=True)
        st.caption('Cada punto = una patente · Líneas punteadas = mediana de la flota · Hover para diagnóstico completo')
        st.markdown('<div class="sec-title">📋 Diagnóstico Individual por Patente</div>', unsafe_allow_html=True)
        _mat_sorted = _mat.sort_values('CUAD_COLOR', key=lambda s: s.map({'#ef4444':0,'#f97316':1,'#f59e0b':2,'#22c55e':3}))
        _diag_cols = st.columns(min(4, len(_mat_sorted)))
        for _i, (_, _r) in enumerate(_mat_sorted.iterrows()):
            with _diag_cols[_i % len(_diag_cols)]:
                _pct_v = _r.get('PCT_VACIOS', 0)
                _n_tot = int(_r.get('N_TOTAL', 0))
                _n_vac = int(_r.get('N_VACIOS', 0))
                _vacios_txt = f"{_n_vac}/{_n_tot} viajes vacíos ({_pct_v:.0f}%)" if _n_tot>0 else "sin datos de viajes"
                _bc = _r['CUAD_COLOR']
                st.markdown(f"""
                <div style="background:#1e293b;border-radius:12px;padding:16px;border-left:5px solid {_bc};margin-bottom:12px;">
                  <div style="font-size:.95rem;font-weight:800;color:#f1f5f9;">{_r['DOMINIO']}</div>
                  <div style="font-size:.72rem;color:#94a3b8;margin-bottom:8px;">{_r['MODELO']}</div>
                  <div style="font-size:1.1rem;font-weight:700;color:{_bc};margin-bottom:6px;">{_r['CUAD_LABEL']}</div>
                  <div style="font-size:.75rem;color:#94a3b8;line-height:1.5;">
                    L/100km: <b style="color:#f1f5f9;">{_r['L100KM']:.2f}</b><br>
                    kg/km: <b style="color:#f1f5f9;">{_r['KG_KM']:.1f}</b><br>
                    Viajes sin peso: <b style="color:#fbbf24;">{_vacios_txt}</b>
                  </div>
                  <div style="font-size:.72rem;color:#94a3b8;margin-top:8px;font-style:italic;">{_r['CUAD_DESC']}</div>
                </div>""", unsafe_allow_html=True)
        with st.expander('📋 Ver tabla completa diagnóstico'):
            _tbl = _mat[['DOMINIO','MODELO','L100KM','KG_KM','PESO_TON','N_TOTAL','N_CARGADOS','N_VACIOS','PCT_VACIOS','CUAD_LABEL']].copy()
            _tbl.columns = ['Patente','Modelo','L/100km','kg/km','Peso total (ton)','Viajes total','Con carga','Sin carga','% sin carga','Cuadrante']
            _tbl['L/100km']=_tbl['L/100km'].round(2)
            _tbl['kg/km']=_tbl['kg/km'].round(1)
            _tbl['Peso total (ton)']=_tbl['Peso total (ton)'].round(1)
            st.dataframe(_tbl.sort_values('% sin carga', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info('Sin datos suficientes para armar la matriz (se necesitan telemetría + carga simultáneas).')
    st.divider()
    st.markdown(f'<div class="sec-title">📦 Peso Entregado por Mes y Patente (toneladas) — {_rango_txt}</div>', unsafe_allow_html=True)
    pivot_carga=(df_carga_anio.pivot_table(index='DOMINIO',columns='MES_STR',values='PESO_TON',aggfunc='sum',fill_value=0).reset_index())
    pivot_carga['TOTAL']=pivot_carga.drop(columns='DOMINIO').sum(axis=1)
    pivot_carga=pivot_carga.sort_values('TOTAL',ascending=False)
    meses_cols=[c for c in pivot_carga.columns if c not in ['DOMINIO','TOTAL']]
    if meses_cols:
        z_vals=pivot_carga[meses_cols].values.tolist(); y_vals=pivot_carga['DOMINIO'].tolist()
        txt_vals=[[f'{v:,.1f}' if v>0 else '' for v in row] for row in z_vals]
        fig_heat_c=go.Figure(go.Heatmap(z=z_vals,x=meses_cols,y=y_vals,text=txt_vals,texttemplate='%{text}',
            colorscale=[[0.0,'#1e293b'],[0.3,'#1d4ed8'],[0.65,'#7c3aed'],[1.0,'#be185d']],
            colorbar=dict(title=dict(text='Ton',font=dict(color='#94a3b8')),tickfont=dict(color='#94a3b8'),bgcolor='rgba(0,0,0,0)'),
            hovertemplate='Patente: <b>%{y}</b><br>Mes: %{x}<br>Peso: <b>%{z:,.1f} ton</b><extra></extra>'))
        fig_heat_c.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
            xaxis=dict(tickfont=dict(color='#94a3b8',size=10),tickangle=-45,side='bottom'),
            yaxis=dict(tickfont=dict(color='#94a3b8',size=10)),height=max(300,len(y_vals)*40),margin=dict(l=10,r=10,t=20,b=60))
        st.plotly_chart(fig_heat_c, use_container_width=True)
    with st.expander('📋 Ver tabla completa de peso entregado (ton)'):
        pivot_show=pivot_carga.copy()
        for c in meses_cols+['TOTAL']: pivot_show[c]=pivot_show[c].apply(lambda x:f'{x:,.1f}' if x>0 else '—')
        pivot_show=pivot_show.rename(columns={'DOMINIO':'Patente','TOTAL':'TOTAL año'})
        st.dataframe(pivot_show, use_container_width=True, hide_index=True)
    st.markdown(f'<div class="sec-title">Evolución Mensual de Peso Entregado por Patente — {_rango_txt}</div>', unsafe_allow_html=True)
    COLORES_PAT=['#3b82f6','#f97316','#22c55e','#a855f7','#f43f5e','#06b6d4','#eab308','#84cc16']
    fig_bar_c=go.Figure()
    for i,(_,row) in enumerate(pivot_carga.iterrows()):
        dom=row['DOMINIO']; vals=[row.get(m,0) for m in meses_cols]
        fig_bar_c.add_trace(go.Bar(name=dom,x=meses_cols,y=vals,marker_color=COLORES_PAT[i%len(COLORES_PAT)],hovertemplate=f'<b>{dom}</b><br>%{{x}}: <b>%{{y:,.1f}} ton</b><extra></extra>'))
    fig_bar_c.update_layout(barmode='stack',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
        xaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8',size=10),tickangle=-45),
        yaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8'),title=dict(text='Toneladas entregadas',font=dict(color='#94a3b8'))),
        legend=dict(bgcolor='rgba(15,23,42,0.8)',bordercolor='#334155',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
        height=420,margin=dict(l=10,r=10,t=50,b=70))
    st.plotly_chart(fig_bar_c, use_container_width=True)
    st.divider()
    st.markdown(f'<div class="sec-title">📐 Detalle ton·km/L (Productividad de Carga) — {_rango_txt}</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="ier-info-box"><b>¿Qué es ton·km/L?</b> Mide cuántas toneladas·kilómetro se transportan por cada litro de combustible.<br><b>Fórmula:</b> ton·km/L = Peso promedio por viaje (ton) × KM recorridos / Litros consumidos.<br>Se usa el peso <b>promedio por viaje</b> (los viajes vacíos cuentan como 0 t), no la suma de todas las toneladas del mes: así hacer más viajes no infla el indicador. Pesos imposibles por viaje (entre 0 y {CARGA_MIN_TON_VIAJE:.0f} t o más de {CARGA_MAX_TON_VIAJE:.0f} t) se descartan como error de carga.</div>""", unsafe_allow_html=True)
    df_op=df[df['KM']>0].copy(); df_op['MES_STR']=df_op['FECHA'].dt.to_period('M').astype(str)
    km_lts_mes=df_op.groupby(['DOMINIO','MES_STR']).agg(KM=('KM','sum'),LITROS=('LITROS','sum')).reset_index()
    if df_viajes_ier is not None and not df_viajes_ier.empty:
        _tv=df_viajes_ier[df_viajes_ier['DOMINIO'].isin(_patentes_ld)].copy()
        _tv['MES_STR']=_tv['MES'].astype(str)
        _tv=_tv.groupby(['DOMINIO','MES_STR'])['PESO_TON'].mean().rename('TON_VIAJE').reset_index()
    else:
        _tv=pd.DataFrame(columns=['DOMINIO','MES_STR','TON_VIAJE'])
    tonkml_mes=km_lts_mes.merge(_tv,on=['DOMINIO','MES_STR'],how='inner')
    tonkml_mes['TONKML']=np.where((tonkml_mes['TON_VIAJE']>0)&(tonkml_mes['LITROS']>0),(tonkml_mes['TON_VIAJE']*tonkml_mes['KM'])/tonkml_mes['LITROS'],np.nan).round(2)
    tonkml_mes['MODELO']=tonkml_mes['DOMINIO'].apply(asignar_modelo)
    tkml_valid=tonkml_mes['TONKML'].dropna()
    if not tkml_valid.empty:
        t1,t2,t3,t4=st.columns(4)
        tkml_prom=tkml_valid.mean(); tkml_max=tkml_valid.max(); tkml_min=tkml_valid.min()
        dom_max=tonkml_mes.loc[tonkml_mes['TONKML'].idxmax(),'DOMINIO']; dom_min=tonkml_mes.loc[tonkml_mes['TONKML'].idxmin(),'DOMINIO']
        kpi2(t1,'','📊 Promedio ton·km/L',f'{tkml_prom:.2f}','toda la flota')
        kpi2(t2,'kpi-green',f'🏆 Mejor — {dom_max}',f'{tkml_max:.2f}','mayor productividad')
        kpi2(t3,'kpi-red',f'⚠️ Peor — {dom_min}',f'{tkml_min:.2f}','menor productividad')
        kpi2(t4,'kpi-purple','📦 Período cubierto',f'{tonkml_mes["MES_STR"].nunique()} meses',f'{_rango_txt}')
    fig_tkml=go.Figure()
    for i,dom in enumerate(tonkml_mes['DOMINIO'].unique()):
        sub=tonkml_mes[tonkml_mes['DOMINIO']==dom].sort_values('MES_STR')
        sub_valid=sub[sub['TONKML'].notna()]
        if sub_valid.empty: continue
        fig_tkml.add_trace(go.Scatter(x=sub_valid['MES_STR'],y=sub_valid['TONKML'],name=dom,mode='lines+markers',line=dict(color=COLORES_PAT[i%len(COLORES_PAT)],width=2.5),marker=dict(size=8,line=dict(color='#fff',width=1.5)),hovertemplate=f'<b>{dom}</b><br>%{{x}}: <b>%{{y:.2f}} ton·km/L</b><extra></extra>'))
    if not tkml_valid.empty:
        fig_tkml.add_hline(y=tkml_valid.mean(),line_dash='dot',line_color='#f59e0b',line_width=2,annotation_text=f'Promedio: {tkml_valid.mean():.2f}',annotation_position='top right',annotation_font_color='#fbbf24',annotation_font_size=11)
    fig_tkml.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
        xaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8',size=10),tickangle=-30),
        yaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8'),title=dict(text='ton·km/L',font=dict(color='#94a3b8'))),
        legend=dict(bgcolor='rgba(15,23,42,0.8)',bordercolor='#334155',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
        height=400,margin=dict(l=10,r=10,t=50,b=50))
    st.plotly_chart(fig_tkml, use_container_width=True)
    _tm_ok=tonkml_mes[tonkml_mes['TONKML'].notna()].assign(TKM=lambda d:d['TON_VIAJE']*d['KM'])
    rank_tkml=_tm_ok.groupby('DOMINIO').agg(TKM=('TKM','sum'),LITROS=('LITROS','sum'),MODELO=('MODELO','first')).reset_index()
    rank_tkml['TONKML_ANUAL']=np.where(rank_tkml['LITROS']>0,rank_tkml['TKM']/rank_tkml['LITROS'],np.nan).round(2)
    rank_tkml=rank_tkml[rank_tkml['TONKML_ANUAL'].notna()].sort_values('TONKML_ANUAL',ascending=True)
    if not rank_tkml.empty:
        fig_rank=go.Figure([go.Bar(y=rank_tkml['DOMINIO'],x=rank_tkml['TONKML_ANUAL'],orientation='h',
            marker_color=['#22c55e' if v==rank_tkml['TONKML_ANUAL'].max() else ('#ef4444' if v==rank_tkml['TONKML_ANUAL'].min() else '#3b82f6') for v in rank_tkml['TONKML_ANUAL']],
            text=[f'{v:.2f}' for v in rank_tkml['TONKML_ANUAL']],textposition='outside',textfont=dict(color='#e2e8f0',size=10),
            hovertemplate='<b>%{y}</b><br>ton·km/L: <b>%{x:.2f}</b><extra></extra>')])
        prom_r=rank_tkml['TONKML_ANUAL'].mean()
        fig_rank.add_vline(x=prom_r,line_dash='dot',line_color='#f59e0b',line_width=2,annotation_text=f'Prom: {prom_r:.2f}',annotation_position='top',annotation_font_color='#fbbf24',annotation_font_size=11)
        fig_rank.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8'),title=dict(text='ton·km/L del período',font=dict(color='#94a3b8'))),
            yaxis=dict(gridcolor='#334155',tickfont=dict(color='#94a3b8',size=10)),
            height=max(300,len(rank_tkml)*50+80),margin=dict(l=10,r=120,t=30,b=30),showlegend=False)
        st.plotly_chart(fig_rank, use_container_width=True)
    st.caption(f'Fuente: reporte_hojas.xlsx (BI Expreso) · Telemetría Google Sheets · Período {_rango_txt}')

# ═══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA — MAPA DE EXCESOS DE VELOCIDAD
# ═══════════════════════════════════════════════════════════════════════════════
elif pg == "🗺️ Mapa Excesos":
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    from streamlit_folium import st_folium

    col_logo_m, col_title_m = st.columns([1,5])
    with col_logo_m: st.image(LOGO_URL, width=130)
    with col_title_m:
        st.markdown(f"""<div style='padding:8px 0;'>
        <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>🗺️ Mapa de Excesos de Velocidad</div>
        <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Geolocalización de eventos &gt;{LIMITE_VELOCIDAD} km/h · {anio_sel}</div>
        </div>""", unsafe_allow_html=True)

    if df_vel_filtrado.empty:
        st.warning('Sin eventos de velocidad en el período filtrado.')
        # Embudo de diagnóstico: muestra en qué filtro se pierden los eventos
        _n_vraw  = len(df_vel_raw)
        _n_vanio = len(df_vel_anio)
        _d_v = st.session_state.get('desde_periodo', None)
        _h_v = st.session_state.get('hasta_periodo', None)
        _periodo_v = f'{_d_v} a {_h_v}' if (_d_v is not None and _h_v is not None) else str(anio_sel)
        if not df_vel_raw.empty and 'FECHA' in df_vel_raw.columns and df_vel_raw['FECHA'].notna().any():
            _fv = df_vel_raw['FECHA'].dropna()
            _rango_v = f'{_fv.min():%Y-%m-%d} → {_fv.max():%Y-%m-%d}'
        else:
            _rango_v = 'sin fechas válidas en la planilla'
        st.caption(
            f'🔎 Diagnóstico · eventos cargados (>{LIMITE_VELOCIDAD} km/h): **{_n_vraw}** '
            f'→ del año {anio_sel}: **{_n_vanio}** → dentro del período {_periodo_v} y patentes filtradas: **0**. '
            f'Rango de fechas de los eventos: **{_rango_v}**. '
            f'Si los eventos no caen en {_periodo_v}, ampliá el filtro *Desde/Hasta*; '
            f'si son de otro año, cambiá el *Año de visualización*.'
        )
        st.stop()

    if 'LAT' not in df_vel_filtrado.columns or 'LON' not in df_vel_filtrado.columns:
        st.error("⚠️ La hoja de velocidades no tiene columnas Latitud/Longitud detectables.")
        st.caption(f"Columnas detectadas: {list(df_vel_filtrado.columns)}")
        st.stop()

    df_map = df_vel_filtrado.dropna(subset=['LAT','LON']).copy()
    # bbox Argentina
    df_map = df_map[(df_map['LAT'].between(-55, -21)) & (df_map['LON'].between(-74, -53))]

    if df_map.empty:
        st.warning('No hay eventos con coordenadas válidas dentro de Argentina.')
        st.stop()

    st.markdown('<div class="sec-title">Filtros del mapa</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        modo = st.selectbox('Vista', ['🔥 Mapa de calor', '📍 Marcadores agrupados', '🎯 Puntos individuales'])
    with fc2:
        pats_map = sorted(df_map['DOMINIO'].unique().tolist())
        pat_filt = st.multiselect('Patente', pats_map, default=[], placeholder="Todas")
    with fc3:
        _max_exc = int(df_map['EXCESO_KMH'].max())
        sev_min = st.slider('Exceso mínimo (km/h sobre límite)', 0, _max_exc if _max_exc>0 else 1, 0)
    with fc4:
        df_map['MODELO_TMP'] = df_map['DOMINIO'].apply(asignar_modelo)
        modelos_map = sorted(df_map['MODELO_TMP'].unique().tolist())
        mod_filt = st.multiselect('Modelo', modelos_map, default=modelos_map)

    df_map['MODELO'] = df_map['DOMINIO'].apply(asignar_modelo)
    if pat_filt: df_map = df_map[df_map['DOMINIO'].isin(pat_filt)]
    if mod_filt: df_map = df_map[df_map['MODELO'].isin(mod_filt)]
    df_map = df_map[df_map['EXCESO_KMH'] >= sev_min]

    if df_map.empty:
        st.warning('Sin eventos con los filtros seleccionados.')
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('📍 Eventos mapeados', f"{len(df_map):,}")
    k2.metric('🚨 Severidad total', f"{df_map['EXCESO_KMH'].sum():.0f} km/h")
    k3.metric('⚡ Vel. máxima', f"{df_map['VELOCIDAD'].max():.0f} km/h")
    k4.metric('🚛 Patentes', f"{df_map['DOMINIO'].nunique()}")

    st.markdown('<br>', unsafe_allow_html=True)

    lat_c = df_map['LAT'].mean()
    lon_c = df_map['LON'].mean()
    m = folium.Map(location=[lat_c, lon_c], zoom_start=5, tiles='CartoDB dark_matter')

    if modo.startswith('🔥'):
        heat_data = [[r['LAT'], r['LON'], float(r['EXCESO_KMH'])] for _, r in df_map.iterrows()]
        HeatMap(heat_data, radius=15, blur=20, max_zoom=10,
                gradient={0.2:'#22c55e', 0.4:'#f59e0b', 0.6:'#f97316', 0.8:'#ef4444', 1.0:'#7f1d1d'}).add_to(m)

    elif modo.startswith('📍'):
        cluster = MarkerCluster().add_to(m)
        for _, r in df_map.iterrows():
            color = '#ef4444' if r['EXCESO_KMH']>=15 else ('#f97316' if r['EXCESO_KMH']>=7 else '#f59e0b')
            ubic = r.get('UBICACION', '—')
            popup = f"""<b>{r['DOMINIO']}</b> ({r['MODELO']})<br>
            Vel: <b>{r['VELOCIDAD']:.0f} km/h</b><br>
            Exceso: +{r['EXCESO_KMH']:.0f} km/h<br>
            Fecha: {r['FECHA'].strftime('%d/%m/%Y %H:%M') if pd.notna(r['FECHA']) else '—'}<br>
            Ubicación: {ubic}"""
            folium.CircleMarker(
                location=[r['LAT'], r['LON']], radius=6,
                color=color, fill=True, fill_color=color, fill_opacity=0.8,
                popup=folium.Popup(popup, max_width=280),
                tooltip=f"{r['DOMINIO']} · {r['VELOCIDAD']:.0f} km/h"
            ).add_to(cluster)

    else:
        for _, r in df_map.iterrows():
            color = '#ef4444' if r['EXCESO_KMH']>=15 else ('#f97316' if r['EXCESO_KMH']>=7 else '#f59e0b')
            radius = 4 + min(r['EXCESO_KMH']/3, 10)
            ubic = r.get('UBICACION', '—')
            popup = f"""<b>{r['DOMINIO']}</b> ({r['MODELO']})<br>
            Vel: <b>{r['VELOCIDAD']:.0f} km/h</b><br>
            Exceso: +{r['EXCESO_KMH']:.0f} km/h<br>
            Fecha: {r['FECHA'].strftime('%d/%m/%Y %H:%M') if pd.notna(r['FECHA']) else '—'}<br>
            Ubicación: {ubic}"""
            folium.CircleMarker(
                location=[r['LAT'], r['LON']], radius=radius,
                color=color, fill=True, fill_color=color, fill_opacity=0.6, weight=1,
                popup=folium.Popup(popup, max_width=280),
                tooltip=f"{r['DOMINIO']} · {r['VELOCIDAD']:.0f} km/h"
            ).add_to(m)

    st_folium(m, use_container_width=True, height=620, returned_objects=[])

    st.caption(f"🟡 Leve (<7) · 🟠 Medio (7–15) · 🔴 Grave (≥15 km/h sobre límite {LIMITE_VELOCIDAD})")

    st.divider()
    st.markdown('<div class="sec-title">🔥 Top Zonas Críticas (grilla ~11km)</div>', unsafe_allow_html=True)
    df_map['LAT_BIN'] = (df_map['LAT']*10).round()/10
    df_map['LON_BIN'] = (df_map['LON']*10).round()/10
    hotspots = (df_map.groupby(['LAT_BIN','LON_BIN'])
                .agg(EVENTOS=('DOMINIO','count'),
                     SEVERIDAD=('EXCESO_KMH','sum'),
                     VEL_MAX=('VELOCIDAD','max'),
                     PATENTES=('DOMINIO',lambda s: ', '.join(sorted(s.unique()))),
                     UBICACION=('UBICACION', lambda s: s.iloc[0] if 'UBICACION' in df_map.columns and len(s)>0 else '—'))
                .reset_index().sort_values('SEVERIDAD',ascending=False).head(15))
    cols_hot = ['LAT_BIN','LON_BIN','EVENTOS','SEVERIDAD','VEL_MAX','PATENTES']
    if 'UBICACION' in hotspots.columns: cols_hot.append('UBICACION')
    hotspots = hotspots[cols_hot]
    hotspots.columns = ['Lat','Lon','Eventos','Severidad acum.','Vel. máx','Patentes'] + (['Ubicación ejemplo'] if 'UBICACION' in df_map.columns else [])
    st.dataframe(hotspots, use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-title">🚛 Ranking por patente (eventos geolocalizados)</div>', unsafe_allow_html=True)
    rank_pat = (df_map.groupby('DOMINIO').agg(
        EVENTOS=('DOMINIO','count'),
        SEVERIDAD=('EXCESO_KMH','sum'),
        VEL_MAX=('VELOCIDAD','max'),
        VEL_PROM=('VELOCIDAD','mean')
    ).reset_index().sort_values('SEVERIDAD',ascending=False))
    rank_pat['MODELO'] = rank_pat['DOMINIO'].apply(asignar_modelo)
    rank_pat = rank_pat[['DOMINIO','MODELO','EVENTOS','SEVERIDAD','VEL_MAX','VEL_PROM']]
    rank_pat.columns = ['Patente','Modelo','Eventos','Severidad (km/h acum.)','Vel. máx','Vel. prom']
    rank_pat['Vel. máx'] = rank_pat['Vel. máx'].round(1)
    rank_pat['Vel. prom'] = rank_pat['Vel. prom'].round(1)
    rank_pat['Severidad (km/h acum.)'] = rank_pat['Severidad (km/h acum.)'].round(1)
    st.dataframe(rank_pat, use_container_width=True, hide_index=True)

    st.caption('Fuente: hoja Velocidades de Google Sheets · Coordenadas Lat/Lon decodificadas como formato AR (-XX,XX)')

# ═══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA — DIAGNÓSTICO
# ═══════════════════════════════════════════════════════════════════════════════
elif pg == "🔧 Diagnóstico":
    col_logo5, col_title5 = st.columns([1,5])
    with col_logo5: st.image(LOGO_URL, width=130)
    with col_title5:
        st.markdown(f"""<div style='padding:8px 0;'>
        <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>🔧 Diagnóstico de Fuentes</div>
        <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Estado de cada fuente de datos · {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>""", unsafe_allow_html=True)
    st.divider()
    def _diag_card(titulo, ok, detalle, sub=''):
        color = '#22c55e' if ok else '#ef4444'
        icon  = '✅' if ok else '❌'
        st.markdown(f"""
        <div style="background:#1e293b;border-radius:12px;padding:14px 18px;border-left:5px solid {color};margin:8px 0;">
          <div style="font-size:1rem;font-weight:700;color:#f1f5f9;">{icon} {titulo}</div>
          <div style="font-size:.85rem;color:#e2e8f0;margin-top:4px;">{detalle}</div>
          <div style="font-size:.75rem;color:#94a3b8;margin-top:2px;font-family:monospace;">{sub}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('## 📡 Telemetría')
    _ok_tel = not df_raw.empty
    _det_tel = f'{len(df_raw):,} filas · {df_raw["DOMINIO"].nunique() if _ok_tel else 0} patentes · {df_raw["FECHA"].min()} → {df_raw["FECHA"].max()}' if _ok_tel else 'Sin datos cargados'
    _diag_card('Telemetría LAD', _ok_tel, _det_tel, f'Fuente: {URL_TEL}')
    st.markdown('## 🚦 Velocidades')
    _ok_vel = not df_vel_raw.empty
    _tiene_coords = _ok_vel and ('LAT' in df_vel_raw.columns) and ('LON' in df_vel_raw.columns) and df_vel_raw[['LAT','LON']].notna().any().any()
    _det_vel = f'{len(df_vel_raw):,} eventos >{LIMITE_VELOCIDAD} km/h · {df_vel_raw["DOMINIO"].nunique() if _ok_vel else 0} patentes con excesos · Coords: {"✅" if _tiene_coords else "❌"}' if _ok_vel else 'Sin eventos de velocidad'
    _diag_card('Excesos de velocidad', _ok_vel, _det_vel, f'Fuente: {URL_VEL}')
    # Diagnóstico extendido velocidades
    st.markdown(f"""
    <div style="background:#0f172a;border:1px solid #334155;border-radius:6px;padding:8px 12px;margin:3px 0;font-size:.78rem;font-family:monospace;color:#e2e8f0;">
    HTTP: <b>{vel_diag.get('status')}</b> · Resultado: <b>{vel_diag.get('err')}</b><br>
    Filas raw: {vel_diag.get('raw_rows')} · Tras parse fecha (no nulas): {vel_diag.get('tras_fecha')} · Vel válidas: {vel_diag.get('n_vel_validas')} · LAT en rango AR: {vel_diag.get('n_lat_validas')}<br>
    Tras filtro &gt;{LIMITE_VELOCIDAD} km/h: <b>{vel_diag.get('tras_velocidad_gt_limite')}</b> · Tras dropna(DOMINIO,FECHA): <b>{vel_diag.get('tras_filtros')}</b><br>
    Columnas raw: <code>{vel_diag.get('raw_cols')}</code><br>
    Columnas mapeadas: <code>{vel_diag.get('mapped_cols')}</code><br>
    Fuente usada: <code>{vel_diag.get('url')}</code> · Fechas ilegibles: <b>{vel_diag.get('n_fechas_invalidas', 0)}</b>
    </div>""", unsafe_allow_html=True)
    if vel_diag.get('muestra_fechas_invalidas'):
        st.caption(f"Ejemplos de fechas que no se pudieron leer: {vel_diag['muestra_fechas_invalidas']}")
    if _ok_vel and 'FECHA' in df_vel_raw.columns:
        _vm = df_vel_raw['FECHA'].dt.to_period('M').value_counts().sort_index()
        st.caption('Eventos por mes: ' + ' · '.join(f'{p}: {n}' for p, n in _vm.items()))
    if vel_diag.get('muestra_raw') is not None:
        st.caption('Muestra raw (primeras 5 filas tal como llegaron):')
        st.dataframe(vel_diag['muestra_raw'], use_container_width=True, hide_index=True)
    if _ok_vel:
        st.caption(f"Columnas finales detectadas: {list(df_vel_raw.columns)}")
    st.markdown('## 📦 Carga (BI)')
    _ok_car = not df_carga_raw.empty
    _det_car = f'{len(df_carga_raw):,} registros mensuales · {df_carga_raw["DOMINIO"].nunique() if _ok_car else 0} patentes con carga · {df_carga_raw["PESO_TON"].sum():,.1f} ton totales' if _ok_car else 'Sin datos de carga del BI'
    _diag_card('Peso entregado', _ok_car, _det_car, f'Fuente: {CARGA_URL}')
    st.markdown('## 🚛 Viajes')
    _ok_vj = not df_viajes_raw.empty
    _det_vj = f'{len(df_viajes_raw):,} viajes · {df_viajes_raw["DOMINIO"].nunique() if _ok_vj else 0} patentes' if _ok_vj else 'Sin datos de viajes'
    _diag_card('Viajes totales (con y sin carga)', _ok_vj, _det_vj, f'Fuente: {CARGA_URL}')
    st.markdown('## 🕵️ Test directo DOMINIO Scania (telemetría vs manejo)')
    def _hexdump(s):
        return ' '.join(f'{ord(ch):04X}' for ch in str(s))
    _tel_doms = set(df_raw['DOMINIO'].dropna().unique()) if not df_raw.empty and 'DOMINIO' in df_raw.columns else set()
    _man_doms = set(df_manejo_raw['DOMINIO'].dropna().unique()) if not df_manejo_raw.empty and 'DOMINIO' in df_manejo_raw.columns else set()
    for _pat in SCANIA_PATENTES:
        _in_tel = _pat in _tel_doms
        _in_man = _pat in _man_doms
        _match = _in_tel and _in_man
        _color = '#22c55e' if _match else '#ef4444'
        _icon  = '✅' if _match else '❌'
        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid {_color};border-radius:6px;padding:8px 12px;margin:4px 0;font-size:.78rem;font-family:monospace;color:#e2e8f0;">
        {_icon} <b>{_pat}</b> · en telemetría: {_in_tel} · en manejo: {_in_man} · match exacto: {_match}<br>
        hex esperado (ASCII): {_hexdump(_pat)}
        </div>""", unsafe_allow_html=True)
    st.caption('Busca coincidencias exactas de string entre lo que llega de telemetría y de la hoja de manejo. Si "match exacto" da False para algún Scania, el problema está en cómo ese valor llega desde la fuente (no en el código).')
    with st.expander('🔬 Ver todos los DOMINIO de telemetría que empiezan con AD o AE (para comparar a mano)'):
        _cands = sorted([d for d in _tel_doms if str(d).upper().startswith(('AD','AE'))])
        for _c in _cands:
            st.code(f"{_c!r}  |  hex: {_hexdump(_c)}  |  len: {len(str(_c))}")
    with st.expander('🔬 Ver todos los DOMINIO de manejo que empiezan con AD o AE (para comparar a mano)'):
        _cands2 = sorted([d for d in _man_doms if str(d).upper().startswith(('AD','AE'))])
        for _c in _cands2:
            st.code(f"{_c!r}  |  hex: {_hexdump(_c)}  |  len: {len(str(_c))}")
    st.markdown('## 🎯 Score Conducción')
    _ok_man = not df_manejo_raw.empty
    _det_man = f'{len(df_manejo_raw):,} registros · {df_manejo_raw["DOMINIO"].nunique() if _ok_man else 0} patentes · {df_manejo_raw["MES"].min()} → {df_manejo_raw["MES"].max()}' if _ok_man else 'Sin datos de conducción'
    _diag_card('Score conducción (3 hojas)', _ok_man, _det_man, f'Sheet: {MANEJO_SHEET_ID}')
    for d in manejo_diag:
        _ok = d['err'] == 'OK'
        _color = '#22c55e' if _ok else '#ef4444'
        _icon  = '✅' if _ok else '❌'
        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:6px;padding:6px 10px;margin:3px 0;font-size:.78rem;font-family:monospace;color:#e2e8f0;">
        {_icon} <b>{d['modelo']}</b> · gid={d['gid']} · HTTP <span style="color:{_color};font-weight:700;">{d['status']}</span> · filas: {d['rows']} · col: <code>{d['col_score']}</code> · {d['err']}
        </div>""", unsafe_allow_html=True)
        if 'dominio_raw_sample' in d:
            st.markdown(f"""
            <div style="background:#0f172a;border:1px dashed #64748b;border-radius:6px;padding:6px 10px;margin:3px 0 10px 0;font-size:.72rem;font-family:monospace;color:#94a3b8;">
            DOMINIO crudo (antes de normalizar) — repr: {d['dominio_raw_sample']}<br>
            largos: {d['dominio_raw_lens']}<br>
            <b style="color:#fbbf24;">FILTRO FINAL:</b> de {d['n_total']} filas → MES parseable: {d['mes_ok']} · DOMINIO válido (len&gt;2): {d['dom_ok']} · SCORE parseable: {d['score_ok']}<br>
            MES crudo — repr: {d['mes_raw_sample']}<br>
            SCORE crudo — repr: {d['score_raw_sample']}
            </div>""", unsafe_allow_html=True)
    st.markdown('## 🔧 Arreglos / Reparaciones')
    _ok_arr = (df_arreglos_raw is not None) and (not df_arreglos_raw.empty)
    if _ok_arr:
        _det_arr = f'{len(df_arreglos_raw):,} arreglos · {df_arreglos_raw["DOMINIO"].nunique()} patentes · ${df_arreglos_raw["MONTO"].sum():,.0f} gasto total'
    else:
        _det_arr = f'Sin datos de arreglos. {arreglos_diag.get("err","")}'
    _diag_card('Gasto en arreglos', _ok_arr, _det_arr, f'Sheet: {ARREGLOS_SHEET_ID} · gid={ARREGLOS_GID}')
    if isinstance(arreglos_diag, dict):
        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:6px;padding:6px 10px;margin:3px 0;font-size:.78rem;font-family:monospace;color:#e2e8f0;">
        Columnas detectadas → patente: <code>{arreglos_diag.get('col_dom')}</code> · fecha: <code>{arreglos_diag.get('col_fecha')}</code> · monto: <code>{arreglos_diag.get('col_monto')}</code><br>
        Columnas en hoja: {arreglos_diag.get('cols')}
        </div>""", unsafe_allow_html=True)
    st.markdown('## ⛽ Precio gasoil (X10 — fuente del precio)')
    _ok_gc = not (gasto_comb_prom is None or (isinstance(gasto_comb_prom,float) and np.isnan(gasto_comb_prom)))
    if _ok_gc:
        _det_gc = f'${gasto_comb_prom:,.0f} = monto estimado X10 promedio del mes {gasto_comb_mes} ({gasto_comb_n} cargas) → usado como precio gasoil'
    else:
        _det_gc = f'Sin datos X10 → cae a valor base manual $2.300. {gasto_comb_diag.get("err","")}'
    _diag_card('Precio gasoil desde X10 (col I, filtro col F = X10, mes más cercano a hoy)', _ok_gc, _det_gc, f'Sheet: {GASTO_COMB_SHEET_ID} · gid={GASTO_COMB_GID}')
    if isinstance(gasto_comb_diag, dict):
        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:6px;padding:6px 10px;margin:3px 0;font-size:.78rem;font-family:monospace;color:#e2e8f0;">
        Filas totales hoja: {gasto_comb_diag.get('rows')} · Filtro tipo: <code>{gasto_comb_diag.get('tipo_filter')}</code> · Mes detectado: <code>{gasto_comb_diag.get('mes')}</code> · Filas del mes: {gasto_comb_diag.get('n_mes')}<br>
        Columnas hoja: {gasto_comb_diag.get('cols')}
        </div>""", unsafe_allow_html=True)
    st.markdown('## ⛽ Precio combustible')
    _diag_card('Precio gasoil', True, f'${precio_gasoil:,.0f} / L', f'Fuente: {precio_fuente}')
    st.divider()
    st.markdown('### 📋 Muestra de telemetría')
    if not df_raw.empty:
        st.dataframe(df_raw.head(15), use_container_width=True, hide_index=True)
    st.markdown('### 📋 Muestra de velocidades (con coords)')
    if not df_vel_raw.empty:
        st.dataframe(df_vel_raw.head(15), use_container_width=True, hide_index=True)
    st.markdown('### 📋 Muestra de carga')
    if not df_carga_raw.empty:
        st.dataframe(df_carga_raw.head(15), use_container_width=True, hide_index=True)
    st.markdown('### 📋 Muestra de viajes')
    if not df_viajes_raw.empty:
        st.dataframe(df_viajes_raw.head(15), use_container_width=True, hide_index=True)
