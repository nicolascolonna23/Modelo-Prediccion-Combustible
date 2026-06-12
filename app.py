# ═══════════════════════════════════════════════════════════════════════════════
#  EXPRESO DIEMAR — Dashboard de Monitoreo de Flota v4 (LIGHT THEME)
#  IER v6.5: Z-Score + Tanh (Eficiencia Productiva: ton·km/L)
#  EXPRESO DIEMAR — Dashboard de Monitoreo de Flota v4
#  IER v4: Z-Score + Tanh  (scoring proporcional e intra-modelo)
# ═══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import streamlit as st
@@ -17,38 +17,22 @@
    page_icon="🚛",
    layout="wide",
)
# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG: URLs DE FUENTES DE DATOS — editá acá si cambian
# ═══════════════════════════════════════════════════════════════════════════════
LOGO_URL     = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/logo_diemar4.png"
IVECO_URL    = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/S-Way-6x2-1.webp"
SCANIA_URL   = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/2016p.png"
STRALIS_URL  = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/image.png"
LOGO_URL    = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/logo_diemar4.png"
IVECO_URL   = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/S-Way-6x2-1.webp"
SCANIA_URL  = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/2016p.png"
STRALIS_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/image.png"
SWAY_PATENTES   = ['AH522SI', 'AH862UB', 'AH938VO', 'AH842GQ']
SCANIA_PATENTES = ['AD247MQ', 'AE423IW']
LIMITE_VELOCIDAD = 88
# Telemetría
BASE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR35NkYPtJrOrdYHLGUH7GIW93s5cPAqQ0zEk5fP1c3gvErwbUW7HJ2OeWBYaBVsYKVmCf0yhLvs6eG/pub?output=csv"
URL_TEL  = f"{BASE_URL}&gid=0"
URL_UNID = f"{BASE_URL}&gid=882343299"
URL_VEL  = f"{BASE_URL}&gid=1563993963"
# Carga
GID_TEL  = "0"
GID_UNID = "882343299"
GID_VEL  = "1563993963"
URL_TEL  = f"{BASE_URL}&gid={GID_TEL}"
URL_UNID = f"{BASE_URL}&gid={GID_UNID}"
URL_VEL  = f"{BASE_URL}&gid={GID_VEL}"
CARGA_URL = "http://bi.sistemaexpreso.com.ar/reporte_hojas.xlsx"
# ── DATOS MANEJO — URL NUEVA ────────────────────────────────────────────────
MANEJO_BASE_NEW = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTcHu-R2zmk-j1QJ_Bu3WnZ5V8OvATxKppcAZPfUtmNamrKBkHkmFVBi2PYsIoadPJH3rTtAEkm50OS/pub?output=csv"
MANEJO_SHEETS = [
    {"gid": "0",          "modelo": "Stralis"},
    {"gid": "738544003",  "modelo": "S-Way"},
    {"gid": "2022308308", "modelo": "Scania"},
]
# ── REPARACIONES ──
REP_SHEET_ID = "1u7cckay0IJ60bfoKk2OZo-TjCvTbH9O1wKxNFdSKDCQ"
REP_GID      = "33208473"
REP_URL      = f"https://docs.google.com/spreadsheets/d/{REP_SHEET_ID}/gviz/tq?tqx=out:csv&gid={REP_GID}&sheet=GASTOS%20REPARACIONES"
# ═══════════════════════════════════════════════════════════════════════════════
#  CSS DARK THEME
# ═══════════════════════════════════════════════════════════════════════════════
LIGHT_CSS = """
DARK_CSS = """
<style>
[data-testid="stAppViewContainer"] { background: #0f172a; }
[data-testid="stSidebar"] { background: #1e293b; }
@@ -72,16 +56,17 @@
    font-size:1.1rem; font-weight:700; color:#e2e8f0;
    border-left:4px solid #2563eb; padding-left:10px; margin:18px 0 10px;
}
.price-badge {
    background:#292524; border:1px solid #f59e0b; border-radius:8px;
    padding:8px 14px; display:inline-block; font-size:0.85rem; color:#fbbf24; font-weight:600;
}
.truck-img-box {
    width:100%; height:280px; border-radius:12px;
    background:linear-gradient(135deg, #f1f5f9 0%, #cbd5e1 100%);
    width:100%; height:280px; border-radius:12px; background:#1e293b;
    display:flex; align-items:center; justify-content:center; overflow:hidden;
    border:1px solid #334155;
}
.truck-img-box img {
    max-width:100%; max-height:100%; width:100%; height:100%;
    object-fit:contain; object-position:center; padding:12px;
    mix-blend-mode: multiply;
}
.rank-row    { display:flex; align-items:center; padding:8px 0; border-bottom:1px solid #334155; }
.rank-num    { width:28px; font-weight:700; font-size:.9rem; color:#94a3b8; }
@@ -110,6 +95,10 @@
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
@@ -125,40 +114,24 @@
.ier-comp-val   { font-weight:700; color:#e2e8f0; }
.ier-comp-bar-bg { width:90px; height:6px; background:#334155; border-radius:3px; margin:0 10px; overflow:hidden; }
.ier-comp-bar    { height:6px; border-radius:3px; }
.diag-ok   { color:#22c55e; font-weight:700; }
.diag-fail { color:#ef4444; font-weight:700; }
.diag-row  {
    background:#0f172a; border:1px solid #334155; border-radius:6px;
    padding:6px 10px; margin:3px 0; font-size:.75rem; font-family:monospace; color:#e2e8f0;
.vel-badge {
    background:#2d1b00; border:1px solid #f97316; border-radius:6px;
    padding:3px 10px; display:inline-block; font-size:.78rem; color:#fb923c; font-weight:700;
}
.best-unit-card {
    background:#1e293b; border-radius:14px; padding:20px 22px;
    border-left:5px solid #22c55e; margin-bottom:12px;
.zscore-badge {
    background:#1e1b4b; border:1px solid #818cf8; border-radius:5px;
    padding:2px 8px; display:inline-block; font-size:.72rem; color:#a5b4fc; font-weight:600;
}
.best-unit-title { font-size:.72rem; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.5px; margin-bottom:6px; }
.best-unit-modelo { font-size:.8rem; color:#94a3b8; margin-bottom:4px; }
.best-unit-dom { font-size:1.5rem; font-weight:800; color:#22c55e; line-height:1.1; margin-bottom:6px; }
.best-unit-ier { font-size:1.15rem; font-weight:700; color:#f1f5f9; }
.best-unit-stats { font-size:.78rem; color:#94a3b8; margin-top:6px; line-height:1.6; }
</style>
"""
# Plotly: paleta dark
PLOTLY_BG      = 'rgba(0,0,0,0)'
PLOTLY_GRID    = '#1e293b'
PLOTLY_FONT    = '#e2e8f0'
PLOTLY_AXIS    = '#94a3b8'
PLOTLY_SUBTLE  = '#64748b'
pg = st.sidebar.radio(
    "Navegacion",
    ["Dashboard Principal", "Modelo Predictivo", "Análisis por Patente", "Datos Operativos", "🔧 Diagnóstico"],
    ["Dashboard Principal", "Modelo Predictivo", "Análisis por Patente", "Datos Operativos"],
    index=0,
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.image(LOGO_URL, width=160)
# ═══════════════════════════════════════════════════════════════════════════════
#  LOADERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=600)
def cargar_datos():
    try:
@@ -172,13 +145,13 @@ def limpiar(df):
                if   "DOMINIO"   in c or "PATENTE"  in c:              cm[c] = "DOMINIO"
                elif "LITROS"    in c or "CONSUMID" in c:              cm[c] = "LITROS"
                elif "DISTANCIA" in c or c == "KM" or "KILOMETR" in c: cm[c] = "KM"
                elif "MARCA"     in c:                                 cm[c] = "MARCA"
                elif "TAG"       in c:                                 cm[c] = "TAG"
                elif "MARCA"     in c:                                  cm[c] = "MARCA"
                elif "TAG"       in c:                                  cm[c] = "TAG"
                elif "FECHA"     in c or "DATE"     in c:              cm[c] = "FECHA"
                elif "L/100"     in c or "CONSUMO C" in c:             cm[c] = "L100KM"
                elif "RALENT"    in c:                                 cm[c] = "RALENTI"
                elif "RALENT"    in c:                                  cm[c] = "RALENTI"
                elif "TIEMPO"    in c and "MOTOR"   in c:              cm[c] = "TIEMPO_MOTOR"
                elif "EMPRESA"   in c:                                 cm[c] = "EMPRESA"
                elif "EMPRESA"   in c:                                  cm[c] = "EMPRESA"
            df = df.rename(columns=cm).loc[:, ~df.rename(columns=cm).columns.duplicated()]
            if "DOMINIO" in df.columns:
                df["DOMINIO"] = df["DOMINIO"].astype(str).str.strip().str.upper()
@@ -221,7 +194,6 @@ def limpiar(df):
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=600)
def cargar_velocidad():
    try:
@@ -260,7 +232,6 @@ def cargar_velocidad():
        return df[keep].dropna(subset=["DOMINIO","FECHA"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["DOMINIO","FECHA","VELOCIDAD","EXCESO_KMH"])

@st.cache_data(ttl=3600)
def cargar_carga():
    try:
@@ -290,160 +261,37 @@ def norm_pat(p):
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

@st.cache_data(ttl=600)
def cargar_datos_manejo():
    """Lee SCORE GENERAL de las 3 hojas raw (Stralis, S-Way, Scania)."""
    dfs = []
    diagnostico = []
    for sheet in MANEJO_SHEETS:
        url = f"{MANEJO_BASE_NEW}&gid={sheet['gid']}"
        try:
            r = requests.get(url, timeout=15)
            status = r.status_code
            if status != 200:
                diagnostico.append({'modelo': sheet['modelo'], 'gid': sheet['gid'], 'status': status, 'rows': 0, 'col_score': '—', 'err': f'HTTP {status}'})
                continue
            df = pd.read_csv(url, header=0)
            df.columns = [str(c).strip() for c in df.columns]
            score_col = next(
                (c for c in df.columns if any(k in c.upper() for k in ['GENERAL','SCORE_COND','SCORE COND'])),
                None
            )
            if score_col is None or len(df.columns) < 2:
                diagnostico.append({'modelo': sheet['modelo'], 'gid': sheet['gid'], 'status': status, 'rows': len(df), 'col_score': '—', 'err': f'No hay columna SCORE/GENERAL. Cols: {list(df.columns)[:6]}'})
                continue
            tmp = df[[df.columns[0], df.columns[1], score_col]].copy()
            tmp.columns = ['MES', 'DOMINIO', 'SCORE_CONDUCCION']
            diagnostico.append({'modelo': sheet['modelo'], 'gid': sheet['gid'], 'status': status, 'rows': len(df), 'col_score': score_col, 'err': 'OK'})
            dfs.append(tmp)
        except Exception as e:
            diagnostico.append({'modelo': sheet['modelo'], 'gid': sheet['gid'], 'status': '?', 'rows': 0, 'col_score': '—', 'err': str(e)[:100]})
            continue
    if not dfs:
        return pd.DataFrame(), diagnostico
    out = pd.concat(dfs, ignore_index=True)
    out['DOMINIO'] = out['DOMINIO'].astype(str).str.strip().str.upper().str.replace(r'\s+', '', regex=True)
    out['MES']     = pd.to_datetime(out['MES'], errors='coerce')
    out['SCORE_CONDUCCION'] = (out['SCORE_CONDUCCION'].astype(str)
                               .str.replace(',', '.').str.replace(r'[^\d.]', '', regex=True))
    out['SCORE_CONDUCCION'] = pd.to_numeric(out['SCORE_CONDUCCION'], errors='coerce')
    out = out[out['MES'].notna() & (out['DOMINIO'].str.len() > 2) & out['SCORE_CONDUCCION'].notna()]
    return out[['DOMINIO','MES','SCORE_CONDUCCION']].reset_index(drop=True), diagnostico

@st.cache_data(ttl=600)
def cargar_reparaciones():
    """Lee hoja gastos reparaciones. Col A=Periodo (MM/YYYY), B=Patente, C=Monto."""
    diag = {'url': REP_URL, 'status': '?', 'rows': 0, 'cols': [], 'err': '', 'sample': []}
    try:
        r = requests.get(REP_URL, timeout=15)
        diag['status'] = r.status_code
        if r.status_code != 200:
            diag['err'] = f'HTTP {r.status_code} — hoja no publicada o gid incorrecto'
            return pd.DataFrame(), diag
        df = pd.read_csv(REP_URL, header=0)
        df.columns = [str(c).strip() for c in df.columns]
        diag['cols'] = list(df.columns)
        diag['rows'] = len(df)
        cols = df.columns.tolist()
        if len(cols) < 3:
            diag['err'] = f'Solo {len(cols)} columnas, se necesitan 3 (Periodo, Patente, Monto)'
            return pd.DataFrame(), diag
        df = df.rename(columns={cols[0]: 'FECHA', cols[1]: 'DOMINIO', cols[2]: 'MONTO'})
        fecha_str = df['FECHA'].astype(str).str.strip()
        df['FECHA'] = pd.to_datetime(fecha_str, format='%m/%Y', errors='coerce')
        mask_na = df['FECHA'].isna()
        if mask_na.any():
            df.loc[mask_na, 'FECHA'] = pd.to_datetime(fecha_str[mask_na], dayfirst=True, errors='coerce')
        df['DOMINIO'] = df['DOMINIO'].astype(str).str.strip().str.upper().str.replace(r'\s+', '', regex=True)
        df['MONTO']   = pd.to_numeric(
            df['MONTO'].astype(str)
            .str.replace(r'[^\d.,]', '', regex=True)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False),
            errors='coerce'
        ).fillna(0)
        df = df[df['FECHA'].notna() & (df['DOMINIO'] != '') & (df['DOMINIO'] != 'NAN')].copy()
        df = df[df['MONTO'] > 0].copy()
        diag['sample'] = df.head(3).to_dict('records')
        diag['err'] = f'OK — {len(df)} registros con monto > 0'
        return df[['FECHA','DOMINIO','MONTO']].reset_index(drop=True), diag
    except Exception as e:
        diag['err'] = str(e)[:200]
        return pd.DataFrame(), diag

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
            gasoil_rows = [
                rec for rec in records
                if any(kw in str(rec.get('producto', '')).upper() for kw in ['GASOIL', 'DIESEL', 'GAS OIL'])
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
                return float(sorted(precios)[len(precios)//2]), "datos.energia.gob.ar (oficial)"
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
    return 2300.0, "referencia estimada"

    return 2025.0, "referencia estimada"
def asignar_modelo(dominio):
    d = str(dominio).strip().upper()
    if d in SWAY_PATENTES:   return 'S-Way'
    if d in SCANIA_PATENTES: return 'Scania'
    return 'Stralis'

def calcular_score_zscore(series, higher_is_better=True, k=0.4, min_sigma_pct=0.05):
    series = pd.to_numeric(series, errors='coerce')
    if series.dropna().count() <= 1:
@@ -461,105 +309,6 @@ def calcular_score_zscore(series, higher_is_better=True, k=0.4, min_sigma_pct=0.
        z = -z
    scores = 1.0 + 1.5 * np.tanh(k * z)
    return scores.clip(0.4, 2.5).fillna(1.0)

def calcular_ier_chofer(df, df_vel=None, df_manejo=None, df_carga=None):
    if 'DOMINIO' not in df.columns or df.empty:
        return pd.DataFrame()
    df_c = df[df['L100KM'] > 0].copy()
    if df_c.empty:
        return pd.DataFrame()
    agg = df_c.groupby('DOMINIO').agg(L100KM=('L100KM','mean'), KM=('KM','sum'), LITROS=('LITROS','sum')).reset_index()
    
    # ── AGREGAR CARGA PARA CALCULAR TONKML ──
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
        
    agg['TONKML'] = np.where((agg['PESO_TON']>0)&(agg['LITROS']>0), (agg['PESO_TON']*agg['KM'])/agg['LITROS'], np.nan)

    if df_vel is not None and not df_vel.empty and 'DOMINIO' in df_vel.columns:
        vel_agg = df_vel.groupby('DOMINIO').agg(
            EXCESOS=('DOMINIO','count'),
            VEL_MAX=('VELOCIDAD','max'),
            SEVERIDAD=('EXCESO_KMH','sum')
        ).reset_index()
        agg = agg.merge(vel_agg, on='DOMINIO', how='left')
        agg['EXCESOS']   = agg['EXCESOS'].fillna(0).astype(int)
        agg['VEL_MAX']   = agg['VEL_MAX'].fillna(0)
        agg['SEVERIDAD'] = agg['SEVERIDAD'].fillna(0)
    else:
        agg['EXCESOS'] = 0; agg['VEL_MAX'] = 0; agg['SEVERIDAD'] = 0.0
    if df_manejo is not None and not df_manejo.empty and 'SCORE_CONDUCCION' in df_manejo.columns:
        manejo_agg = df_manejo.groupby('DOMINIO')['SCORE_CONDUCCION'].mean().reset_index()
        agg = agg.merge(manejo_agg, on='DOMINIO', how='left')
    else:
        agg['SCORE_CONDUCCION'] = np.nan
    agg['TIENE_MANEJO'] = agg['SCORE_CONDUCCION'].notna()
    agg['MODELO'] = agg['DOMINIO'].apply(asignar_modelo)
    
    def _safe_mean(x): v = x.dropna(); return v.mean() if len(v)>0 else np.nan
    modelo_avgs = agg.groupby('MODELO').agg(
        L100KM_MOD=('L100KM','mean'),
        TONKML_MOD=('TONKML',_safe_mean),
        SEVERIDAD_MOD=('SEVERIDAD',_safe_mean),
        SCORE_MANEJO_MOD=('SCORE_CONDUCCION',_safe_mean)).reset_index()
    agg = agg.merge(modelo_avgs, on='MODELO', how='left')
    
    agg['SCORE_CONSUMO'] = 1.0; agg['SCORE_VEL'] = 1.0; agg['SCORE_MANEJO'] = 1.0
    
    for modelo in agg['MODELO'].unique():
        mask = agg['MODELO'] == modelo
        idx  = agg.index[mask]
        if mask.sum() == 0: continue
        
        # ── NUEVO SCORE CONSUMO BASADO EN TON·km/L ──
        agg.loc[idx,'SCORE_CONSUMO'] = calcular_score_zscore(
            agg.loc[idx,'TONKML'].fillna(0), 
            higher_is_better=True, 
            k=0.4, 
            min_sigma_pct=0.10
        ).values
        
        sev_log = np.log1p(agg.loc[idx,'SEVERIDAD'].astype(float))
        agg.loc[idx,'SCORE_VEL'] = calcular_score_zscore(sev_log, higher_is_better=False, k=0.4, min_sigma_pct=0.30).values
        
        manejo_idx = idx[agg.loc[idx,'SCORE_CONDUCCION'].notna()]
        if len(manejo_idx) > 0:
            agg.loc[manejo_idx,'SCORE_MANEJO'] = calcular_score_zscore(
                agg.loc[manejo_idx,'SCORE_CONDUCCION'], higher_is_better=True, k=0.4, min_sigma_pct=0.05).values

    def _ier_row(r):
        if r['TIENE_MANEJO']:
            return 0.50 * r['SCORE_CONSUMO'] + 0.40 * r['SCORE_MANEJO'] + 0.10 * r['SCORE_VEL']
        else:
            return 0.80 * r['SCORE_CONSUMO'] + 0.20 * r['SCORE_VEL']

    agg['IER'] = agg.apply(_ier_row, axis=1) * 100
    agg['IER'] = agg['IER'].round(1).fillna(100.0)
    for modelo in agg['MODELO'].unique():
        mask = agg['MODELO'] == modelo
        grp_mean = agg.loc[mask,'IER'].mean()
        if not np.isnan(grp_mean) and grp_mean > 0 and mask.sum() > 1:
            agg.loc[mask,'IER'] = (agg.loc[mask,'IER'] - grp_mean + 100).round(1)
    def clasif(v):
        if   v>=105: return '🟢 Eficiente'
        elif v>= 95: return '🟡 Normal'
        elif v>= 85: return '🟠 Atención'
        else:        return '🔴 Crítico'
    agg['CLASIFICACION'] = agg['IER'].apply(clasif)
    return agg[['DOMINIO','MODELO','IER','CLASIFICACION','L100KM','L100KM_MOD',
                'EXCESOS','SEVERIDAD','SEVERIDAD_MOD','VEL_MAX','KM','TONKML','TONKML_MOD',
                'SCORE_CONDUCCION','SCORE_MANEJO_MOD','TIENE_MANEJO',
                'SCORE_CONSUMO','SCORE_MANEJO','SCORE_VEL']].sort_values('IER', ascending=False).reset_index(drop=True)

def calcular_ier(df, df_vel=None, df_carga=None):
    if 'DOMINIO' not in df.columns or df.empty:
        return pd.DataFrame()
@@ -583,7 +332,7 @@ def calcular_ier(df, df_vel=None, df_carga=None):
        vel_counts = df_vel.groupby('DOMINIO').agg(
            EXCESOS=('DOMINIO','count'),
            VEL_MAX=('VELOCIDAD','max'),
            SEVERIDAD=('EXCESO_KMH','sum')
            SEVERIDAD=('EXCESO_KMH','sum')  # km/h acumulados sobre el límite: combina frecuencia + magnitud
        ).reset_index()
        agg = agg.merge(vel_counts, on='DOMINIO', how='left')
        agg['EXCESOS']   = agg['EXCESOS'].fillna(0).astype(int)
@@ -592,20 +341,35 @@ def calcular_ier(df, df_vel=None, df_carga=None):
    else:
        agg['EXCESOS'] = 0; agg['VEL_MAX'] = 0; agg['SEVERIDAD'] = 0.0
    agg['MODELO'] = agg['DOMINIO'].apply(asignar_modelo)
    if df_carga is not None and not df_carga.empty and 'MES_PERIODO' in df.columns:
        meses_activos = df['MES_PERIODO'].dropna().unique()
        carga_periodo = df_carga[df_carga['MES'].isin(meses_activos)]
        if not carga_periodo.empty:
    # Alineación temporal: TONKML usa SOLO meses con telemetría Y carga simultánea
    agg['KM_CARGA']     = 0.0
    agg['LITROS_CARGA'] = 0.0
    if df_carga is not None and not df_carga.empty and 'MES_PERIODO' in df_c.columns and 'MES' in df_carga.columns:
        meses_telem  = set(df_c['MES_PERIODO'].dropna().unique())
        meses_carga  = set(df_carga['MES'].dropna().unique())
        meses_comunes = meses_telem & meses_carga
        if meses_comunes:
            carga_periodo = df_carga[df_carga['MES'].isin(meses_comunes)]
            df_c_carga    = df_c[df_c['MES_PERIODO'].isin(meses_comunes)]
            carga_agg = carga_periodo.groupby('DOMINIO')['PESO_TON'].sum().reset_index()
            km_lts_carga = df_c_carga.groupby('DOMINIO').agg(
                KM_CARGA=('KM','sum'), LITROS_CARGA=('LITROS','sum')
            ).reset_index()
            agg = agg.drop(columns=['KM_CARGA','LITROS_CARGA'])
            agg = agg.merge(carga_agg, on='DOMINIO', how='left')
            agg['PESO_TON'] = agg['PESO_TON'].fillna(0)
            agg = agg.merge(km_lts_carga, on='DOMINIO', how='left')
            agg['PESO_TON']     = agg['PESO_TON'].fillna(0)
            agg['KM_CARGA']     = agg['KM_CARGA'].fillna(0)
            agg['LITROS_CARGA'] = agg['LITROS_CARGA'].fillna(0)
        else:
            agg['PESO_TON'] = 0.0
    else:
        agg['PESO_TON'] = 0.0
    # TONKML alineado: usa KM y LITROS de los mismos meses que PESO_TON
    agg['TONKML'] = np.where(
        (agg['PESO_TON']>0)&(agg['LITROS']>0),
        (agg['PESO_TON']*agg['KM'])/agg['LITROS'], np.nan)
        (agg['PESO_TON']>0)&(agg['LITROS_CARGA']>0),
        (agg['PESO_TON']*agg['KM_CARGA'])/agg['LITROS_CARGA'], np.nan)
    tiene_carga = agg['PESO_TON'].sum() > 0
    def _safe_mean(x):
        v = x.dropna(); return v.mean() if len(v)>0 else np.nan
    modelo_avgs = agg.groupby('MODELO').agg(
@@ -614,38 +378,45 @@ def _safe_mean(x):
        SEVERIDAD_MOD=('SEVERIDAD','mean'),
        TONKML_MOD=('TONKML',_safe_mean)).reset_index()
    agg = agg.merge(modelo_avgs, on='MODELO', how='left')
    for col in ['SCORE_CONSUMO','SCORE_KM','SCORE_VEL','SCORE_RAL']:
    for col in ['SCORE_CONSUMO','SCORE_KM','SCORE_VEL','SCORE_CARGA']:
        agg[col] = 1.0
    for modelo in agg['MODELO'].unique():
        mask = agg['MODELO']==modelo
        idx  = agg.index[mask]
        if mask.sum()==0: continue
        
        # ── NUEVO SCORE CONSUMO BASADO EN TON·km/L ──
        agg.loc[idx,'SCORE_CONSUMO'] = calcular_score_zscore(
            agg.loc[idx,'TONKML'].fillna(0), 
            higher_is_better=True, 
            k=0.4, 
            min_sigma_pct=0.10
        ).values
        
        agg.loc[idx,'SCORE_CONSUMO'] = calcular_score_zscore(agg.loc[idx,'L100KM'], higher_is_better=False, k=0.4, min_sigma_pct=0.05).values
        agg.loc[idx,'SCORE_KM'] = calcular_score_zscore(agg.loc[idx,'KM'], higher_is_better=True, k=0.4, min_sigma_pct=0.05).values
        # Excesos: SEVERIDAD = suma de km/h sobre el límite (frecuencia × magnitud combinadas)
        # Ej: 5 eventos a 95 km/h (sum=35) > 10 eventos a 89 km/h (sum=10) → 95 penaliza más
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
        if tiene_carga:
            carga_vals = agg.loc[idx,'TONKML']
            valid_c    = carga_vals.dropna()
            valid_c    = valid_c[valid_c>0]
            if len(valid_c)>1:
                sc = calcular_score_zscore(carga_vals.where(carga_vals>0), higher_is_better=True, k=0.4, min_sigma_pct=0.10).fillna(1.0)
                agg.loc[idx,'SCORE_CARGA'] = sc.values
    # Ponderación con renormalización si no hay carga:
    #   CON carga:  40% L/100km · 40% ton·km/L · 10% KM · 10% excesos severidad
    #   SIN carga:  80% L/100km · 10% KM · 10% excesos severidad
    # Ralentí excluido del cálculo (dato S-Way no confiable)
    if tiene_carga:
        agg['IER'] = (
            0.40 * agg['SCORE_CONSUMO'] +
            0.40 * agg['SCORE_CARGA']   +
            0.10 * agg['SCORE_KM']      +
            0.10 * agg['SCORE_VEL']
        ).mul(100).round(1)
    else:
        agg['IER'] = (
            0.80 * agg['SCORE_CONSUMO'] +
            0.10 * agg['SCORE_KM']      +
            0.10 * agg['SCORE_VEL']
        ).mul(100).round(1)
    agg['IER'] = agg['IER'].fillna(100.0)
    for modelo in agg['MODELO'].unique():
        mask     = agg['MODELO']==modelo
        grp_mean = agg.loc[mask,'IER'].mean()
        if not np.isnan(grp_mean) and grp_mean>0 and mask.sum()>1:
            agg.loc[mask,'IER'] = (agg.loc[mask,'IER']-grp_mean+100).round(1)
    # Doble centrado eliminado: Z-score ya centra cada modelo en ~100.
    # Re-restar grp_mean achataba la varianza tras el clip [0.4, 2.5].
    def clasif(v):
        if   v>=105: return '🟢 Eficiente'
        elif v>= 95: return '🟡 Normal'
@@ -658,27 +429,20 @@ def clasif(v):
            'L100KM','L100KM_MOD','RALENTI_PCT','RAL_MOD',
            'KM','KM_MOD','LITROS','EXCESOS','SEVERIDAD','SEVERIDAD_MOD','VEL_MAX','EXCESOS_MOD',
            'PESO_TON','TONKML','TONKML_MOD',
            'SCORE_CONSUMO','SCORE_KM','SCORE_VEL','SCORE_RAL']
            'SCORE_CONSUMO','SCORE_KM','SCORE_VEL','SCORE_CARGA']
    if 'MESES' in agg.columns: keep.append('MESES')
    return agg[keep].sort_values('IER', ascending=False).reset_index(drop=True)

with st.spinner('Cargando datos...'):
    df_raw, df_unid       = cargar_datos()
    df_vel_raw            = cargar_velocidad()
    df_carga_raw          = cargar_carga()
    df_viajes_raw         = cargar_viajes_todos()
    df_rep_raw, rep_diag  = cargar_reparaciones()
    df_manejo_raw, manejo_diag = cargar_datos_manejo()

with st.spinner('Cargando telemetría, velocidades y datos de carga...'):
    df_raw, df_unid = cargar_datos()
    df_vel_raw      = cargar_velocidad()
    df_carga_raw    = cargar_carga()
if df_raw.empty:
    st.warning('No se pudieron cargar datos.')
    st.stop()

precio_gasoil, precio_fuente = obtener_precio_gasoil()
st.markdown(LIGHT_CSS, unsafe_allow_html=True)
st.markdown(DARK_CSS, unsafe_allow_html=True)
if 'DOMINIO' in df_raw.columns:
    df_raw['MODELO'] = df_raw['DOMINIO'].apply(asignar_modelo)

df_full = df_raw.copy()
anios_disponibles = (sorted(df_full['FECHA'].dt.year.dropna().unique().tolist(), reverse=True)
                     if 'FECHA' in df_full.columns else [2025])
@@ -689,10 +453,6 @@ def clasif(v):
    df_vel_anio = df_vel_raw[df_vel_raw['FECHA'].dt.year==anio_sel].copy()
else:
    df_vel_anio = df_vel_raw.copy()

desde_periodo = None
hasta_periodo = None
patentes_sel = []
if 'FECHA' in df.columns and df['FECHA'].notna().any():
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-filter-header">🔍 Filtros</div>', unsafe_allow_html=True)
@@ -710,11 +470,9 @@ def clasif(v):
    patentes_sel  = st.sidebar.multiselect('Patente', patentes_disp, default=[], placeholder="Todas las patentes")
    if marcas_sel   and 'MARCA'   in df.columns: df = df[df['MARCA'].isin(marcas_sel)]
    if patentes_sel and 'DOMINIO' in df.columns: df = df[df['DOMINIO'].isin(patentes_sel)]

if df.empty:
    st.warning(f'Sin datos para {anio_sel} con los filtros seleccionados.')
    st.stop()

df['MES_PERIODO'] = df['FECHA'].dt.to_period('M')
df['MES_NUM']     = df['FECHA'].dt.month
meses_df = df.groupby('MES_PERIODO').agg(LITROS=('LITROS','sum'),KM=('KM','sum')).reset_index().sort_values('MES_PERIODO')
@@ -729,100 +487,22 @@ def clasif(v):
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

if not df_manejo_raw.empty and 'MES' in df_manejo_raw.columns:
    _manejo_periodos = df_manejo_raw['MES'].dt.to_period('M')
    if desde_periodo is not None and hasta_periodo is not None:
        df_manejo_filtrado = df_manejo_raw[(_manejo_periodos >= desde_periodo) & (_manejo_periodos <= hasta_periodo)].copy()
    else:
        df_manejo_filtrado = df_manejo_raw.copy()
else:
    df_manejo_filtrado = df_manejo_raw.copy()

df_ier        = calcular_ier(df, df_vel_filtrado, df_carga=df_carga_raw)
df_ier_chofer = calcular_ier_chofer(df, df_vel_filtrado, df_manejo_filtrado, df_carga=df_carga_raw)

df_ier = calcular_ier(df, df_vel_filtrado, df_carga=df_carga_raw)
total_excesos  = len(df_vel_filtrado) if not df_vel_filtrado.empty else 0
vel_max_global = (df_vel_filtrado['VELOCIDAD'].max() if not df_vel_filtrado.empty and 'VELOCIDAD' in df_vel_filtrado.columns else 0)

def layout_light(fig, **kwargs):
    fig.update_layout(
        paper_bgcolor=PLOTLY_BG,
        plot_bgcolor='rgba(30,41,59,0.6)',
        font=dict(color=PLOTLY_FONT),
        **kwargs
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA DIAGNÓSTICO
# ═══════════════════════════════════════════════════════════════════════════════
if pg == "🔧 Diagnóstico":
    st.markdown('# 🔧 Diagnóstico de Fuentes de Datos')
    st.markdown(f'**Fecha:** `{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}`')
    st.divider()
    st.markdown('## 📊 Datos Manejo (Score Conducción)')
    st.markdown(f'**URL base:** `{MANEJO_BASE_NEW}`')
    for d in manejo_diag:
        status_ok = d['status'] == 200 and d['err'] == 'OK'
        cls = 'diag-ok' if status_ok else 'diag-fail'
        icon = '✅' if status_ok else '❌'
        st.markdown(f'<div class="diag-row">{icon} <b>{d["modelo"]}</b> · gid={d["gid"]} · HTTP <span class="{cls}">{d["status"]}</span> · filas: {d["rows"]} · col score: <code>{d["col_score"]}</code> · {d["err"]}</div>', unsafe_allow_html=True)
    if df_manejo_raw.empty:
        st.error(f'❌ No se cargaron datos de manejo. Revisá los gids y que la hoja esté publicada como CSV.')
    else:
        st.success(f'✅ {len(df_manejo_raw)} registros cargados · {df_manejo_raw["DOMINIO"].nunique()} patentes únicas')
        with st.expander('Ver muestra de datos manejo'):
            st.dataframe(df_manejo_raw.head(20), use_container_width=True)
    st.divider()
    st.markdown('## 🔧 Reparaciones')
    st.markdown(f'**URL:** `{rep_diag["url"]}`')
    status_rep = rep_diag['status']
    cls_rep = 'diag-ok' if status_rep == 200 else 'diag-fail'
    st.markdown(f'<div class="diag-row">HTTP <span class="{cls_rep}">{status_rep}</span> · filas crudas: {rep_diag["rows"]} · resultado: {rep_diag["err"]}</div>', unsafe_allow_html=True)
    if rep_diag['cols']:
        st.markdown(f'**Columnas detectadas:** `{rep_diag["cols"]}`')
    if rep_diag['sample']:
        st.markdown('**Muestra (primeras 3 filas):**')
        st.dataframe(pd.DataFrame(rep_diag['sample']), use_container_width=True)
    if df_rep_raw.empty:
        st.error(f'❌ No se cargaron reparaciones. Causas probables:')
        st.markdown('''
        - El gid `33208473` no está publicado en la URL nueva.
        - La hoja está vacía o no tiene el formato Fecha | Patente | Monto.
        - La URL apunta a otro documento.
        **Pasos:**
        1. Abrí el Google Sheet con la URL pub.
        2. En la pestaña de reparaciones, mirá la URL del navegador → copiá el número después de `gid=`.
        3. Reemplazá `33208473` en `REP_URL` (línea 49 del código) por ese número.
        ''')
    else:
        st.success(f'✅ {len(df_rep_raw)} registros · {df_rep_raw["DOMINIO"].nunique()} patentes · total: ${df_rep_raw["MONTO"].sum():,.0f}')
        with st.expander('Ver muestra de reparaciones'):
            st.dataframe(df_rep_raw.head(20), use_container_width=True)
    st.divider()
    st.markdown('## 📡 Resto de fuentes')
    st.markdown(f'<div class="diag-row">{"✅" if not df_raw.empty else "❌"} <b>Telemetría</b> · {len(df_raw)} filas · {df_raw["DOMINIO"].nunique() if not df_raw.empty else 0} patentes</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="diag-row">{"✅" if not df_vel_raw.empty else "❌"} <b>Velocidades</b> · {len(df_vel_raw)} eventos >{LIMITE_VELOCIDAD} km/h</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="diag-row">{"✅" if not df_carga_raw.empty else "❌"} <b>Carga (BI)</b> · {len(df_carga_raw)} registros mensuales</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="diag-row">{"✅" if not df_viajes_raw.empty else "❌"} <b>Viajes totales</b> · {len(df_viajes_raw)} viajes</div>', unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 1 — DASHBOARD PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
@@ -834,6 +514,7 @@ def layout_light(fig, **kwargs):
        <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>Expreso Diemar &mdash; Dashboard LAD {anio_sel}</div>
        <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Telemetría flota LAD &middot; Año {anio_sel} &middot; Actualización automática</div>
        </div>""", unsafe_allow_html=True)
    st.markdown(f'<div style="margin-bottom:12px;"><span class="price-badge">&#9981; Precio gasoil: <b>${precio_gasoil:,.0f}/L</b></span>&nbsp;&nbsp;<span style="font-size:.75rem;color:#64748b;">Fuente: {precio_fuente}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">Métricas Globales — {anio_sel}</div>', unsafe_allow_html=True)
    lts_total  = df['LITROS'].sum() if 'LITROS' in df.columns else 0
    kms_total  = df['KM'].sum()     if 'KM'     in df.columns else 0
@@ -858,42 +539,14 @@ def kpi(cont, color, label, value, sub=''):
    kpi(k5,'kpi-green','🚛 Unidades activas',f'{n_unidades}','dominios únicos')
    _ral_sub = (f'{ralenti_total:,.0f} L · {ralenti_delta_txt}' if ralenti_delta_txt else f'{ralenti_total:,.0f} L en ralentí')
    kpi(k6,'kpi-amber','⏱️ % Ralentí',f'{ralenti_pct:.1f}%',_ral_sub)

    patentes_filtradas = df['DOMINIO'].dropna().unique().tolist() if 'DOMINIO' in df.columns else []
    if not df_rep_raw.empty and patentes_filtradas and desde_periodo is not None and hasta_periodo is not None:
        rep_periodo = df_rep_raw['FECHA'].dt.to_period('M')
        mask = (
            (rep_periodo >= desde_periodo) &
            (rep_periodo <= hasta_periodo) &
            (df_rep_raw['DOMINIO'].isin(patentes_filtradas))
        )
        df_rep_anio = df_rep_raw[mask].copy()
    else:
        df_rep_anio = pd.DataFrame()
    k7,k8,k9 = st.columns(3)
    if not df_rep_anio.empty:
        total_rep   = df_rep_anio['MONTO'].sum()
        n_eventos   = len(df_rep_anio)
        pat_mayor   = df_rep_anio.groupby('DOMINIO')['MONTO'].sum().idxmax()
        monto_mayor = df_rep_anio.groupby('DOMINIO')['MONTO'].sum().max()
        prom_rep    = total_rep / max(df_rep_anio['DOMINIO'].nunique(), 1)
        kpi(k7,'kpi-red','🔧 Total reparaciones',f'${total_rep/1e6:.2f}M',f'{n_eventos} registros')
        kpi(k8,'kpi-amber','⚠️ Mayor gasto',pat_mayor,f'${monto_mayor:,.0f}')
        kpi(k9,'','📊 Prom. por patente',f'${prom_rep:,.0f}',f'{df_rep_anio["DOMINIO"].nunique()} patentes')
    else:
        kpi(k7,'kpi-red','🔧 Total reparaciones','—','sin datos en el rango')
        kpi(k8,'','⚠️ Mayor gasto','—','')
        kpi(k9,'','📊 Prom. por patente','—','')
        if df_rep_raw.empty:
            st.warning(f'⚠️ Reparaciones no disponibles. HTTP {rep_diag["status"]}: {rep_diag["err"]}. Andá a la pestaña **🔧 Diagnóstico** para más detalles.')
    st.divider()
    st.markdown(f'<div class="sec-title">Rendimiento por Modelo — {anio_sel}</div>', unsafe_allow_html=True)
    def stats_modelo(patentes_lista):
        if 'DOMINIO' not in df.columns: return {'l100':0,'lts':0,'kms':0,'n':0,'total':len(patentes_lista)}
        if 'DOMINIO' not in df.columns: return {'l100':0,'lts':0,'kms':0,'n':0}
        sub = df[df['DOMINIO'].isin(patentes_lista)]
        if sub.empty: return {'l100':0,'lts':0,'kms':0,'n':0,'total':len(patentes_lista)}
        if sub.empty: return {'l100':0,'lts':0,'kms':0,'n':0}
        lts=sub['LITROS'].sum(); kms=sub['KM'].sum()
        return {'l100':round(lts/kms*100,2) if kms>0 else 0,'lts':lts,'kms':kms,'n':sub['DOMINIO'].nunique(),'total':len(patentes_lista)}
        return {'l100':round(lts/kms*100,2) if kms>0 else 0,'lts':lts,'kms':kms,'n':sub['DOMINIO'].nunique()}
    todas_patentes   = df['DOMINIO'].dropna().unique().tolist() if 'DOMINIO' in df.columns else []
    stralis_patentes = [p for p in todas_patentes if p not in SWAY_PATENTES and p not in SCANIA_PATENTES]
    s_sway=stats_modelo(SWAY_PATENTES); s_scania=stats_modelo(SCANIA_PATENTES); s_stralis=stats_modelo(stralis_patentes)
@@ -906,10 +559,9 @@ def stats_modelo(patentes_lista):
            st.markdown(f'<div class="truck-img-box"><img src="{img_url}" alt="{modelo}" /></div>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            sc1,sc2,sc3=st.columns(3)
            sc1.metric('L/100km', f"{s['l100']:.1f}" if s['l100']>0 else '—')
            sc2.metric('Unidades', f"{s['total']}")
            lts_fmt = f"{s['lts']/1000:.1f}k" if s['lts']>=1000 else f"{s['lts']:.0f}"
            sc3.metric(f'Lts {anio_sel}', lts_fmt if s['lts']>0 else '—')
            sc1.metric(f'{modelo} — L/100km',f"{s['l100']:.1f}" if s['l100']>0 else '—')
            sc2.metric('Unidades',f"{s['n']}")
            sc3.metric(f'Litros {anio_sel}',f"{s['lts']:,.0f}" if s['lts']>0 else '—')
            st.caption(f"Patentes: {pats_label} | {s['kms']:,.0f} km")
    st.divider()
    st.markdown(f'<div class="sec-title">Ranking de Eficiencia — {anio_sel}</div>', unsafe_allow_html=True)
@@ -920,7 +572,7 @@ def render_ranking(col,titulo,df_rank,color_fn):
            if df_rank.empty: st.info('Sin datos.'); return
            vmin,vmax=df_rank['L100KM'].min(),df_rank['L100KM'].max()
            rh='<div style="background:#1e293b;border-radius:12px;padding:16px;">'
            rh+='<div style="font-size:.72rem;display:flex;justify-content:space-between;color:#64748b;margin-bottom:6px;"><span>Unidad</span><span>L/100km</span></div>'
            rh+='<div style="font-size:.72rem;display:flex;justify-content:space-between;color:#94a3b8;margin-bottom:6px;"><span>Unidad</span><span>L/100km</span></div>'
            for i,(_,r) in enumerate(df_rank.iterrows(),1):
                v=r['L100KM']; pct=int((v-vmin)/(vmax-vmin)*100) if vmax!=vmin else 50; cb=color_fn(i)
                rh+=(f'<div class="rank-row"><div class="rank-num">#{i}</div><div class="rank-dom">{r["DOMINIO"]}</div>'
@@ -935,104 +587,102 @@ def render_ranking(col,titulo,df_rank,color_fn):
        render_ranking(rcol2,'TOP 10 menos eficientes (mayor L/100km)',base.sort_values('L100KM',ascending=False).head(10),
                       lambda i:'#ef4444' if i<=3 else ('#f59e0b' if i<=6 else '#22c55e'))
    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    #  SECCIÓN IER-CHOFER
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-title">📊 IER-Chofer — Lo que controla el conductor — {anio_sel}</div>', unsafe_allow_html=True)
    tiene_manejo_disp = not df_ier_chofer.empty and df_ier_chofer['TIENE_MANEJO'].any()
    pond_manejo = ('' if tiene_manejo_disp else '⚠️ sin datos este período')

    st.markdown(f'<div class="sec-title">📊 Índice de Eficiencia Relativa (IER v4) — {anio_sel}</div>', unsafe_allow_html=True)
    tiene_vel       = total_excesos>0
    tiene_carga_ier = (not df_ier.empty and 'PESO_TON' in df_ier.columns and df_ier['PESO_TON'].sum()>0)
    pond_txt = (
        f"<b>40%</b> L/100km &nbsp;·&nbsp; "
        f"<b>40%</b> ton·km/L {'📦' if tiene_carga_ier else '⚠️ sin datos'} &nbsp;·&nbsp; "
        f"<b>10%</b> KM totales &nbsp;·&nbsp; "
        f"<b>10%</b> Severidad vel. {'✅' if tiene_vel else '⚠️'} "
        f"&nbsp;·&nbsp; <i>Ralentí: excluido (dato S-Way no confiable)</i>"
    )
    st.markdown(f"""<div class="ier-info-box">
    <b>¿Qué mide?</b> Eficiencia productiva, estilo de conducción y velocidad.<br>
    <b>Ponderación:</b> <b>50%</b> Eficiencia (ton·km/L) &nbsp;·&nbsp; <b>40%</b> Score conducción {pond_manejo} &nbsp;·&nbsp; <b>10%</b> Severidad velocidad<br>
    <b>Eficiencia (ton·km/L):</b> Premia mover carga gastando menos. Reemplaza al clásico L/100km para no castigar a los camiones que van pesados.<br>
    <b>¿Qué es el IER v5?</b> Métrica estadísticamente justa: cada camión se compara <b>solo contra el promedio de su propio modelo</b> — Stralis vs Stralis, S‑Way vs S‑Way, Scania vs Scania.<br>
    <b>Scoring:</b> Z-Score + Tanh. El promedio del grupo obtiene IER ≈ 100. Mayor IER = mejor rendimiento relativo.<br>
    <b>Velocidad:</b> mide <b>severidad promedio</b> (km/h sobre el límite en promedio) — no cantidad de eventos. Ir siempre a 89 pesa menos que ir pocas veces a 95.<br>
    <b>Ponderación:</b>&nbsp;{pond_txt}<br>
    <b>Escala:</b>&nbsp;🟢 Eficiente ≥105 &nbsp;·&nbsp; 🟡 Normal 95–105 &nbsp;·&nbsp; 🟠 Atención 85–95 &nbsp;·&nbsp; 🔴 Crítico &lt;85
    </div>""", unsafe_allow_html=True)

    if not df_ier_chofer.empty:
        cats_c=df_ier_chofer['CLASIFICACION'].value_counts()
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
        ic1.metric('🟢 Eficiente',int(cats_c.get('🟢 Eficiente',0)),'IER ≥ 105')
        ic2.metric('🟡 Normal',int(cats_c.get('🟡 Normal',0)),'IER 95–105')
        ic3.metric('🟠 Atención',int(cats_c.get('🟠 Atención',0)),'IER 85–95')
        ic4.metric('🔴 Crítico',int(cats_c.get('🔴 Crítico',0)),'IER < 85')
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

        df_ch_sorted = df_ier_chofer.sort_values(['MODELO','IER'], ascending=[True, True])

        fig_ch=go.Figure()
        MODELO_COLOR={'S-Way':'#3b82f6','Scania':'#f97316','Stralis':'#a78bfa'}
        df_ier_sorted = df_ier.sort_values(['MODELO','IER'],ascending=[True,False])
        fig_ier=go.Figure()
        MODELO_COLOR={'S-Way':'#60a5fa','Scania':'#f97316','Stralis':'#a78bfa'}
        for modelo in MODELO_COLOR:
            subset=df_ch_sorted[df_ch_sorted['MODELO']==modelo]
            subset=df_ier_sorted[df_ier_sorted['MODELO']==modelo]
            if subset.empty: continue
            hover=[]
            for _,row in subset.iterrows():
                sev=row.get('SEVERIDAD',0)
                sc_m = row.get('SCORE_CONDUCCION', np.nan)
                sc_m_txt = f"{sc_m:.2f}/10 · score: {row['SCORE_MANEJO']:.2f}" if not np.isnan(sc_m) else "sin datos"
                hover.append(f"<b>{row['DOMINIO']}</b> ({row['MODELO']})<br>IER-Chofer: <b>{row['IER']:.1f}</b><br>"
                             f"ton·km/L: {row.get('TONKML',0):.2f} (L/100km: {row['L100KM']:.2f})<br>"
                             f"Score conducción: {sc_m_txt}<br>"
                             f"Severidad: {sev:.0f} km/h")
            fig_ch.add_trace(go.Bar(y=subset['DOMINIO'],x=subset['IER'],name=modelo,orientation='h',
                marker=dict(color=[ier_bar_color(v) for v in subset['IER']],line=dict(color='rgba(15,23,42,0.15)',width=1)),
                text=[f"{v:.1f}" for v in subset['IER']],textposition='outside',textfont=dict(color=PLOTLY_FONT,size=10),
                tkml_txt=(f"Carga: {row['TONKML']:.1f} ton·km/L (prom: {row['TONKML_MOD']:.1f})" if row['TONKML']>0 else "Carga: sin datos")
                ral_txt = f"⚠️ dato no confiable (S-Way)" if row['MODELO']=='S-Way' else f"{row['RALENTI_PCT']:.1f}%"
                severidad = row.get('SEVERIDAD', 0)
                hover.append(f"<b>{row['DOMINIO']}</b> ({row['MODELO']})<br>IER: <b>{row['IER']:.1f}</b> — {row['CLASIFICACION']}<br>"
                             f"L/100km: {row['L100KM']:.2f} (prom {row['MODELO']}: {row['L100KM_MOD']:.2f}) score: {row['SCORE_CONSUMO']:.2f}<br>"
                             f"% Ralentí: {ral_txt} (display solo, no entra en IER)<br>"
                             f"Vel. severidad: {severidad:.0f} km/h acum. · {int(row['EXCESOS'])} eventos · score: {row['SCORE_VEL']:.2f}<br>"
                             f"Vel. máx: {row['VEL_MAX']:.0f} km/h<br>KM total: {row['KM']:,.0f}  score: {row['SCORE_KM']:.2f}<br>"
                             f"{tkml_txt}  score: {row['SCORE_CARGA']:.2f}")
            fig_ier.add_trace(go.Bar(y=subset['DOMINIO'],x=subset['IER'],name=modelo,orientation='h',
                marker=dict(color=[ier_bar_color(v) for v in subset['IER']],line=dict(color='rgba(255,255,255,0.15)',width=1)),
                text=[f"{v:.1f}" for v in subset['IER']],textposition='outside',textfont=dict(color='#e2e8f0',size=10),
                hovertemplate='%{customdata}<extra></extra>',customdata=hover))

        ch_min=max(50,df_ch_sorted['IER'].min()-10); ch_max=min(175,df_ch_sorted['IER'].max()+25)
        fig_ch.add_vline(x=100,line_dash='solid',line_color='#f59e0b',line_width=2.5,
                         annotation_text='Base 100',annotation_position='top',annotation_font_color='#fbbf24',annotation_font_size=11)
        layout_light(fig_ch,
            barmode='overlay',
            xaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS),title=dict(text='IER-Chofer (100 = promedio de su modelo)',font=dict(color=PLOTLY_AXIS)),range=[ch_min,ch_max]),
            yaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS,size=10),categoryorder='array',categoryarray=df_ch_sorted['DOMINIO'].tolist()),
            height=max(380,len(df_ch_sorted)*44),margin=dict(l=10,r=130,t=60,b=30),showlegend=False)
        st.plotly_chart(fig_ch, use_container_width=True)

        if df_manejo_raw.empty:
            st.warning(f'⚠️ Score conducción no disponible. Andá a **🔧 Diagnóstico** para ver qué falla.')

        st.markdown(f'<div class="sec-title">🏆 Unidad más eficiente por grupo — {anio_sel}</div>', unsafe_allow_html=True)

        modelos_presentes = df_ch_sorted['MODELO'].unique().tolist()
        best_cols = st.columns(len(modelos_presentes)) if modelos_presentes else []

        COLOR_MODELO_BADGE = {'S-Way': '#3b82f6', 'Scania': '#f97316', 'Stralis': '#a78bfa'}

        for col_b, modelo_b in zip(best_cols, modelos_presentes):
            grupo = df_ch_sorted[df_ch_sorted['MODELO'] == modelo_b]
            if grupo.empty:
                continue
            best = grupo.loc[grupo['IER'].idxmax()]
            badge_color = COLOR_MODELO_BADGE.get(modelo_b, '#2563eb')
            sc_val = best.get('SCORE_CONDUCCION', np.nan)
            sc_txt = f"{sc_val:.2f}/10" if not (isinstance(sc_val, float) and np.isnan(sc_val)) else "—"
            sev_val = best.get('SEVERIDAD', 0)

            with col_b:
                st.markdown(f"""
                <div class="best-unit-card" style="border-left-color:{badge_color};">
                  <div class="best-unit-title">🏆 Mejor en {modelo_b}</div>
                  <div class="best-unit-modelo">{modelo_b}</div>
                  <div class="best-unit-dom">{best['DOMINIO']}</div>
                  <div class="best-unit-ier">IER {best['IER']:.1f} &nbsp; {best['CLASIFICACION']}</div>
                  <div class="best-unit-stats">
                    ⚖️ ton·km/L: <b style="color:#f1f5f9;">{best.get('TONKML',0):.2f}</b> (⛽ {best['L100KM']:.2f} L/100km)<br>
                    🎯 Score conducción: <b style="color:#f1f5f9;">{sc_txt}</b><br>
                    🚦 Severidad velocidad: <b style="color:#f1f5f9;">{sev_val:.0f} km/h</b>
                  </div>
                </div>""", unsafe_allow_html=True)

        ier_min=max(50,df_ier_sorted['IER'].min()-10); ier_max=min(175,df_ier_sorted['IER'].max()+25)
        fig_ier.add_vline(x=100,line_dash='solid',line_color='#f59e0b',line_width=2.5,
                          annotation_text='Base 100',annotation_position='top',annotation_font_color='#fbbf24',annotation_font_size=11)
        fig_ier.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),barmode='overlay',
            xaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8'),title=dict(text='IER  (100 = promedio de su modelo)',font=dict(color='#64748b')),range=[ier_min,ier_max]),
            yaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8',size=10),categoryorder='array',categoryarray=df_ier_sorted['DOMINIO'].tolist()),
            height=max(380,len(df_ier_sorted)*44),margin=dict(l=10,r=130,t=60,b=30),showlegend=False)
        st.plotly_chart(fig_ier, use_container_width=True)
        st.caption('Verde = mejor que su modelo · Rojo = peor · Línea amarilla = base 100 · Hover para detalle completo')
        with st.expander('📋 Ver tabla detallada IER (todos los componentes)'):
            show_cols=['DOMINIO','MODELO','IER','CLASIFICACION','L100KM','L100KM_MOD','RALENTI_PCT','RAL_MOD',
                       'EXCESOS','SEVERIDAD','SEVERIDAD_MOD','VEL_MAX','KM','PESO_TON','TONKML','TONKML_MOD',
                       'SCORE_CONSUMO','SCORE_KM','SCORE_VEL','SCORE_CARGA']
            ier_show=df_ier[[c for c in show_cols if c in df_ier.columns]].copy()
            col_rename={'DOMINIO':'Patente','MODELO':'Modelo','IER':'IER','CLASIFICACION':'Clasificación',
                'L100KM':'L/100km','L100KM_MOD':'Prom L/100km',
                'RALENTI_PCT':'% Ralentí (⚠️ no entra en IER)','RAL_MOD':'% Ral Prom',
                'EXCESOS':f'Cant. Excesos >{LIMITE_VELOCIDAD}km/h',
                'SEVERIDAD':'Severidad total (km/h acum.)','SEVERIDAD_MOD':'Sev. total prom mod.',
                'VEL_MAX':'Vel. Máx (km/h)',
                'KM':'KM total','PESO_TON':'Peso (ton)','TONKML':'ton·km/L','TONKML_MOD':'ton·km/L prom mod.',
                'SCORE_CONSUMO':'S.Consumo (40%)','SCORE_KM':'S.KM (10%)','SCORE_VEL':'S.Vel (10%)','SCORE_CARGA':'S.Carga (40%)'}
            ier_show=ier_show.rename(columns=col_rename)
            for c in ['IER','L/100km','Prom L/100km','Severidad total (km/h acum.)','Sev. total prom mod.']:
                if c in ier_show.columns: ier_show[c]=ier_show[c].round(1)
            for c in ['S.Consumo (40%)','S.KM (10%)','S.Vel (10%)','S.Carga (40%)']:
                if c in ier_show.columns: ier_show[c]=ier_show[c].round(3)
            if 'KM total' in ier_show.columns: ier_show['KM total']=ier_show['KM total'].apply(lambda x:f'{x:,.0f}')
            st.dataframe(ier_show, use_container_width=True, hide_index=True)
    else:
        st.info('Sin datos suficientes para calcular el IER.')
    if not df_vel_filtrado.empty and 'DOMINIO' in df_vel_filtrado.columns:
        st.divider()
        st.markdown(f'<div class="sec-title">🚨 Ranking Severidad Velocidad >{LIMITE_VELOCIDAD} km/h — {anio_sel}</div>', unsafe_allow_html=True)
        st.caption(f'Métrica: suma total de km/h sobre el límite (frecuencia × magnitud). 5 eventos a 95 km/h (sum=35) es más grave que 10 eventos a 89 km/h (sum=10).')
        vel_rank=(df_vel_filtrado.groupby('DOMINIO').agg(
            CANTIDAD=('DOMINIO','count'),
            VEL_MAX=('VELOCIDAD','max'),
@@ -1042,15 +692,25 @@ def ier_bar_color(v):
        vel_rank['MODELO']=vel_rank['DOMINIO'].apply(asignar_modelo)
        fig_vel=go.Figure([go.Bar(x=vel_rank['DOMINIO'],y=vel_rank['SEVERIDAD'].round(0),
            marker_color=['#ef4444' if v==vel_rank['SEVERIDAD'].max() else '#f97316' for v in vel_rank['SEVERIDAD']],
            text=vel_rank['SEVERIDAD'].round(0).astype(int),textposition='outside',textfont=dict(color=PLOTLY_FONT,size=10),
            hovertemplate='<b>%{x}</b><br>Severidad: +%{y:.0f} km/h<extra></extra>')])
        layout_light(fig_vel,
            xaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS,size=10),tickangle=-45),
            yaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS),title=dict(text=f'km/h acumulados sobre {LIMITE_VELOCIDAD}',font=dict(color=PLOTLY_AXIS))),
            text=vel_rank['SEVERIDAD'].round(0).astype(int),textposition='outside',textfont=dict(color='#e2e8f0',size=10),
            hovertemplate='<b>%{x}</b><br>Severidad total: +%{y:.0f} km/h acumulados sobre límite<extra></extra>')])
        fig_vel.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8',size=10),tickangle=-45),
            yaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8'),title=dict(text=f'km/h acumulados sobre {LIMITE_VELOCIDAD} km/h',font=dict(color='#64748b'))),
            height=380,margin=dict(l=10,r=10,t=20,b=80),showlegend=False)
        st.plotly_chart(fig_vel, use_container_width=True)
    st.caption(f'Datos {anio_sel}: Google Sheets Expreso Diemar | Precio: {precio_fuente}')

        with st.expander('Ver tabla de excesos por unidad'):
            vel_show=vel_rank[['DOMINIO','MODELO','SEVERIDAD','CANTIDAD','VEL_MAX','VEL_PROM']].copy()
            vel_show.columns=['Patente','Modelo',f'Severidad total (km/h acum.)',f'Cant. eventos >{LIMITE_VELOCIDAD}km/h','Vel. Máx (km/h)','Vel. Prom (km/h)']
            vel_show['Vel. Máx (km/h)']=vel_show['Vel. Máx (km/h)'].round(1)
            vel_show['Vel. Prom (km/h)']=vel_show['Vel. Prom (km/h)'].round(1)
            vel_show['Severidad total (km/h acum.)']=vel_show['Severidad total (km/h acum.)'].round(1)
            st.dataframe(vel_show, use_container_width=True, hide_index=True)
    st.divider()
    with st.expander(f'Ver datos completos {anio_sel}'):
        cols_s=[c for c in ['DOMINIO','MARCA','MODELO','FECHA','KM','LITROS','L100KM','RALENTI'] if c in df.columns]
        st.dataframe(df[cols_s], use_container_width=True, height=380)
    st.caption(f'Datos {anio_sel}: Google Sheets Expreso Diemar | Precio: {precio_fuente} | Excesos: satelital >{LIMITE_VELOCIDAD} km/h | Actualización cada 10 min')
# ═══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 2 — MODELO PREDICTIVO
# ═══════════════════════════════════════════════════════════════════════════════
@@ -1060,46 +720,40 @@ def ier_bar_color(v):
    with col_title2:
        st.markdown("""<div style='padding:8px 0;'>
        <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>Modelo Predictivo &mdash; LAD</div>
        <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Regresión polinomial &middot; Simulador What-If</div>
        <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Entrenado con todo el histórico &middot; Regresión polinomial &middot; Simulador What-If</div>
        </div>""", unsafe_allow_html=True)
    st.markdown(f'<span class="price-badge">&#9981; Precio gasoil: <b>${precio_gasoil:,.0f}/L</b></span>&nbsp;&nbsp;<span style="font-size:.75rem;color:#64748b;">Fuente: {precio_fuente}</span>', unsafe_allow_html=True)
    anos_en_hist=(sorted(df_full_clean['FECHA'].dt.year.unique().tolist()) if 'FECHA' in df_full_clean.columns else [])
    anos_str=" · ".join(str(a) for a in anos_en_hist)
    st.markdown(f'<div class="training-badge">🧠 Modelo entrenado con {n_meses_entrenamiento} meses ({anos_str})</div>', unsafe_allow_html=True)
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
        st.info(f'📐 Grado polinomial: {degree} | R² = {r2_l100:.3f} | σ residuos = {std_res:.2f} L/100km')
        st.markdown(f'<div class="sec-title">Predicción meses restantes {ultimo.year} ({n_pred} meses)</div>', unsafe_allow_html=True)
        def render_pred_card(mes, l100_p, lts_p, precio_gasoil):
            costo_p = lts_p * precio_gasoil
            return (
                f'<div class="kpi-card" style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 10px;text-align:center;">'
                f'<div style="font-size:.72rem;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;">{mes}</div>'
                f'<div style="font-size:1.55rem;font-weight:700;color:#ea580c;line-height:1.1;">{l100_p:.2f}</div>'
                f'<div style="font-size:.78rem;color:#94a3b8;margin-bottom:6px;">L/100km</div>'
                f'<div style="border-top:1px solid #e2e8f0;padding-top:6px;margin-top:2px;">'
                f'<div style="font-size:.82rem;color:#e2e8f0;font-weight:600;">{lts_p:,.0f} L</div>'
                f'<div style="font-size:.75rem;color:#94a3b8;">${costo_p/1e6:.2f}M est.</div>'
                f'</div></div>'
            )
        n_kpi=min(4,n_pred); kpi_cols=st.columns(n_kpi)
        for c,mes,l100_p,lts_p in zip(kpi_cols,meses_fut[:n_kpi],pred_l100[:n_kpi],pred_lts[:n_kpi]):
            with c: st.markdown(render_pred_card(mes,l100_p,lts_p,precio_gasoil), unsafe_allow_html=True)
            costo_p=lts_p*precio_gasoil
            c.metric(label=f'Predicción {mes}',value=f'{l100_p:.2f} L/100km',delta=f'{lts_p:,.0f} L | ${costo_p/1e6:.2f}M')
        if n_pred>4:
            kpi_cols2=st.columns(min(4,n_pred-4))
            for c,mes,l100_p,lts_p in zip(kpi_cols2,meses_fut[4:],pred_l100[4:],pred_lts[4:]):
                with c: st.markdown(render_pred_card(mes,l100_p,lts_p,precio_gasoil), unsafe_allow_html=True)
                costo_p=lts_p*precio_gasoil
                c.metric(label=f'Predicción {mes}',value=f'{l100_p:.2f} L/100km',delta=f'{lts_p:,.0f} L | ${costo_p/1e6:.2f}M')
        st.divider()
        st.markdown('<div class="sec-title">Evolución histórica con Proyección</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Evolución histórica completa con Proyección</div>', unsafe_allow_html=True)
        all_labels=[str(p) for p in hist['MES_PERIODO']]+meses_fut
        all_hist=hist['L100'].tolist()+[None]*n_pred
        all_pred=[None]*(len(hist)-1)+[float(hist['L100'].iloc[-1])]+[float(v) for v in pred_l100]
@@ -1108,36 +762,61 @@ def render_pred_card(mes, l100_p, lts_p, precio_gasoil):
        pred_start=len(hist)-1; pred_labels=all_labels[pred_start:]
        upper_clean=[upper_vals[i] for i in range(pred_start,len(all_labels))]
        lower_clean=[lower_vals[i] for i in range(pred_start,len(all_labels))]
        unique_years=sorted(set(p.year for p in hist['MES_PERIODO']))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=pred_labels+pred_labels[::-1],y=upper_clean+lower_clean[::-1],fill='toself',fillcolor='rgba(59,130,246,0.12)',line=dict(color='rgba(0,0,0,0)'),name='Intervalo ±1.5σ',hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=pred_labels,y=upper_clean,mode='lines',line=dict(color='#3b82f6',width=1,dash='dot'),name='CI sup'))
        fig.add_trace(go.Scatter(x=pred_labels,y=lower_clean,mode='lines',line=dict(color='#3b82f6',width=1,dash='dot'),name='CI inf'))
        fig.add_trace(go.Scatter(x=pred_labels+pred_labels[::-1],y=upper_clean+lower_clean[::-1],fill='toself',fillcolor='rgba(59,130,246,0.15)',line=dict(color='rgba(0,0,0,0)'),name='Intervalo ±1.5σ',hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=pred_labels,y=upper_clean,mode='lines',line=dict(color='#3b82f6',width=1,dash='dot'),name='CI sup',hovertemplate='CI sup: %{y:.2f} L/100km<extra></extra>'))
        fig.add_trace(go.Scatter(x=pred_labels,y=lower_clean,mode='lines',line=dict(color='#3b82f6',width=1,dash='dot'),name='CI inf',hovertemplate='CI inf: %{y:.2f} L/100km<extra></extra>'))
        hist_x=[all_labels[i] for i,v in enumerate(all_hist) if v is not None]; hist_y=[v for v in all_hist if v is not None]
        fig.add_trace(go.Scatter(x=hist_x,y=hist_y,mode='lines+markers',line=dict(color='#ef4444',width=2.5),marker=dict(size=7,color='#ef4444'),name='Histórico'))
        fig.add_trace(go.Scatter(x=hist_x,y=hist_y,mode='lines+markers',line=dict(color='#ef4444',width=2.5),marker=dict(size=7,color='#ef4444',line=dict(color='#fff',width=1.5)),name='Histórico',hovertemplate='%{x}<br>Real: <b>%{y:.2f} L/100km</b><extra></extra>'))
        pred_x=[all_labels[i] for i,v in enumerate(all_pred) if v is not None]; pred_y=[v for v in all_pred if v is not None]
        fig.add_trace(go.Scatter(x=pred_x,y=pred_y,mode='lines+markers',line=dict(color='#f59e0b',width=4,dash='dash'),marker=dict(size=11,color='#f59e0b',symbol='diamond'),name='Predicción'))
        layout_light(fig,
        fig.add_trace(go.Scatter(x=pred_x,y=pred_y,mode='lines+markers',line=dict(color='#60a5fa',width=2.5,dash='dash'),marker=dict(size=9,color='#60a5fa',symbol='diamond',line=dict(color='#fff',width=1.5)),name='Predicción',hovertemplate='%{x}<br>Pred: <b>%{y:.2f} L/100km</b><extra></extra>'))
        for yr in unique_years[1:]:
            yr_label=f'Ene {yr}'
            if yr_label in all_labels:
                fig.add_vline(x=yr_label,line_width=1,line_dash='dot',line_color='#334155',annotation_text=str(yr),annotation_position='top',annotation_font_color='#64748b',annotation_font_size=10)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
            legend=dict(bgcolor='rgba(15,23,42,0.8)',bordercolor='#334155',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
            xaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS,size=10),tickangle=-45),
            yaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS,size=11),title=dict(text='L/100km',font=dict(color=PLOTLY_AXIS))),
            xaxis=dict(gridcolor='#1e293b',linecolor='#334155',tickfont=dict(color='#94a3b8',size=10),title=dict(text='Período',font=dict(color='#64748b')),tickangle=-45),
            yaxis=dict(gridcolor='#1e293b',linecolor='#334155',tickfont=dict(color='#94a3b8',size=11),title=dict(text='L/100km',font=dict(color='#64748b'))),
            height=450,margin=dict(l=10,r=10,t=50,b=60),hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f'±1.5σ intervalo de confianza | Línea roja = histórico ({n_meses_entrenamiento} meses) | Línea azul = predicción')
        st.divider()
        st.markdown('<div class="sec-title">🚨 Alerta de Desvío — Predicción vs. Real</div>', unsafe_allow_html=True)
        if len(hist)>=3:
            ultimo_real_mes=str(hist['MES_PERIODO'].iloc[-1]); ultimo_real_l100=float(hist['L100'].iloc[-1])
            hist_prev=hist.iloc[:-1].copy(); hist_prev['T']=range(len(hist_prev))
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
        st.divider()
        st.markdown('<div class="sec-title">🎨 Simulador What-If</div>', unsafe_allow_html=True)
        delta_precio_pct=st.slider('💸 Variación precio combustible (%)',min_value=-30,max_value=50,value=0,step=1)
        precio_sim=precio_gasoil*(1+delta_precio_pct/100)
        wf1,wf2=st.columns(2)
        with wf1:
            st.markdown(f'<div class="kpi-card kpi-amber"><div class="kpi-label">Precio Simulado</div><div class="kpi-value">${precio_sim:,.0f}/L</div><div class="kpi-sub">{delta_precio_pct:+.1f}%</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-card kpi-amber"><div class="kpi-label">Precio Simulado</div><div class="kpi-value">${precio_sim:,.0f}/L</div><div class="kpi-sub">{delta_precio_pct:+.1f}% vs hoy</div></div>', unsafe_allow_html=True)
        with wf2:
            costo_sim_m1=pred_lts[0]*precio_sim/1e6; costo_base_m1=pred_lts[0]*precio_gasoil/1e6; diff_costo=costo_sim_m1-costo_base_m1
            color_wf2='kpi-red' if diff_costo>0 else 'kpi-green'
            st.markdown(f'<div class="kpi-card {color_wf2}"><div class="kpi-label">Costo {meses_fut[0]}</div><div class="kpi-value">${costo_sim_m1:.2f}M</div><div class="kpi-sub">{diff_costo:+.2f}M</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-card {color_wf2}"><div class="kpi-label">Costo {meses_fut[0]}</div><div class="kpi-value">${costo_sim_m1:.2f}M</div><div class="kpi-sub">{diff_costo:+.2f}M vs base</div></div>', unsafe_allow_html=True)
        cost_df=pd.DataFrame({'Mes':meses_fut,'L/100km pred.':[round(v,2) for v in pred_l100],'Litros est.':[round(v,0) for v in pred_lts],
            'Costo base M$':[round(v*precio_gasoil/1e6,2) for v in pred_lts],'Costo sim M$':[round(v*precio_sim/1e6,2) for v in pred_lts],
            'Costo base M$':[round(v*precio_gasoil/1e6,2) for v in pred_lts],'Costo simulado M$':[round(v*precio_sim/1e6,2) for v in pred_lts],
            'Dif. M$':[round(v*(precio_sim-precio_gasoil)/1e6,2) for v in pred_lts]})
        st.dataframe(cost_df, use_container_width=True, hide_index=True)

    else:
        st.info('Se necesitan al menos 2 meses de datos históricos para el modelo predictivo.')
    st.caption(f'Modelo entrenado con {n_meses_entrenamiento} meses | Precio: {precio_fuente} | Actualización cada 10 min')
# ═══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 3 — ANÁLISIS POR PATENTE
# ═══════════════════════════════════════════════════════════════════════════════
@@ -1147,139 +826,152 @@ def render_pred_card(mes, l100_p, lts_p, precio_gasoil):
    with col_title3:
        st.markdown(f"""<div style='padding:8px 0;'>
        <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>Análisis por Patente — {anio_sel}</div>
        <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Consumo · Eficiencia · Excesos · Promedios</div>
        <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Consumo · IER v4 · Excesos velocidad · Promedios</div>
        </div>""", unsafe_allow_html=True)
    if df.empty or 'DOMINIO' not in df.columns: st.warning('Sin datos.'); st.stop()
    if df.empty or 'DOMINIO' not in df.columns: st.warning('Sin datos disponibles.'); st.stop()
    resumen=df.groupby('DOMINIO').agg(LITROS_TOTAL=('LITROS','sum'),KM_TOTAL=('KM','sum'),MESES=('MES_PERIODO','nunique')).reset_index()
    resumen['L100KM_PROM']=(resumen['LITROS_TOTAL']/resumen['KM_TOTAL'].replace(0,np.nan)*100).round(2)
    resumen['LITROS_PROM_MES']=(resumen['LITROS_TOTAL']/resumen['MESES'].replace(0,np.nan)).round(0)
    resumen=resumen[resumen['KM_TOTAL']>0].sort_values('L100KM_PROM',ascending=False)
    resumen['MODELO']=resumen['DOMINIO'].apply(asignar_modelo)
    
    if not df_ier.empty:
        resumen=resumen.merge(df_ier[['DOMINIO','IER','CLASIFICACION','EXCESOS','VEL_MAX','TONKML']],on='DOMINIO',how='left')
        resumen=resumen.merge(df_ier[['DOMINIO','IER','CLASIFICACION','EXCESOS','VEL_MAX']],on='DOMINIO',how='left')
    else:
        resumen['IER']='—'; resumen['CLASIFICACION']='—'; resumen['EXCESOS']=0; resumen['VEL_MAX']=0; resumen['TONKML']=0
        
    if not df_ier_chofer.empty:
        resumen=resumen.merge(df_ier_chofer[['DOMINIO','SCORE_CONDUCCION']],on='DOMINIO',how='left')
    else:
        resumen['SCORE_CONDUCCION']=np.nan

    if resumen.empty: st.warning('Sin datos.'); st.stop()
        resumen['IER']='—'; resumen['CLASIFICACION']='—'; resumen['EXCESOS']=0; resumen['VEL_MAX']=0
    if resumen.empty: st.warning('Sin datos suficientes.'); st.stop()
    patente_max=resumen.iloc[0]; patente_min=resumen.iloc[-1]
    st.markdown(f'<div class="sec-title">⚡ Destacados {anio_sel}</div>', unsafe_allow_html=True)
    hc1,hc2=st.columns(2)
    with hc1:
        st.markdown(f'<div class="highlight-max"><b>🔴 Mayor consumo — {patente_max["DOMINIO"]}</b> ({patente_max["MODELO"]})<br>Prom: <b>{patente_max["L100KM_PROM"]:.2f} L/100km</b> | Total: <b>{patente_max["LITROS_TOTAL"]:,.0f} L</b> | {patente_max["KM_TOTAL"]:,.0f} km</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="highlight-max"><b>🔴 Mayor consumo — {patente_max["DOMINIO"]}</b> <span style="color:#94a3b8;font-size:.8rem;">({patente_max["MODELO"]})</span><br>Promedio: <b>{patente_max["L100KM_PROM"]:.2f} L/100km</b> &nbsp;|&nbsp; Total: <b>{patente_max["LITROS_TOTAL"]:,.0f} L</b> &nbsp;|&nbsp; {patente_max["KM_TOTAL"]:,.0f} km &nbsp;|&nbsp; {int(patente_max["MESES"])} meses activa</div>', unsafe_allow_html=True)
    with hc2:
        st.markdown(f'<div class="highlight-min"><b>🟢 Menor consumo — {patente_min["DOMINIO"]}</b> ({patente_min["MODELO"]})<br>Prom: <b>{patente_min["L100KM_PROM"]:.2f} L/100km</b> | Total: <b>{patente_min["LITROS_TOTAL"]:,.0f} L</b> | {patente_min["KM_TOTAL"]:,.0f} km</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="highlight-min"><b>🟢 Menor consumo — {patente_min["DOMINIO"]}</b> <span style="color:#94a3b8;font-size:.8rem;">({patente_min["MODELO"]})</span><br>Promedio: <b>{patente_min["L100KM_PROM"]:.2f} L/100km</b> &nbsp;|&nbsp; Total: <b>{patente_min["LITROS_TOTAL"]:,.0f} L</b> &nbsp;|&nbsp; {patente_min["KM_TOTAL"]:,.0f} km &nbsp;|&nbsp; {int(patente_min["MESES"])} meses activa</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown(f'<div class="sec-title">Promedio L/100km por Patente — {anio_sel}</div>', unsafe_allow_html=True)
    colors_bar=[('#ef4444' if r['DOMINIO']==patente_max['DOMINIO'] else ('#22c55e' if r['DOMINIO']==patente_min['DOMINIO'] else '#3b82f6')) for _,r in resumen.iterrows()]
    fig_bar=go.Figure([go.Bar(x=resumen['DOMINIO'],y=resumen['L100KM_PROM'],marker_color=colors_bar,
        text=resumen['L100KM_PROM'].apply(lambda v:f'{v:.1f}'),textposition='outside',textfont=dict(color=PLOTLY_FONT,size=10))])
        text=resumen['L100KM_PROM'].apply(lambda v:f'{v:.1f}'),textposition='outside',textfont=dict(color='#e2e8f0',size=10),
        hovertemplate='<b>%{x}</b><br>L/100km: %{y:.2f}<extra></extra>')])
    promedio_flota=resumen['L100KM_PROM'].mean()
    fig_bar.add_hline(y=promedio_flota,line_dash='dot',line_color='#f59e0b',line_width=2,annotation_text=f'Prom: {promedio_flota:.2f}',annotation_font_color='#fbbf24')
    layout_light(fig_bar,
        xaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS,size=10),tickangle=-45),
        yaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS),title=dict(text='L/100km',font=dict(color=PLOTLY_AXIS))),
    fig_bar.add_hline(y=promedio_flota,line_dash='dot',line_color='#f59e0b',line_width=2,annotation_text=f'Promedio flota: {promedio_flota:.2f}',annotation_position='top right',annotation_font_color='#fbbf24',annotation_font_size=11)
    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
        xaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8',size=10),tickangle=-45),
        yaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8'),title=dict(text='L/100km',font=dict(color='#64748b'))),
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
    pat_sel=st.selectbox('Seleccioná patente',resumen['DOMINIO'].tolist())
    pat_sel=st.selectbox('Seleccioná una patente para ver su evolución',resumen['DOMINIO'].tolist())
    if pat_sel:
        df_pat=df[df['DOMINIO']==pat_sel].copy()
        if 'MES_PERIODO' in df_pat.columns:
            df_pat_mes=df_pat.groupby('MES_PERIODO').agg(LITROS=('LITROS','sum'),KM=('KM','sum')).reset_index().sort_values('MES_PERIODO')
            df_pat_mes['L100']=(df_pat_mes['LITROS']/df_pat_mes['KM'].replace(0,np.nan)*100).round(2)
            df_pat_mes['MES_STR']=df_pat_mes['MES_PERIODO'].astype(str)
            
            # --- AGREGADO DE CARGA PARA LA PATENTE INDIVIDUAL ---
            if not df_carga_raw.empty:
                carga_pat = df_carga_raw[(df_carga_raw['DOMINIO'] == pat_sel) & (df_carga_raw['MES'].apply(lambda p: p.year) == anio_sel)].copy()
                if not carga_pat.empty:
                    carga_pat_agg = carga_pat.groupby('MES')['PESO_TON'].sum().reset_index()
                    carga_pat_agg.rename(columns={'MES': 'MES_PERIODO'}, inplace=True)
                    df_pat_mes = df_pat_mes.merge(carga_pat_agg, on='MES_PERIODO', how='left')
                    df_pat_mes['PESO_TON'] = df_pat_mes['PESO_TON'].fillna(0)
                else:
                    df_pat_mes['PESO_TON'] = 0.0
            else:
                df_pat_mes['PESO_TON'] = 0.0
                
            df_pat_mes['TONKML'] = np.where((df_pat_mes['PESO_TON']>0)&(df_pat_mes['LITROS']>0), (df_pat_mes['PESO_TON']*df_pat_mes['KM'])/df_pat_mes['LITROS'], 0)
            
            l100_prom_pat=df_pat_mes['L100'].mean(); lts_total_pat=df_pat_mes['LITROS'].sum(); kms_total_pat=df_pat_mes['KM'].sum()
            marca_pat=df_pat['MARCA'].iloc[0] if 'MARCA' in df_pat.columns else '—'
            modelo_pat=df_pat['MODELO'].iloc[0] if 'MODELO' in df_pat.columns else '—'
            
            tonkml_pat = resumen.loc[resumen['DOMINIO'] == pat_sel, 'TONKML'].values
            tonkml_val = f"{tonkml_pat[0]:.2f}" if len(tonkml_pat) > 0 and pd.notnull(tonkml_pat[0]) else "0.00"
            score_pat = resumen.loc[resumen['DOMINIO'] == pat_sel, 'SCORE_CONDUCCION'].values
            score_val = f"{score_pat[0]:.2f}/10" if len(score_pat) > 0 and pd.notnull(score_pat[0]) else "—"

            dpk1, dpk2, dpk3, dpk4 = st.columns(4)
            dpk1.metric('Patente', pat_sel)
            dpk2.metric('Marca', marca_pat)
            dpk3.metric('Modelo', modelo_pat)
            dpk4.metric('Meses activos', f"{len(df_pat_mes)}")
            
            dpk5, dpk6, dpk7, dpk8 = st.columns(4)
            dpk5.metric('L/100km promedio', f'{l100_prom_pat:.2f}')
            dpk6.metric('ton·km/L prom.', tonkml_val)
            dpk7.metric('Score Conducción', score_val)
            dpk8.metric('Litros totales', f'{lts_total_pat:,.0f}')

            st.markdown('<br><div class="sec-title">📈 Evolución: L/100km y Productividad (ton·km/L)</div>', unsafe_allow_html=True)
            c_g1, c_g2 = st.columns(2)
            
            with c_g1:
                fig_pat=go.Figure()
                fig_pat.add_trace(go.Bar(x=df_pat_mes['MES_STR'],y=df_pat_mes['LITROS'],name='Litros',marker_color='rgba(59,130,246,0.5)',yaxis='y2'))
                fig_pat.add_trace(go.Scatter(x=df_pat_mes['MES_STR'],y=df_pat_mes['L100'],name='L/100km',mode='lines+markers',line=dict(color='#ef4444',width=2.5),marker=dict(size=8,color='#ef4444')))
                fig_pat.add_hline(y=l100_prom_pat,line_dash='dot',line_color='#f59e0b',annotation_text=f'Prom: {l100_prom_pat:.2f}',annotation_font_color='#fbbf24')
                layout_light(fig_pat,
                    xaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS,size=10),tickangle=-30),
                    yaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS),title=dict(text='L/100km',font=dict(color='#ef4444'))),
                    yaxis2=dict(overlaying='y',side='right',tickfont=dict(color='#3b82f6'),title=dict(text='Litros',font=dict(color='#3b82f6')),showgrid=False),
                    legend=dict(bgcolor='rgba(15,23,42,0.8)',bordercolor='#334155',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
                    height=370,margin=dict(l=10,r=50,t=40,b=50))
                st.plotly_chart(fig_pat, use_container_width=True)

            with c_g2:
                fig_pat_tkml = go.Figure()
                fig_pat_tkml.add_trace(go.Bar(x=df_pat_mes['MES_STR'], y=df_pat_mes['PESO_TON'], name='Toneladas', marker_color='rgba(168,85,247,0.5)', yaxis='y2'))
                fig_pat_tkml.add_trace(go.Scatter(x=df_pat_mes['MES_STR'], y=df_pat_mes['TONKML'], name='ton·km/L', mode='lines+markers', line=dict(color='#22c55e',width=2.5), marker=dict(size=8,color='#22c55e')))
                
                tonkml_prom = df_pat_mes[df_pat_mes['TONKML']>0]['TONKML'].mean()
                if not np.isnan(tonkml_prom):
                    fig_pat_tkml.add_hline(y=tonkml_prom, line_dash='dot', line_color='#f59e0b', annotation_text=f'Prom: {tonkml_prom:.2f}', annotation_font_color='#fbbf24')
                    
                layout_light(fig_pat_tkml,
                    xaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS,size=10),tickangle=-30),
                    yaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS),title=dict(text='ton·km/L',font=dict(color='#22c55e'))),
                    yaxis2=dict(overlaying='y',side='right',tickfont=dict(color='#a855f7'),title=dict(text='Toneladas',font=dict(color='#a855f7')),showgrid=False),
                    legend=dict(bgcolor='rgba(15,23,42,0.8)',bordercolor='#334155',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
                    height=370,margin=dict(l=10,r=50,t=40,b=50))
                st.plotly_chart(fig_pat_tkml, use_container_width=True)
                
            pk1,pk2,pk3,pk4,pk5=st.columns(5)
            pk1.metric('Patente',pat_sel); pk2.metric('Marca',marca_pat); pk3.metric('Modelo',modelo_pat)
            pk4.metric('L/100km promedio',f'{l100_prom_pat:.2f}'); pk5.metric('Litros totales',f'{lts_total_pat:,.0f}')
            if not df_ier.empty and pat_sel in df_ier['DOMINIO'].values:
                ier_row=df_ier[df_ier['DOMINIO']==pat_sel].iloc[0]
                st.markdown('<div class="sec-title">📊 Índice de Eficiencia Relativa (IER v5)</div>', unsafe_allow_html=True)
                ier_v=ier_row['IER']
                sc_color=('#22c55e' if ier_v>=105 else ('#f59e0b' if ier_v>=95 else ('#f97316' if ier_v>=85 else '#ef4444')))
                ia1,ia2,ia3=st.columns([1,2,2])
                with ia1:
                    st.markdown(f'<div class="ier-gauge-wrap"><div class="kpi-label">IER v5</div><div class="ier-score-big" style="color:{sc_color};">{ier_v:.1f}</div><div class="ier-clasif">{ier_row["CLASIFICACION"]}</div><div style="font-size:.72rem;color:#64748b;margin-top:6px;">base 100 = prom. {modelo_pat}</div></div>', unsafe_allow_html=True)
                with ia2:
                    st.markdown('<div style="font-size:.8rem;color:#94a3b8;font-weight:600;margin-bottom:6px;">Componentes del IER (40/40/10/10)</div>', unsafe_allow_html=True)
                    def comp_bar(label,score,peso):
                        pct=min(int(score*50),100); bc='#22c55e' if score>=1 else '#ef4444'
                        st.markdown(f'<div class="ier-comp-row"><div class="ier-comp-label">{label} <span style="color:#475569;">({peso}%)</span></div><div class="ier-comp-bar-bg"><div class="ier-comp-bar" style="width:{pct}%;background:{bc}"></div></div><div class="ier-comp-val" style="color:{bc};">{score*100:.0f}</div></div>', unsafe_allow_html=True)
                    comp_bar('⛽ L/100km',ier_row['SCORE_CONSUMO'],40)
                    comp_bar('📦 ton·km/L',ier_row['SCORE_CARGA'],40)
                    comp_bar('🛣️ KM totales',ier_row['SCORE_KM'],10)
                    comp_bar(f'🚨 Severidad vel.',ier_row['SCORE_VEL'],10)
                    # Ralentí: solo display
                    ral_display = "⚠️ dato no confiable" if modelo_pat=='S-Way' else f"{ier_row['RALENTI_PCT']:.1f}%"
                    st.markdown(f'<div class="ier-comp-row" style="opacity:.55;"><div class="ier-comp-label">⏱️ Ralentí <span style="color:#475569;">(excluido)</span></div><div class="ier-comp-val" style="color:#64748b;">{ral_display}</div></div>', unsafe_allow_html=True)
                with ia3:
                    st.markdown(f'<div style="font-size:.8rem;color:#94a3b8;font-weight:600;margin-bottom:6px;">Esta unidad vs. promedio {modelo_pat}</div>', unsafe_allow_html=True)
                    delta_l100=ier_row['L100KM']-ier_row['L100KM_MOD']
                    severidad_u = ier_row.get('SEVERIDAD', 0)
                    severidad_m = ier_row.get('SEVERIDAD_MOD', 0)
                    delta_sev = severidad_u - severidad_m
                    st.metric('L/100km',f"{ier_row['L100KM']:.2f}",f"{delta_l100:+.2f} vs prom. {modelo_pat} ({ier_row['L100KM_MOD']:.2f})",delta_color='inverse')
                    st.metric(f'Severidad vel. (km/h acum. sobre {LIMITE_VELOCIDAD})',f"{severidad_u:.0f}",f"{delta_sev:+.0f} vs prom. {modelo_pat} ({severidad_m:.0f})",delta_color='inverse')
                    st.metric(f'Eventos >{LIMITE_VELOCIDAD} km/h',f"{int(ier_row['EXCESOS'])} eventos",'ref. — la severidad usa km/h acumulados')
                    if ier_row.get('PESO_TON',0)>0:
                        delta_tkml=ier_row['TONKML']-ier_row['TONKML_MOD']
                        st.metric('📦 ton·km/L',f"{ier_row['TONKML']:.1f}",f"{delta_tkml:+.1f} vs prom. {modelo_pat} ({ier_row['TONKML_MOD']:.1f})",delta_color='normal')
                    else:
                        st.metric('📦 ton·km/L','sin datos','score neutral (1.0)')
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
                xaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8',size=10),tickangle=-30),
                yaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8'),title=dict(text='L/100km',font=dict(color='#ef4444'))),
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
    cols_show=['DOMINIO','MODELO','LITROS_TOTAL','KM_TOTAL','L100KM_PROM','TONKML','LITROS_PROM_MES','MESES']
    col_names=['Patente','Modelo','Litros','KM','L/100km','ton·km/L','L/Mes','Meses']
    for c,n in [('IER','IER'),('CLASIFICACION','Clasif'),('EXCESOS','Excesos'),('VEL_MAX','Vel.Max'),('SCORE_CONDUCCION','Score Cond.')]:
    st.markdown(f'<div class="sec-title">Tabla Resumen — Todas las Patentes {anio_sel}</div>', unsafe_allow_html=True)
    cols_show=['DOMINIO','MODELO','LITROS_TOTAL','KM_TOTAL','L100KM_PROM','LITROS_PROM_MES','MESES']
    col_names=['Patente','Modelo','Litros Total','KM Total','L/100km Prom','Litros/Mes Prom','Meses Activa']
    for c,n in [('IER','IER'),('CLASIFICACION','Clasificación IER'),(f'EXCESOS',f'Excesos >{LIMITE_VELOCIDAD}km/h'),('VEL_MAX','Vel. Máx (km/h)')]:
        if c in resumen.columns: cols_show.append(c); col_names.append(n)
    resumen_show=resumen[cols_show].copy(); resumen_show.columns=col_names
    resumen_show['Litros']=resumen_show['Litros'].apply(lambda x:f'{x:,.0f}')
    resumen_show['KM']=resumen_show['KM'].apply(lambda x:f'{x:,.0f}')
    resumen_show['L/Mes']=resumen_show['L/Mes'].apply(lambda x:f'{x:,.0f}')
    if 'ton·km/L' in resumen_show.columns:
        resumen_show['ton·km/L'] = resumen_show['ton·km/L'].apply(lambda x: f'{x:.2f}' if pd.notnull(x) else '—')
    if 'Score Cond.' in resumen_show.columns:
        resumen_show['Score Cond.'] = resumen_show['Score Cond.'].apply(lambda x: f'{x:.2f}' if pd.notnull(x) else '—')
    resumen_show['Litros Total']=resumen_show['Litros Total'].apply(lambda x:f'{x:,.0f}')
    resumen_show['KM Total']=resumen_show['KM Total'].apply(lambda x:f'{x:,.0f}')
    resumen_show['Litros/Mes Prom']=resumen_show['Litros/Mes Prom'].apply(lambda x:f'{x:,.0f}')
    st.dataframe(resumen_show, use_container_width=True, hide_index=True)

    st.caption(f'Datos {anio_sel} · Google Sheets Expreso Diemar · Actualización cada 10 min')
# ═══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 4 — DATOS OPERATIVOS
# ═══════════════════════════════════════════════════════════════════════════════
@@ -1289,181 +981,32 @@ def render_pred_card(mes, l100_p, lts_p, precio_gasoil):
    with col_title4:
        st.markdown(f"""<div style='padding:8px 0;'>
        <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>Datos Operativos — {anio_sel}</div>
        <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Peso entregado · Ton·km/L</div>
        <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Peso entregado por patente &middot; Ton·km/L &middot; Productividad de carga</div>
        </div>""", unsafe_allow_html=True)
    if df_carga_raw is None or df_carga_raw.empty:
        st.warning('⚠️ No hay datos de carga.'); st.stop()
        st.warning('⚠️ No hay datos de carga disponibles. Verificá la conexión al sistema BI (reporte_hojas.xlsx).')
        st.stop()
    # Solo patentes LAD (las mismas del df de telemetría)
    _patentes_ld = df['DOMINIO'].dropna().unique()
    df_carga_anio=df_carga_raw[
        (df_carga_raw['MES'].apply(lambda p:p.year)==anio_sel) &
        (df_carga_raw['DOMINIO'].isin(_patentes_ld))
    ].copy() if not df_carga_raw.empty else pd.DataFrame()
    if df_carga_anio.empty: st.warning(f'Sin carga {anio_sel}.'); st.stop()
    if df_carga_anio.empty: st.warning(f'Sin datos de carga para {anio_sel}.'); st.stop()
    df_carga_anio['MES_STR']=df_carga_anio['MES'].astype(str)
    df_carga_anio['MODELO']=df_carga_anio['DOMINIO'].apply(asignar_modelo)
    st.markdown(f'<div class="sec-title">Resumen Carga — {anio_sel}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">Resumen de Carga — {anio_sel}</div>', unsafe_allow_html=True)
    peso_total=df_carga_anio['PESO_TON'].sum(); n_pat_con_carga=df_carga_anio['DOMINIO'].nunique()
    peso_prom_pat=peso_total/n_pat_con_carga if n_pat_con_carga>0 else 0; meses_con_carga=df_carga_anio['MES'].nunique()
    def kpi2(cont,color,label,value,sub=''):
        cont.markdown(f'<div class="kpi-card {color}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)
    ck1,ck2,ck3,ck4=st.columns(4)
    kpi2(ck1,'kpi-purple','📦 Peso Total',f'{peso_total:,.1f}','toneladas')
    kpi2(ck2,'','🚛 Patentes',f'{n_pat_con_carga}',f'de {df["DOMINIO"].nunique()}')
    kpi2(ck3,'kpi-green','📊 Prom/Patente',f'{peso_prom_pat:,.1f}','ton')
    kpi2(ck4,'kpi-amber','📅 Meses',f'{meses_con_carga}',f'{anio_sel}')
    kpi2(ck1,'kpi-purple','📦 Peso Total Entregado',f'{peso_total:,.1f}',f'toneladas {anio_sel}')
    kpi2(ck2,'','🚛 Patentes con Carga',f'{n_pat_con_carga}',f'de {df["DOMINIO"].nunique()} activas')
    kpi2(ck3,'kpi-green','📊 Prom. por Patente',f'{peso_prom_pat:,.1f}','toneladas anuales')
    kpi2(ck4,'kpi-amber','📅 Meses con datos',f'{meses_con_carga}',f'de {anio_sel}')
    st.divider()
    st.markdown(f'<div class="sec-title">🔬 Diagnóstico de Carga — Matriz L/100km vs kg/km — {anio_sel}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ier-info-box">
    <b>¿Cómo leer esta matriz?</b>
    Cada punto es una patente. Los ejes separan consumo (L/100km) y densidad de carga (kg transportados por km recorrido).<br>
    La línea divisoria es la <b>mediana de la flota</b> en cada eje.<br>
    <b>% viajes sin peso:</b> viajes finalizados en el sistema BI con Peso Entregado = 0 — proxy de retornos en vacío o sin registro.
    </div>""", unsafe_allow_html=True)
    _km_pat   = df[df['KM']>0].groupby('DOMINIO')['KM'].sum().reset_index()
    _l100_pat = df[df['L100KM']>0].groupby('DOMINIO')['L100KM'].mean().reset_index()
    _tons_pat = df_carga_anio.groupby('DOMINIO')['PESO_TON'].sum().reset_index()
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
        _mat['N_TOTAL'] = 0; _mat['N_CARGADOS'] = 0; _mat['N_VACIOS'] = 0; _mat['PCT_VACIOS'] = 0.0
    if not _mat.empty and len(_mat) >= 2:
        _l100_med = _mat['L100KM'].median()
        _kgkm_med = _mat['KG_KM'].median()
        def _cuadrante(row):
            bajo = row['L100KM'] <= _l100_med
            alto  = row['KG_KM'] >= _kgkm_med
            if   bajo and alto:  return '🟢 Ideal',         '#22c55e', 'Eficiente y bien cargado. Rendimiento óptimo.'
            elif bajo and not alto: return '🟡 Subutilizado','#f59e0b', 'Consumo eficiente pero baja densidad de carga. Revisar asignación de rutas o retornos en vacío.'
            elif not bajo and alto: return '🟠 Consumo alto','#f97316', 'Bien cargado pero consume en exceso para el peso transportado. Revisar mecánica o conducción.'
            else:                   return '🔴 Crítico',     '#ef4444', 'Consumo alto y baja carga simultáneamente. Intervención urgente en mecánica y operación.'
        _mat[['CUAD_LABEL','CUAD_COLOR','CUAD_DESC']] = pd.DataFrame(
            _mat.apply(_cuadrante, axis=1).tolist(), index=_mat.index)
        _x_min = _mat['L100KM'].min() * 0.95; _x_max = _mat['L100KM'].max() * 1.05
        _y_min = _mat['KG_KM'].min()  * 0.90; _y_max = _mat['KG_KM'].max()  * 1.10
        fig_mat = go.Figure()
        _quad_cfg = [
            ([_x_min, _l100_med], [_kgkm_med, _y_max], 'rgba(34,197,94,0.12)',  '#22c55e', '🟢 IDEAL',
             'Eficiente + bien cargado', _x_min, _y_max, 'top left'),
            ([_l100_med, _x_max], [_kgkm_med, _y_max], 'rgba(249,115,22,0.12)', '#f97316', '🟠 CONSUMO ALTO',
             'Carga OK · consumo excesivo', _x_max, _y_max, 'top right'),
            ([_x_min, _l100_med], [_y_min, _kgkm_med], 'rgba(245,158,11,0.12)', '#f59e0b', '🟡 SUBUTILIZADO',
             'Eficiente · poca carga/retornos vacíos', _x_min, _y_min, 'bottom left'),
            ([_l100_med, _x_max], [_y_min, _kgkm_med], 'rgba(239,68,68,0.12)',  '#ef4444', '🔴 CRÍTICO',
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
                bgcolor='rgba(15,23,42,0.75)', borderpad=4
            )
        fig_mat.add_vline(x=_l100_med, line_dash='dash', line_color='#64748b', line_width=1.5)
        fig_mat.add_hline(y=_kgkm_med, line_dash='dash', line_color='#64748b', line_width=1.5)
        fig_mat.add_annotation(x=_l100_med, y=_y_max, text=f'mediana L/100km = {_l100_med:.1f}',
            showarrow=False, yanchor='bottom', font=dict(size=9, color='#64748b'),
            bgcolor='rgba(15,23,42,0.7)', borderpad=3)
        fig_mat.add_annotation(x=_x_min, y=_kgkm_med, text=f'mediana kg/km = {_kgkm_med:.0f}',
            showarrow=False, xanchor='left', font=dict(size=9, color='#64748b'),
            bgcolor='rgba(15,23,42,0.7)', borderpad=3)
        fig_mat.add_trace(go.Scatter(
            x=_mat['L100KM'], y=_mat['KG_KM'],
            mode='markers+text',
            text=_mat['DOMINIO'],
            textposition='top center',
            textfont=dict(size=10, color='#f1f5f9', family='monospace'),
            marker=dict(size=18, color=_mat['CUAD_COLOR'],
                        line=dict(color='white', width=2),
                        symbol='circle'),
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
        layout_light(fig_mat,
            xaxis=dict(
                gridcolor=PLOTLY_GRID, tickfont=dict(color=PLOTLY_AXIS, size=11),
                title=dict(
                    text='L / 100 km   —   litros consumidos por cada 100 km recorridos   (← menor = más eficiente)',
                    font=dict(color=PLOTLY_AXIS, size=11)
                ),
                range=[_x_min, _x_max]
            ),
            yaxis=dict(
                gridcolor=PLOTLY_GRID, tickfont=dict(color=PLOTLY_AXIS, size=11),
                title=dict(
                    text='kg / km   —   kg de carga entregados por km recorrido   (↑ mayor = más productivo)',
                    font=dict(color=PLOTLY_AXIS, size=11)
                ),
                range=[_y_min, _y_max]
            ),
            height=560, margin=dict(l=70, r=50, t=40, b=70)
        )
        st.plotly_chart(fig_mat, use_container_width=True)
        st.caption(f'Cada punto = una patente · Hover para diagnóstico completo · Líneas punteadas = mediana de la flota · L/100km: dato de telemetría · kg/km: carga del BI ÷ km telemetría')
        st.markdown('<div class="sec-title">📋 Diagnóstico Individual por Patente</div>', unsafe_allow_html=True)
        _mat_sorted = _mat.sort_values('CUAD_COLOR', key=lambda s: s.map({'#ef4444':0,'#f97316':1,'#f59e0b':2,'#22c55e':3}))
        _diag_cols = st.columns(min(4, len(_mat_sorted)))
        for _i, (_, _r) in enumerate(_mat_sorted.iterrows()):
            with _diag_cols[_i % len(_diag_cols)]:
                _pct_v = _r.get('PCT_VACIOS', 0)
                _n_tot = int(_r.get('N_TOTAL', 0))
                _n_vac = int(_r.get('N_VACIOS', 0))
                _vacios_txt = f"{_n_vac}/{_n_tot} viajes sin peso ({_pct_v:.0f}%)" if _n_tot > 0 else "sin datos de viajes"
                _bc = _r['CUAD_COLOR']
                st.markdown(f"""
                <div style="background:#1e293b;border-radius:12px;padding:16px;border-left:5px solid {_bc};margin-bottom:12px;">
                  <div style="font-size:.95rem;font-weight:800;color:#f1f5f9;">{_r['DOMINIO']}</div>
                  <div style="font-size:.72rem;color:#64748b;margin-bottom:8px;">{_r['MODELO']}</div>
                  <div style="font-size:1.2rem;font-weight:700;color:{_bc};margin-bottom:6px;">{_r['CUAD_LABEL']}</div>
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
            _tbl['L/100km']         = _tbl['L/100km'].round(2)
            _tbl['kg/km']           = _tbl['kg/km'].round(1)
            _tbl['Peso total (ton)'] = _tbl['Peso total (ton)'].round(1)
            st.dataframe(_tbl.sort_values('% sin carga', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info('Sin datos suficientes para armar la matriz (se necesitan datos de telemetría y carga simultáneos).')
    st.divider()
    st.markdown(f'<div class="sec-title">📦 Peso por Mes y Patente</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">📦 Peso Entregado por Mes y Patente (toneladas) — {anio_sel}</div>', unsafe_allow_html=True)
    pivot_carga=(df_carga_anio.pivot_table(index='DOMINIO',columns='MES_STR',values='PESO_TON',aggfunc='sum',fill_value=0).reset_index())
    pivot_carga['TOTAL']=pivot_carga.drop(columns='DOMINIO').sum(axis=1)
    pivot_carga=pivot_carga.sort_values('TOTAL',ascending=False)
@@ -1472,32 +1015,76 @@ def _cuadrante(row):
        z_vals=pivot_carga[meses_cols].values.tolist(); y_vals=pivot_carga['DOMINIO'].tolist()
        txt_vals=[[f'{v:,.1f}' if v>0 else '' for v in row] for row in z_vals]
        fig_heat_c=go.Figure(go.Heatmap(z=z_vals,x=meses_cols,y=y_vals,text=txt_vals,texttemplate='%{text}',
            colorscale=[[0.0,'#f8fafc'],[0.3,'#3b82f6'],[0.65,'#a78bfa'],[1.0,'#be185d']],
            colorbar=dict(title=dict(text='Ton',font=dict(color=PLOTLY_AXIS)),tickfont=dict(color=PLOTLY_AXIS),bgcolor='rgba(0,0,0,0)')))
        layout_light(fig_heat_c,
            xaxis=dict(tickfont=dict(color=PLOTLY_AXIS,size=10),tickangle=-45,side='bottom'),
            yaxis=dict(tickfont=dict(color=PLOTLY_AXIS,size=10)),height=max(300,len(y_vals)*40),margin=dict(l=10,r=10,t=20,b=60))
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
    st.markdown(f'<div class="sec-title">Evolución Mensual de Peso Entregado por Patente — {anio_sel}</div>', unsafe_allow_html=True)
    COLORES_PAT=['#3b82f6','#f97316','#22c55e','#a855f7','#f43f5e','#06b6d4','#eab308','#84cc16']
    fig_bar_c=go.Figure()
    for i,(_,row) in enumerate(pivot_carga.iterrows()):
        dom=row['DOMINIO']; vals=[row.get(m,0) for m in meses_cols]
        fig_bar_c.add_trace(go.Bar(name=dom,x=meses_cols,y=vals,marker_color=COLORES_PAT[i%len(COLORES_PAT)],hovertemplate=f'<b>{dom}</b><br>%{{x}}: <b>%{{y:,.1f}} ton</b><extra></extra>'))
    fig_bar_c.update_layout(barmode='stack',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
        xaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8',size=10),tickangle=-45),
        yaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8'),title=dict(text='Toneladas entregadas',font=dict(color='#64748b'))),
        legend=dict(bgcolor='rgba(15,23,42,0.8)',bordercolor='#334155',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
        height=420,margin=dict(l=10,r=10,t=50,b=70))
    st.plotly_chart(fig_bar_c, use_container_width=True)
    st.divider()
    st.markdown(f'<div class="sec-title">📐 ton·km/L</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">📐 Detalle ton·km/L (Productividad de Carga) — {anio_sel}</div>', unsafe_allow_html=True)
    st.markdown("""<div class="ier-info-box"><b>¿Qué es ton·km/L?</b> Mide cuántas toneladas·kilómetro se transportan por cada litro de combustible.<br><b>Fórmula:</b> ton·km/L = Peso entregado (ton) × KM recorridos / Litros consumidos</div>""", unsafe_allow_html=True)
    df_op=df[df['KM']>0].copy(); df_op['MES_STR']=df_op['FECHA'].dt.to_period('M').astype(str)
    km_lts_mes=df_op.groupby(['DOMINIO','MES_STR']).agg(KM=('KM','sum'),LITROS=('LITROS','sum')).reset_index()
    df_carga_str=df_carga_anio[['DOMINIO','MES_STR','PESO_TON']].copy()
    tonkml_mes=km_lts_mes.merge(df_carga_str,on=['DOMINIO','MES_STR'],how='left')
    tonkml_mes['PESO_TON']=tonkml_mes['PESO_TON'].fillna(0)
    tonkml_mes['TONKML']=np.where((tonkml_mes['PESO_TON']>0)&(tonkml_mes['LITROS']>0),(tonkml_mes['PESO_TON']*tonkml_mes['KM'])/tonkml_mes['LITROS'],np.nan).round(2)
    tonkml_mes['MODELO']=tonkml_mes['DOMINIO'].apply(asignar_modelo)
    COLORES_PAT=['#3b82f6','#f97316','#22c55e','#a855f7','#ef4444','#06b6d4','#eab308','#84cc16']
    tkml_valid=tonkml_mes['TONKML'].dropna()
    if not tkml_valid.empty:
        t1,t2,t3,t4=st.columns(4)
        tkml_prom=tkml_valid.mean(); tkml_max=tkml_valid.max(); tkml_min=tkml_valid.min()
        dom_max=tonkml_mes.loc[tonkml_mes['TONKML'].idxmax(),'DOMINIO']; dom_min=tonkml_mes.loc[tonkml_mes['TONKML'].idxmin(),'DOMINIO']
        kpi2(t1,'','📊 Promedio ton·km/L',f'{tkml_prom:.2f}','toda la flota')
        kpi2(t2,'kpi-green',f'🏆 Mejor — {dom_max}',f'{tkml_max:.2f}','mayor productividad')
        kpi2(t3,'kpi-red',f'⚠️ Peor — {dom_min}',f'{tkml_min:.2f}','menor productividad')
        kpi2(t4,'kpi-purple','📦 Período cubierto',f'{tonkml_mes["MES_STR"].nunique()} meses',f'{anio_sel}')
    fig_tkml=go.Figure()
    for i,dom in enumerate(tonkml_mes['DOMINIO'].unique()):
        sub=tonkml_mes[tonkml_mes['DOMINIO']==dom].sort_values('MES_STR')
        sub_valid=sub[sub['TONKML'].notna()]
        if sub_valid.empty: continue
        fig_tkml.add_trace(go.Scatter(x=sub_valid['MES_STR'],y=sub_valid['TONKML'],name=dom,mode='lines+markers',line=dict(color=COLORES_PAT[i%len(COLORES_PAT)],width=2.5),marker=dict(size=8)))
    layout_light(fig_tkml,
        xaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS,size=10),tickangle=-30),
        yaxis=dict(gridcolor=PLOTLY_GRID,tickfont=dict(color=PLOTLY_AXIS),title=dict(text='ton·km/L',font=dict(color=PLOTLY_AXIS))),
        fig_tkml.add_trace(go.Scatter(x=sub_valid['MES_STR'],y=sub_valid['TONKML'],name=dom,mode='lines+markers',line=dict(color=COLORES_PAT[i%len(COLORES_PAT)],width=2.5),marker=dict(size=8,line=dict(color='#fff',width=1.5)),hovertemplate=f'<b>{dom}</b><br>%{{x}}: <b>%{{y:.2f}} ton·km/L</b><extra></extra>'))
    if not tkml_valid.empty:
        fig_tkml.add_hline(y=tkml_valid.mean(),line_dash='dot',line_color='#f59e0b',line_width=2,annotation_text=f'Promedio: {tkml_valid.mean():.2f}',annotation_position='top right',annotation_font_color='#fbbf24',annotation_font_size=11)
    fig_tkml.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
        xaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8',size=10),tickangle=-30),
        yaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8'),title=dict(text='ton·km/L',font=dict(color='#64748b'))),
        legend=dict(bgcolor='rgba(15,23,42,0.8)',bordercolor='#334155',borderwidth=1,orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
        height=400,margin=dict(l=10,r=10,t=50,b=50))
    st.plotly_chart(fig_tkml, use_container_width=True)
    st.caption(f'BI Expreso · Año {anio_sel}')
    rank_tkml=tonkml_mes.groupby('DOMINIO').agg(PESO_TON=('PESO_TON','sum'),KM=('KM','sum'),LITROS=('LITROS','sum'),MODELO=('MODELO','first')).reset_index()
    rank_tkml['TONKML_ANUAL']=np.where((rank_tkml['PESO_TON']>0)&(rank_tkml['LITROS']>0),(rank_tkml['PESO_TON']*rank_tkml['KM'])/rank_tkml['LITROS'],np.nan).round(2)
    rank_tkml=rank_tkml[rank_tkml['TONKML_ANUAL'].notna()].sort_values('TONKML_ANUAL',ascending=True)
    if not rank_tkml.empty:
        fig_rank=go.Figure([go.Bar(y=rank_tkml['DOMINIO'],x=rank_tkml['TONKML_ANUAL'],orientation='h',
            marker_color=['#22c55e' if v==rank_tkml['TONKML_ANUAL'].max() else ('#ef4444' if v==rank_tkml['TONKML_ANUAL'].min() else '#3b82f6') for v in rank_tkml['TONKML_ANUAL']],
            text=[f'{v:.2f}' for v in rank_tkml['TONKML_ANUAL']],textposition='outside',textfont=dict(color='#e2e8f0',size=10),
            hovertemplate='<b>%{y}</b><br>ton·km/L: <b>%{x:.2f}</b><extra></extra>')])
        prom_r=rank_tkml['TONKML_ANUAL'].mean()
        fig_rank.add_vline(x=prom_r,line_dash='dot',line_color='#f59e0b',line_width=2,annotation_text=f'Prom: {prom_r:.2f}',annotation_position='top',annotation_font_color='#fbbf24',annotation_font_size=11)
        fig_rank.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,41,59,0.6)',font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8'),title=dict(text='ton·km/L acumulado año',font=dict(color='#64748b'))),
            yaxis=dict(gridcolor='#1e293b',tickfont=dict(color='#94a3b8',size=10)),
            height=max(300,len(rank_tkml)*50+80),margin=dict(l=10,r=120,t=30,b=30),showlegend=False)
        st.plotly_chart(fig_rank, use_container_width=True)
    st.caption(f'Fuente: reporte_hojas.xlsx (BI Expreso) · Telemetría Google Sheets · Año {anio_sel}')
