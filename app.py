import pandas as pd
import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Expreso Diemar — Predicción Combustible",
    page_icon="🚛",
    layout="wide",
)

LOGO_URL  = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/logo_diemar4.png"
IVECO_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/S-Way-6x2-1.webp"
SCANIA_URL= "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/2016p.png"

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
.kpi-sub   { font-size:0.75rem; color:#64748b; margin-top:4px; }
.kpi-red   { border-left-color:#ef4444; }
.kpi-green { border-left-color:#22c55e; }
.kpi-amber { border-left-color:#f59e0b; }
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
.alert-box { background:#450a0a; border:1px solid #ef4444; border-radius:10px; padding:14px 18px; margin:10px 0; }
.alert-ok  { background:#052e16; border:1px solid #22c55e; border-radius:10px; padding:14px 18px; margin:10px 0; }
.highlight-max { background:#450a0a; border:1px solid #ef4444; border-radius:10px; padding:14px 18px; margin:6px 0; }
.highlight-min { background:#052e16; border:1px solid #22c55e; border-radius:10px; padding:14px 18px; margin:6px 0; }
.training-badge {
    background:#1e1b4b; border:1px solid #6366f1; border-radius:8px;
    padding:6px 12px; display:inline-block; font-size:0.8rem; color:#a5b4fc; font-weight:600;
    margin-bottom: 12px;
}
.sidebar-filter-header {
    font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.5px;
    color:#64748b; margin-bottom:10px; padding:6px 0; border-bottom:1px solid #334155;
}
[data-testid="stSidebar"] [data-testid="stDateInput"] label,
[data-testid="stSidebar"] [data-testid="stMultiSelect"] label {
    font-size:.78rem !important; color:#94a3b8 !important; font-weight:600 !important;
}
</style>
"""

pg = st.sidebar.radio(
    "Navegacion",
    ["Dashboard Principal", "Modelo Predictivo", "Análisis por Patente"],
    index=0,
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.image(LOGO_URL, width=160)

BASE_URL  = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR35NkYPtJrOrdYHLGUH7GIW93s5cPAqQ0zEk5fP1c3gvErwbUW7HJ2OeWBYaBVsYKVmCf0yhLvs6eG/pub?output=csv"
GID_TEL   = "0"
GID_UNID  = "882343299"
URL_TEL   = f"{BASE_URL}&gid={GID_TEL}"
URL_UNID  = f"{BASE_URL}&gid={GID_UNID}"


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
                if   "DOMINIO" in c or "PATENTE" in c:          cm[c] = "DOMINIO"
                elif "LITROS"  in c or "CONSUMID" in c:         cm[c] = "LITROS"
                elif "DISTANCIA" in c or c == "KM" or "KILOMETR" in c: cm[c] = "KM"
                elif "MARCA"   in c:                             cm[c] = "MARCA"
                elif "TAG"     in c:                             cm[c] = "TAG"
                elif "FECHA"   in c or "DATE" in c:             cm[c] = "FECHA"
                elif "L/100"   in c or "CONSUMO C" in c:        cm[c] = "L100KM"
                elif "RALENT"  in c:                             cm[c] = "RALENTI"
                elif "TIEMPO"  in c and "MOTOR" in c:           cm[c] = "TIEMPO_MOTOR"
                elif "EMPRESA" in c:                             cm[c] = "EMPRESA"
            df = df.rename(columns=cm)
            df = df.loc[:, ~df.columns.duplicated()]
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

        extra = [c for c in ["DOMINIO", "RALENTI"] if c in df2.columns]
        if len(extra) > 1:
            df2r = df2[extra].groupby("DOMINIO").sum(numeric_only=True).reset_index()
            df1  = pd.merge(df1, df2r, on="DOMINIO", how="left", suffixes=("", "_u"))
            if "RALENTI_u" in df1.columns:
                df1["RALENTI"] = df1["RALENTI"].combine_first(df1["RALENTI_u"])
                df1.drop(columns=["RALENTI_u"], inplace=True)

        if "EMPRESA" in df1.columns:
            df1 = df1[df1["EMPRESA"].str.upper().str.contains("LAD|DIEMAR", na=False)]

        if "DOMINIO" in df2.columns and not df2.empty:
            lad_units = df2["DOMINIO"].dropna().unique()
            if len(lad_units) > 0:
                df1 = df1[df1["DOMINIO"].isin(lad_units)]

        # NO filtramos por año — guardamos todo el histórico para entrenamiento
        return df1, df2

    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600)
def obtener_precio_gasoil():
    try:
        import re
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get("https://surtidores.com.ar/precios/", headers=headers, timeout=10)
        soup  = BeautifulSoup(r.text, "html.parser")
        texto = soup.get_text(separator=" ")
        idx_2026 = texto.find("2026")
        if idx_2026 != -1:
            segmento  = texto[idx_2026:idx_2026 + 600]
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
    return 2025.0, "referencia estimada (mar 2026)"


# ── Carga inicial ──────────────────────────────────────────────────────────────
with st.spinner('Cargando telemetría...'):
    df_raw, df_unid = cargar_datos()

if df_raw.empty:
    st.warning('No se pudieron cargar datos.')
    st.stop()

precio_gasoil, precio_fuente = obtener_precio_gasoil()
st.markdown(DARK_CSS, unsafe_allow_html=True)

# df_full = TODO el histórico (para modelo)
# df = filtrado al año de visualización seleccionado (para dashboard/patentes)
df_full = df_raw.copy()

# Año de visualización — sidebar
anios_disponibles = sorted(df_full['FECHA'].dt.year.dropna().unique().tolist(), reverse=True) if 'FECHA' in df_full.columns else [2025]
anio_sel = st.sidebar.selectbox('Año de visualización', anios_disponibles, index=0)

df = df_full[df_full['FECHA'].dt.year == anio_sel].copy() if 'FECHA' in df_full.columns else df_full.copy()

# ── Filtros sidebar ────────────────────────────────────────────────────────────
if 'FECHA' in df.columns and df['FECHA'].notna().any():
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-filter-header">🔍 Filtros</div>', unsafe_allow_html=True)
    fmin  = df['FECHA'].min().date()
    fmax  = df['FECHA'].max().date()
    desde = st.sidebar.date_input('Desde', value=fmin, min_value=fmin, max_value=fmax)
    hasta = st.sidebar.date_input('Hasta', value=fmax, min_value=fmin, max_value=fmax)
    marcas_disp  = sorted(df['MARCA'].dropna().unique().tolist()) if 'MARCA' in df.columns else []
    marcas_sel   = st.sidebar.multiselect('Marca', marcas_disp, default=marcas_disp)
    patentes_disp = sorted(df['DOMINIO'].dropna().unique().tolist()) if 'DOMINIO' in df.columns else []
    patentes_sel  = st.sidebar.multiselect('Patente', patentes_disp, default=[], placeholder="Todas las patentes")
    df = df[(df['FECHA'].dt.date >= desde) & (df['FECHA'].dt.date <= hasta)]
    if marcas_sel   and 'MARCA'   in df.columns: df = df[df['MARCA'].isin(marcas_sel)]
    if patentes_sel and 'DOMINIO' in df.columns: df = df[df['DOMINIO'].isin(patentes_sel)]

if df.empty:
    st.warning(f'Sin datos para {anio_sel} con los filtros seleccionados.')
    st.stop()

df['MES_PERIODO'] = df['FECHA'].dt.to_period('M')
df['MES_NUM']     = df['FECHA'].dt.month

meses_df = df.groupby('MES_PERIODO').agg(
    LITROS=('LITROS', 'sum'),
    KM=('KM', 'sum')
).reset_index().sort_values('MES_PERIODO')
meses_df['L100'] = (meses_df['LITROS'] / meses_df['KM'].replace(0, np.nan) * 100).round(2)

ralenti_total    = df['RALENTI'].sum() if 'RALENTI' in df.columns else 0
ralenti_delta_txt = ''
if 'RALENTI' in df.columns and 'MES_PERIODO' in df.columns:
    _mg = df.groupby('MES_PERIODO').agg(_RAL=('RALENTI','sum'), _LTS=('LITROS','sum')).reset_index().sort_values('MES_PERIODO')
    if len(_mg) >= 2:
        _curr = _mg.iloc[-1]; _prev = _mg.iloc[-2]
        _pct_curr = _curr['_RAL'] / _curr['_LTS'] * 100 if _curr['_LTS'] > 0 else 0
        _pct_prev = _prev['_RAL'] / _prev['_LTS'] * 100 if _prev['_LTS'] > 0 else 0
        _dr = _pct_curr - _pct_prev
        ralenti_delta_txt = f"{'▲' if _dr > 0 else '▼'} {abs(_dr):.1f}pp vs mes ant."

# ── Preparar datos históricos para el modelo (todo el histórico disponible) ──
df_full_clean = df_full[df_full['FECHA'].notna() & (df_full['KM'] > 0)].copy()
df_full_clean['MES_PERIODO'] = df_full_clean['FECHA'].dt.to_period('M')
meses_hist_full = df_full_clean.groupby('MES_PERIODO').agg(
    LITROS=('LITROS', 'sum'),
    KM=('KM', 'sum')
).reset_index().sort_values('MES_PERIODO')
meses_hist_full['L100'] = (meses_hist_full['LITROS'] / meses_hist_full['KM'].replace(0, np.nan) * 100).round(2)
meses_hist_full = meses_hist_full[meses_hist_full['KM'] > 0].copy()
n_meses_entrenamiento = len(meses_hist_full)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
if pg == "Dashboard Principal":

    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image(LOGO_URL, width=130)
    with col_title:
        st.markdown(f"""
        <div style='padding:8px 0;'>
            <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>Expreso Diemar &mdash; Dashboard LAD {anio_sel}</div>
            <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Telemetria flota LAD &middot; Año {anio_sel} &middot; Actualizacion automatica</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<span class="price-badge">&#9981; Precio gasoil: <b>${precio_gasoil:,.0f}/L</b></span>'
        f'&nbsp;&nbsp;<span style="font-size:.75rem;color:#64748b;">Fuente: {precio_fuente}</span></div>',
        unsafe_allow_html=True)

    st.markdown(f'<div class="sec-title">Métricas Globales — {anio_sel}</div>', unsafe_allow_html=True)

    lts_total  = df['LITROS'].sum()  if 'LITROS'  in df.columns else 0
    kms_total  = df['KM'].sum()      if 'KM'      in df.columns else 0
    l100_prom  = round(lts_total / kms_total * 100, 2) if kms_total > 0 else 0
    costo_est  = lts_total * precio_gasoil
    n_unidades = df['DOMINIO'].nunique() if 'DOMINIO' in df.columns else 0
    ralenti_pct = round(ralenti_total / lts_total * 100, 1) if lts_total > 0 else 0

    if len(meses_df) >= 2:
        delta_l100 = meses_df['L100'].iloc[-1] - meses_df['L100'].iloc[-2]
        delta_txt  = f"{'▲' if delta_l100>0 else '▼'} {abs(delta_l100):.2f} vs mes anterior"
        delta_col  = 'kpi-red' if delta_l100 > 0 else 'kpi-green'
    else:
        delta_txt, delta_col = '', ''

    def kpi(cont, color, label, value, sub=''):
        cont.markdown(
            f'<div class="kpi-card {color}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>', unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3)
    kpi(k1, '',          '⛽ Litros totales',   f'{lts_total:,.0f}',      f'litros {anio_sel}')
    kpi(k2, '',          '🛣️ KM recorridos',    f'{kms_total:,.0f}',      f'kilómetros {anio_sel}')
    kpi(k3, delta_col,   '📊 L/100km flota',    f'{l100_prom:.2f}',       delta_txt)

    k4, k5, k6 = st.columns(3)
    kpi(k4, 'kpi-amber', '💰 Costo estimado',   f'${costo_est/1e6:.1f}M', f'@ ${precio_gasoil:,.0f}/L')
    kpi(k5, 'kpi-green', '🚛 Unidades activas', f'{n_unidades}',          'dominios únicos')
    _ral_sub = (f'{ralenti_total:,.0f} L · {ralenti_delta_txt}' if ralenti_delta_txt
                else f'{ralenti_total:,.0f} L en ralentí')
    kpi(k6, 'kpi-amber', '⏱️ % Ralentí',        f'{ralenti_pct:.1f}%',    _ral_sub)

    st.divider()

    st.markdown(f'<div class="sec-title">Rendimiento por Marca — {anio_sel}</div>', unsafe_allow_html=True)

    def stats_marca(marca):
        sub = df[df['MARCA'].str.upper() == marca.upper()] if 'MARCA' in df.columns else pd.DataFrame()
        if sub.empty:
            return {'l100': 0, 'lts': 0, 'kms': 0, 'n': 0}
        lts = sub['LITROS'].sum(); kms = sub['KM'].sum()
        return {'l100': round(lts/kms*100, 2) if kms > 0 else 0, 'lts': lts, 'kms': kms, 'n': sub['DOMINIO'].nunique()}

    tc1, tc2 = st.columns(2)
    for col_t, marca, img_url, modelo in [
        (tc1, 'IVECO',  IVECO_URL,  'S-Way 6x2'),
        (tc2, 'SCANIA', SCANIA_URL, 'Serie P 2016')
    ]:
        s = stats_marca(marca)
        with col_t:
            st.markdown(f'<div class="truck-img-box"><img src="{img_url}" alt="{marca}" /></div>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric(f'{marca} — L/100km', f"{s['l100']:.1f}")
            sc2.metric('Unidades',           f"{s['n']}")
            sc3.metric('Litros {}'.format(anio_sel), f"{s['lts']:,.0f}")
            st.caption(f"Modelo: {modelo} | {s['kms']:,.0f} km")

    st.divider()

    st.markdown(f'<div class="sec-title">Ranking de Eficiencia — {anio_sel}</div>', unsafe_allow_html=True)
    rcol1, rcol2 = st.columns(2)

    def render_ranking(col, titulo, df_rank, color_fn):
        with col:
            st.markdown(f'**{titulo}**')
            if df_rank.empty:
                st.info('Sin datos.')
                return
            vmin, vmax = df_rank['L100KM'].min(), df_rank['L100KM'].max()
            rh = '<div style="background:#1e293b;border-radius:12px;padding:16px;">'
            rh += '<div style="font-size:.72rem;display:flex;justify-content:space-between;color:#94a3b8;margin-bottom:6px;"><span>Unidad</span><span>L/100km</span></div>'
            for i, (_, r) in enumerate(df_rank.iterrows(), 1):
                v   = r['L100KM']
                pct = int((v - vmin) / (vmax - vmin) * 100) if vmax != vmin else 50
                cb  = color_fn(i)
                rh += (f'<div class="rank-row">'
                       f'<div class="rank-num">#{i}</div>'
                       f'<div class="rank-dom">{r["DOMINIO"]}</div>'
                       f'<div class="rank-bar-bg"><div class="rank-bar" style="width:{pct}%;background:{cb}"></div></div>'
                       f'<div class="rank-val" style="color:{cb}">{v:.2f}</div>'
                       f'</div>')
            rh += '</div>'
            st.markdown(rh, unsafe_allow_html=True)

    if 'DOMINIO' in df.columns and 'L100KM' in df.columns:
        base = df[df['L100KM'] > 0].groupby('DOMINIO')['L100KM'].mean().round(2).reset_index()
        rank_eff  = base.sort_values('L100KM').head(10)
        rank_inef = base.sort_values('L100KM', ascending=False).head(10)
        render_ranking(rcol1, 'TOP 10 más eficientes (menor L/100km)', rank_eff,
                       lambda i: '#22c55e' if i<=3 else ('#f59e0b' if i<=6 else '#ef4444'))
        render_ranking(rcol2, 'TOP 10 menos eficientes (mayor L/100km)', rank_inef,
                       lambda i: '#ef4444' if i<=3 else ('#f59e0b' if i<=6 else '#22c55e'))

    st.divider()

    with st.expander(f'Ver datos completos {anio_sel}'):
        cols_s = [c for c in ['DOMINIO','MARCA','FECHA','KM','LITROS','L100KM','RALENTI'] if c in df.columns]
        st.dataframe(df[cols_s], use_container_width=True, height=380)

    st.caption(f'Datos {anio_sel}: Google Sheets Expreso Diemar | Precio: {precio_fuente} | Actualización cada 10 min')


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO PREDICTIVO  (entrenado con TODO el histórico, predice meses futuros)
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "Modelo Predictivo":

    col_logo2, col_title2 = st.columns([1, 5])
    with col_logo2:
        st.image(LOGO_URL, width=130)
    with col_title2:
        st.markdown("""
        <div style='padding:8px 0;'>
            <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>Modelo Predictivo &mdash; LAD</div>
            <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Entrenado con todo el histórico &middot; Regresión polinomial &middot; Simulador What-If</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<span class="price-badge">&#9981; Precio gasoil: <b>${precio_gasoil:,.0f}/L</b></span>'
        f'&nbsp;&nbsp;<span style="font-size:.75rem;color:#64748b;">Fuente: {precio_fuente}</span>',
        unsafe_allow_html=True)

    anos_en_hist = sorted(df_full_clean['FECHA'].dt.year.unique().tolist()) if 'FECHA' in df_full_clean.columns else []
    anos_str = " · ".join(str(a) for a in anos_en_hist)
    st.markdown(
        f'<div class="training-badge">🧠 Modelo entrenado con {n_meses_entrenamiento} meses históricos ({anos_str})</div>',
        unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    hist = meses_hist_full.copy()
    hist['T'] = range(len(hist))

    if len(hist) >= 2:
        X      = hist['T'].values.reshape(-1, 1)
        y_l100 = hist['L100'].values
        y_lts  = hist['LITROS'].values

        # Grado 2 si hay suficientes puntos, sino 1
        degree = min(2, len(hist) - 1)
        poly   = PolynomialFeatures(degree=degree)
        Xp     = poly.fit_transform(X)
        model_l100 = LinearRegression().fit(Xp, y_l100)
        model_lts  = LinearRegression().fit(Xp, y_lts)
        r2_l100    = model_l100.score(Xp, y_l100)
        residuals  = y_l100 - model_l100.predict(Xp)
        std_res    = np.std(residuals)

        t_max   = hist['T'].max()
        t_fut   = np.array([t_max+1, t_max+2, t_max+3]).reshape(-1, 1)
        Xf      = poly.transform(t_fut)
        pred_l100 = np.clip(model_l100.predict(Xf), 0, 100)
        pred_lts  = np.clip(model_lts.predict(Xf),  0, None)

        ultimo    = hist['MES_PERIODO'].iloc[-1]
        meses_fut = [(ultimo + i + 1).strftime('%b %Y') for i in range(3)]

        st.info(f'📐 Grado polinomial: {degree} | R² = {r2_l100:.3f} | σ residuos = {std_res:.2f} L/100km')

        # ── Predicción por mes ────────────────────────────────────────────────
        st.markdown('<div class="sec-title">Predicción próximos 3 meses</div>', unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)
        for c, mes, l100_p, lts_p in zip([pc1,pc2,pc3], meses_fut, pred_l100, pred_lts):
            costo_p = lts_p * precio_gasoil
            c.metric(label=f'Predicción {mes}', value=f'{l100_p:.2f} L/100km',
                     delta=f'{lts_p:,.0f} L | ${costo_p/1e6:.2f}M')

        st.divider()

        # ── Gráfico Plotly — histórico completo + predicción ─────────────────
        st.markdown('<div class="sec-title">Evolución histórica completa con Proyección</div>', unsafe_allow_html=True)

        all_labels = [str(p) for p in hist['MES_PERIODO']] + meses_fut
        all_hist   = hist['L100'].tolist() + [None, None, None]
        all_pred   = [None]*(len(hist)-1) + [float(hist['L100'].iloc[-1])] + [float(v) for v in pred_l100]
        upper_vals = ([None]*(len(hist)-1)
                      + [float(hist['L100'].iloc[-1]) + 1.5*std_res]
                      + [float(v) + 1.5*std_res for v in pred_l100])
        lower_vals = ([None]*(len(hist)-1)
                      + [float(hist['L100'].iloc[-1]) - 1.5*std_res]
                      + [float(v) - 1.5*std_res for v in pred_l100])

        pred_start  = len(hist) - 1
        pred_labels = all_labels[pred_start:]
        upper_clean = [upper_vals[i] for i in range(pred_start, len(all_labels))]
        lower_clean = [lower_vals[i] for i in range(pred_start, len(all_labels))]

        # Colorear histórico por año
        hist_years = [p.year for p in hist['MES_PERIODO']]
        unique_years = sorted(set(hist_years))
        year_colors = {y: c for y, c in zip(unique_years, ['#94a3b8','#f97316','#ef4444','#22c55e','#60a5fa'])}

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred_labels+pred_labels[::-1], y=upper_clean+lower_clean[::-1],
            fill='toself', fillcolor='rgba(59,130,246,0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='Intervalo ±1.5σ', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=pred_labels, y=upper_clean, mode='lines',
            line=dict(color='#3b82f6',width=1,dash='dot'), name='CI sup',
            hovertemplate='CI sup: %{y:.2f} L/100km<extra></extra>'))
        fig.add_trace(go.Scatter(x=pred_labels, y=lower_clean, mode='lines',
            line=dict(color='#3b82f6',width=1,dash='dot'), name='CI inf',
            hovertemplate='CI inf: %{y:.2f} L/100km<extra></extra>'))

        # Línea histórica completa
        hist_x = [all_labels[i] for i,v in enumerate(all_hist) if v is not None]
        hist_y = [v for v in all_hist if v is not None]
        fig.add_trace(go.Scatter(x=hist_x, y=hist_y, mode='lines+markers',
            line=dict(color='#ef4444',width=2.5),
            marker=dict(size=7,color='#ef4444',line=dict(color='#fff',width=1.5)),
            name='Histórico', hovertemplate='%{x}<br>Real: <b>%{y:.2f} L/100km</b><extra></extra>'))

        # Línea predicción
        pred_x = [all_labels[i] for i,v in enumerate(all_pred) if v is not None]
        pred_y = [v for v in all_pred if v is not None]
        fig.add_trace(go.Scatter(x=pred_x, y=pred_y, mode='lines+markers',
            line=dict(color='#60a5fa',width=2.5,dash='dash'),
            marker=dict(size=9,color='#60a5fa',symbol='diamond',line=dict(color='#fff',width=1.5)),
            name='Predicción', hovertemplate='%{x}<br>Pred: <b>%{y:.2f} L/100km</b><extra></extra>'))

        # Líneas verticales para separar años
        for yr in unique_years[1:]:
            yr_label = f'Ene {yr}'
            if yr_label in all_labels:
                fig.add_vline(x=yr_label, line_width=1, line_dash="dot", line_color="#334155",
                              annotation_text=str(yr), annotation_position="top",
                              annotation_font_color="#64748b", annotation_font_size=10)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,41,59,0.6)',
            font=dict(color='#e2e8f0', family='sans-serif'),
            legend=dict(bgcolor='rgba(15,23,42,0.8)', bordercolor='#334155', borderwidth=1,
                        orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis=dict(gridcolor='#1e293b', linecolor='#334155',
                       tickfont=dict(color='#94a3b8',size=10), title=dict(text='Período',font=dict(color='#64748b')),
                       tickangle=-45),
            yaxis=dict(gridcolor='#1e293b', linecolor='#334155',
                       tickfont=dict(color='#94a3b8',size=11), title=dict(text='L/100km',font=dict(color='#64748b'))),
            height=450, margin=dict(l=10,r=10,t=50,b=60), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f'± 1.5σ intervalo de confianza | Línea roja = histórico ({n_meses_entrenamiento} meses) | Línea azul = predicción')

        st.divider()

        # ── Alerta de desvío ──────────────────────────────────────────────────
        st.markdown('<div class="sec-title">🚨 Alerta de Desvío — Predicción vs. Real</div>', unsafe_allow_html=True)
        if len(hist) >= 3:
            ultimo_real_mes  = str(hist['MES_PERIODO'].iloc[-1])
            ultimo_real_l100 = float(hist['L100'].iloc[-1])
            hist_prev = hist.iloc[:-1].copy()
            hist_prev['T'] = range(len(hist_prev))
            degree_prev = min(2, len(hist_prev) - 1)
            poly_prev  = PolynomialFeatures(degree=degree_prev)
            Xprev  = poly_prev.fit_transform(hist_prev['T'].values.reshape(-1,1))
            m_prev = LinearRegression().fit(Xprev, hist_prev['L100'].values)
            X_pred_prev = poly_prev.transform(np.array([[len(hist_prev)]]).reshape(-1,1))
            pred_ultimo = float(np.clip(m_prev.predict(X_pred_prev), 0, 100)[0])
            desvio     = ultimo_real_l100 - pred_ultimo
            desvio_pct = (desvio / pred_ultimo * 100) if pred_ultimo > 0 else 0
            umbral     = 1.5 * std_res
            if abs(desvio) > umbral:
                dir_txt = 'SUPERIOR' if desvio > 0 else 'INFERIOR'
                st.markdown(
                    f'<div class="alert-box"><b>🚨 DESVÍO DETECTADO — {ultimo_real_mes}</b><br>'
                    f'Consumo real: <b>{ultimo_real_l100:.2f} L/100km</b> &nbsp;|&nbsp; '
                    f'Predicción: <b>{pred_ultimo:.2f} L/100km</b><br>'
                    f'Desvío: <b>{desvio:+.2f} L/100km ({desvio_pct:+.1f}%)</b> — {dir_txt} al intervalo esperado (±{umbral:.2f})<br>'
                    f'🔍 Investigar posible problema mecánico o de conducción.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="alert-ok"><b>✅ Sin desvío — {ultimo_real_mes}</b><br>'
                    f'Consumo real: <b>{ultimo_real_l100:.2f} L/100km</b> &nbsp;|&nbsp; '
                    f'Predicción: <b>{pred_ultimo:.2f} L/100km</b><br>'
                    f'Desvío: <b>{desvio:+.2f} L/100km ({desvio_pct:+.1f}%)</b> — dentro del intervalo esperado (±{umbral:.2f})</div>',
                    unsafe_allow_html=True)

        st.divider()

        # ── Simulador What-If ─────────────────────────────────────────────────
        st.markdown('<div class="sec-title">🎨 Simulador What-If</div>', unsafe_allow_html=True)
        delta_precio_pct = st.slider(
            '💸 Variación precio combustible (%)',
            min_value=-30, max_value=50, value=0, step=1)
        precio_sim = precio_gasoil * (1 + delta_precio_pct / 100)

        wf1, wf2 = st.columns(2)
        with wf1:
            st.markdown(
                '<div class="kpi-card kpi-amber">'
                '<div class="kpi-label">Precio Simulado</div>'
                '<div class="kpi-value">${:,.0f}/L</div>'
                '<div class="kpi-sub">{:+.1f}% vs hoy</div>'
                '</div>'.format(precio_sim, delta_precio_pct),
                unsafe_allow_html=True)
        with wf2:
            costo_sim_m1  = pred_lts[0] * precio_sim / 1e6
            costo_base_m1 = pred_lts[0] * precio_gasoil / 1e6
            diff_costo    = costo_sim_m1 - costo_base_m1
            color_wf2     = 'kpi-red' if diff_costo > 0 else 'kpi-green'
            st.markdown(
                f'<div class="kpi-card {color_wf2}">'
                f'<div class="kpi-label">Costo {meses_fut[0]}</div>'
                f'<div class="kpi-value">${costo_sim_m1:.2f}M</div>'
                f'<div class="kpi-sub">{diff_costo:+.2f}M vs base</div>'
                f'</div>',
                unsafe_allow_html=True)

        cost_df = pd.DataFrame({
            'Mes':               meses_fut,
            'L/100km pred.':     [round(v, 2) for v in pred_l100],
            'Litros est.':       [round(v, 0) for v in pred_lts],
            'Costo base M$':     [round(v * precio_gasoil / 1e6, 2) for v in pred_lts],
            'Costo simulado M$': [round(v * precio_sim    / 1e6, 2) for v in pred_lts],
            'Dif. M$':           [round(v * (precio_sim - precio_gasoil) / 1e6, 2) for v in pred_lts]
        })
        st.dataframe(cost_df, use_container_width=True, hide_index=True)

    else:
        st.info('Se necesitan al menos 2 meses de datos históricos para el modelo predictivo.')

    st.caption(f'Modelo entrenado con {n_meses_entrenamiento} meses | Precio: {precio_fuente} | Actualización cada 10 min')


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS POR PATENTE
# ══════════════════════════════════════════════════════════════════════════════
else:
    col_logo3, col_title3 = st.columns([1, 5])
    with col_logo3:
        st.image(LOGO_URL, width=130)
    with col_title3:
        st.markdown(f"""
        <div style='padding:8px 0;'>
            <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>Análisis por Patente — {anio_sel}</div>
            <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>Consumo mensual por unidad &middot; Promedios &middot; Destacados</div>
        </div>""", unsafe_allow_html=True)

    if df.empty or 'DOMINIO' not in df.columns:
        st.warning('Sin datos disponibles.')
        st.stop()

    # ── Tabla resumen por patente ─────────────────────────────────────────────
    resumen = df.groupby('DOMINIO').agg(
        LITROS_TOTAL=('LITROS', 'sum'),
        KM_TOTAL=('KM', 'sum'),
        MESES=('MES_PERIODO', 'nunique')
    ).reset_index()
    resumen['L100KM_PROM'] = (resumen['LITROS_TOTAL'] / resumen['KM_TOTAL'].replace(0, np.nan) * 100).round(2)
    resumen['LITROS_PROM_MES'] = (resumen['LITROS_TOTAL'] / resumen['MESES'].replace(0, np.nan)).round(0)
    resumen = resumen[resumen['KM_TOTAL'] > 0].sort_values('L100KM_PROM', ascending=False)

    if resumen.empty:
        st.warning('Sin datos suficientes para analizar patentes.')
        st.stop()

    patente_max = resumen.iloc[0]
    patente_min = resumen.iloc[-1]

    # ── Destacados ────────────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-title">⚡ Destacados {anio_sel}</div>', unsafe_allow_html=True)
    hc1, hc2 = st.columns(2)
    with hc1:
        st.markdown(
            f'<div class="highlight-max">'
            f'<b>🔴 Mayor consumo — {patente_max["DOMINIO"]}</b><br>'
            f'Promedio: <b>{patente_max["L100KM_PROM"]:.2f} L/100km</b> &nbsp;|&nbsp; '
            f'Total: <b>{patente_max["LITROS_TOTAL"]:,.0f} L</b> &nbsp;|&nbsp; '
            f'{patente_max["KM_TOTAL"]:,.0f} km &nbsp;|&nbsp; {int(patente_max["MESES"])} meses activa<br>'
            f'⚠️ Revisión mecánica recomendada.</div>',
            unsafe_allow_html=True)
    with hc2:
        st.markdown(
            f'<div class="highlight-min">'
            f'<b>🟢 Menor consumo — {patente_min["DOMINIO"]}</b><br>'
            f'Promedio: <b>{patente_min["L100KM_PROM"]:.2f} L/100km</b> &nbsp;|&nbsp; '
            f'Total: <b>{patente_min["LITROS_TOTAL"]:,.0f} L</b> &nbsp;|&nbsp; '
            f'{patente_min["KM_TOTAL"]:,.0f} km &nbsp;|&nbsp; {int(patente_min["MESES"])} meses activa<br>'
            f'✅ Referencia de eficiencia de la flota.</div>',
            unsafe_allow_html=True)

    st.divider()

    # ── Gráfico barras: promedio L/100km por patente ──────────────────────────
    st.markdown(f'<div class="sec-title">Promedio L/100km por Patente — {anio_sel}</div>', unsafe_allow_html=True)

    colors_bar = []
    for _, row in resumen.iterrows():
        if row['DOMINIO'] == patente_max['DOMINIO']:
            colors_bar.append('#ef4444')
        elif row['DOMINIO'] == patente_min['DOMINIO']:
            colors_bar.append('#22c55e')
        else:
            colors_bar.append('#3b82f6')

    fig_bar = go.Figure(go.Bar(
        x=resumen['DOMINIO'],
        y=resumen['L100KM_PROM'],
        marker_color=colors_bar,
        text=resumen['L100KM_PROM'].apply(lambda v: f'{v:.1f}'),
        textposition='outside',
        textfont=dict(color='#e2e8f0', size=10),
        hovertemplate='<b>%{x}</b><br>L/100km: %{y:.2f}<extra></extra>'
    ))
    promedio_flota = resumen['L100KM_PROM'].mean()
    fig_bar.add_hline(y=promedio_flota, line_dash='dot', line_color='#f59e0b', line_width=2,
                      annotation_text=f'Promedio flota: {promedio_flota:.2f}',
                      annotation_position='top right',
                      annotation_font_color='#fbbf24', annotation_font_size=11)
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,41,59,0.6)',
        font=dict(color='#e2e8f0'),
        xaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#94a3b8', size=10), tickangle=-45),
        yaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#94a3b8'), title=dict(text='L/100km', font=dict(color='#64748b'))),
        height=420, margin=dict(l=10, r=10, t=30, b=80),
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption('🔴 Mayor consumo · 🟢 Menor consumo · 🔵 Resto · Línea amarilla = promedio flota')

    st.divider()

    # ── Heatmap: consumo mensual por patente ──────────────────────────────────
    st.markdown(f'<div class="sec-title">Consumo Mensual por Patente (L/100km) — {anio_sel}</div>', unsafe_allow_html=True)

    if 'MES_PERIODO' in df.columns:
        pivot = df[df['L100KM'] > 0].groupby(['DOMINIO', 'MES_PERIODO'])['L100KM'].mean().round(2).reset_index()
        pivot['MES_STR'] = pivot['MES_PERIODO'].astype(str)
        pivot_wide = pivot.pivot(index='DOMINIO', columns='MES_STR', values='L100KM')
        pivot_wide = pivot_wide.reindex(index=resumen['DOMINIO'].tolist())
        pivot_wide = pivot_wide.dropna(how='all')

        if not pivot_wide.empty:
            z_vals  = pivot_wide.values.tolist()
            x_vals  = list(pivot_wide.columns)
            y_vals  = list(pivot_wide.index)

            text_vals = []
            for row_data in z_vals:
                row_text = []
                for v in row_data:
                    try:
                        row_text.append(f'{float(v):.1f}' if v is not None and not np.isnan(float(v)) else '')
                    except (TypeError, ValueError):
                        row_text.append('')
                text_vals.append(row_text)

            fig_heat = go.Figure(go.Heatmap(
                z=z_vals, x=x_vals, y=y_vals,
                text=text_vals, texttemplate='%{text}',
                colorscale=[
                    [0.0,  '#052e16'],
                    [0.35, '#16a34a'],
                    [0.65, '#f59e0b'],
                    [1.0,  '#7f1d1d']
                ],
                colorbar=dict(
                    title=dict(text='L/100km', font=dict(color='#94a3b8')),
                    tickfont=dict(color='#94a3b8'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                hovertemplate='Patente: <b>%{y}</b><br>Mes: %{x}<br>L/100km: <b>%{z:.2f}</b><extra></extra>'
            ))
            fig_heat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,41,59,0.6)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(tickfont=dict(color='#94a3b8', size=10), tickangle=-45, side='bottom'),
                yaxis=dict(tickfont=dict(color='#94a3b8', size=10)),
                height=max(300, len(y_vals) * 40),
                margin=dict(l=10, r=10, t=20, b=60)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption('Verde = eficiente · Amarillo = moderado · Rojo = alto consumo · Celda vacía = sin dato ese mes')

    st.divider()

    # ── Detalle individual por patente ────────────────────────────────────────
    st.markdown('<div class="sec-title">🔍 Detalle Individual por Patente</div>', unsafe_allow_html=True)
    patentes_lista = resumen['DOMINIO'].tolist()
    pat_sel = st.selectbox('Seleccioná una patente para ver su evolución', patentes_lista)

    if pat_sel:
        df_pat = df[df['DOMINIO'] == pat_sel].copy()
        if 'MES_PERIODO' in df_pat.columns:
            df_pat_mes = df_pat.groupby('MES_PERIODO').agg(
                LITROS=('LITROS','sum'), KM=('KM','sum')
            ).reset_index().sort_values('MES_PERIODO')
            df_pat_mes['L100'] = (df_pat_mes['LITROS'] / df_pat_mes['KM'].replace(0, np.nan) * 100).round(2)
            df_pat_mes['MES_STR'] = df_pat_mes['MES_PERIODO'].astype(str)

            l100_prom_pat = df_pat_mes['L100'].mean()
            lts_total_pat = df_pat_mes['LITROS'].sum()
            kms_total_pat = df_pat_mes['KM'].sum()
            marca_pat = df_pat['MARCA'].iloc[0] if 'MARCA' in df_pat.columns else '—'

            pk1, pk2, pk3, pk4 = st.columns(4)
            pk1.metric('Patente', pat_sel)
            pk2.metric('Marca', marca_pat)
            pk3.metric('L/100km promedio', f'{l100_prom_pat:.2f}')
            pk4.metric('Litros totales', f'{lts_total_pat:,.0f}')

            fig_pat = go.Figure()
            fig_pat.add_trace(go.Bar(
                x=df_pat_mes['MES_STR'], y=df_pat_mes['LITROS'],
                name='Litros', marker_color='rgba(59,130,246,0.5)',
                yaxis='y2', hovertemplate='%{x}<br>Litros: <b>%{y:,.0f}</b><extra></extra>'
            ))
            fig_pat.add_trace(go.Scatter(
                x=df_pat_mes['MES_STR'], y=df_pat_mes['L100'],
                name='L/100km', mode='lines+markers',
                line=dict(color='#ef4444', width=2.5),
                marker=dict(size=8, color='#ef4444', line=dict(color='#fff', width=1.5)),
                hovertemplate='%{x}<br>L/100km: <b>%{y:.2f}</b><extra></extra>'
            ))
            fig_pat.add_hline(y=l100_prom_pat, line_dash='dot', line_color='#f59e0b',
                              annotation_text=f'Prom: {l100_prom_pat:.2f}',
                              annotation_font_color='#fbbf24')
            fig_pat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,41,59,0.6)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#94a3b8', size=10), tickangle=-30),
                yaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#94a3b8'),
                           title=dict(text='L/100km', font=dict(color='#ef4444'))),
                yaxis2=dict(overlaying='y', side='right',
                            tickfont=dict(color='#3b82f6'),
                            title=dict(text='Litros', font=dict(color='#3b82f6')),
                            showgrid=False),
                legend=dict(bgcolor='rgba(15,23,42,0.8)', bordercolor='#334155', borderwidth=1,
                            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=370, margin=dict(l=10,r=50,t=40,b=50)
            )
            st.plotly_chart(fig_pat, use_container_width=True)
            st.caption(f'Barras azules = litros/mes (eje derecho) · Línea roja = L/100km (eje izquierdo) · Línea amarilla = promedio')

            with st.expander(f'Ver tabla mensual — {pat_sel}'):
                df_show = df_pat_mes[['MES_STR','LITROS','KM','L100']].rename(columns={
                    'MES_STR':'Mes','LITROS':'Litros','KM':'KM','L100':'L/100km'})
                df_show['Litros'] = df_show['Litros'].apply(lambda x: f'{x:,.0f}')
                df_show['KM']     = df_show['KM'].apply(lambda x: f'{x:,.0f}')
                st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.divider()

    # ── Tabla resumen completa ────────────────────────────────────────────────
    st.markdown(f'<div class="sec-title">Tabla Resumen — Todas las Patentes {anio_sel}</div>', unsafe_allow_html=True)
    resumen_show = resumen[['DOMINIO','LITROS_TOTAL','KM_TOTAL','L100KM_PROM','LITROS_PROM_MES','MESES']].copy()
    resumen_show.columns = ['Patente','Litros Total','KM Total','L/100km Prom','Litros/Mes Prom','Meses Activa']
    resumen_show['Litros Total']    = resumen_show['Litros Total'].apply(lambda x: f'{x:,.0f}')
    resumen_show['KM Total']        = resumen_show['KM Total'].apply(lambda x: f'{x:,.0f}')
    resumen_show['Litros/Mes Prom'] = resumen_show['Litros/Mes Prom'].apply(lambda x: f'{x:,.0f}')
    st.dataframe(resumen_show, use_container_width=True, hide_index=True)

    st.caption(f'Datos {anio_sel} · Google Sheets Expreso Diemar · Actualización cada 10 min')
