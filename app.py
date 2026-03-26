import pandas as pd
import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Expreso Diemar — Predicción Combustible",
    page_icon="🚛",
    layout="wide",
)

LOGO_URL   = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/logo_diemar4.png"
IVECO_URL  = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/S-Way-6x2-1.webp"
SCANIA_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/image.img.90.768.jpeg"

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f0f4f8; }
  .kpi-card {
    background: white; border-radius: 14px; padding: 18px 22px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08); text-align: center;
    border-left: 5px solid #2563eb; margin-bottom: 6px;
  }
  .kpi-label { font-size: 0.78rem; color: #64748b; font-weight: 600;
               text-transform: uppercase; letter-spacing: .5px; margin-bottom: 4px; }
  .kpi-value { font-size: 2rem; font-weight: 800; color: #0f172a; line-height:1.1; }
  .kpi-sub   { font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }
  .kpi-red   { border-left-color: #ef4444; }
  .kpi-green { border-left-color: #22c55e; }
  .kpi-amber { border-left-color: #f59e0b; }
  .kpi-purple{ border-left-color: #a855f7; }
  .sec-title { font-size: 1.1rem; font-weight: 700; color: #1e293b;
               border-left: 4px solid #2563eb; padding-left: 10px; margin: 18px 0 10px; }
  .pred-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 14px; padding: 20px; color: white; margin-bottom: 8px;
    box-shadow: 0 4px 15px rgba(37,99,235,0.3);
  }
  .pred-month { font-size: 0.8rem; opacity: .8; margin-bottom: 4px; }
  .pred-value { font-size: 1.8rem; font-weight: 800; }
  .pred-unit  { font-size: 0.75rem; opacity: .7; }
  .truck-card {
    background: white; border-radius: 14px; overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 10px;
  }
  .truck-img  { width: 100%; height: 170px; object-fit: cover; }
  .truck-body { padding: 14px 16px; }
  .price-badge {
    background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px;
    padding: 8px 14px; display: inline-block; font-size: 0.85rem;
    color: #92400e; font-weight: 600;
  }
  .rank-row { display:flex; align-items:center; padding: 8px 0;
              border-bottom: 1px solid #f1f5f9; }
  .rank-num { width:28px; font-weight:700; font-size:.9rem; color:#64748b; }
  .rank-dom { flex:1; font-size:.88rem; color:#1e293b; font-weight:600; }
  .rank-val { font-size:.88rem; font-weight:700; }
  .rank-bar-bg { width:80px; height:6px; background:#f1f5f9; border-radius:3px;
                 margin:0 10px; overflow:hidden; }
  .rank-bar { height:6px; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# FUENTE DE DATOS
BASE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR35NkYPtJrOrdYHLGUH7GIW93s5cPAqQ0zEk5fP1c3gvErwbUW7HJ2OeWBYaBVsYKVmCf0yhLvs6eG/pub?output=csv"
GID_TEL  = "0"
GID_UNID = "882343299"
URL_TEL  = f"{BASE_URL}&gid={GID_TEL}"
URL_UNID = f"{BASE_URL}&gid={GID_UNID}"

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
                if "DOMINIO" in c or "PATENTE" in c:        cm[c] = "DOMINIO"
                elif "LITROS" in c or "CONSUMID" in c:      cm[c] = "LITROS"
                elif "DISTANCIA" in c or c == "KM" or "KILOMETR" in c: cm[c] = "KM"
                elif "CO2" in c or "EMISION" in c:          cm[c] = "CO2"
                elif "MARCA" in c:                          cm[c] = "MARCA"
                elif "TAG" in c:                            cm[c] = "TAG"
                elif "FECHA" in c or "DATE" in c:           cm[c] = "FECHA"
                elif "L/100" in c or "CONSUMO C" in c:      cm[c] = "L100KM"
                elif "RALENT" in c:                         cm[c] = "RALENTI"
                elif "TIEMPO" in c and "MOTOR" in c:        cm[c] = "TIEMPO_MOTOR"
            df = df.rename(columns=cm)
            df = df.loc[:, ~df.columns.duplicated()]
            if "DOMINIO" in df.columns:
                df["DOMINIO"] = df["DOMINIO"].astype(str).str.strip().str.upper()
            for col in ["LITROS", "KM", "CO2", "L100KM", "RALENTI"]:
                if col in df.columns:
                    serie = df[col]
                    if isinstance(serie, pd.DataFrame):
                        serie = serie.iloc[:, 0]
                    serie = (serie.astype(str)
                             .str.replace(".", "", regex=False)
                             .str.replace(",", ".", regex=False))
                    df[col] = pd.to_numeric(serie, errors="coerce").fillna(0)
            if "FECHA" in df.columns:
                df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce", dayfirst=True)
            return df

        df1 = limpiar(df1)
        df2 = limpiar(df2)

        if "L100KM" not in df1.columns and "LITROS" in df1.columns and "KM" in df1.columns:
            df1["L100KM"] = (df1["LITROS"] / df1["KM"].replace(0, np.nan) * 100).round(2)

        extra = [c for c in ["DOMINIO", "CO2", "RALENTI"] if c in df2.columns]
        if len(extra) > 1:
            df2r = df2[extra].groupby("DOMINIO").sum(numeric_only=True).reset_index()
            df1 = pd.merge(df1, df2r, on="DOMINIO", how="left", suffixes=("", "_u"))
            for col in ["CO2", "RALENTI"]:
                if f"{col}_u" in df1.columns:
                    df1[col] = df1[col].combine_first(df1[f"{col}_u"])
                    df1.drop(columns=[f"{col}_u"], inplace=True)
        return df1, df2
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600)
def obtener_precio_gasoil():
    try:
        import re
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get("https://www.surtidores.com.ar/precios/", headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        texto = soup.get_text(separator=" ")
        matches = re.findall(r'[Gg]as[aoi][io]l[^\d]*(\d{3,4}[.,]\d{0,2})', texto[:8000])
        if matches:
            precio = float(matches[0].replace(",", "."))
            if 500 < precio < 5000:
                return precio, "surtidores.com.ar"
    except Exception:
        pass
    return 1420.0, "referencia estimada"

# CARGA
with st.spinner('Cargando telemetria...'):
    df_raw, df_unid = cargar_datos()

if df_raw.empty:
    st.warning('No se pudieron cargar los datos.')
    st.stop()

precio_gasoil, precio_fuente = obtener_precio_gasoil()

# HEADER
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image(LOGO_URL, width=130)
with col_title:
    st.markdown("""
    <div style='padding:8px 0;'>
      <div style='font-size:1.6rem;font-weight:800;color:#0f172a;'>
        Expreso Diemar &mdash; Prediccion de Consumo de Combustible
      </div>
      <div style='font-size:.9rem;color:#64748b;margin-top:4px;'>
        Telemetria flota completa &middot; Modelo predictivo ML &middot; Actualizacion automatica
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown(
    f'<div style="margin-bottom:12px;">'
    f'<span class="price-badge">&#9981; Precio gasoil referencia: <b>${precio_gasoil:,.0f}/L</b></span>'
    f'&nbsp;&nbsp;<span style="font-size:.75rem;color:#94a3b8;">Fuente: {precio_fuente}</span></div>',
    unsafe_allow_html=True
)

# FILTROS
df = df_raw.copy()
if 'FECHA' in df.columns and df['FECHA'].notna().any():
    with st.expander('Filtros', expanded=True):
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        fmin = df['FECHA'].min().date()
        fmax = df['FECHA'].max().date()
        desde = fc1.date_input('Desde', value=fmin, min_value=fmin, max_value=fmax)
        hasta = fc2.date_input('Hasta', value=fmax, min_value=fmin, max_value=fmax)
        marcas_disp = sorted(df['MARCA'].dropna().unique().tolist()) if 'MARCA' in df.columns else []
        marcas_sel  = fc3.multiselect('Marca', marcas_disp, default=marcas_disp)
        df = df[(df['FECHA'].dt.date >= desde) & (df['FECHA'].dt.date <= hasta)]
        if marcas_sel and 'MARCA' in df.columns:
            df = df[df['MARCA'].isin(marcas_sel)]
    if df.empty:
        st.warning('Sin datos para los filtros seleccionados.')
        st.stop()

df['MES_PERIODO'] = df['FECHA'].dt.to_period('M')
df['MES_NUM'] = (df['FECHA'].dt.year - df['FECHA'].dt.year.min()) * 12 + df['FECHA'].dt.month

# KPIs
st.markdown('<div class="sec-title">Metricas Globales de la Flota</div>', unsafe_allow_html=True)

lts_total  = df['LITROS'].sum() if 'LITROS' in df.columns else 0
kms_total  = df['KM'].sum()     if 'KM'     in df.columns else 0
co2_total  = df['CO2'].sum()    if 'CO2'    in df.columns else 0
l100_prom  = round(lts_total / kms_total * 100, 2) if kms_total > 0 else 0
costo_est  = lts_total * precio_gasoil
n_unidades = df['DOMINIO'].nunique() if 'DOMINIO' in df.columns else 0
ralenti_total = df['RALENTI'].sum() if 'RALENTI' in df.columns else 0

meses_df = df.groupby('MES_PERIODO').agg(
    LITROS=('LITROS', 'sum'), KM=('KM', 'sum')
).reset_index().sort_values('MES_PERIODO')
meses_df['L100'] = (meses_df['LITROS'] / meses_df['KM'].replace(0, np.nan) * 100).round(2)

if len(meses_df) >= 2:
    delta_l100 = meses_df['L100'].iloc[-1] - meses_df['L100'].iloc[-2]
    delta_txt  = f"{'▲' if delta_l100>0 else '▼'} {abs(delta_l100):.2f} vs mes anterior"
    delta_col  = 'kpi-red' if delta_l100 > 0 else 'kpi-green'
else:
    delta_txt = ''
    delta_col = ''

k1, k2, k3, k4, k5, k6 = st.columns(6)

def kpi(container, color, label, value, sub=''):
    container.markdown(
        f'<div class="kpi-card {color}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>', unsafe_allow_html=True)

kpi(k1, '',          '&#9981; Litros totales',    f'{lts_total:,.0f}',     'litros consumidos')
kpi(k2, '',          '&#128739; KM recorridos',   f'{kms_total:,.0f}',     'kilometros')
kpi(k3, delta_col,   '&#128202; L/100km flota',   f'{l100_prom:.2f}',      delta_txt)
kpi(k4, 'kpi-amber', '&#128176; Costo estimado',  f'${costo_est/1e6:.1f}M', f'@ ${precio_gasoil:,.0f}/L')
kpi(k5, 'kpi-purple','&#127807; CO2 emitido',     f'{co2_total/1000:.1f}t', 'toneladas CO2')
kpi(k6, 'kpi-green', '&#128666; Unidades activas',f'{n_unidades}',          'dominios unicos')

st.divider()

# TARJETAS IVECO / SCANIA
st.markdown('<div class="sec-title">Rendimiento por Marca Principal</div>', unsafe_allow_html=True)

def stats_marca(marca):
    sub = df[df['MARCA'].str.upper() == marca.upper()] if 'MARCA' in df.columns else pd.DataFrame()
    if sub.empty:
        return {'l100': 0, 'lts': 0, 'kms': 0, 'n': 0, 'co2': 0, 'ralenti': 0}
    lts = sub['LITROS'].sum()
    kms = sub['KM'].sum()
    return {
        'l100': round(lts / kms * 100, 2) if kms > 0 else 0,
        'lts': lts, 'kms': kms,
        'n': sub['DOMINIO'].nunique(),
        'co2': sub['CO2'].sum() if 'CO2' in sub.columns else 0,
        'ralenti': sub['RALENTI'].sum() if 'RALENTI' in sub.columns else 0
    }

tc1, tc2 = st.columns(2)
for col_t, marca, img_url, color, modelo in [
    (tc1, 'IVECO',  IVECO_URL,  '#2563eb', 'S-Way 6x2'),
    (tc2, 'SCANIA', SCANIA_URL, '#dc2626', 'Serie R/S')
]:
    s = stats_marca(marca)
    with col_t:
        st.image(img_url, use_container_width=True)
        sub_col1, sub_col2, sub_col3 = st.columns(3)
        sub_col1.metric(f'{marca} - L/100km', f"{s['l100']:.1f}")
        sub_col2.metric('Unidades',           f"{s['n']}")
        sub_col3.metric('Litros totales',     f"{s['lts']:,.0f}")
        st.caption(f"Modelo: {modelo} | {s['kms']:,.0f} km | CO2: {s['co2']/1000:.1f}t")

st.divider()

# MODELO PREDICTIVO
st.markdown('<div class="sec-title">Modelo Predictivo &mdash; Consumo Proximos 3 Meses</div>', unsafe_allow_html=True)

hist = meses_df[meses_df['KM'] > 0].copy()
hist['T'] = range(len(hist))

pred_col1, pred_col2 = st.columns([3, 2])

with pred_col1:
    if len(hist) >= 4:
        X = hist['T'].values.reshape(-1, 1)
        y_l100 = hist['L100'].values
        y_lts  = hist['LITROS'].values

        poly = PolynomialFeatures(degree=2)
        Xp   = poly.fit_transform(X)
        model_l100 = LinearRegression().fit(Xp, y_l100)
        model_lts  = LinearRegression().fit(Xp, y_lts)

        r2_l100 = model_l100.score(Xp, y_l100)

        t_max   = hist['T'].max()
        t_fut   = np.array([t_max+1, t_max+2, t_max+3]).reshape(-1, 1)
        Xf      = poly.transform(t_fut)
        pred_l100 = np.clip(model_l100.predict(Xf), 0, 100)
        pred_lts  = np.clip(model_lts.predict(Xf),  0, None)

        ultimo_periodo = hist['MES_PERIODO'].iloc[-1]
        meses_fut = [(ultimo_periodo + i + 1).strftime('%b %Y') for i in range(3)]

        st.caption(f'Modelo de regresion polinomial grado 2 | R2 = {r2_l100:.3f} | Entrenado con {len(hist)} meses de datos')

        pc1, pc2, pc3 = st.columns(3)
        for c, mes, l100_p, lts_p in zip([pc1, pc2, pc3], meses_fut, pred_l100, pred_lts):
            costo_p = lts_p * precio_gasoil
            c.metric(
                label=f'Prediccion {mes}',
                value=f'{l100_p:.2f} L/100km',
                delta=f'{lts_p:,.0f} L | ${costo_p/1e6:.2f}M'
            )

        # Grafico con st.line_chart
        import io
        hist_chart = hist[['MES_PERIODO', 'L100']].copy()
        hist_chart['MES_PERIODO'] = hist_chart['MES_PERIODO'].astype(str)

        # Armar DataFrame para visualizacion
        all_labels  = [str(p) for p in hist['MES_PERIODO']] + meses_fut
        all_hist    = hist['L100'].tolist() + [None, None, None]
        all_pred_partial = [None] * (len(hist) - 1) + [float(hist['L100'].iloc[-1])] + [float(v) for v in pred_l100]
        chart_df = pd.DataFrame({
            'Mes': all_labels,
            'Historico': all_hist,
            'Prediccion': all_pred_partial
        }).set_index('Mes')
        st.line_chart(chart_df, use_container_width=True)

    else:
        st.info('Se necesitan al menos 4 meses de datos para el modelo predictivo.')

with pred_col2:
    st.markdown('**TOP 10 Unidades mas eficientes (menor L/100km)**')
    if 'DOMINIO' in df.columns and 'L100KM' in df.columns:
        rank = (df[df['L100KM'] > 0]
                .groupby('DOMINIO')['L100KM']
                .mean().round(2).reset_index()
                .sort_values('L100KM').head(10))
        vmin = rank['L100KM'].min()
        vmax = rank['L100KM'].max()
        rank_html = '<div style="background:white;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.07);">'
        rank_html += '<div style="font-size:.72rem;display:flex;justify-content:space-between;color:#94a3b8;margin-bottom:6px;"><span>Unidad</span><span>L/100km</span></div>'
        for i, (_, r) in enumerate(rank.iterrows(), 1):
            v   = r['L100KM']
            pct = int((v - vmin) / (vmax - vmin) * 100) if vmax != vmin else 50
            col_b = '#22c55e' if i <= 3 else ('#f59e0b' if i <= 6 else '#ef4444')
            rank_html += f'<div class="rank-row">'
            rank_html += f'<div class="rank-num">#{i}</div>'
            rank_html += f'<div class="rank-dom">{r["DOMINIO"]}</div>'
            rank_html += f'<div class="rank-bar-bg"><div class="rank-bar" style="width:{pct}%;background:{col_b}"></div></div>'
            rank_html += f'<div class="rank-val" style="color:{col_b}">{v:.2f}</div></div>'
        rank_html += '</div>'
        st.markdown(rank_html, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('**TOP 10 Unidades MENOS eficientes**')
    if 'DOMINIO' in df.columns and 'L100KM' in df.columns:
        rank_worst = (df[df['L100KM'] > 0]
                      .groupby('DOMINIO')['L100KM']
                      .mean().round(2).reset_index()
                      .sort_values('L100KM', ascending=False).head(10))
        st.dataframe(rank_worst, use_container_width=True, hide_index=True)

st.divider()

# ANALISIS DETALLADO
st.markdown('<div class="sec-title">Analisis Detallado</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(['Por Ruta/TAG', 'Por Vehiculo', 'Evolucion Mensual', 'Ralenti y CO2'])

with tab1:
    if 'TAG' in df.columns and 'L100KM' in df.columns:
        t_tag = (
            df[df['L100KM'] > 0].groupby('TAG')
            .agg(L100KM=('L100KM','mean'), LITROS=('LITROS','sum'), KM=('KM','sum'), UNIDADES=('DOMINIO','nunique'))
            .round(2).reset_index().sort_values('L100KM')
        )
        t_tag['COSTO_M$'] = (t_tag['LITROS'] * precio_gasoil / 1e6).round(2)
        t_tag.columns = ['Ruta/TAG', 'L/100km', 'Litros', 'KM', 'Unidades', 'Costo est. M$']
        st.dataframe(t_tag, use_container_width=True, hide_index=True)
    else:
        st.info('No hay columna TAG disponible.')

with tab2:
    if 'DOMINIO' in df.columns:
        grp_cols = ['DOMINIO', 'MARCA'] if 'MARCA' in df.columns else ['DOMINIO']
        t_veh = (
            df[df['L100KM'] > 0].groupby(grp_cols)
            .agg(L100KM=('L100KM','mean'), LITROS=('LITROS','sum'), KM=('KM','sum'), MESES=('MES_PERIODO','nunique'))
            .round(2).reset_index().sort_values('L100KM')
        )
        t_veh['COSTO_M$'] = (t_veh['LITROS'] * precio_gasoil / 1e6).round(2)
        st.dataframe(t_veh, use_container_width=True, hide_index=True, height=420)

with tab3:
    c_ev1, c_ev2 = st.columns(2)
    with c_ev1:
        st.markdown('**Litros consumidos por mes**')
        if not meses_df.empty:
            ms = meses_df.copy()
            ms['MES_PERIODO'] = ms['MES_PERIODO'].astype(str)
            st.bar_chart(ms.set_index('MES_PERIODO')['LITROS'], use_container_width=True)
    with c_ev2:
        st.markdown('**Eficiencia L/100km por mes**')
        if not meses_df.empty:
            ms2 = meses_df.copy()
            ms2['MES_PERIODO'] = ms2['MES_PERIODO'].astype(str)
            st.line_chart(ms2.set_index('MES_PERIODO')['L100'], use_container_width=True)
    if len(meses_df) >= 2:
        ms3 = meses_df.copy()
        ms3['MES_PERIODO'] = ms3['MES_PERIODO'].astype(str)
        ms3['Costo_M$'] = (ms3['LITROS'] * precio_gasoil / 1e6).round(2)
        st.markdown('**Costo estimado por mes**')
        st.bar_chart(ms3.set_index('MES_PERIODO')['Costo_M$'], use_container_width=True)

with tab4:
    has_env = ('RALENTI' in df.columns and df['RALENTI'].sum() > 0) or ('CO2' in df.columns and df['CO2'].sum() > 0)
    if has_env:
        agg_dict = {}
        if 'CO2' in df.columns:     agg_dict['CO2']     = ('CO2',     'sum')
        if 'RALENTI' in df.columns: agg_dict['RALENTI']  = ('RALENTI', 'sum')
        t_env = df.groupby('DOMINIO').agg(**agg_dict).round(2).reset_index()
        if 'CO2' in t_env.columns:
            t_env = t_env.sort_values('CO2', ascending=False)
        st.dataframe(t_env.head(30), use_container_width=True, hide_index=True)
        e1, e2 = st.columns(2)
        if 'RALENTI' in df.columns and df['RALENTI'].sum() > 0:
            e1.metric('Litros en Ralenti', f"{df['RALENTI'].sum():,.0f} L",
                      delta=f"${df['RALENTI'].sum() * precio_gasoil / 1e6:.2f}M costo")
        if 'CO2' in df.columns and df['CO2'].sum() > 0:
            e2.metric('CO2 Total (ton)', f"{df['CO2'].sum() / 1000:.1f}")
    else:
        st.info('Datos de ralenti/CO2 no disponibles en el periodo seleccionado.')

st.divider()

# DATOS CRUDOS
with st.expander('Ver datos completos'):
    cols_s = [c for c in ['DOMINIO', 'MARCA', 'TAG', 'FECHA', 'KM', 'LITROS', 'L100KM', 'CO2', 'RALENTI']
              if c in df.columns]
    st.dataframe(df[cols_s], use_container_width=True, height=380)

st.caption(f'Datos: Google Sheets Expreso Diemar | Precio combustible: {precio_fuente} | Actualizacion cada 10 min')
