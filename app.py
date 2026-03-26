import pandas as pd
import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Expreso Diemar — Predicción Combustible",
    page_icon="🚛",
    layout="wide",
)

LOGO_URL   = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/logo_diemar4.png"
IVECO_URL  = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/S-Way-6x2-1.webp"
SCANIA_URL = "https://raw.githubusercontent.com/nicolascolonna23/Modelo-Prediccion-Combustible/main/2016p.png"

# ── TEMA OSCURO ─────────────────────────────────────────────────────────────
DARK_CSS = """
<style>
[data-testid=\"stAppViewContainer\"] { background: #0f172a; }
[data-testid=\"stSidebar\"]          { background: #1e293b; }
section[data-testid=\"stMain\"]      { background: #0f172a; }
.stMarkdown, .stCaption, label, p, span, div { color: #e2e8f0 !important; }
[data-testid=\"stMetricValue\"]      { color: #f1f5f9 !important; }
[data-testid=\"stMetricDelta\"]      { color: #94a3b8 !important; }
.kpi-card {
    background: #1e293b; border-radius: 14px; padding: 18px 22px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.4); text-align: center;
    border-left: 5px solid #2563eb; margin-bottom: 6px;
}
.kpi-label { font-size:0.78rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:.5px; margin-bottom:4px; }
.kpi-value { font-size:2rem; font-weight:800; color:#f1f5f9; line-height:1.1; }
.kpi-sub   { font-size:0.75rem; color:#64748b; margin-top:4px; }
.kpi-red    { border-left-color:#ef4444; }
.kpi-green  { border-left-color:#22c55e; }
.kpi-amber  { border-left-color:#f59e0b; }
.sec-title { font-size:1.1rem; font-weight:700; color:#e2e8f0; border-left:4px solid #2563eb; padding-left:10px; margin:18px 0 10px; }
.price-badge { background:#292524; border:1px solid #f59e0b; border-radius:8px; padding:8px 14px; display:inline-block; font-size:0.85rem; color:#fbbf24; font-weight:600; }
.rank-row { display:flex; align-items:center; padding:8px 0; border-bottom:1px solid #334155; }
.rank-num { width:28px; font-weight:700; font-size:.9rem; color:#94a3b8; }
.rank-dom { flex:1; font-size:.88rem; color:#e2e8f0; font-weight:600; }
.rank-val { font-size:.88rem; font-weight:700; }
.rank-bar-bg { width:80px; height:6px; background:#334155; border-radius:3px; margin:0 10px; overflow:hidden; }
.rank-bar    { height:6px; border-radius:3px; }
</style>
"""

# ── NAVEGACION ──────────────────────────────────────────────────────────────
pg = st.sidebar.radio(
    "Navegacion",
    ["Dashboard Principal", "Modelo Predictivo"],
    index=0,
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.image(LOGO_URL, width=160)

# ── FUENTE DE DATOS ─────────────────────────────────────────────────────────
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
                if "DOMINIO" in c or "PATENTE" in c: cm[c] = "DOMINIO"
                elif "LITROS" in c or "CONSUMID" in c: cm[c] = "LITROS"
                elif "DISTANCIA" in c or c == "KM" or "KILOMETR" in c: cm[c] = "KM"
                elif "MARCA" in c: cm[c] = "MARCA"
                elif "TAG" in c: cm[c] = "TAG"
                elif "FECHA" in c or "DATE" in c: cm[c] = "FECHA"
                elif "L/100" in c or "CONSUMO C" in c: cm[c] = "L100KM"
                elif "RALENT" in c: cm[c] = "RALENTI"
                elif "TIEMPO" in c and "MOTOR" in c: cm[c] = "TIEMPO_MOTOR"
            df = df.rename(columns=cm)
            df = df.loc[:, ~df.columns.duplicated()]
            if "DOMINIO" in df.columns:
                df["DOMINIO"] = df["DOMINIO"].astype(str).str.strip().str.upper()
            for col in ["LITROS", "KM", "L100KM", "RALENTI"]:
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

        extra = [c for c in ["DOMINIO", "RALENTI"] if c in df2.columns]
        if len(extra) > 1:
            df2r = df2[extra].groupby("DOMINIO").sum(numeric_only=True).reset_index()
            df1 = pd.merge(df1, df2r, on="DOMINIO", how="left", suffixes=("", "_u"))
            for col in ["RALENTI"]:
                if f"{col}_u" in df1.columns:
                    df1[col] = df1[col].combine_first(df1[f"{col}_u"])
                    df1.drop(columns=[f"{col}_u"], inplace=True)

        if "EMPRESA" in df1.columns:
            df1 = df1[df1["EMPRESA"].str.upper().str.contains("LAD|DIEMAR", na=False)]

        return df1, df2
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600)
def obtener_precio_gasoil():
    """Extrae el ultimo precio de Gasoil de 2026 desde surtidores.com.ar/precios/"""
    try:
        import re
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get("https://surtidores.com.ar/precios/", headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        texto = soup.get_text(separator=" ")
        idx_2026 = texto.find("2026")
        if idx_2026 != -1:
            segmento = texto[idx_2026:idx_2026 + 600]
            gasoil_idx = segmento.lower().find("gasoil")
            if gasoil_idx != -1:
                linea = segmento[gasoil_idx:gasoil_idx + 120]
                numeros = re.findall(r'\\b(\\d{3,4})\\b', linea)
                numeros = [int(n) for n in numeros if 500 <= int(n) <= 5000]
                if numeros:
                    return float(numeros[-1]), "surtidores.com.ar (2026)"
        matches = re.findall(r'[Gg]as[oi][il][^\\d]*(\\d{3,4})', texto[:8000])
        if matches:
            precio = float(matches[0])
            if 500 < precio < 5000:
                return precio, "surtidores.com.ar"
    except Exception:
        pass
    return 2025.0, "referencia estimada (mar 2026)"

with st.spinner('Cargando telemetría...'):
    df_raw, df_unid = cargar_datos()

if df_raw.empty:
    st.warning('No se pudieron cargar los datos.')
    st.stop()

precio_gasoil, precio_fuente = obtener_precio_gasoil()

st.markdown(DARK_CSS, unsafe_allow_html=True)

# ── FILTROS COMUNES ─────────────────────────────────────────────────────────
df = df_raw.copy()
if 'FECHA' in df.columns and df['FECHA'].notna().any():
    with st.sidebar.expander('Filtros', expanded=True):
        fmin = df['FECHA'].min().date()
        fmax = df['FECHA'].max().date()
        desde = st.date_input('Desde', value=fmin, min_value=fmin, max_value=fmax)
        hasta = st.date_input('Hasta', value=fmax, min_value=fmin, max_value=fmax)
        marcas_disp = sorted(df['MARCA'].dropna().unique().tolist()) if 'MARCA' in df.columns else []
        marcas_sel  = st.multiselect('Marca', marcas_disp, default=marcas_disp)
        df = df[(df['FECHA'].dt.date >= desde) & (df['FECHA'].dt.date <= hasta)]
        if marcas_sel and 'MARCA' in df.columns:
            df = df[df['MARCA'].isin(marcas_sel)]

if df.empty:
    st.warning('Sin datos para los filtros seleccionados.')
    st.stop()

df['MES_PERIODO'] = df['FECHA'].dt.to_period('M')
df['MES_NUM']     = (df['FECHA'].dt.year - df['FECHA'].dt.year.min()) * 12 + df['FECHA'].dt.month

meses_df = df.groupby('MES_PERIODO').agg(
    LITROS=('LITROS', 'sum'),
    KM=('KM', 'sum')
).reset_index().sort_values('MES_PERIODO')
meses_df['L100'] = (meses_df['LITROS'] / meses_df['KM'].replace(0, np.nan) * 100).round(2)

# ── PAGINA: DASHBOARD PRINCIPAL ─────────────────────────────────────────────
if pg == "Dashboard Principal":
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image(LOGO_URL, width=130)
    with col_title:
        st.markdown("""
        <div style='padding:8px 0;'>
          <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>
            Expreso Diemar &mdash; Dashboard de Consumo LAD
          </div>
          <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>
            Telemetria flota LAD &middot; Actualizacion automatica
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<span class="price-badge">&#9981; Precio gasoil: <b>${precio_gasoil:,.0f}/L</b></span>'
        f'&nbsp;&nbsp;<span style="font-size:.75rem;color:#64748b;">Fuente: {precio_fuente}</span></div>',
        unsafe_allow_html=True
    )

    # KPIs
    st.markdown('<div class="sec-title">Metricas Globales de la Flota</div>', unsafe_allow_html=True)
    lts_total  = df['LITROS'].sum() if 'LITROS' in df.columns else 0
    kms_total  = df['KM'].sum() if 'KM' in df.columns else 0
    l100_prom  = round(lts_total / kms_total * 100, 2) if kms_total > 0 else 0
    costo_est  = lts_total * precio_gasoil
    n_unidades = df['DOMINIO'].nunique() if 'DOMINIO' in df.columns else 0

    if len(meses_df) >= 2:
        delta_l100 = meses_df['L100'].iloc[-1] - meses_df['L100'].iloc[-2]
        delta_txt  = f"{'\u25b2' if delta_l100>0 else '\u25bc'} {abs(delta_l100):.2f} vs mes anterior"
        delta_col  = 'kpi-red' if delta_l100 > 0 else 'kpi-green'
    else:
        delta_txt, delta_col = '', ''

    k1, k2, k3, k4, k5 = st.columns(5)
    def kpi(container, color, label, value, sub=''):
        container.markdown(
            f'<div class="kpi-card {color}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True)

    kpi(k1, '',          '&#9981; Litros totales',     f'{lts_total:,.0f}',       'litros consumidos')
    kpi(k2, '',          '&#128739; KM recorridos',    f'{kms_total:,.0f}',       'kilometros')
    kpi(k3, delta_col,   '&#128202; L/100km flota',    f'{l100_prom:.2f}',        delta_txt)
    kpi(k4, 'kpi-amber', '&#128176; Costo estimado',   f'${costo_est/1e6:.1f}M',  f'@ ${precio_gasoil:,.0f}/L')
    kpi(k5, 'kpi-green', '&#128666; Unidades activas', f'{n_unidades}',           'dominios unicos')

    st.divider()

    # Tarjetas por marca
    st.markdown('<div class="sec-title">Rendimiento por Marca Principal</div>', unsafe_allow_html=True)
    def stats_marca(marca):
        sub = df[df['MARCA'].str.upper() == marca.upper()] if 'MARCA' in df.columns else pd.DataFrame()
        if sub.empty:
            return {'l100': 0, 'lts': 0, 'kms': 0, 'n': 0}
        lts = sub['LITROS'].sum()
        kms = sub['KM'].sum()
        return {'l100': round(lts/kms*100,2) if kms>0 else 0, 'lts':lts, 'kms':kms, 'n':sub['DOMINIO'].nunique()}

    tc1, tc2 = st.columns(2)
    for col_t, marca, img_url, modelo in [
        (tc1, 'IVECO',  IVECO_URL,  'S-Way 6x2'),
        (tc2, 'SCANIA', SCANIA_URL, 'Serie P 2016')
    ]:
        s = stats_marca(marca)
        with col_t:
            st.image(img_url, use_container_width=True)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric(f'{marca} - L/100km', f"{s['l100']:.1f}")
            sc2.metric('Unidades', f"{s['n']}")
            sc3.metric('Litros totales', f"{s['lts']:,.0f}")
            st.caption(f"Modelo: {modelo} | {s['kms']:,.0f} km")

    st.divider()

    # Analisis detallado
    st.markdown('<div class="sec-title">Analisis Detallado</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(['Por Ruta/TAG', 'Por Vehiculo', 'Evolucion Mensual', 'Ralenti'])

    with tab1:
        if 'TAG' in df.columns and 'L100KM' in df.columns:
            t_tag = (df[df['L100KM']>0].groupby('TAG')
                     .agg(L100KM=('L100KM','mean'),LITROS=('LITROS','sum'),KM=('KM','sum'),UNIDADES=('DOMINIO','nunique'))
                     .round(2).reset_index().sort_values('L100KM'))
            t_tag['COSTO_M$'] = (t_tag['LITROS']*precio_gasoil/1e6).round(2)
            t_tag.columns = ['Ruta/TAG','L/100km','Litros','KM','Unidades','Costo est. M$']
            st.dataframe(t_tag, use_container_width=True, hide_index=True)
        else:
            st.info('No hay columna TAG disponible.')

    with tab2:
        if 'DOMINIO' in df.columns:
            grp = ['DOMINIO','MARCA'] if 'MARCA' in df.columns else ['DOMINIO']
            t_veh = (df[df['L100KM']>0].groupby(grp)
                     .agg(L100KM=('L100KM','mean'),LITROS=('LITROS','sum'),KM=('KM','sum'),MESES=('MES_PERIODO','nunique'))
                     .round(2).reset_index().sort_values('L100KM'))
            t_veh['COSTO_M$'] = (t_veh['LITROS']*precio_gasoil/1e6).round(2)
            st.dataframe(t_veh, use_container_width=True, hide_index=True, height=420)

    with tab3:
        ce1, ce2 = st.columns(2)
        with ce1:
            st.markdown('**Litros por mes**')
            if not meses_df.empty:
                ms = meses_df.copy(); ms['MES_PERIODO'] = ms['MES_PERIODO'].astype(str)
                st.bar_chart(ms.set_index('MES_PERIODO')['LITROS'], use_container_width=True)
        with ce2:
            st.markdown('**L/100km por mes**')
            if not meses_df.empty:
                ms2 = meses_df.copy(); ms2['MES_PERIODO'] = ms2['MES_PERIODO'].astype(str)
                st.line_chart(ms2.set_index('MES_PERIODO')['L100'], use_container_width=True)
        if len(meses_df) >= 2:
            ms3 = meses_df.copy(); ms3['MES_PERIODO'] = ms3['MES_PERIODO'].astype(str)
            ms3['Costo_M$'] = (ms3['LITROS']*precio_gasoil/1e6).round(2)
            st.markdown('**Costo estimado por mes**')
            st.bar_chart(ms3.set_index('MES_PERIODO')['Costo_M$'], use_container_width=True)

    with tab4:
        if 'RALENTI' in df.columns and df['RALENTI'].sum() > 0:
            t_ral = df.groupby('DOMINIO').agg(RALENTI=('RALENTI','sum')).round(2).reset_index().sort_values('RALENTI', ascending=False)
            st.dataframe(t_ral.head(30), use_container_width=True, hide_index=True)
            st.metric('Litros en Ralenti', f"{df['RALENTI'].sum():,.0f} L",
                      delta=f"${df['RALENTI'].sum()*precio_gasoil/1e6:.2f}M costo")
        else:
            st.info('Datos de ralenti no disponibles.')

    st.divider()

    # Ranking eficiencia
    st.markdown('<div class="sec-title">Ranking de Eficiencia por Unidad</div>', unsafe_allow_html=True)
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.markdown('**TOP 10 mas eficientes (menor L/100km)**')
        if 'DOMINIO' in df.columns and 'L100KM' in df.columns:
            rank = (df[df['L100KM']>0].groupby('DOMINIO')['L100KM'].mean().round(2).reset_index().sort_values('L100KM').head(10))
            vmin, vmax = rank['L100KM'].min(), rank['L100KM'].max()
            rh = '<div style="background:#1e293b;border-radius:12px;padding:16px;">'
            rh += '<div style="font-size:.72rem;display:flex;justify-content:space-between;color:#94a3b8;margin-bottom:6px;"><span>Unidad</span><span>L/100km</span></div>'
            for i,(_, r) in enumerate(rank.iterrows(),1):
                v=r['L100KM']; pct=int((v-vmin)/(vmax-vmin)*100) if vmax!=vmin else 50
                cb='#22c55e' if i<=3 else('#f59e0b' if i<=6 else '#ef4444')
                rh+=f'<div class="rank-row"><div class="rank-num">#{i}</div>'
                rh+=f'<div class="rank-dom">{r["DOMINIO"]}</div>'
                rh+=f'<div class="rank-bar-bg"><div class="rank-bar" style="width:{pct}%;background:{cb}"></div></div>'
                rh+=f'<div class="rank-val" style="color:{cb}">{v:.2f}</div></div>'
            rh+='</div>'
            st.markdown(rh, unsafe_allow_html=True)
    with rcol2:
        st.markdown('**TOP 10 MENOS eficientes**')
        if 'DOMINIO' in df.columns and 'L100KM' in df.columns:
            rw = (df[df['L100KM']>0].groupby('DOMINIO')['L100KM'].mean().round(2).reset_index().sort_values('L100KM', ascending=False).head(10))
            st.dataframe(rw, use_container_width=True, hide_index=True)

    st.divider()
    with st.expander('Ver datos completos'):
        cols_s = [c for c in ['DOMINIO','MARCA','TAG','FECHA','KM','LITROS','L100KM','RALENTI'] if c in df.columns]
        st.dataframe(df[cols_s], use_container_width=True, height=380)
    st.caption(f'Datos: Google Sheets Expreso Diemar | Precio: {precio_fuente} | Actualización cada 10 min')

# ── PAGINA: MODELO PREDICTIVO ───────────────────────────────────────────────
else:  # pg == 'Modelo Predictivo'
    col_logo2, col_title2 = st.columns([1, 5])
    with col_logo2:
        st.image(LOGO_URL, width=130)
    with col_title2:
        st.markdown("""
        <div style='padding:8px 0;'>
          <div style='font-size:1.6rem;font-weight:800;color:#f1f5f9;'>
            Modelo Predictivo &mdash; Consumo Próximos 3 Meses
          </div>
          <div style='font-size:.9rem;color:#94a3b8;margin-top:4px;'>
            Regresión polinomial grado 2 &middot; Flota LAD
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<span class="price-badge">&#9981; Precio gasoil: <b>${precio_gasoil:,.0f}/L</b></span>'
        f'&nbsp;&nbsp;<span style="font-size:.75rem;color:#64748b;">Fuente: {precio_fuente}</span>',
        unsafe_allow_html=True
    )
    st.markdown('<br>', unsafe_allow_html=True)

    hist = meses_df[meses_df['KM'] > 0].copy()
    hist['T'] = range(len(hist))

    if len(hist) >= 4:
        X = hist['T'].values.reshape(-1, 1)
        y_l100 = hist['L100'].values
        y_lts  = hist['LITROS'].values
        poly   = PolynomialFeatures(degree=2)
        Xp     = poly.fit_transform(X)
        model_l100 = LinearRegression().fit(Xp, y_l100)
        model_lts  = LinearRegression().fit(Xp, y_lts)
        r2_l100 = model_l100.score(Xp, y_l100)

        t_max   = hist['T'].max()
        t_fut   = np.array([t_max+1, t_max+2, t_max+3]).reshape(-1, 1)
        Xf      = poly.transform(t_fut)
        pred_l100 = np.clip(model_l100.predict(Xf), 0, 100)
        pred_lts  = np.clip(model_lts.predict(Xf), 0, None)

        ultimo_periodo = hist['MES_PERIODO'].iloc[-1]
        meses_fut = [(ultimo_periodo + i + 1).strftime('%b %Y') for i in range(3)]

        st.info(f'🧠 Modelo entrenado con {len(hist)} meses | R² = {r2_l100:.3f}')

        # Predicciones mes a mes
        st.markdown('<div class="sec-title">Prediccion por Mes</div>', unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)
        for c, mes, l100_p, lts_p in zip([pc1, pc2, pc3], meses_fut, pred_l100, pred_lts):
            costo_p = lts_p * precio_gasoil
            c.metric(
                label=f'Predicción {mes}',
                value=f'{l100_p:.2f} L/100km',
                delta=f'{lts_p:,.0f} L | ${costo_p/1e6:.2f}M'
            )

        st.markdown('<div class="sec-title">Evolucion con Proyeccion</div>', unsafe_allow_html=True)
        all_labels = [str(p) for p in hist['MES_PERIODO']] + meses_fut
        all_hist   = hist['L100'].tolist() + [None, None, None]
        all_pred   = [None]*(len(hist)-1) + [float(hist['L100'].iloc[-1])] + [float(v) for v in pred_l100]
        chart_df   = pd.DataFrame({'Mes': all_labels, 'Historico': all_hist, 'Prediccion': all_pred}).set_index('Mes')
        st.line_chart(chart_df, use_container_width=True)

        st.markdown('<div class="sec-title">Costo Proyectado</div>', unsafe_allow_html=True)
        cost_df = pd.DataFrame({
            'Mes': meses_fut,
            'L/100km': [round(v, 2) for v in pred_l100],
            'Litros est.': [round(v, 0) for v in pred_lts],
            'Costo est. M$': [round(v * precio_gasoil / 1e6, 2) for v in pred_lts]
        })
        st.dataframe(cost_df, use_container_width=True, hide_index=True)
    else:
        st.info('Se necesitan al menos 4 meses de datos para el modelo predictivo.')

    st.caption(f'Datos: Google Sheets Expreso Diemar | Precio: {precio_fuente} | Actualización cada 10 min')
