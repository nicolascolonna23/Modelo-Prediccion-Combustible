import pandas as pd
import streamlit as st
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────
st.set_page_config(page_title="Expreso Diemar - Consumo", page_icon="🚛", layout="wide")

st.markdown("""
<style>
  .brand-card {
    background: white; border-radius: 12px; padding: 20px 24px;
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-top: 4px solid #2563eb; margin-bottom: 8px;
  }
  .brand-name  { font-size: 1.3rem; font-weight: 700; margin-bottom: 4px; }
  .brand-label { font-size: 0.78rem; color: #64748b; margin-bottom: 8px; }
  .brand-value { font-size: 1.9rem; font-weight: 800; color: #1e293b; }
  .brand-unit  { font-size: 0.82rem; color: #64748b; }
  .tbl-card    { background: white; border-radius: 12px; padding: 20px;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
  .tbl-title   { font-size: 1rem; font-weight: 700; color: #1e293b; margin-bottom: 10px; }
  .tbl-row     { display: flex; align-items: center; padding: 7px 0;
                 border-bottom: 1px solid #f1f5f9; }
  .tbl-label   { min-width: 100px; font-size: 0.85rem; color: #334155; }
  .tbl-bar-bg  { flex: 1; background: #f1f5f9; border-radius: 4px; height: 7px;
                 margin: 0 10px; overflow: hidden; }
  .tbl-bar     { height: 7px; border-radius: 4px; }
  .tbl-val     { min-width: 55px; text-align: right; font-size: 0.85rem;
                 font-weight: 600; color: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ─── FUENTE DE DATOS ──────────────────────────────────────
BASE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR35NkYPtJrOrdYHLGUH7GIW93s5cPAqQ0zEk5fP1c3gvErwbUW7HJ2OeWBYaBVsYKVmCf0yhLvs6eG/pub?output=csv"
GID_TEL  = "1044040871"
GID_UNID = "882343299"
URL_TEL  = f"{BASE_URL}&gid={GID_TEL}"
URL_UNID = f"{BASE_URL}&gid={GID_UNID}"

@st.cache_data(ttl=300)
def cargar_datos():
    try:
        df1 = pd.read_csv(URL_TEL)
        df2 = pd.read_csv(URL_UNID)

        def limpiar(df):
            df.columns = df.columns.str.strip().str.upper()
            cm = {}
            for c in df.columns:
                if "DOMINIO" in c or "PATENTE" in c:                   cm[c] = "DOMINIO"
                elif "LITROS" in c or "CONSUMID" in c:                 cm[c] = "LITROS"
                elif "DISTANCIA" in c or "KM" in c or "KILOMETR" in c: cm[c] = "KM"
                elif "CO2" in c or "EMISION" in c:                     cm[c] = "CO2"
                elif "MARCA" in c:                                      cm[c] = "MARCA"
                elif "TAG" in c or "RUTA" in c or "SITE" in c:         cm[c] = "TAG"
                elif "FECHA" in c or "DATE" in c:                      cm[c] = "FECHA"
                elif "L/100" in c or "CONSUMO C" in c:                 cm[c] = "L100KM"
            df = df.rename(columns=cm)
            if "DOMINIO" in df.columns:
                df["DOMINIO"] = df["DOMINIO"].astype(str).str.strip().str.upper()
            for col in ["LITROS", "KM", "CO2", "L100KM"]:
                if col in df.columns:
                    df[col] = (df[col].astype(str)
                               .str.replace(".", "", regex=False)
                               .str.replace(",", ".", regex=False))
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if "FECHA" in df.columns:
                df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce", dayfirst=True)
            return df

        df1 = limpiar(df1)
        df2 = limpiar(df2)

        if "L100KM" not in df1.columns and "LITROS" in df1.columns and "KM" in df1.columns:
            df1["L100KM"] = (df1["LITROS"] / df1["KM"].replace(0, np.nan) * 100).round(2)

        extra = [c for c in ["DOMINIO", "MARCA", "CO2", "TAG"] if c in df2.columns]
        if extra:
            df2r = df2[extra].groupby("DOMINIO").first().reset_index()
            df1  = pd.merge(df1, df2r, on="DOMINIO", how="left", suffixes=("", "_y"))
            for col in ["MARCA", "CO2", "TAG"]:
                if f"{col}_y" in df1.columns:
                    df1[col] = df1[col].combine_first(df1[f"{col}_y"])
                    df1.drop(columns=[f"{col}_y"], inplace=True)
        return df1
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

df_raw = cargar_datos()
if df_raw.empty:
    st.warning("No se pudieron cargar los datos.")
    st.stop()

# ─── HEADER ───────────────────────────────────────────────
st.markdown("""
<div style="background:#0f172a;padding:14px 24px;border-radius:10px;margin-bottom:20px;
            display:flex;align-items:center;gap:12px;">
  <span style="font-size:1.6rem">🚛</span>
  <span style="color:white;font-size:1.25rem;font-weight:700">
    Expreso Diemar — Dashboard de Consumo de Combustible
  </span>
</div>
""", unsafe_allow_html=True)

# ─── FILTRO DE FECHA ──────────────────────────────────────
df = df_raw.copy()
if "FECHA" in df.columns and df["FECHA"].notna().any():
    f1, f2, _ = st.columns([1, 1, 2])
    fmin  = df["FECHA"].min().date()
    fmax  = df["FECHA"].max().date()
    desde = f1.date_input("Desde", value=fmin, min_value=fmin, max_value=fmax)
    hasta = f2.date_input("Hasta", value=fmax, min_value=fmin, max_value=fmax)
    df    = df[(df["FECHA"].dt.date >= desde) & (df["FECHA"].dt.date <= hasta)]

if df.empty:
    st.warning("Sin datos para el período seleccionado.")
    st.stop()

st.divider()

# ─── HELPERS ──────────────────────────────────────────────
def color_bar(val, vmin, vmax):
    if vmax == vmin: return "#f59e0b"
    p = (val - vmin) / (vmax - vmin)
    return "#22c55e" if p <= 0.33 else ("#f59e0b" if p <= 0.66 else "#ef4444")

def tabla_html(rows_df, col_label, col_val, titulo):
    if rows_df.empty:
        return f'<div class="tbl-card"><div class="tbl-title">🔥 {titulo}</div><p style="color:#94a3b8">Sin datos</p></div>'
    vmin, vmax = rows_df[col_val].min(), rows_df[col_val].max()
    html = f'<div class="tbl-card"><div class="tbl-title">🔥 {titulo}</div>'
    html += '<div style="display:flex;justify-content:space-between;font-size:.73rem;color:#94a3b8;margin-bottom:4px"><span>Unidad</span><span>L/100km</span></div>'
    for _, r in rows_df.iterrows():
        v   = r[col_val]
        pct = int((v - vmin) / (vmax - vmin) * 100) if vmax != vmin else 50
        col = color_bar(v, vmin, vmax)
        html += f"""<div class="tbl-row">
          <div class="tbl-label">{r[col_label]}</div>
          <div class="tbl-bar-bg"><div class="tbl-bar" style="width:{pct}%;background:{col}"></div></div>
          <div class="tbl-val">{v:.2f}</div></div>"""
    html += "</div>"
    return html

# ─── TARJETAS POR MARCA ───────────────────────────────────
st.markdown("#### Consumo promedio por marca")
MARCAS_ORD = ["SCANIA", "IVECO", "MERCEDES", "VOLKSWAGEN", "MAN"]
if "MARCA" in df.columns and "L100KM" in df.columns:
    md = df[df["L100KM"] > 0].groupby("MARCA")["L100KM"].mean().round(2)
    mostrar = [m for m in MARCAS_ORD if m in md.index]
    mostrar += [m for m in md.index if m not in MARCAS_ORD]
    mostrar  = mostrar[:4]
    cols = st.columns(len(mostrar) if mostrar else 1)
    for i, m in enumerate(mostrar):
        cols[i].markdown(f"""<div class="brand-card">
          <div class="brand-name">{m}</div>
          <div class="brand-label">Consumo promedio flota</div>
          <div class="brand-value">{md[m]:.2f}<span class="brand-unit"> ltr/100</span></div>
        </div>""", unsafe_allow_html=True)
else:
    st.info("Sin columna MARCA — necesaria para las tarjetas por marca.")

st.markdown("<br>", unsafe_allow_html=True)

# ─── TRES TABLAS ──────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    if "L100KM" in df.columns:
        t1 = (df[df["L100KM"] > 0].groupby("DOMINIO")["L100KM"]
              .mean().round(2).reset_index().sort_values("L100KM").head(12))
        st.markdown(tabla_html(t1, "DOMINIO", "L100KM", "Consumo por camión"), unsafe_allow_html=True)

with c2:
    if "TAG" in df.columns and "L100KM" in df.columns:
        t2 = (df[df["L100KM"] > 0].groupby("TAG")["L100KM"]
              .mean().round(2).reset_index().sort_values("L100KM"))
        st.markdown(tabla_html(t2, "TAG", "L100KM", "Consumo por ruta / TAG"), unsafe_allow_html=True)
    elif "FECHA" in df.columns and "LITROS" in df.columns and "KM" in df.columns:
        df["MES"] = df["FECHA"].dt.strftime("%b %Y")
        t2 = (df.groupby("MES").apply(lambda g: pd.Series({
              "L100KM": round(g["LITROS"].sum() / g["KM"].sum() * 100, 2) if g["KM"].sum() > 0 else 0
              })).reset_index())
        st.markdown(tabla_html(t2, "MES", "L100KM", "Consumo por mes"), unsafe_allow_html=True)
    else:
        st.info("Sin datos de ruta / TAG")

with c3:
    if "KM" in df.columns and "L100KM" in df.columns:
        dfk    = df[df["KM"] > 0].copy()
        bins   = [0, 5000, 10000, 15000, 20000, 999999]
        labels = ["0-5k km", "5k-10k km", "10k-15k km", "15k-20k km", "+20k km"]
        dfk["RANGO"] = pd.cut(dfk["KM"], bins=bins, labels=labels, right=True)
        t3 = (dfk[dfk["L100KM"] > 0].groupby("RANGO", observed=True)["L100KM"]
              .mean().round(2).reset_index().dropna())
        t3.columns = ["RANGO", "L100KM"]
        st.markdown(tabla_html(t3, "RANGO", "L100KM", "Consumo por rango de km"), unsafe_allow_html=True)

st.divider()

# ─── KPIs GLOBALES ────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
lts  = df["LITROS"].sum() if "LITROS" in df.columns else 0
kms  = df["KM"].sum()     if "KM"     in df.columns else 0
co2  = df["CO2"].sum()    if "CO2"    in df.columns else 0
l100 = round(lts / kms * 100, 2) if kms > 0 else 0
k1.metric("⛽ Litros totales",   f"{lts:,.0f} L")
k2.metric("🛣️ Km totales",      f"{kms:,.0f} km")
k3.metric("📊 L/100km promedio", f"{l100:.2f}")
k4.metric("🌿 CO₂ total",       f"{co2:,.0f} kg")

with st.expander("📋 Ver datos completos"):
    cols_s = [c for c in ["DOMINIO","MARCA","TAG","FECHA","KM","LITROS","L100KM","CO2"] if c in df.columns]
    st.dataframe(df[cols_s], use_container_width=True, height=380)

st.caption("Fuente: Google Sheets — Expreso Diemar | Refresco automático cada 5 min")
