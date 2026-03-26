import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import io

# Configuración de página
st.set_page_config(page_title="Expreso Diemar - Predicción", layout="wide")
st.title("🚛 Dashboard de Consumo de Flota")

# ==============================
# 1. FUENTES DE DATOS (GOOGLE SHEETS)
# ==============================
base_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR35NkYPtJrOrdYHLGUH7GIW93s5cPAqQ0zEk5fP1c3gvErwbUW7HJ2OeWBYaBVsYKVmCf0yhLvs6eG/pub?output=csv"

# GIDs actualizados según tu configuración
gid_telemetria = "1044040871"
gid_unidades = "882343299" 

url1 = f"{base_url}&gid={gid_telemetria}"
url2 = f"{base_url}&gid={gid_unidades}"

# ==============================
# 2. CARGAR Y LIMPIAR DATOS
# ==============================
@st.cache_data(ttl=600)
def cargar_datos():
    try:
        df1 = pd.read_csv(url1)
        df2 = pd.read_csv(url2)

        # Limpieza de nombres (sacar espacios y tildes)
        for d in [df1, df2]:
            d.columns = d.columns.str.strip().str.replace('í', 'i').str.replace('á', 'a')

        # MAPEO DE COLUMNAS: De nombre largo a nombre corto para el código
        # Esto soluciona el KeyError: 'KM'
        rename_dict = {
            'DISTANCIA RECORRIDA TELEMETRIA': 'KM',
            'Ralenti (Lts)': 'Ralenti_Lts'
        }
        df1 = df1.rename(columns=rename_dict)
        df2 = df2.rename(columns=rename_dict)

        # Unión por Fecha y Dominio
        df_merged = pd.merge(df1, df2, on=["FECHA", "DOMINIO"])
        
        # Limpieza de valores nulos o en cero para que no falle el modelo
        df_merged = df_merged.dropna(subset=["KM", "LITROS CONSUMIDOS", "Ralenti_Lts"])
        df_merged = df_merged[(df_merged["KM"] > 0) & (df_merged["LITROS CONSUMIDOS"] > 0)]
        
        return df_merged
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

df = cargar_datos()

if df.empty:
    st.stop()

# ==============================
# 3. PROCESAMIENTO DE FECHAS
# ==============================
df["FECHA"] = pd.to_datetime(df["FECHA"], errors='coerce')
df["MES"] = df["FECHA"].dt.month
df["AÑO"] = df["FECHA"].dt.year

# ==============================
# 4. PRECIOS GASOIL (SCRAPING)
# ==============================
def obtener_tabla_precios():
    url_combustible = "https://surtidores.com.ar/precios/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url_combustible, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        data = []
        tablas = soup.find_all("table")
        for tabla in tablas:
            if "2025" in tabla.text or "2026" in tabla.text:
                año = 2026 if "2026" in tabla.text else 2025
                for fila in tabla.find_all("tr"):
                    if "Gasoil" in fila.text:
                        columnas = fila.find_all("td")
                        valores = [int(col.text.strip()) for col in columnas if col.text.strip().isdigit()]
                        for i, precio in enumerate(valores):
                            data.append({"AÑO": año, "MES": i+1, "PRECIO": precio})
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

df_precios = obtener_tabla_precios()

# Cruzar precios o usar default si falla el scraping
if not df_precios.empty:
    df = pd.merge(df, df_precios, on=["AÑO", "MES"], how="left")
df["PRECIO"] = df["PRECIO"].fillna(1200) # Precio base de seguridad

# ==============================
# 5. CÁLCULOS DE COSTOS
# ==============================
df["COSTO"] = df["LITROS CONSUMIDOS"] * df["PRECIO"]
df["COSTO_RALENTI"] = df["Ralenti_Lts"] * df["PRECIO"]

litros_totales = df["LITROS CONSUMIDOS"].sum()
costo_total = df["COSTO"].sum()
costo_ralenti = df["COSTO_RALENTI"].sum()
ahorro_potencial = costo_ralenti * 0.2 # 20% de ahorro sugerido

# ==============================
# 6. MODELO DE PREDICCIÓN (IA)
# ==============================
# Usamos las columnas ya renombradas
X = df[["KM", "Consumo c/ 100km TELEMETRIA", "Ralenti_Lts"]]
y = df["LITROS CONSUMIDOS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# 7. DASHBOARD VISUAL
# ==============================
c1, c2, c3 = st.columns(3)
c1.metric("⛽ Litros Totales", f"{litros_totales:,.0f} L")
c2.metric("💰 Costo Total", f"$ {costo_total:,.0f}")
c3.metric("🛑 Costo Ralentí", f"$ {costo_ralenti:,.0f}")

st.divider()

c4, c5 = st.columns(2)
c4.metric("💸 Ahorro Potencial (20% Ralentí)", f"$ {ahorro_potencial:,.0f}", delta_color="normal")
c5.metric("⛽ Precio Promedio Aplicado", f"$ {df['PRECIO'].mean():,.0f}")

st.divider()

# Gráficos
st.subheader("📊 Consumo por Unidad (Dominios)")
st.bar_chart(df.groupby("DOMINIO")["LITROS CONSUMIDOS"].sum())

st.subheader("📈 Evolución de Costos")
df_evol = df.groupby(["AÑO", "MES"])["COSTO"].sum().reset_index()
df_evol["FECHA_STR"] = df_evol["AÑO"].astype(str) + "-" + df_evol["MES"].astype(str)
st.line_chart(df_evol.set_index("FECHA_STR")["COSTO"])

# Detalle
st.subheader("📋 Datos Procesados")
st.dataframe(df[["FECHA", "DOMINIO", "KM", "LITROS CONSUMIDOS", "Ralenti_Lts", "COSTO"]], use_container_width=True)
