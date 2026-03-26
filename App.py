import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("🚛 Dashboard de Consumo de Flota")

# ==============================
# 1. GOOGLE SHEETS (TU LINK)
# ==============================

base_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR35NkYPtJrOrdYHLGUH7GIW93s5cPAqQ0zEk5fP1c3gvErwbUW7HJ2OeWBYaBVsYKVmCf0yhLvs6eG/pub?output=csv"

# ⚠️ TENÉS QUE AJUSTAR ESTOS GID
gid_telemetria = "1044040871"
gid_unidades = "882343299"  # cambiar por el real

url1 = base_url + f"&gid={gid_telemetria}"
url2 = base_url + f"&gid={gid_unidades}"

# ==============================
# 2. CARGAR DATOS
# ==============================

df1 = pd.read_csv(url1)
df2 = pd.read_csv(url2)

df = pd.merge(df1, df2, on=["FECHA", "DOMINIO"])

df = df.dropna()
df = df[df["KM"] > 0]
df = df[df["LITROS CONSUMIDOS"] > 0]

# ==============================
# 3. FECHAS
# ==============================

df["FECHA"] = pd.to_datetime(df["FECHA"])
df["MES"] = df["FECHA"].dt.month
df["AÑO"] = df["FECHA"].dt.year

# ==============================
# 4. PRECIOS GASOIL
# ==============================

def obtener_tabla_precios():
    url_combustible = "https://surtidores.com.ar/precios/"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url_combustible, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    data = []

    meses = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
             "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]

    tablas = soup.find_all("table")

    for tabla in tablas:
        texto = tabla.text

        if "2025" in texto or "2026" in texto:
            filas = tabla.find_all("tr")

            for fila in filas:
                if "Gasoil" in fila.text:
                    columnas = fila.find_all("td")

                    valores = []
                    for col in columnas:
                        val = col.text.strip()
                        if val.isdigit():
                            valores.append(int(val))

                    if valores:
                        año = 2026 if "2026" in texto else 2025

                        for i, precio in enumerate(valores):
                            if i < len(meses):
                                data.append({
                                    "AÑO": año,
                                    "MES": i+1,
                                    "PRECIO": precio
                                })

    return pd.DataFrame(data)

df_precios = obtener_tabla_precios()

df = pd.merge(df, df_precios, on=["AÑO", "MES"], how="left")
df["PRECIO"].fillna(1000, inplace=True)

# ==============================
# 5. METRICAS
# ==============================

df["consumo_km"] = df["LITROS CONSUMIDOS"] / df["KM"]
df["ralenti_pct"] = df["Ralentí (Lts)"] / df["LITROS CONSUMIDOS"]

df["COSTO"] = df["LITROS CONSUMIDOS"] * df["PRECIO"]
df["COSTO_RALENTI"] = df["Ralentí (Lts)"] * df["PRECIO"]

# ==============================
# 6. TOTALES
# ==============================

litros_totales = df["LITROS CONSUMIDOS"].sum()
costo_total = df["COSTO"].sum()
costo_ralenti = df["COSTO_RALENTI"].sum()
ahorro_potencial = costo_ralenti * 0.2

# ==============================
# 7. MODELO
# ==============================

X = df[["KM", "Consumo c/ 100km TELEMETRIA", "Ralentí (Lts)"]]
y = df["LITROS CONSUMIDOS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# 8. DASHBOARD
# ==============================

col1, col2, col3 = st.columns(3)

col1.metric("⛽ Litros Totales", round(litros_totales, 0))
col2.metric("💰 Costo Total", f"${round(costo_total, 0)}")
col3.metric("🛑 Costo Ralentí", f"${round(costo_ralenti, 0)}")

st.markdown("---")

col4, col5 = st.columns(2)

col4.metric("💸 Ahorro Potencial (20%)", f"${round(ahorro_potencial, 0)}")

precio_prom = df["PRECIO"].mean()
col5.metric("⛽ Precio Promedio", f"${round(precio_prom, 0)}")

st.markdown("---")

# ==============================
# 9. GRAFICOS
# ==============================

st.subheader("Consumo por Camión")
consumo_camion = df.groupby("DOMINIO")["LITROS CONSUMIDOS"].sum()
st.bar_chart(consumo_camion)

st.subheader("Costo por Mes")
costo_mes = df.groupby(["AÑO","MES"])["COSTO"].sum()
st.line_chart(costo_mes)

# ==============================
# 10. TABLA
# ==============================

st.subheader("Datos Detallados")
st.dataframe(df)
