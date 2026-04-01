from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import io

st.set_page_config(layout="wide")
# Configuración de página
st.set_page_config(page_title="Expreso Diemar - Predicción", layout="wide")
st.title("🚛 Dashboard de Consumo de Flota")

# ==============================
# 1. GOOGLE SHEETS (TU LINK)
# 1. FUENTES DE DATOS (GOOGLE SHEETS)
# ==============================

base_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR35NkYPtJrOrdYHLGUH7GIW93s5cPAqQ0zEk5fP1c3gvErwbUW7HJ2OeWBYaBVsYKVmCf0yhLvs6eG/pub?output=csv"

# ⚠️ TENÉS QUE AJUSTAR ESTOS GID
# GIDs actualizados según tu configuración
gid_telemetria = "1044040871"
gid_unidades = "882343299"  # cambiar por el real
gid_unidades = "882343299" 

url1 = base_url + f"&gid={gid_telemetria}"
url2 = base_url + f"&gid={gid_unidades}"
url1 = f"{base_url}&gid={gid_telemetria}"
url2 = f"{base_url}&gid={gid_unidades}"

# ==============================
# 2. CARGAR DATOS
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

df1 = pd.read_csv(url1)
df2 = pd.read_csv(url2)
        # Unión por Fecha y Dominio
        df_merged = pd.merge(df1, df2, on=["FECHA", "DOMINIO"])
        
        # Limpieza de valores nulos o en cero para que no falle el modelo
        df_merged = df_merged.dropna(subset=["KM", "LITROS CONSUMIDOS", "Ralenti_Lts"])
        df_merged = df_merged[(df_merged["KM"] > 0) & (df_merged["LITROS CONSUMIDOS"] > 0)]
        
        return df_merged
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

df = pd.merge(df1, df2, on=["FECHA", "DOMINIO"])
df = cargar_datos()

df = df.dropna()
df = df[df["KM"] > 0]
df = df[df["LITROS CONSUMIDOS"] > 0]
if df.empty:
    st.stop()

# ==============================
# 3. FECHAS
# 3. PROCESAMIENTO DE FECHAS
# ==============================

df["FECHA"] = pd.to_datetime(df["FECHA"])
df["FECHA"] = pd.to_datetime(df["FECHA"], errors='coerce')
df["MES"] = df["FECHA"].dt.month
df["AÑO"] = df["FECHA"].dt.year

# ==============================
# 4. PRECIOS GASOIL
# 4. PRECIOS GASOIL (SCRAPING)
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
                            if i < len(meses):
                                data.append({
                                    "AÑO": año,
                                    "MES": i+1,
                                    "PRECIO": precio
                                })

    return pd.DataFrame(data)
                            data.append({"AÑO": año, "MES": i+1, "PRECIO": precio})
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

df_precios = obtener_tabla_precios()

df = pd.merge(df, df_precios, on=["AÑO", "MES"], how="left")
df["PRECIO"].fillna(1000, inplace=True)
# Cruzar precios o usar default si falla el scraping
if not df_precios.empty:
    df = pd.merge(df, df_precios, on=["AÑO", "MES"], how="left")
df["PRECIO"] = df["PRECIO"].fillna(1200) # Precio base de seguridad

# ==============================
# 5. METRICAS
# 5. CÁLCULOS DE COSTOS
# ==============================

df["consumo_km"] = df["LITROS CONSUMIDOS"] / df["KM"]
df["ralenti_pct"] = df["Ralentí (Lts)"] / df["LITROS CONSUMIDOS"]

df["COSTO"] = df["LITROS CONSUMIDOS"] * df["PRECIO"]
df["COSTO_RALENTI"] = df["Ralentí (Lts)"] * df["PRECIO"]

# ==============================
# 6. TOTALES
# ==============================
df["COSTO_RALENTI"] = df["Ralenti_Lts"] * df["PRECIO"]

litros_totales = df["LITROS CONSUMIDOS"].sum()
costo_total = df["COSTO"].sum()
costo_ralenti = df["COSTO_RALENTI"].sum()
ahorro_potencial = costo_ralenti * 0.2
ahorro_potencial = costo_ralenti * 0.2 # 20% de ahorro sugerido

# ==============================
# 7. MODELO
# 6. MODELO DE PREDICCIÓN (IA)
# ==============================

X = df[["KM", "Consumo c/ 100km TELEMETRIA", "Ralentí (Lts)"]]
# Usamos las columnas ya renombradas
X = df[["KM", "Consumo c/ 100km TELEMETRIA", "Ralenti_Lts"]]
y = df["LITROS CONSUMIDOS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# 8. DASHBOARD
# 7. DASHBOARD VISUAL
# ==============================
c1, c2, c3 = st.columns(3)
c1.metric("⛽ Litros Totales", f"{litros_totales:,.0f} L")
c2.metric("💰 Costo Total", f"$ {costo_total:,.0f}")
c3.metric("🛑 Costo Ralentí", f"$ {costo_ralenti:,.0f}")

col1, col2, col3 = st.columns(3)

col1.metric("⛽ Litros Totales", round(litros_totales, 0))
col2.metric("💰 Costo Total", f"${round(costo_total, 0)}")
col3.metric("🛑 Costo Ralentí", f"${round(costo_ralenti, 0)}")

st.markdown("---")
st.divider()

col4, col5 = st.columns(2)
c4, c5 = st.columns(2)
c4.metric("💸 Ahorro Potencial (20% Ralentí)", f"$ {ahorro_potencial:,.0f}", delta_color="normal")
c5.metric("⛽ Precio Promedio Aplicado", f"$ {df['PRECIO'].mean():,.0f}")

col4.metric("💸 Ahorro Potencial (20%)", f"${round(ahorro_potencial, 0)}")
st.divider()

precio_prom = df["PRECIO"].mean()
col5.metric("⛽ Precio Promedio", f"${round(precio_prom, 0)}")
# Gráficos
st.subheader("📊 Consumo por Unidad (Dominios)")
st.bar_chart(df.groupby("DOMINIO")["LITROS CONSUMIDOS"].sum())

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
st.subheader("📈 Evolución de Costos")
df_evol = df.groupby(["AÑO", "MES"])["COSTO"].sum().reset_index()
df_evol["FECHA_STR"] = df_evol["AÑO"].astype(str) + "-" + df_evol["MES"].astype(str)
st.line_chart(df_evol.set_index("FECHA_STR")["COSTO"])

st.subheader("Datos Detallados")
st.dataframe(df)
# Detalle
st.subheader("📋 Datos Procesados")
st.dataframe(df[["FECHA", "DOMINIO", "KM", "LITROS CONSUMIDOS", "Ralenti_Lts", "COSTO"]], use_container_width=True)
