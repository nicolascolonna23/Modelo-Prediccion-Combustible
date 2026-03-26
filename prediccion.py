import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================
# 1. LINKS DE ONEDRIVE
# ==============================

url1 = "https://expresodiemar-my.sharepoint.com/:x:/g/personal/nicolascolonna_expresodiemar_onmicrosoft_com/IQCCJG7r9T2JTb0eAdpQU1ggAcTn9ZELfjq58Xk9-eqj58o?e=qOAaQe=download?"
url2 = "https://expresodiemar-my.sharepoint.com/:x:/g/personal/nicolascolonna_expresodiemar_onmicrosoft_com/IQAWlrsay0HVT622_ANLB-bWAfMlRi4IHHFMH6DJBzVW3BU?e=M1unfp=download?"

# ==============================
# 2. CARGAR DESDE WEB
# ==============================

df1 = pd.read_excel(url1, sheet_name="TELEMETRIA")
df2 = pd.read_excel(url2)

# ==============================
# 3. UNIR DATOS
# ==============================

df = pd.merge(df1, df2, on=["FECHA", "DOMINIO"])

# ==============================
# 4. LIMPIEZA
# ==============================

df = df.dropna()

df = df[df["KM"] > 0]
df = df[df["LITROS CONSUMIDOS"] > 0]

# ==============================
# 5. METRICAS
# ==============================

df["consumo_km"] = df["LITROS CONSUMIDOS"] / df["KM"]
df["ralenti_pct"] = df["Ralentí (Lts)"] / df["LITROS CONSUMIDOS"]

# ==============================
# 6. MODELO
# ==============================

X = df[[
    "KM",
    "Consumo c/ 100km TELEMETRIA",
    "Ralentí (Lts)"
]]

y = df["LITROS CONSUMIDOS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# 7. RESULTADOS
# ==============================

y_pred = model.predict(X_test)

print("Error medio:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# ==============================
# 8. IMPACTO
# ==============================

coef = pd.DataFrame({
    "Variable": X.columns,
    "Impacto": model.coef_
})

print("\nImpacto de variables:\n")
print(coef)

# ==============================
# 9. GRAFICO
# ==============================

plt.scatter(y_test, y_pred)
plt.xlabel("Real")
plt.ylabel("Predicho")
plt.title("Modelo consumo")
plt.show()
