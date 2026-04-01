import pandas as pd
import streamlit as st
import numpy as np
import os # Para verificar si los archivos existen
# ... (mantener el resto de tus imports: sklearn, plotly, etc.)

@st.cache_data(ttl=600)
def cargar_datos():
    # Nombres de los archivos que subiste a GitHub
    FILE_TEL = "telemetria.xlsx"
    FILE_UNID = "unidades.xlsx"

    try:
        # Verificamos si los archivos están en la carpeta
        if not os.path.exists(FILE_TEL) or not os.path.exists(FILE_UNID):
            st.error(f"❌ No se encontraron los archivos {FILE_TEL} o {FILE_UNID} en el repositorio.")
            return pd.DataFrame(), pd.DataFrame()

        # Lectura local (mucho más rápida y sin errores 403)
        df1 = pd.read_excel(FILE_TEL, engine="openpyxl")
        df2 = pd.read_excel(FILE_UNID, engine="openpyxl")

        def limpiar(df):
            df.columns = [str(c).strip().upper() for c in df.columns]
            
            # Mapeo inteligente (se mantiene igual para seguridad)
            rename_dict = {}
            for c in df.columns:
                if any(x in c for x in ["DOMINIO", "PATENTE", "UNIDAD"]): rename_dict[c] = "DOMINIO"
                elif any(x in c for x in ["LITROS", "CONSUMID"]): rename_dict[c] = "LITROS"
                elif any(x in c for x in ["KM", "DISTANCIA", "KILOMETR"]): rename_dict[c] = "KM"
                elif "MARCA" in c: rename_dict[c] = "MARCA"
                elif "FECHA" in c: rename_dict[c] = "FECHA"
                elif "L/100" in c: rename_dict[c] = "L100KM"
                elif "RALENT" in c: rename_dict[c] = "RALENTI"
                elif "EMPRESA" in c: rename_dict[c] = "EMPRESA"
            
            df = df.rename(columns=rename_dict)
            
            if "FECHA" in df.columns:
                df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
            
            for col in ["LITROS", "KM", "L100KM", "RALENTI"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            return df

        df1 = limpiar(df1)
        df2 = limpiar(df2)

        # Filtros automáticos
        if "EMPRESA" in df1.columns:
            df1 = df1[df1["EMPRESA"].str.upper().str.contains("LAD|DIEMAR", na=False)]
        
        # Solo 2026
        if "FECHA" in df1.columns:
            df1 = df1[df1["FECHA"].dt.year == 2026]

        return df1, df2

    except Exception as e:
        st.error(f"❌ Error al leer archivos locales: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ... (El resto del código de la UI y los gráficos sigue igual)
