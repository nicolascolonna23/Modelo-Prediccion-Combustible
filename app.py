@st.cache_data(ttl=300)
def cargar_datos():
    archivos = os.listdir('.')
    file_tel = next((f for f in archivos if "ANALISIS" in f.upper() and f.endswith('.xlsx')), None)
    file_unid = next((f for f in archivos if "UNIDADES" in f.upper() and f.endswith('.xlsx')), None)

    if not file_tel or not file_unid:
        st.error(f"❌ No se detectan los archivos. Archivos en repo: {archivos}")
        st.stop()

    try:
        df1 = pd.read_excel(file_tel, engine="openpyxl")
        df2 = pd.read_excel(file_unid, engine="openpyxl")

        def limpiar(df):
            # 1. Normalizar nombres de columnas
            df.columns = [str(c).strip().upper() for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]
            
            # 2. Mapeo de columnas críticas
            cm = {}
            for c in df.columns:
                if any(x in c for x in ["DOMINIO", "PATENTE", "UNIDAD"]): cm[c] = "DOMINIO"
                elif any(x in c for x in ["LITROS", "CONSUMID"]): cm[c] = "LITROS"
                elif any(x in c for x in ["KM", "DISTANCIA", "KILOMETR"]): cm[c] = "KM"
                elif "MARCA" in c: cm[c] = "MARCA"
                elif "FECHA" in c: cm[c] = "FECHA"
                elif "EMPRESA" in c: cm[c] = "EMPRESA"
            df = df.rename(columns=cm)

            # 3. Tratamiento de FECHAS (Aquí es donde ignoramos 2024/2025)
            if "FECHA" in df.columns:
                # Convertimos a fecha, lo que no es fecha se vuelve NaT (Not a Time)
                df["FECHA"] = pd.to_datetime(df["FECHA"], errors='coerce')
                # Eliminamos filas sin fecha válida antes de filtrar
                df = df.dropna(subset=["FECHA"])
                # FILTRO CRÍTICO: Solo el año 2026
                df = df[df["FECHA"].dt.year == 2026]

            # 4. Tratamiento de NÚMEROS
            for col in ["LITROS", "KM"]:
                if col in df.columns:
                    # Convertimos a numérico, ignorando errores de texto
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            
            return df

        df1 = limpiar(df1)
        df2 = limpiar(df2)

        # Validación post-filtrado
        if df1.empty:
            st.warning("⚠️ El archivo se leyó bien, pero NO hay datos del año 2026. (Datos de 2024/2025 ignorados).")
            # Para que no tire error el resto de la app, devolvemos el df vacío pero con columnas
            return df1, df2

        # Lógica de Empresa (LAD/Diemar)
        if "EMPRESA" in df1.columns:
            df1 = df1[df1["EMPRESA"].astype(str).str.upper().str.contains("LAD|DIEMAR", na=False)]

        # Recalcular consumo L/100
        if "KM" in df1.columns and "LITROS" in df1.columns:
            df1["L100KM"] = (df1["LITROS"] / df1["KM"].replace(0, np.nan) * 100).round(2).fillna(0)

        return df1, df2

    except Exception as e:
        st.error(f"❌ Error procesando los datos: {e}")
        return pd.DataFrame(), pd.DataFrame()
