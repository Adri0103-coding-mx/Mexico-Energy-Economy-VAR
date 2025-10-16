def analizar_multicolinealidad_para_VAR(series, umbral_corr=0.95, verbose=True):
    """
    Evalúa la presencia de multicolinealidad en un conjunto de series temporales multivariadas
    mediante el análisis de correlaciones lineales entre todas las variables.

    Para cada par de variables, calcula el coeficiente de correlación de Pearson absoluto y
    reporta aquellos que superan un umbral especificado, indicando posible redundancia informativa
    que podría afectar la estimación de modelos VAR.

    Parámetros:
        series (pd.DataFrame): Conjunto de series temporales multivariadas sin transformar.
        umbral_corr (float): Umbral de correlación para considerar que existe multicolinealidad (por defecto 0.95).
        verbose (bool): Si True, imprime los pares de variables con alta correlación.

    Retorna:
        correlaciones_altas (list of tuple): Lista de tuplas (var1, var2, coef) con los pares de variables
                                             cuya correlación absoluta supera el umbral especificado.
          """
    correlaciones = series.corr().abs()
    correlaciones_altas = []

    for i in range(len(correlaciones.columns)):
        for j in range(i):
            coef = correlaciones.iloc[i, j]
            if coef > umbral_corr:
                var1 = correlaciones.columns[i]
                var2 = correlaciones.columns[j]
                correlaciones_altas.append((var1, var2, coef))

    # Ordenar de mayor a menor
    correlaciones_altas.sort(key=lambda x: x[2], reverse=True)

    # Imprimir después de ordenar
    if verbose:
        for var1, var2, coef in correlaciones_altas:
            print(f"⚠️ Alta correlación: {var1} y {var2} -> {coef:.3f}")
    else:
        print("No hay alta correlación.")

    return correlaciones_altas

# Función de procesamiento (diferenciación - estandarización) de las series para el modelo VAR
def preprocesar_series_para_VAR(series, alpha=0.05, umbral_corr=0.90, verbose=True):
    """
    Verifica estacionariedad, aplica diferenciación si es necesario,
    detecta multicolinealidad y estandariza todas las series.

    Parámetros:
        series (pd.DataFrame): Serie multivariada con índice temporal.
        alpha (float): Nivel de significancia para ADF.
        umbral_corr (float): Umbral para identificar multicolinealidad extrema.
        verbose (bool): Si True, imprime información diagnóstica.

    Retorna:
        series_scaled (pd.DataFrame): Series transformadas (diferenciadas si es necesario y estandarizadas).
        estacionariedad (dict): Si la serie original era estacionaria.
        scaler (StandardScaler): Objeto para revertir la estandarización.
        correlaciones_altas (list of tuple): Pares de variables con alta correlación.
        variables_diferenciadas (list): Lista de variables que fueron diferenciadas.
    """
    series_var = series.copy()
    estacionariedad = {}

    # 1. Verificar estacionariedad y diferenciar
    for col in series_var.columns:
        resultado = adfuller(series_var[col].dropna(), autolag='AIC')
        p_valor = resultado[1]
        es_estacionaria = p_valor < alpha
        estacionariedad[col] = es_estacionaria
        if verbose:
            print(f"{col}: {'Estacionaria' if es_estacionaria else 'NO estacionaria'} (p-valor = {p_valor:.4f})")
        if not es_estacionaria:
            series_var[col] = series_var[col].diff()

    series_var = series_var.dropna()
    variables_diferenciadas = [col for col, est in estacionariedad.items() if not est]

    # 2. Verificar multicolinealidad
    correlaciones_altas = analizar_multicolinealidad_para_VAR(series_var, umbral_corr=umbral_corr, verbose=verbose)

    return series_var, estacionariedad, correlaciones_altas, variables_diferenciadas

def estandarizar_train_test(train_df, test_df):
    """
    Estandariza las variables del conjunto de entrenamiento y aplica la transformación al conjunto de prueba.

    Parámetros:
        train_df (pd.DataFrame): Conjunto de entrenamiento (series diferenciadas o originales).
        test_df (pd.DataFrame): Conjunto de prueba (mismo orden y columnas que train_df).

    Retorna:
        train_scaled (pd.DataFrame): Entrenamiento estandarizado.
        test_scaled (pd.DataFrame): Prueba estandarizada.
        scaler (StandardScaler): Objeto scaler ajustado sobre el entrenamiento.
    """
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df),
        columns=train_df.columns,
        index=train_df.index
    )

    test_scaled = pd.DataFrame(
        scaler.transform(test_df),
        columns=test_df.columns,
        index=test_df.index
    )

    return train_scaled, test_scaled, scaler
