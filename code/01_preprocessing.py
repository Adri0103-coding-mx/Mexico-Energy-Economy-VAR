# === 3. Heuristica para definir las series que deben llevar log-transform ===
def candidatos_para_log(df, umbral_ratio=5, umbral_media=1000):
    """
    Detecta automáticamente columnas candidatas para log-transform.

    Requisitos:
    - No deben tener ceros o negativos.
    - Deben tener una razón entre máximo y mínimo significativa.
    - Deben tener un valor medio razonablemente alto.

    Retorna:
    --------
    Lista de nombres de columnas candidatas.
    """
    candidatos = []
    for col in df.columns:
        serie = pd.to_numeric(df[col], errors='coerce').dropna()
        if (serie <= 0).any():
            continue  # ❌ No apta para log
        if len(serie) < 3:
            continue  # No suficiente información
        rango = serie.max() / serie.min()
        if rango > umbral_ratio and serie.mean() > umbral_media:
            candidatos.append(col)
    return candidatos

# === 4. Función principal de interpolación segura y escalonada para cada serie ===
def interpolar_escalonado(serie_original, frecuencia_final='D', metodo='linear', escalonado=True, log_transform=False, revert_log=True):
    """
    Interpola una serie temporal desde frecuencia anual hacia una frecuencia más alta,
    con opción de hacerlo en pasos (escalonado) o directamente.

    Parámetros:
    -----------
    serie_original : pd.Series
        Serie temporal con índice datetime y frecuencia anual.
    frecuencia_final : str
        Frecuencia de destino: 'Q' (trimestral), 'M' (mensual), 'D' (diaria).
    metodo : str
        Método de interpolación: 'linear', 'spline', 'polynomial', 'ffill'.
    escalonado : bool
        Si True, realiza interpolaciones intermedias (ej. anual→trimestral→mensual→diaria).
    log_transform : bool
        Si True, aplica log-transform antes de interpolar.
    revert_log : bool
        Si True, aplica np.exp() al final. Si False, conserva la escala logarítmica.

    Retorna:
    --------
    pd.Series con índice en frecuencia_final e interpolada.
    """
    if not isinstance(serie_original, pd.Series):
        raise ValueError("La entrada debe ser una pd.Series")

    serie = serie_original.copy()
    serie.index = pd.to_datetime(serie.index)
    serie = serie.sort_index()

    pasos = ['A', 'Q', 'M', 'D']
    if frecuencia_final not in pasos:
        raise ValueError(f"Frecuencia destino '{frecuencia_final}' no válida. Usa una de: {pasos}")
    target_idx = pasos.index(frecuencia_final)

    freq_inferida = pd.infer_freq(serie.index)
    if not freq_inferida or freq_inferida[0] != 'A':
        print("⚠️ Advertencia: frecuencia inicial no parece ser anual.")

    if metodo == 'ffill':
        nueva_frecuencia = pd.date_range(start=serie.index.min(), end=serie.index.max(), freq=frecuencia_final)
        serie = serie.reindex(nueva_frecuencia).ffill()
        serie.index.name = 'fecha'
        return serie

    if log_transform:
        if (serie <= 0).any():
            raise ValueError("La serie contiene ceros o negativos: no se puede aplicar log-transform.")
        serie = np.log(serie)

    for paso in pasos[1:target_idx+1] if escalonado else [frecuencia_final]:
        idx_nuevo = pd.date_range(start=serie.index.min(), end=serie.index.max(), freq=paso)
        serie = serie.reindex(idx_nuevo)

        if metodo in ['spline', 'polynomial'] and serie.notna().sum() < 4:
            raise ValueError(f"Interpolación con '{metodo}' requiere al menos 4 puntos no nulos")

        if metodo == 'polynomial':
            serie = serie.interpolate(method=metodo, order=2)
        elif metodo == 'spline':
            serie = serie.interpolate(method=metodo, order=3)
        else:
            serie = serie.interpolate(method=metodo)

    if log_transform and revert_log:
        serie = np.exp(serie)

    serie.index.name = 'fecha'
    return serie
