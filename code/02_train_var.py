# Ajuste y estabilidad del modelo VAR
def ajustar_VAR(series_var, p):
    """
    Ajusta un modelo VAR con rezagos `p` y verifica estabilidad.

    Parámetros:
        df (pd.DataFrame): Serie temporal multivariada (preprocesada).
        p (int): Número de rezagos.

    Retorna:
        modelo_fitted (VARResults): Modelo ajustado.
        es_estable (bool): True si el modelo es estable, False en caso contrario.
    """
    modelo = VAR(series_var)
    modelo_fitted = modelo.fit(p)
    es_estable = modelo_fitted.is_stable()

    print(f"🔧 Modelo VAR(p={p}) ajustado.")
    print(f"✅ Estabilidad del modelo: {'Sí' if es_estable else 'No'}")

    return modelo_fitted, es_estable
