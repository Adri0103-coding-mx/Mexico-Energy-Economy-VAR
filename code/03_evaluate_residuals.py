def evaluar_residuos_varios_modelos(modelos_dict, lags=20, lags_ljung=10, mostrar_graficas=True):
    """
    Evalúa y visualiza los residuos de varios modelos VAR.

    Para cada modelo VAR:
        - Muestra residuos, ACF y PACF horizontalmente por variable (opcional).
        - Realiza la prueba de Ljung-Box para cada variable.

    Parámetros:
        modelos_dict (dict): Diccionario con modelos ajustados {p: modelo_fitted}.
        lags (int): Número de rezagos para ACF y PACF (por defecto 20).
        lags_ljung (int): Rezago para prueba de Ljung-Box.
        mostrar_graficas (bool): Si True, muestra gráficos de residuos.

    Retorna:
        df_ljung (pd.DataFrame): Resultados de Ljung-Box por modelo y variable.
    """
    resultados_ljungbox = []

    for p, modelo in modelos_dict.items():
        residuos = modelo.resid
        print(f"\n📊 Evaluación de residuos para modelo VAR(p={p}):")

        for col in residuos.columns:
            if mostrar_graficas:
                fig, axes = plt.subplots(1, 3, figsize=(15, 3))

                # Gráfico de residuos
                axes[0].plot(residuos[col])
                axes[0].set_title(f'Residuos - {col}')
                axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)

                # ACF
                plot_acf(residuos[col], lags=lags, ax=axes[1])
                axes[1].set_title('ACF')

                # PACF
                plot_pacf(residuos[col], lags=lags, ax=axes[2], method='ywm')
                axes[2].set_title('PACF')

                plt.suptitle(f'VAR(p={p}) - {col}', fontsize=12)
                plt.tight_layout()
                plt.show()

            # Prueba de Ljung-Box
            ljungbox = acorr_ljungbox(residuos[col], lags=[lags_ljung], return_df=True)
            pval = ljungbox['lb_pvalue'].iloc[0]
            resultados_ljungbox.append({
                'p': p,
                'Variable': col,
                'p-valor': pval,
                'Autocorrelación': 'No' if pval > 0.05 else 'Sí'
            })

    df_ljung = pd.DataFrame(resultados_ljungbox)
    return df_ljung.sort_values(by=["p", "Variable"])
