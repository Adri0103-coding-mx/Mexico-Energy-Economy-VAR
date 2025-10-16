def evaluar_metricas_VAR_train_test(modelos_dict, train_series, test_series, forecast_steps,
                                    variable_objetivo, scaler, series_original):
    """
    Evalúa modelos VAR entrenados sobre series diferenciadas y estandarizadas,
    revirtiendo ambas transformaciones antes de calcular métricas reales en train y test.
    """
    metricas_train = {}
    metricas_test = {}

    for p, modelo in modelos_dict.items():
        try:
            # === Predicción en TRAIN ===
            input_train = train_series.values[-p:]
            pred_train = modelo.forecast(input_train, steps=p)
            pred_df_train = pd.DataFrame(pred_train, columns=train_series.columns)

            pred_train_descaled = pd.DataFrame(scaler.inverse_transform(pred_df_train), columns=train_series.columns)
            ultimo_real_train = series_original[variable_objetivo].iloc[len(train_series) - 1 - p]
            y_pred_train = np.cumsum(pred_train_descaled[variable_objetivo].values) + ultimo_real_train
            y_real_train = series_original[variable_objetivo].iloc[len(train_series) - p : len(train_series)].values

            # Métricas train
            mae_train = mean_absolute_error(y_real_train, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_real_train, y_pred_train))

            # Validar si R2 es computable
            if len(y_real_train) >= 2 and not np.allclose(y_real_train, y_real_train[0]):
                r2_train = r2_score(y_real_train, y_pred_train)
            else:
                print(f"⚠️ Menos de 2 puntos o constante en y_real_train para p={p}: {y_real_train}")
                r2_train = np.nan

            metricas_train[p] = {'MAE': mae_train, 'RMSE': rmse_train, 'R2': r2_train}

            # === Predicción en TEST ===
            input_test = test_series.values[:p]
            pred_test = modelo.forecast(input_test, steps=forecast_steps)
            pred_df_test = pd.DataFrame(pred_test, columns=test_series.columns)

            pred_test_descaled = pd.DataFrame(scaler.inverse_transform(pred_df_test), columns=test_series.columns)
            ultimo_real_test = series_original[variable_objetivo].iloc[len(train_series) - 1]
            y_pred_test = np.cumsum(pred_test_descaled[variable_objetivo].values) + ultimo_real_test
            y_real_test = series_original[variable_objetivo].iloc[len(train_series):len(train_series)+forecast_steps].values

            # Métricas test
            mae_test = mean_absolute_error(y_real_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_real_test, y_pred_test))
            r2_test = r2_score(y_real_test, y_pred_test)

            metricas_test[p] = {'MAE': mae_test, 'RMSE': rmse_test, 'R2': r2_test}

        except Exception as e:
            print(f"⚠️ Error en evaluación para p={p}")
            traceback.print_exc()
            metricas_train[p] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
            metricas_test[p] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}

    return metricas_train, metricas_test
