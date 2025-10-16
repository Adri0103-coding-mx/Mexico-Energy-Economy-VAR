def GIRF(modelo, horizontes, ruta="./", guardar_figuras=True, guardar_datos=True):
    """
    Calcula y grafica Generalized IRFs (GIRFs) para un modelo VAR ajustado.

    Parámetros:
    -----------
    modelo : VARResults
        Modelo VAR ajustado (fit).
    horizontes : list[int]
        Lista de horizontes a analizar (ej. [3, 18, 40]).
    ruta : str
        Carpeta donde guardar las figuras y datos.
    guardar_figuras : bool
        Si True, guarda las gráficas en PNG.
    guardar_datos : bool
        Si True, guarda las respuestas en CSV.
    """

    for h in horizontes:
        # 1. Calcular GIRF hasta el horizonte h
        girf = modelo.irf(h)
        fig = girf.plot(orth=False)

        # 2. Ajustes de visualización
        fig.set_size_inches(24, 24)
        plt.suptitle(f"Generalized Impulse Response Functions (GIRFs) - Horizonte {h}",
                     fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

        # 3. Extraer respuestas (array → DataFrame)
        girf_responses = girf.irfs  # shape: (h+1, nvars, nvars)
        df_girf = pd.DataFrame(
            girf_responses.reshape(h+1, -1),
            columns=[f"{resp}_to_{imp}" for imp in modelo.names for resp in modelo.names]
        )

        # 4. Guardar resultados
        if guardar_figuras:
            fig.savefig(f"{ruta}/GIRF_h{h}.pdf", dpi=300, bbox_inches="tight")
        if guardar_datos:
            df_girf.to_csv(f"{ruta}/GIRF_h{h}.csv", index=False)

    print("✅ GIRFs generados correctamente.")

def graficar_IRF(modelo_fitted, nombres_personalizados: dict,
                      steps=40, nombre_modelo='', carpeta_salida=None,
                      tol_convergencia=0.01, max_steps=40):
    """
    Calcula y grafica IRF y FEVD con nombres personalizados, exportando resultados en CSV y PNG.
    Si steps es None, lo calcula automáticamente según convergencia de la varianza (FEVD).

    Parámetros:
        modelo_fitted (VARResults): Modelo VAR entrenado.
        nombres_personalizados (dict): Diccionario {nombre_original: nombre_personalizado}.
        steps (int or None): Número de pasos a futuro. Si None, se calcula automáticamente.
        nombre_modelo (str): Nombre del modelo (para guardar archivos).
        carpeta_salida (str): Carpeta donde guardar los archivos. Si None, no guarda.
        tol_convergencia (float): Tolerancia para definir convergencia en FEVD.
        max_steps (int): Máximo de pasos a evaluar para sugerencia automática.
    """
    if carpeta_salida:
      os.makedirs(carpeta_salida, exist_ok=True)

    nombres = modelo_fitted.names
    nombres_cortos = [nombres_personalizados.get(n, n) for n in nombres]

    # === IRF ===
    irf = modelo_fitted.irf(steps)
    irf_data = {}
    for i, respuesta in enumerate(nombres):
        for j, impulso in enumerate(nombres):
            valores = irf.irfs[:, i, j]
            col = f"{nombres_personalizados.get(respuesta, respuesta)}_resp_to_{nombres_personalizados.get(impulso, impulso)}"
            irf_data[col] = valores
    irf_df = pd.DataFrame(irf_data)
    irf_df.index.name = 'Step'
    if carpeta_salida:
        irf_df.to_csv(os.path.join(carpeta_salida, f'IRF_{nombre_modelo}.csv'))

    # === Graficar IRF ===
    fig_irf = irf.plot(orth=False, figsize=(24, len(nombres) * 2.5))
    plt.suptitle(f"Impulse Response Functions (IRFs) - Horizonte {h}",
                     fontsize=16, y=1.02)
    palette_irf = plt.get_cmap("Set2")
    for i, ax in enumerate(fig_irf.axes):
        fila = i // len(nombres)
        col = i % len(nombres)
        resp = nombres_cortos[fila]
        imp = nombres_cortos[col]
        ax.set_title(f'{resp} resp to {imp}')
        ax.tick_params(axis='x', rotation=45)
        for j, line in enumerate(ax.get_lines()):
            line.set_color(palette_irf(j % palette_irf.N))
        for j, collection in enumerate(ax.collections):
            color = palette_irf(j % palette_irf.N)
            collection.set_facecolor(color)
            collection.set_edgecolor(color)
    plt.tight_layout()
    plt.show()
    if carpeta_salida:
        fig_irf.savefig(os.path.join(carpeta_salida, f'IRF_{nombre_modelo}.png'))

    return irf_df, irf
def calcular_FEVD_cholesky(modelo_fitted, steps=40, nombre_modelo='', carpeta_salida=None):
    """
    Calcula la Descomposición de la Varianza del Error de Pronóstico (FEVD) usando descomposición de Cholesky.

    Basado en:
    Lütkepohl (2005) - New Introduction to Multiple Time Series Analysis.

    Args:
        modelo_fitted (VARResultsWrapper): Modelo VAR ajustado.
        steps (int): Número de pasos hacia adelante para el análisis.
        nombre_modelo (str): Nombre base para el archivo CSV.
        carpeta_salida (str or None): Carpeta para guardar el archivo. Si None, no guarda.

    Returns:
        pd.DataFrame: Contribuciones normalizadas acumuladas por paso y variable.
    """
    if carpeta_salida:
        os.makedirs(carpeta_salida, exist_ok=True)

    irf = modelo_fitted.irf(steps)
    psi = irf.irfs  # (steps, nvars, nvars)
    nombres = modelo_fitted.names
    num_vars = len(nombres)

    sigma_u = modelo_fitted.sigma_u
    P = np.linalg.cholesky(sigma_u)  # (nvars, nvars)

    fevd = np.zeros((steps, num_vars, num_vars))  # (steps, caused_by, affected_var)

    for h in range(steps):
        suma_total = np.zeros(num_vars)
        if h == 0:
            theta = psi[0] @ P
            for i in range(num_vars):
                for j in range(num_vars):
                    fevd[h, j, i] = 1.0 if i == j else 0.0
        else:
            for s in range(h + 1):
                theta = psi[s] @ P
                for i in range(num_vars):
                    for j in range(num_vars):
                        fevd[h, j, i] += theta[i, j] ** 2
                        suma_total[i] += theta[i, j] ** 2

            for i in range(num_vars):
                if suma_total[i] == 0:
                    fevd[h, :, i] = 1.0 / num_vars
                else:
                    fevd[h, :, i] /= suma_total[i]

    datos = {}
    for i, afectada in enumerate(nombres):
        for j, causa in enumerate(nombres):
            col = f"{afectada}_caused_by_{causa}"
            datos[col] = fevd[:, j, i]

    fevd_df = pd.DataFrame(datos)
    fevd_df.index.name = "Step"

    if carpeta_salida:
        ruta = os.path.join(carpeta_salida, f'FEVD_{nombre_modelo}.csv')
        fevd_df.to_csv(ruta)
        print(f"✅ FEVD guardado en: {ruta}")

    return fevd_df


# Gráfica de la FEVD desde la salida (df) de calcular_FEVD_cholesky()
def graficar_FEVD_desde_df(fevd_df, nombre_modelo='', nombres_personalizados=None, carpeta_salida=None):
    """
    Grafica y guarda la descomposición de la varianza del forecast (FEVD) desde un DataFrame generado manualmente.

    Args:
        fevd_df (pd.DataFrame): DataFrame con columnas 'variable_caused_by_variable'.
        nombre_modelo (str): Nombre del archivo para guardar (sin extensión).
        nombres_personalizados (dict): Diccionario para reemplazar nombres en la gráfica.
        carpeta_salida (str): Carpeta donde se guarda el archivo PNG. Si None, no se guarda.
    """
    if nombres_personalizados is None:
        nombres_personalizados = {}

    variables = sorted(set(col.split('_caused_by_')[0] for col in fevd_df.columns))
    causas = sorted(set(col.split('_caused_by_')[1] for col in fevd_df.columns))
    colores = plt.cm.tab10(np.linspace(0, 1, len(causas)))

    fig, axs = plt.subplots(len(variables), 1, figsize=(14, 3 * len(variables)), sharex=True)
    if len(variables) == 1:
        axs = [axs]

    for i, var in enumerate(variables):
        ax = axs[i]
        bottom = np.zeros(fevd_df.shape[0])
        for j, causa in enumerate(causas):
            col = f"{var}_caused_by_{causa}"
            label = nombres_personalizados.get(causa, causa)
            contribucion = fevd_df[col].values
            ax.bar(fevd_df.index, contribucion, bottom=bottom, label=label, color=colores[j])
            bottom += contribucion

        ax.set_title(nombres_personalizados.get(var, var))
        ax.set_ylabel("Contribución acumulada")
        ax.set_ylim(0, 1.05)

    axs[-1].set_xlabel("Step")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=min(4, len(causas)))
    fig.suptitle("FEVD: Participación acumulada por variable causante", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Guardar como imagen si se especifica la carpeta
    if carpeta_salida:
        os.makedirs(carpeta_salida, exist_ok=True)
        ruta = os.path.join(carpeta_salida, f"FEVD_{nombre_modelo}.png")
        fig.savefig(ruta, dpi=300)
        print(f"✅ Imagen guardada en: {ruta}")

    plt.show()
