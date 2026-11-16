# Implementación y evaluación de filtros digitales

**Actividad formativa 3 · Señales y Sistemas · UCNL**  
**Estudiante:** Rashmon Bentzamin Fraga Rodríguez

## Objetivo
Diseñar y comparar filtros digitales **pasa bajos**, **pasa altos** y **pasa bandas** de las familias **IIR Butterworth** y **FIR con ventana Hamming**, con un ejemplo adicional de **IIR Chebyshev tipo I**. Se analiza una señal compuesta de 15 Hz, 80 Hz y 220 Hz contaminada con ruido blanco. Se contrastan dominio del tiempo, espectro, respuesta en frecuencia y relación señal/error (SNR).

## Cómo ejecutarlo

Requiere Python 3.10 o superior. En una terminal situada dentro de esta carpeta:

```bash
python -m pip install -r requirements.txt
python codigo/filtros_digitales.py
```

En Windows, si `python` no se reconoce, puede utilizarse `py` en su lugar. No es necesario descargar datos: la señal se genera automáticamente y siempre igual gracias a la semilla 2026.

## Parámetros de la prueba

- Muestreo: 1000 Hz; duración: 4 segundos (4000 muestras).
- Señales de 15 Hz (amplitud 0.90), 80 Hz (0.70), 220 Hz (0.55).
- Ruido blanco gaussiano con desviación estándar de 0.35.
- IIR Butterworth de orden 6; FIR Hamming con 301 coeficientes.
- Pasa bajos: corte 35 Hz (se desea conservar 15 Hz).
- Pasa altos: corte 150 Hz (se desea conservar 220 Hz).
- Pasa bandas: cortes 60 a 105 Hz (se desea conservar 80 Hz).
- Extra: Chebyshev tipo I de orden 6, rizado de 0.5 dB, pasa bajos de 35 Hz.

## Resultados generados

Las imágenes se guardan en `graficas/` y las métricas en `resultados/`.

| Archivo | Contenido |
|---|---|
| `01_entrada_tiempo_y_espectro.png` | Entrada ruidosa en tiempo y frecuencia |
| `02_respuesta_en_frecuencia.png` | Comparación de las funciones de transferencia FIR/IIR |
| `03_comparacion_pasa_bajos.png` | Efecto del filtro pasa bajos |
| `04_comparacion_pasa_altos.png` | Efecto del filtro pasa altos |
| `05_comparacion_pasa_bandas.png` | Efecto del filtro pasa bandas |
| `06_comparacion_snr.png` | Comparación de la relación señal/error |
| `07_butterworth_vs_chebyshev.png` | Ejemplo adicional de Chebyshev tipo I |
| `resultados/metricas_filtros.csv` | Tabla de resultados cuantitativos |
| `resultados/resumen_experimento.json` | Configuración y métricas reutilizables |

**Interpretación:** las gráficas de respuesta en frecuencia muestran la **ganancia de una sola pasada**. Para comparar señales sin retraso temporal en esta práctica se utilizan `sosfiltfilt` y `filtfilt` (pasada hacia adelante y atrás), de modo que la atenuación efectiva queda aproximadamente duplicada en dB. Esta técnica no es causal: sirve para análisis de señales ya registradas, no para procesamiento estrictamente en tiempo real.

La SNR indicada aquí se calcula como `10·log10(potencia de la componente objetivo ideal / potencia del error total respecto a ella)` sobre el intervalo sin extremos de 0.3 a 3.7 s. No es una medida de SNR universal para cualquier señal real: es válida aquí porque conocemos la componente limpia creada en la simulación.

## Verificación de ejecución

El código original se ejecutó con Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 y Matplotlib 3.10.8. Las siete gráficas y los archivos CSV y JSON resultantes coinciden byte por byte con los incluidos en el paquete académico original.

**Precisión sobre el orden:** en el pasa bandas Butterworth, el parámetro `N=6` produce un filtro de orden total 12 (seis secciones de segundo orden). En pasa bajos y pasa altos, el orden total es 6. Los resultados originales se conservan sin modificaciones.

## Aplicación práctica

La extracción de componentes de una señal puede resultar útil al separar frecuencias de vibración de una máquina, atenuar ruido de baja frecuencia o aislar una banda de interés. Para sistemas reales se requieren diseños ajustados al muestreo, latencia, respuesta de fase y especificaciones del sistema.

## Fuentes de consulta

- SciPy. (s. f.). *Signal processing (scipy.signal).* https://docs.scipy.org/doc/scipy/reference/signal.html
- SciPy. (s. f.). *butter.* https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
- SciPy. (s. f.). *sosfiltfilt.* https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html
- SciPy. (s. f.). *Signal processing tutorial.* https://docs.scipy.org/doc/scipy/tutorial/signal.html
