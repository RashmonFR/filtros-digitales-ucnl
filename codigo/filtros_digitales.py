"""Actividad formativa 3: implementacion y evaluacion de filtros digitales.

Autor: Rashmon Bentzamin Fraga Rodriguez
Asignatura: Senales y Sistemas | Universidad Ciudadana de Nuevo Leon

Uso:
    python codigo/filtros_digitales.py

Las figuras y metricas se generan automaticamente. No usa archivos de entrada.
Las senales son sinteticas y los experimentos son reproducibles (semilla fija).
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Generar graficas aunque no exista una pantalla.
import matplotlib.pyplot as plt
from scipy import signal

# Carpetas calculadas desde el archivo, no desde la terminal actual.
BASE = Path(__file__).resolve().parents[1]
FIGS = BASE / "graficas"
RESULTS = BASE / "resultados"
FIGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# 1. Definicion de la senal de entrada.
FS = 1000                   # Muestras por segundo, Hz.
DURATION = 4.0              # Duracion de la prueba, segundos.
TIME = np.arange(int(FS * DURATION)) / FS
RNG = np.random.default_rng(2026)   # Semilla fija para repetir resultados.
FREQS = (15.0, 80.0, 220.0)
AMPS = (0.90, 0.70, 0.55)
NOISE_STD = 0.35            # Desviacion estandar del ruido blanco gaussiano.
COMPONENTS = {f: a * np.sin(2 * np.pi * f * TIME)
              for f, a in zip(FREQS, AMPS)}
CLEAN = sum(COMPONENTS.values())
NOISE = RNG.normal(0.0, NOISE_STD, len(TIME))
INPUT = CLEAN + NOISE
TRIM = int(0.30 * FS)        # Evita extremos al calcular metricas.
SLICE = slice(TRIM, -TRIM)

# 2. Casos: la frecuencia objetivo debe permanecer; las demas se rechazan.
CASES = {
    "pasa_bajos": {
        "title": "Pasa bajos", "btype": "lowpass", "cutoff": 35,
        "target": 15.0, "filename": "03_comparacion_pasa_bajos.png",
        "description": "Conserva 15 Hz; reduce 80 y 220 Hz."},
    "pasa_altos": {
        "title": "Pasa altos", "btype": "highpass", "cutoff": 150,
        "target": 220.0, "filename": "04_comparacion_pasa_altos.png",
        "description": "Conserva 220 Hz; reduce 15 y 80 Hz."},
    "pasa_bandas": {
        "title": "Pasa bandas", "btype": "bandpass", "cutoff": [60, 105],
        "target": 80.0, "filename": "05_comparacion_pasa_bandas.png",
        "description": "Conserva 80 Hz; reduce 15 y 220 Hz."},
}

IIR_ORDER = 6             # Butterworth IIR de sexto orden.
FIR_TAPS = 301             # FIR Hamming: 301 coeficientes, orden 300.

def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "figure.dpi": 125,
        "savefig.dpi": 170,
    })


def spectrum(x):
    """Devuelve amplitud aproximada FFT con ventana Hann para visualizacion."""
    window = np.hanning(len(x))
    fft = np.fft.rfft(x * window)
    hz = np.fft.rfftfreq(len(x), d=1 / FS)
    amp = 2 * np.abs(fft) / window.sum()
    return hz, amp


def tone_amplitude(x, f):
    """Proyecta la senal en una frecuencia conocida, sin usar la FFT global."""
    xt = x[SLICE]
    tt = TIME[SLICE]
    return float(2 * np.abs(np.mean(xt * np.exp(-2j * np.pi * f * tt))))


def snr_db(y, clean_target):
    """Relacion senal/error en dB frente a la componente objetivo ideal."""
    err = (y - clean_target)[SLICE]
    sig = clean_target[SLICE]
    return float(10 * np.log10(np.mean(sig ** 2) / np.mean(err ** 2)))


def designs(case):
    """Butterworth IIR en secciones de orden 2 y FIR de ventana Hamming."""
    typ, cutoff = case["btype"], case["cutoff"]
    sos = signal.butter(IIR_ORDER, cutoff, btype=typ, fs=FS, output="sos")
    # En firwin, pass_zero=True -> pasa bajos; False -> pasa altos o bandas.
    taps = signal.firwin(FIR_TAPS, cutoff, pass_zero=(typ == "lowpass"),
                         fs=FS, window="hamming")
    return sos, taps


def freq_response_sos(sos):
    """Compatibilidad entre versiones modernas y anteriores de SciPy."""
    if hasattr(signal, "freqz_sos"):
        return signal.freqz_sos(sos, worN=4096, fs=FS)
    return signal.sosfreqz(sos, worN=4096, fs=FS)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_input():
    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    mask = TIME <= 0.24
    ax[0].plot(TIME[mask], INPUT[mask], color="#53637D", lw=1.05, label="Entrada con ruido")
    ax[0].plot(TIME[mask], CLEAN[mask], color="#D28D33", lw=1.4,
               alpha=0.85, label="Suma ideal sin ruido")
    ax[0].set(xlabel="Tiempo (s)", ylabel="Amplitud", title="Señal compuesta: dominio del tiempo")
    ax[0].legend(loc="upper right")
    hz, amp = spectrum(INPUT)
    ax[1].plot(hz, amp, color="#53637D", lw=1.1)
    for f, a in zip(FREQS, AMPS):
        ax[1].axvline(f, color="#CE8545", ls="--", lw=1.0)
        ax[1].annotate(f"{int(f)} Hz", (f, a), xytext=(5, 8),
                       textcoords="offset points", fontsize=9)
    ax[1].set(xlim=(0, 300), xlabel="Frecuencia (Hz)", ylabel="Amplitud espectral",
              title="Espectro de la señal antes del filtrado")
    save_figure(fig, FIGS / "01_entrada_tiempo_y_espectro.png")


def plot_responses(all_designs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=True)
    for ax, (key, case) in zip(axes, CASES.items()):
        sos, taps = all_designs[key]
        wi, hi = freq_response_sos(sos)
        wf, hf = signal.freqz(taps, worN=4096, fs=FS)
        # Respuesta de una sola pasada (sin filtfilt): por eso no coincide
        # exactamente con la atenuacion de los resultados filtrados doblemente.
        ax.plot(wi, 20 * np.log10(np.maximum(np.abs(hi), 1e-9)),
                label="IIR Butterworth", color="#087E91", lw=2)
        ax.plot(wf, 20 * np.log10(np.maximum(np.abs(hf), 1e-9)),
                label="FIR Hamming", color="#E38F44", lw=1.8)
        cuts = case["cutoff"] if isinstance(case["cutoff"], list) else [case["cutoff"]]
        for c in cuts:
            ax.axvline(c, color="#677582", lw=1, ls="--")
        ax.set(xlim=(0, 300), ylim=(-110, 5), xlabel="Frecuencia (Hz)",
               title=case["title"])
    axes[0].set_ylabel("Ganancia por pasada (dB)")
    axes[2].legend(loc="lower left", fontsize=8)
    fig.suptitle("Respuesta en frecuencia: FIR vs IIR", fontsize=15, fontweight="bold")
    save_figure(fig, FIGS / "02_respuesta_en_frecuencia.png")


def plot_case(case, fir_y, iir_y):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5))
    mask = (TIME >= 1.0) & (TIME <= 1.24)
    ax[0].plot(TIME[mask], INPUT[mask], label="Entrada con ruido",
               color="#A4ACB5", alpha=0.78, lw=0.95)
    ax[0].plot(TIME[mask], COMPONENTS[case["target"]][mask],
               label=f"Componente ideal {case['target']:.0f} Hz",
               color="#272A3A", ls="--", lw=1.65)
    ax[0].plot(TIME[mask], iir_y[mask], label="IIR Butterworth",
               color="#087E91", lw=1.4)
    ax[0].plot(TIME[mask], fir_y[mask], label="FIR Hamming",
               color="#E38F44", alpha=0.88, lw=1.25)
    ax[0].set(xlabel="Tiempo (s)", ylabel="Amplitud", title="Comparación en el tiempo")
    ax[0].legend(loc="upper right", ncol=2, fontsize=8.5)
    for y, label, color in [
            (INPUT, "Original", "#A4ACB5"),
            (iir_y, "IIR", "#087E91"),
            (fir_y, "FIR", "#E38F44")]:
        hz, amp = spectrum(y)
        ax[1].plot(hz, amp, label=label, color=color, lw=1.4,
                   alpha=0.85 if label == "Original" else 1)
    for freq in FREQS:
        ax[1].axvline(freq, color="#888888", lw=0.7, ls=":")
    ax[1].set(xlim=(0, 300), xlabel="Frecuencia (Hz)", ylabel="Amplitud espectral",
              title="Comparación en la frecuencia")
    ax[1].legend(loc="upper right")
    fig.suptitle(f"Filtro {case['title'].lower()} — {case['description']}",
                 fontsize=13.2, fontweight="bold")
    save_figure(fig, FIGS / case["filename"])


def plot_metrics(rows):
    input_rows = {row["filtro"]: row["snr_entrada_db"] for row in rows}
    keys = list(CASES)
    pos = np.arange(len(keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(pos - width, [input_rows[k] for k in keys], width,
           label="Sin filtrar", color="#A5B0BE")
    for offset, typ, color in [(0, "IIR Butterworth", "#087E91"),
                               (width, "FIR Hamming", "#E38F44")]:
        vals = [next(r["snr_salida_db"] for r in rows
                     if r["filtro"] == k and r["metodo"] == typ) for k in keys]
        ax.bar(pos + offset, vals, width, label=typ, color=color)
    ax.axhline(0, color="#657587", lw=0.8)
    ax.set_xticks(pos, ["Pasa bajos\n15 Hz", "Pasa altos\n220 Hz", "Pasa bandas\n80 Hz"])
    ax.set_ylabel("Relación señal/error (dB)")
    ax.set_title("Calidad frente a la componente que se desea conservar")
    ax.legend()
    save_figure(fig, FIGS / "06_comparacion_snr.png")


def chebyshev_comparison(sos_butter):
    """Ejemplo adicional: IIR Chebyshev I frente a Butterworth en pasa bajos."""
    sos_cheby = signal.cheby1(6, 0.5, 35, btype="lowpass", fs=FS, output="sos")
    y = signal.sosfiltfilt(sos_cheby, INPUT)
    w, hc = freq_response_sos(sos_cheby)
    _, hb = freq_response_sos(sos_butter)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(w, 20 * np.log10(np.maximum(np.abs(hb), 1e-9)),
               label="Butterworth orden 6", color="#087E91", lw=2)
    ax[0].plot(w, 20 * np.log10(np.maximum(np.abs(hc), 1e-9)),
               label="Chebyshev I orden 6; 0.5 dB", color="#C26E94", lw=1.8)
    ax[0].set(xlim=(0, 110), ylim=(-110, 4), xlabel="Frecuencia (Hz)",
              ylabel="Ganancia por pasada (dB)", title="Respuesta en frecuencia")
    ax[0].legend(fontsize=8)
    mask = (TIME >= 1.0) & (TIME <= 1.30)
    ax[1].plot(TIME[mask], COMPONENTS[15.0][mask], ls="--", c="#333333",
               label="Referencia 15 Hz")
    ax[1].plot(TIME[mask], signal.sosfiltfilt(sos_butter, INPUT)[mask],
               c="#087E91", label="Butterworth")
    ax[1].plot(TIME[mask], y[mask], c="#C26E94", label="Chebyshev I", alpha=0.9)
    ax[1].set(xlabel="Tiempo (s)", ylabel="Amplitud",
              title="Señal filtrada (ejemplo pasa bajos)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Ejemplo IIR adicional: Butterworth y Chebyshev tipo I",
                 fontsize=13, fontweight="bold")
    save_figure(fig, FIGS / "07_butterworth_vs_chebyshev.png")
    return {
        "orden": 6, "rizado_pasobanda_db": 0.5,
        "snr_entrada_db": round(snr_db(INPUT, COMPONENTS[15.0]), 3),
        "snr_chebyshev_db": round(snr_db(y, COMPONENTS[15.0]), 3),
    }


def main():
    setup_style()
    plot_input()
    filters = {key: designs(cfg) for key, cfg in CASES.items()}
    plot_responses(filters)
    rows = []
    for key, case in CASES.items():
        sos, taps = filters[key]
        # 3. Aplicacion sin desfase (adelante/atras): solo analisis offline.
        yiir = signal.sosfiltfilt(sos, INPUT)
        yfir = signal.filtfilt(taps, [1.0], INPUT)
        plot_case(case, yfir, yiir)
        for name, y in [("IIR Butterworth", yiir), ("FIR Hamming", yfir)]:
            target = COMPONENTS[case["target"]]
            snr_before = snr_db(INPUT, target)
            snr_after = snr_db(y, target)
            row = {
                "filtro": key, "metodo": name,
                "objetivo_hz": int(case["target"]),
                "corte_hz": str(case["cutoff"]),
                "snr_entrada_db": round(snr_before, 3),
                "snr_salida_db": round(snr_after, 3),
                "mejora_snr_db": round(snr_after - snr_before, 3),
                "amplitud_objetivo_entrada": round(tone_amplitude(INPUT, case["target"]), 4),
                "amplitud_objetivo_salida": round(tone_amplitude(y, case["target"]), 4),
                "amplitud_15hz_salida": round(tone_amplitude(y, 15), 4),
                "amplitud_80hz_salida": round(tone_amplitude(y, 80), 4),
                "amplitud_220hz_salida": round(tone_amplitude(y, 220), 4),
            }
            rows.append(row)
    plot_metrics(rows)
    cheby = chebyshev_comparison(filters["pasa_bajos"][0])

    # 4. Documentacion de resultados para revisar o graficar posteriormente.
    out_csv = RESULTS / "metricas_filtros.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (RESULTS / "resumen_experimento.json").open("w", encoding="utf-8") as fp:
        json.dump({"fs_hz": FS, "duracion_s": DURATION,
                   "frecuencias_hz": FREQS, "amplitudes": AMPS,
                   "ruido_std": NOISE_STD, "semilla": 2026,
                   "iir_orden": IIR_ORDER, "fir_taps": FIR_TAPS,
                   "chebyshev": cheby, "metricas": rows},
                  fp, indent=2, ensure_ascii=False)
    print("Simulacion terminada. Figuras:", FIGS)
    print("Metricas:", out_csv)
    for row in rows:
        print(f"{row['filtro']:12} {row['metodo']:18} "
              f"SNR={row['snr_salida_db']:6.2f} dB | "
              f"mejora={row['mejora_snr_db']:+6.2f} dB")

if __name__ == "__main__":
    main()
