"""
Experimento de anticipacion (P3): compara la version 1 (frecuencias) con la version 2 (LSTM).

Simula una persona con un habito imperfecto: cuando aparece el objeto senal, casi siempre pide el
mismo objeto. Mientras tanto otros objetos aparecen y desaparecen (varios en el mismo momento y en
orden al azar) y a veces pide otras cosas. Se mide, cada vez que aparece la senal, la confianza con
la que el modelo anticipa el pedido del habito, y en los momentos sin senal, la confianza de su
mejor prediccion (falsos positivos).

    python experimentos/anticipacion.py --repeticiones 30 --semillas 20
"""
from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asistente.cerebro.anticipacion import AnticipacionFrecuencias, AnticipacionSecuencias
from asistente.tipos import Evento

MODELOS = {"Frecuencias (v1)": AnticipacionFrecuencias, "LSTM (v2)": AnticipacionSecuencias}
COLORES = {"Frecuencias (v1)": "#eb6834", "LSTM (v2)": "#2a78d6"}
SENAL, PEDIDO = "objeto_senal", "traer:objeto_pedido"
PERSONA = "persona_sim"


def simular(Modelo, repeticiones: int, semilla: int, p_cumple: float = 0.8,
            p_senal: float = 0.35, p_cambio: float = 0.3, p_otro_pedido: float = 0.15):
    """Devuelve (confianza con senal por repeticion, confianza media sin senal por repeticion)."""
    rng = random.Random(semilla)
    modelo = Modelo()
    objetos = [f"objeto_{i}" for i in range(1, 7)]
    otros_pedidos = [f"traer:{o}" for o in objetos[:3]] + [f"apartar:{o}" for o in objetos[3:]]
    en_mesa: set[str] = set()
    eventos: list[Evento] = []
    con_senal, sin_senal, pendientes_sin = [], [], []
    t, senal_en_mesa = 0.0, False

    while len(con_senal) < repeticiones:
        t += 5.0
        cambios = []
        for o in objetos:
            if rng.random() < p_cambio:
                cambios.append(("desaparecio" if o in en_mesa else "aparecio", o))
                en_mesa ^= {o}
        if senal_en_mesa:
            cambios.append(("desaparecio", SENAL))
            senal_en_mesa = False
        elif rng.random() < p_senal:
            cambios.append(("aparecio", SENAL))
            senal_en_mesa = True
        rng.shuffle(cambios)
        eventos += [Evento(tipo, detalle, PERSONA, t + 0.01 * i) for i, (tipo, detalle) in enumerate(cambios)]

        recientes = [e for e in eventos if e.t > t - 300]
        pedido, prob = modelo.predecir(recientes)
        if senal_en_mesa:
            con_senal.append(prob if pedido == PEDIDO else 0.0)
            sin_senal.append(float(np.mean(pendientes_sin)) if pendientes_sin else np.nan)
            pendientes_sin = []
        else:
            pendientes_sin.append(prob)

        if senal_en_mesa and rng.random() < p_cumple:
            real = PEDIDO
        elif not senal_en_mesa and rng.random() < p_otro_pedido:
            real = rng.choice(otros_pedidos)
        else:
            continue
        eventos.append(Evento("pedido", real, PERSONA, t + 2.0))
        modelo.actualizar(recientes + [eventos[-1]], real)

    return np.array(con_senal), np.array(sin_senal)


def _corrida(args):
    nombre, repeticiones, semilla = args
    torch.set_num_threads(1)
    return nombre, simular(MODELOS[nombre], repeticiones, semilla)


def graficar(resultados: dict, repeticiones: int, umbral: float, salida: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tinta, tinta_suave, superficie = "#0b0b0b", "#52514e", "#fcfcfb"
    x = np.arange(1, repeticiones + 1)
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor=superficie)
    titulos = ["Confianza al aparecer la senal", "Confianza sin la senal (falsos positivos)"]
    for eje, titulo, clave in zip(ejes, titulos, ["con", "sin"]):
        eje.set_facecolor(superficie)
        finales = {}
        for nombre, r in resultados.items():
            media = np.nanmean(r[clave], axis=0)
            eje.plot(x, media, color=COLORES[nombre], linewidth=2, marker="o", markersize=4,
                     label=nombre)
            finales[nombre] = media[~np.isnan(media)][-1]
        (a, ya), (b, yb) = sorted(finales.items(), key=lambda kv: kv[1])
        separacion = max(0.0, 0.07 - (yb - ya)) / 2
        for nombre, y in ((a, ya - separacion), (b, yb + separacion)):
            eje.annotate(nombre, (x[-1], y), xytext=(6, 0), textcoords="offset points",
                         va="center", fontsize=8, color=tinta_suave)
        eje.axhline(umbral, color=tinta_suave, linewidth=1, linestyle="--")
        eje.text(1, umbral + 0.02, f"umbral {umbral:.0%}", fontsize=8, color=tinta_suave)
        eje.set_title(titulo, fontsize=10, color=tinta, loc="left")
        eje.set_xlabel("Veces que aparecio la senal", fontsize=9, color=tinta_suave)
        eje.set_xlim(1, repeticiones * 1.15)
        eje.set_ylim(0, 1.02)
        eje.grid(axis="y", color="#e4e3df", linewidth=0.8)
        eje.tick_params(colors=tinta_suave, labelsize=8)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            eje.spines[lado].set_color("#c3c2b7")
    ejes[0].set_ylabel("Probabilidad del pedido anticipado", fontsize=9, color=tinta_suave)
    ejes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeticiones", type=int, default=30)
    ap.add_argument("--semillas", type=int, default=20)
    ap.add_argument("--umbral", type=float, default=0.7)
    ap.add_argument("--salida", default="docs/figuras/anticipacion_frecuencias_vs_lstm.png")
    args = ap.parse_args()

    tareas = [(nombre, args.repeticiones, s) for nombre in MODELOS for s in range(args.semillas)]
    with ProcessPoolExecutor() as ejecutor:
        corridas = list(ejecutor.map(_corrida, tareas))

    bloques = [(1, 5), (6, 10), (11, 20), (21, args.repeticiones)]
    print("Confianza media al aparecer la senal, por bloque de repeticiones:")
    resultados = {}
    for nombre in MODELOS:
        con = np.stack([c for n, (c, _) in corridas if n == nombre])
        sin = np.stack([s for n, (_, s) in corridas if n == nombre])
        resultados[nombre] = {"con": con, "sin": sin}
        media = con.mean(axis=0)
        texto = "  ".join(f"{i}-{j}: {media[i - 1:j].mean():.2f}" for i, j in bloques if i <= j)
        falsos = np.nanmean(sin[:, 1:] > args.umbral)
        print(f"  {nombre:<17} {texto}  | momentos sin senal sobre {args.umbral:.0%}: {falsos:.1%}")

    salida = Path(args.salida)
    graficar(resultados, args.repeticiones, args.umbral, salida)
    print(f"Figura guardada en {salida}")


if __name__ == "__main__":
    main()
