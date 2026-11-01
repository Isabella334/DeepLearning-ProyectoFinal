"""
Curvas del reconocedor con pocos ejemplos (P2 · Parte 3): aciertos con 1, 2, 3 y 5 ejemplos
por objeto, comparando v1 (prototipo + coseno) contra v2 (cabeza MLP por persona).

Para cada semilla y cada nivel de ruido de la percepcion:
  1. El robot "mira" la mesa un rato (todos los objetos, sin nombres): eso llena su fondo.
  2. La persona A le ensena k ejemplos de 3 de los 5 objetos (elegidos al azar).
  3. Se mide con vistas NUEVAS:
       - acierto:  objetos ensenados reconocidos con su nombre correcto;
       - rechazo:  objetos NO ensenados que salen como "desconocido" (no los confunde);
       - otra persona: lo de A, visto por B, sale "desconocido".
  Ademas, con los 5 objetos ensenados, el acierto "todos ensenados".

    python experimentos/curvas_reconocedor.py              # 10 semillas
    python experimentos/curvas_reconocedor.py --semillas 3

Salidas: docs/figuras/p2_reconocedor.png, experimentos/resultados/p2_reconocedor.json
"""
from __future__ import annotations

import argparse
import json

import comun
import numpy as np

from asistente.mocks import PercepcionFalsa
from asistente.tipos import DESCONOCIDO
from asistente.vision.reconocedor import ReconocedorPocosEjemplos

TIPOS = PercepcionFalsa.TIPOS              # el aspecto oculto de cada tipo (solo para simular)
MODOS = {"prototipo": "v1 · prototipo + coseno", "cabeza": "v2 · cabeza MLP por persona"}


def vista(tipo, ruido, rng):
    return list(np.asarray(TIPOS[tipo]) + rng.normal(0, ruido, len(TIPOS[tipo])))


def una_corrida(modo, k, ruido, semilla, ensenados, n_fondo=20, n_prueba=40):
    rng = np.random.default_rng(semilla)
    r = ReconocedorPocosEjemplos(modo=modo, semilla=semilla)
    for _ in range(n_fondo):                              # mirar la mesa antes de que le ensenen
        for t in TIPOS:
            r.reconocer("A", vista(t, ruido, rng))
    for t in ensenados:
        for _ in range(k):
            r.ensenar("A", t, vista(t, ruido, rng))
    r.aprender_del_fondo = False                          # medir sin "mirar el examen"
    acierto = np.mean([r.reconocer("A", vista(t, ruido, rng))[0] == t
                       for t in ensenados for _ in range(n_prueba)])
    otros = [t for t in TIPOS if t not in ensenados]
    rechazo = np.mean([r.reconocer("A", vista(t, ruido, rng))[0] == DESCONOCIDO
                       for t in otros for _ in range(n_prueba)]) if otros else np.nan
    otra_persona = np.mean([r.reconocer("B", vista(t, ruido, rng))[0] == DESCONOCIDO
                            for t in ensenados for _ in range(10)])
    return float(acierto), float(rechazo), float(otra_persona)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=10)
    ap.add_argument("--ejemplos", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--ruidos", type=float, nargs="+", default=[0.03, 0.10],
                    help="0.03 es el ruido de mocks.PercepcionFalsa; 0.10 es una percepcion peor")
    ap.add_argument("--sin-graficas", action="store_true")
    args = ap.parse_args()

    res = {}
    for modo in MODOS:
        for ruido in args.ruidos:
            fila = {"k": args.ejemplos, "acierto_todos": [], "acierto": [], "rechazo": [], "otra_persona": []}
            for k in args.ejemplos:
                todos, parciales = [], []
                for s in range(args.semillas):
                    sub = list(np.random.default_rng(1000 + s).choice(list(TIPOS), 3, replace=False))
                    todos.append(una_corrida(modo, k, ruido, s, list(TIPOS))[0])
                    parciales.append(una_corrida(modo, k, ruido, s, sub))
                P = np.array(parciales)
                fila["acierto_todos"].append(float(np.mean(todos)))
                fila["acierto"].append(float(P[:, 0].mean()))
                fila["rechazo"].append(float(P[:, 1].mean()))
                fila["otra_persona"].append(float(P[:, 2].mean()))
                print(f"{modo:9s} ruido={ruido:.2f} k={k}: acierto(todos)={fila['acierto_todos'][-1]:.0%} "
                      f"acierto(3 de 5)={fila['acierto'][-1]:.0%} rechazo={fila['rechazo'][-1]:.0%} "
                      f"otra persona={fila['otra_persona'][-1]:.0%}")
            res[f"{modo}|{ruido}"] = fila

    comun.RESULTADOS.mkdir(parents=True, exist_ok=True)
    (comun.RESULTADOS / "p2_reconocedor.json").write_text(
        json.dumps({"semillas": args.semillas, "resultados": res}, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.sin_graficas:
        graficar(res, args)


def graficar(res, args):
    from comun import SERIE, TEXTO_2, estilo, guardar_figura
    plt = estilo()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    estilos = ["-", "--", ":"]
    paneles = [("acierto_todos", "Reconoce lo que le enseñaron", "objetos reconocidos con su nombre (%)"),
               ("rechazo", "No confunde lo que no le enseñaron", "objetos no enseñados → \"desconocido\" (%)")]
    for ax, (clave, titulo, ylabel) in zip(axes, paneles):
        for c, modo in enumerate(MODOS):
            for e, ruido in enumerate(args.ruidos):
                f = res[f"{modo}|{ruido}"]
                ax.plot(f["k"], np.array(f[clave]) * 100, color=SERIE[c], ls=estilos[e % 3], marker="o", ms=5,
                        label=f"{MODOS[modo]}, ruido {ruido:.2f}")
        ax.set(xticks=args.ejemplos, xlabel="ejemplos enseñados por objeto", ylabel=ylabel, title=titulo,
               ylim=(0, 104))
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.text(0.01, -0.04, f"Promedio de {args.semillas} semillas. Izquierda: los 5 objetos enseñados. "
             "Derecha: solo 3 de 5 enseñados; se mide si los otros 2 salen como \"desconocido\". "
             "Umbral = config.yaml (reconocimiento.umbral_similitud).", fontsize=8, color=TEXTO_2)
    guardar_figura(fig, "p2_reconocedor.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
