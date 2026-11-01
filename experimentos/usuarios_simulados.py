"""
Usuarios simulados y curvas de aprendizaje de las preferencias (P2 · Parte 3).

Un "usuario" es una regla oculta (experimentos/usuarios.yaml): que accion le gusta para cada
tipo de objeto. El robot propone una accion, el usuario dice "bien" o "mal" segun su regla y el
robot aprende (PreferenciasRed.actualizar). Se mide el porcentaje de aciertos en situaciones
nuevas a medida que avanzan las interacciones.

Dos usuarios con reglas distintas entrenan dos COPIAS del mismo robot (clonar) viendo las
mismas mesas: las copias terminan distintas.

    python experimentos/usuarios_simulados.py                 # 10 semillas x 100 interacciones
    python experimentos/usuarios_simulados.py --semillas 3    # rapido

Salidas: docs/figuras/p2_preferencias_curvas.png, docs/figuras/p2_preferencias_dos_robots.png
         experimentos/resultados/p2_preferencias.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import comun  # noqa: F401  (agrega la raiz del repo al path)
import numpy as np
import torch
import yaml

from asistente.cerebro.preferencias import ACCIONES, PreferenciasRed
from asistente.mocks import PercepcionFalsa
from asistente.tipos import Accion, Contexto
from asistente.vision.reconocedor import ReconocedorPocosEjemplos

RUTA_USUARIOS = Path(__file__).with_name("usuarios.yaml")
HORA_FIJA = time.mktime((2026, 10, 26, 10, 0, 0, 0, 0, -1))   # todas las mesas, a las 10 a. m.


class UsuarioSimulado:
    """Aprueba o corrige segun su regla oculta. El robot nunca ve la regla."""

    def __init__(self, nombre: str, descripcion: str = "", reglas: dict | None = None,
                 condiciones: list | None = None, por_defecto: str = "ignorar"):
        self.nombre, self.descripcion = nombre, descripcion
        self.reglas = {t: Accion(a) for t, a in (reglas or {}).items()}
        self.condiciones = [dict(c, entonces=Accion(c["entonces"]), si_no=Accion(c["si_no"]))
                            for c in (condiciones or [])]
        self.por_defecto = Accion(por_defecto)

    @classmethod
    def cargar_todos(cls, ruta=RUTA_USUARIOS) -> dict[str, "UsuarioSimulado"]:
        datos = yaml.safe_load(Path(ruta).read_text(encoding="utf-8"))["usuarios"]
        return {n: cls(n, **d) for n, d in datos.items()}

    def accion_preferida(self, tipo: str, tipos_en_mesa: set[str]) -> Accion:
        for c in self.condiciones:
            if c["tipo"] == tipo:
                return c["entonces"] if c["si_hay"] in tipos_en_mesa else c["si_no"]
        return self.reglas.get(tipo, self.por_defecto)

    def calificar(self, tipo: str, tipos_en_mesa: set[str], accion: Accion) -> bool:
        return accion == self.accion_preferida(tipo, tipos_en_mesa)


# ---------------------------------------------------------------- el mundo de cada persona
def mundo_ensenado(persona: str, semilla: int, ejemplos: int = 3) -> PercepcionFalsa:
    """Una mesa falsa donde la persona ya le enseno el nombre de cada objeto `ejemplos` veces
    (con lo que va apareciendo en la mesa, como en la demo). Cada persona usa sus propias
    palabras: aqui el nombre que ensena es el tipo oculto."""
    perc = PercepcionFalsa(ReconocedorPocosEjemplos(modo="prototipo"), persona_id=persona, semilla=semilla)
    vistos = Counter()
    while min(vistos[t] for t in PercepcionFalsa.TIPOS) < ejemplos:
        for o in perc.observar().objetos:
            tipo = perc.tipo_oculto[o.id]
            if vistos[tipo] < ejemplos:
                perc.ensenar(persona, o.id, tipo)
                vistos[tipo] += 1
    return perc


def situacion(perc: PercepcionFalsa, persona: str, rng: random.Random):
    """Una mesa nueva y UN objeto sobre el que el robot tiene que decidir."""
    escena = perc.observar()
    while not escena.objetos:
        escena = perc.observar()
    escena.t = HORA_FIJA                                 # la hora del reloj no debe cambiar los resultados
    obj = rng.choice(escena.objetos)
    tipos = {perc.tipo_oculto[o.id] for o in escena.objetos}
    return Contexto(escena=escena, persona_id=persona, eventos_recientes=[]), obj.id, perc.tipo_oculto[obj.id], tipos


def mejor_accion(modelo: PreferenciasRed, ctx, objeto_id) -> Accion:
    p = modelo.puntajes(ctx, objeto_id)
    return max(p, key=p.get)


def evaluar(modelo, usuario: UsuarioSimulado, prueba) -> float:
    return float(np.mean([mejor_accion(modelo, c, o) == usuario.accion_preferida(t, ts)
                          for c, o, t, ts in prueba]))


def distancia_pesos(a: PreferenciasRed, b: PreferenciasRed) -> float:
    with torch.no_grad():
        return float(sum(((pa - pb) ** 2).sum() for pa, pb in zip(a.red.parameters(), b.red.parameters())).sqrt())


# ---------------------------------------------------------------- una corrida
def correr(usuarios: dict[str, UsuarioSimulado], semilla: int, n: int, cada: int, eps: float,
           n_prueba: int = 200):
    """Las dos personas ven las MISMAS mesas (misma semilla); solo cambia su gusto."""
    rng_mesa = {u: random.Random(semilla) for u in usuarios}            # mismas mesas y objetos
    rng_robot = {u: random.Random(semilla + 1) for u in usuarios}       # exploracion del robot
    mundos = {u: mundo_ensenado(u, semilla) for u in usuarios}
    pruebas = {}
    for u in usuarios:                                   # situaciones nuevas, solo para medir
        perc_prueba = PercepcionFalsa(mundos[u].reconocedor, persona_id=u, semilla=10_000 + semilla)
        r = random.Random(10_000 + semilla)
        pruebas[u] = [situacion(perc_prueba, u, r) for _ in range(n_prueba)]

    base = PreferenciasRed(semilla=semilla)
    modelos = {u: base.clonar() for u in usuarios}      # el mismo robot de partida para todos
    curvas = {u: [evaluar(modelos[u], usuarios[u], pruebas[u])] for u in usuarios}
    aprobaciones = {u: [] for u in usuarios}
    distancia = [0.0]
    for i in range(1, n + 1):
        for u, usuario in usuarios.items():
            ctx, oid, tipo, tipos = situacion(mundos[u], u, rng_mesa[u])
            accion = modelos[u].elegir_accion(ctx, oid, exploracion=eps, rng=rng_robot[u])
            ok = usuario.calificar(tipo, tipos, accion)
            modelos[u].actualizar(ctx, oid, accion, ok)
            aprobaciones[u].append(ok)
            if i % cada == 0:
                curvas[u].append(evaluar(modelos[u], usuario, pruebas[u]))
        if i % cada == 0:
            a, b = list(modelos.values())[:2]
            distancia.append(distancia_pesos(a, b))

    cruce = {u: {v: evaluar(modelos[u], usuarios[v], pruebas[u]) for v in usuarios} for u in usuarios}
    return dict(curvas=curvas, aprobaciones=aprobaciones, distancia=distancia, cruce=cruce,
                modelos=modelos, pruebas=pruebas)


def mapa_preferencias(modelo, usuario, prueba, filas):
    """Puntaje promedio de cada accion por fila (tipo, o tipo con/sin el objeto de la condicion)."""
    M = np.full((len(filas), len(ACCIONES)), np.nan)
    preferida = [None] * len(filas)
    for i, (tipo, cond, hay) in enumerate(filas):
        casos = [(c, o, t, ts) for c, o, t, ts in prueba if t == tipo and (cond is None or (cond in ts) == hay)]
        if casos:
            M[i] = np.mean([[modelo.puntajes(c, o)[a] for a in ACCIONES] for c, o, _, _ in casos], axis=0)
            preferida[i] = ACCIONES.index(usuario.accion_preferida(tipo, casos[0][3]))
    return M, preferida


# ---------------------------------------------------------------- graficas
def graficar(res, usuarios, n, cada):
    from comun import SERIE, TEXTO, TEXTO_2, SECUENCIAL, estilo, guardar_figura
    plt = estilo()
    x = np.arange(0, n + 1, cada)
    nombres = list(usuarios)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    for k, u in enumerate(nombres):
        Y = np.array([r["curvas"][u] for r in res]) * 100
        m = Y.mean(0)
        ax1.fill_between(x, np.percentile(Y, 10, axis=0), np.percentile(Y, 90, axis=0), color=SERIE[k], alpha=0.15, lw=0)
        ax1.plot(x, m, color=SERIE[k], label=f"{u}: {usuarios[u].descripcion}")
        ax1.annotate(f"{m[-1]:.0f}%", (x[-1], m[-1]), xytext=(4, 6 if k == 0 else -6), textcoords="offset points",
                     va="center", fontsize=8.5, color=TEXTO)
    ax1.axhline(80, color=TEXTO_2, lw=1, ls="--")
    ax1.text(1, 81.5, "meta 80%", color=TEXTO_2, fontsize=8)
    ax1.axhline(25, color=TEXTO_2, lw=0.8, ls=":")
    ax1.text(1, 26.5, "al azar (1 de 4 acciones)", color=TEXTO_2, fontsize=8)
    ax1.axvline(30, color=TEXTO_2, lw=0.8, ls=":")
    ax1.set(xlabel="interacciones (bien / mal)", ylabel="aciertos en situaciones nuevas (%)",
            ylim=(0, 104), xlim=(0, n + 8), title="El robot aprende el gusto de cada persona")
    ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.16), fontsize=8)
    D = np.array([r["distancia"] for r in res])
    ax2.fill_between(x, np.percentile(D, 10, axis=0), np.percentile(D, 90, axis=0), color=SERIE[2], alpha=0.15, lw=0)
    ax2.plot(x, D.mean(0), color=SERIE[2])
    ax2.set(xlabel="interacciones", ylabel="distancia entre los pesos (norma L2)",
            title="Las dos copias del robot se separan")
    fig.text(0.01, -0.2, f"Promedio de {len(res)} semillas; banda: percentiles 10-90. "
             "Las dos copias parten del mismo robot y ven las mismas mesas.", fontsize=8, color=TEXTO_2)
    guardar_figura(fig, "p2_preferencias_curvas.png")
    plt.close(fig)

    # Mapa de lo que aprendio cada copia + matriz cruzada
    tipos = list(PercepcionFalsa.TIPOS)
    filas = []
    conds = {c["tipo"]: c["si_hay"] for us in usuarios.values() for c in us.condiciones}
    for t in tipos:
        if t in conds:
            filas += [(t, conds[t], True), (t, conds[t], False)]
        else:
            filas.append((t, None, None))
    etiquetas = [t if c is None else f"{t} ({'con' if h else 'sin'} {c})" for t, c, h in filas]
    r0 = res[0]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1, 1, 0.62]})
    for ax, u in zip(axes[:2], nombres):
        M, pref = mapa_preferencias(r0["modelos"][u], usuarios[u], r0["pruebas"][u], filas)
        ax.imshow(M, cmap=SECUENCIAL, vmin=0, vmax=1, aspect="auto")
        for i in range(len(filas)):
            for j in range(len(ACCIONES)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > 0.6 else TEXTO,
                            fontweight="bold" if pref[i] == j else "normal")
            if pref[i] is not None:
                ax.add_patch(plt.Rectangle((pref[i] - 0.5, i - 0.5), 1, 1, fill=False, ec=SERIE[1], lw=2))
        ax.set_xticks(range(len(ACCIONES)), [a.value for a in ACCIONES])
        ax.set_yticks(range(len(filas)), etiquetas if ax is axes[0] else [""] * len(filas))
        ax.grid(False)
        ax.set_title(f"Robot de '{u}'")
    C = np.array([[np.mean([r["cruce"][u][v] for r in res]) * 100 for v in nombres] for u in nombres])
    ax = axes[2]
    ax.imshow(C, cmap=SECUENCIAL, vmin=0, vmax=100)
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            ax.text(j, i, f"{C[i, j]:.0f}%", ha="center", va="center", fontsize=10,
                    color="white" if C[i, j] > 60 else TEXTO)
    ax.set_xticks(range(len(nombres)), nombres)
    ax.set_yticks(range(len(nombres)), [f"robot de\n{u}" for u in nombres])
    ax.set_xlabel("gusto con el que se evalua")
    ax.grid(False)
    ax.set_title("Aciertos cruzados")
    fig.text(0.01, -0.04, "Izquierda: probabilidad de que le guste cada accion (semilla 0, situaciones nuevas); "
             "recuadro naranja = lo que esa persona prefiere. Derecha: promedio de todas las semillas.",
             fontsize=8, color=TEXTO_2)
    fig.tight_layout()
    guardar_figura(fig, "p2_preferencias_dos_robots.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=10)
    ap.add_argument("--interacciones", type=int, default=100)
    ap.add_argument("--cada", type=int, default=5, help="medir cada tantas interacciones")
    ap.add_argument("--exploracion", type=float, default=0.1, help="probabilidad de probar una accion al azar")
    ap.add_argument("--sin-graficas", action="store_true")
    args = ap.parse_args()

    usuarios = UsuarioSimulado.cargar_todos()
    res = []
    for s in range(args.semillas):
        r = correr(usuarios, s, args.interacciones, args.cada, args.exploracion)
        res.append(r)
        print(f"semilla {s}: " + "  ".join(f"{u}={r['curvas'][u][-1]:.0%}" for u in usuarios))

    x = list(range(0, args.interacciones + 1, args.cada))
    resumen = {"semillas": args.semillas, "interacciones": args.interacciones, "x": x, "usuarios": {}}
    for u in usuarios:
        Y = np.array([r["curvas"][u] for r in res])
        m = Y.mean(0)
        pasa = next((xi for xi, v in zip(x, m) if v >= 0.8), None)
        resumen["usuarios"][u] = {
            "descripcion": usuarios[u].descripcion,
            "aciertos_promedio": m.round(4).tolist(),
            "aciertos_a_las_30": float(m[x.index(30)]) if 30 in x else None,
            "aciertos_final": float(m[-1]),
            "primera_vez_80": pasa,
            "aprobaciones_en_linea": float(np.mean([np.mean(r["aprobaciones"][u]) for r in res])),
            "cruce": {v: float(np.mean([r["cruce"][u][v] for r in res])) for v in usuarios},
        }
    resumen["distancia_pesos_final"] = float(np.mean([r["distancia"][-1] for r in res]))
    RESULTADOS = comun.RESULTADOS
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    (RESULTADOS / "p2_preferencias.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    for u, d in resumen["usuarios"].items():
        print(f"{u}: a las 30 = {d['aciertos_a_las_30']:.0%}, final = {d['aciertos_final']:.0%}, "
              f"supera 80% desde la interaccion {d['primera_vez_80']}, cruce = "
              + ", ".join(f"{v}:{a:.0%}" for v, a in d["cruce"].items()))
    if not args.sin_graficas:
        graficar(res, usuarios, args.interacciones, args.cada)


if __name__ == "__main__":
    main()
