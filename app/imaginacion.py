from __future__ import annotations

import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import yaml                       # noqa: E402
from matplotlib import patches    # noqa: E402

from asistente.tipos import DESCONOCIDO, Accion, Decision, Escena   # noqa: E402

RUTA_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
DISTANCIA_DETRAS = 0.17     # la misma que usa Empujador en sim_mujoco.py
COLOR_ZONA = {"apartar": "#3a9d4f", "persona": "#3a62c9"}
COLOR_PLAN = "#e8710a"


def _cfg(cfg):
    if cfg is not None:
        return cfg
    with open(RUTA_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _color(nombre: str):
    if nombre == DESCONOCIDO:
        return "#9a9a9a"
    return plt.get_cmap("tab10")(zlib.crc32(nombre.encode()) % 10)


def _flecha(ax, a, b, estilo="-", ancho=2.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=COLOR_PLAN, lw=ancho,
                                                     linestyle=estilo, shrinkA=0, shrinkB=4))


def _texto_decision(decision, objeto) -> str:
    if decision is None:
        return "Sin plan"
    que = decision.accion.value.upper()
    if objeto is not None:
        que += f" {'¿?' if objeto.nombre == DESCONOCIDO else objeto.nombre}"
    texto = f"{que}  (confianza {decision.confianza:.0%})"
    if decision.preguntar_antes:
        texto += "  · pregunta antes"
    return texto


def dibujar_imaginacion(escena: Escena, decision: Decision | None = None, cfg: dict | None = None,
                        ancho_px: int = 720) -> np.ndarray:
    """Devuelve una imagen RGB (uint8)."""
    cfg = _cfg(cfg)
    ancho, alto = cfg["cartulina"]["ancho"], cfg["cartulina"]["alto"]
    margen = 0.06
    fig_w = 7.2
    fig = plt.figure(figsize=(fig_w, fig_w * (alto + 2 * margen) / (ancho + 2 * margen) + 0.5), dpi=ancho_px / fig_w)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.86])
    ax.set_xlim(-ancho / 2 - margen, ancho / 2 + margen)
    ax.set_ylim(-alto / 2 - margen, alto / 2 + margen)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(patches.Rectangle((-ancho / 2, -alto / 2), ancho, alto, fc="#f4efe1", ec="#333", lw=1.5))
    for nombre in ("apartar", "persona"):
        x, y, m = cfg["zonas"][nombre]
        ax.add_patch(patches.Rectangle((x - m, y - m), 2 * m, 2 * m, fc=COLOR_ZONA[nombre], alpha=0.25,
                                       ec=COLOR_ZONA[nombre], lw=1))
        ax.text(x, y + m + 0.008, nombre, ha="center", va="bottom", fontsize=8, color=COLOR_ZONA[nombre])

    objeto = None
    if decision is not None and decision.objeto_id is not None:
        objeto = next((o for o in escena.objetos if o.id == decision.objeto_id), None)

    for o in escena.objetos:
        es_obj = objeto is not None and o.id == objeto.id
        ax.add_patch(patches.Circle((o.x, o.y), 0.018, fc=_color(o.nombre), ec="#222", lw=0.8, zorder=3))
        if es_obj:
            ax.add_patch(patches.Circle((o.x, o.y), 0.03, fc="none", ec=COLOR_PLAN, lw=2.5, zorder=4))
        etiqueta = "¿?" if o.nombre == DESCONOCIDO else f"{o.nombre} ({o.confianza:.0%})"
        ax.text(o.x, o.y - 0.03, etiqueta, ha="center", va="top", fontsize=8, zorder=5,
                fontweight="bold" if es_obj else "normal",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    c = escena.carrito
    frente = np.array([np.cos(c.angulo), np.sin(c.angulo)])
    lado = np.array([-frente[1], frente[0]])
    pc = np.array([c.x, c.y])
    tri = [pc + 0.05 * frente, pc - 0.035 * frente + 0.035 * lado, pc - 0.035 * frente - 0.035 * lado]
    ax.add_patch(patches.Polygon(tri, fc="#2759b8", ec="#10295e", lw=1, zorder=6))

    if decision is not None:
        if decision.accion in (Accion.APARTAR, Accion.TRAER) and objeto is not None:
            zx, zy, _ = cfg["zonas"]["apartar" if decision.accion == Accion.APARTAR else "persona"]
            po, pz = np.array([objeto.x, objeto.y]), np.array([zx, zy])
            u = (pz - po) / max(np.linalg.norm(pz - po), 1e-6)
            detras = po - DISTANCIA_DETRAS * u
            _flecha(ax, pc, detras, estilo="--", ancho=1.5)
            ax.plot(*detras, "o", ms=4, color=COLOR_PLAN, zorder=5)
            _flecha(ax, po, pz, ancho=2.5)
        elif decision.accion == Accion.ACERCARSE:
            zx, zy, _ = cfg["zonas"]["persona"]
            _flecha(ax, pc, (zx, zy), estilo="--", ancho=2.0)

    fig.text(0.5, 0.965, "Lo que imagino", ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.915, _texto_decision(decision, objeto), ha="center", va="top", fontsize=10, color="#444")

    fig.canvas.draw()
    imagen = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return imagen


def vista_real(mundo, ancho_px: int = 640) -> np.ndarray:
    """Foto cenital de la simulacion (lo que de verdad hay)."""
    import mujoco
    ancho, alto = mundo.cfg["cartulina"]["ancho"], mundo.cfg["cartulina"]["alto"]
    alto_px = int(ancho_px * alto / ancho)
    with mujoco.Renderer(mundo.model, height=alto_px, width=ancho_px) as r:
        r.update_scene(mundo.data, camera="cenital")
        return r.render()


def _demo(carpeta: Path):
    from asistente.cuerpo_sim import CuerpoSim, MundoSim, PercepcionSim
    from asistente.mocks import ReconocedorFalso

    mundo = MundoSim(semilla=1)
    percepcion = PercepcionSim(mundo, ReconocedorFalso())
    escena = percepcion.observar()
    elegido = escena.objetos[0]
    percepcion.ensenar("persona_A", elegido.id, "mi objeto")   # el resto queda como "¿?"
    escena = percepcion.observar()
    decision = Decision(Accion.TRAER, elegido.id, confianza=0.72, preguntar_antes=True, motivo="demo")

    carpeta.mkdir(parents=True, exist_ok=True)
    plt.imsave(carpeta / "imaginacion_demo.png", dibujar_imaginacion(escena, decision))
    real = vista_real(mundo)
    plt.imsave(carpeta / "vista_real_antes.png", real)
    print(CuerpoSim(mundo).ejecutar(decision.accion, elegido, escena))
    plt.imsave(carpeta / "vista_real_despues.png", vista_real(mundo))
    print(f"Guardado en {carpeta}/")


if __name__ == "__main__":
    import sys
    _demo(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "figuras")
