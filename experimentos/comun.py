"""Rutas y estilo comun de las graficas de los experimentos de P2."""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:                      # para correrlo como: python experimentos/xxx.py
    sys.path.insert(0, str(RAIZ))

FIGURAS = RAIZ / "docs" / "figuras"
RESULTADOS = RAIZ / "experimentos" / "resultados"

# Paleta categorica validada (daltonismo) y tintas de texto
SERIE = ["#2a78d6", "#eb6834", "#1baf7a"]          # azul, naranja, aqua (en ese orden, siempre)
TEXTO, TEXTO_2, GRILLA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SECUENCIAL = "Blues"


def estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "axes.edgecolor": GRILLA, "axes.labelcolor": TEXTO_2, "axes.titlecolor": TEXTO,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "xtick.color": TEXTO_2, "ytick.color": TEXTO_2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "grid.color": GRILLA, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5, "text.color": TEXTO,
        "lines.linewidth": 2, "font.family": "DejaVu Sans",
    })
    return plt


def guardar_figura(fig, nombre: str):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS / nombre
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    print(f"  figura: {ruta.relative_to(RAIZ)}")
