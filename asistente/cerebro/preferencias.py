"""
Modelo de preferencias de UNA persona (P2 · Parte 2). Cumple ModeloPreferencias (contratos.py).

Idea: dada la situacion y un objeto, una red da la probabilidad de que a esta persona le guste
cada accion. Cada "bien" o "mal" es un ejemplo de entrenamiento.

  vector()     convierte (contexto, objeto) en numeros. Sin nombres escritos a mano: el aspecto
               del objeto sale de su embedding, y los nombres aprendidos ocupan "casillas" que se
               van llenando con lo que la persona ensena ("es:lapicero rojo", "hay:cuaderno verde").
  red          MLP pequena (PyTorch) con 4 salidas sigmoide, una por accion.
  actualizar() guarda el ejemplo en un buffer y reentrena unos pasos con todo el buffer
               (entropia cruzada binaria SOLO sobre la accion que se califico).

Uso:
    p = PreferenciasRed()
    p.puntajes(contexto, "obj_2")  -> {Accion.APARTAR: 0.81, Accion.TRAER: 0.10, ...}
    p.actualizar(contexto, "obj_2", Accion.APARTAR, aprobado=True)
    nuevo = p.clonar()             # copia independiente (otra persona, otro robot)
    p.guardar("modelos/pref_persona_A.pt")   # + pref_persona_A_pesos.npz con la historia de pesos (P1)
"""
from __future__ import annotations

import copy
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..configuracion import cargar_config
from ..tipos import DESCONOCIDO, Accion, Contexto
from .contratos import ModeloPreferencias

ACCIONES = list(Accion)                  # orden de las salidas de la red
SEGUNDOS_RECIENTE = 30.0                 # "aparecio hace poco"

CARACTERISTICAS_BASE = [
    "sin_objeto",            # 1 si la decision no es sobre un objeto
    "obj_conocido",          # 1 si la persona ya le enseno su nombre
    "obj_confianza",         # que tan seguro esta del nombre
    "dist_persona",          # distancia del objeto a la zona de la persona (0 a 1)
    "dist_apartar",          # distancia del objeto a la zona de apartar (0 a 1)
    "dist_carrito",          # distancia del objeto al carrito (0 a 1)
    "en_zona_persona",       # el objeto ya esta frente a la persona
    "en_zona_apartar",       # el objeto ya esta apartado
    "n_objetos",             # cuantos objetos hay en la mesa (/10)
    "hay_persona",           # hay alguien usando el robot
    "aparecio_hace_poco",    # el objeto aparecio en los ultimos 30 s
    "hay_pedido",            # hubo una orden explicita
    "pedido_es_este_objeto", # la orden menciona este objeto
    "hora_sin", "hora_cos",  # hora del dia (circular)
]


def _red(entradas: int, ocultas: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(entradas, ocultas), nn.ReLU(),
                         nn.Linear(ocultas, ocultas), nn.ReLU(),
                         nn.Linear(ocultas, len(ACCIONES)))


class PreferenciasRed(ModeloPreferencias):
    """Una instancia por persona. Empieza sin saber nada (todas las acciones ~0.5)."""

    def __init__(self, dim_aspecto: int = 8, max_nombres: int = 12, ocultas: int = 64,
                 pasos_por_actualizacion: int = 60, lr: float = 0.01, tam_buffer: int = 400,
                 semilla: int = 0, guardar_historial_pesos: bool = True):
        cfg = cargar_config()
        cart = cfg.get("cartulina", {})
        zonas = cfg.get("zonas", {})
        self.diagonal = math.hypot(cart.get("ancho", 0.65), cart.get("alto", 0.50))
        self.zona_persona = list(zonas.get("persona", [-0.25, 0.17, 0.07]))
        self.zona_apartar = list(zonas.get("apartar", [0.25, 0.17, 0.07]))

        self.dim_aspecto, self.max_nombres, self.ocultas = dim_aspecto, max_nombres, ocultas
        self.pasos, self.lr, self.semilla = pasos_por_actualizacion, lr, semilla
        self.vocabulario: list[str] = []          # nombres aprendidos, en el orden en que llegaron
        self._proyecciones: dict[int, np.ndarray] = {}

        torch.manual_seed(semilla)
        self.red = _red(self.n_entradas, ocultas)
        self.opt = torch.optim.Adam(self.red.parameters(), lr=lr, weight_decay=1e-4)
        self.buffer: deque[tuple[list[float], int, float]] = deque(maxlen=tam_buffer)
        self.conteos: dict[tuple[str, str], int] = {}     # (nombre del objeto, accion) -> veces calificada
        self.n_actualizaciones = 0
        self.guardar_historial = guardar_historial_pesos
        self.historial_pesos: list[dict] = [self._foto()] if guardar_historial_pesos else []

    # ---------------------------------------------------------------- contrato
    def puntajes(self, contexto: Contexto, objeto_id: str | None) -> dict[Accion, float]:
        p = self.predecir_vectores(np.asarray([self.vector(contexto, objeto_id)]))[0]
        return {a: float(v) for a, v in zip(ACCIONES, p)}

    def actualizar(self, contexto: Contexto, objeto_id: str | None, accion: Accion,
                   aprobado: bool) -> None:
        self.buffer.append((self.vector(contexto, objeto_id), ACCIONES.index(Accion(accion)),
                            1.0 if aprobado else 0.0))
        clave = (self._nombre_de(contexto, objeto_id), Accion(accion).value)
        self.conteos[clave] = self.conteos.get(clave, 0) + 1
        X = torch.tensor([b[0] for b in self.buffer], dtype=torch.float32)
        A = torch.tensor([b[1] for b in self.buffer]).unsqueeze(1)
        Y = torch.tensor([b[2] for b in self.buffer], dtype=torch.float32).unsqueeze(1)
        perdida = nn.BCEWithLogitsLoss()
        self.red.train()
        for _ in range(self.pasos):
            self.opt.zero_grad()
            perdida(self.red(X).gather(1, A), Y).backward()    # solo la accion calificada
            self.opt.step()
        self.red.eval()
        self.n_actualizaciones += 1
        if self.guardar_historial:
            self.historial_pesos.append(self._foto())

    def vector(self, contexto: Contexto, objeto_id: str | None) -> list[float]:
        escena = contexto.escena
        objetos = escena.objetos
        obj = next((o for o in objetos if o.id == objeto_id), None) if objeto_id else None
        for o in objetos:                                  # cada nombre nuevo recibe su casilla
            self._casilla(o.nombre)

        base = dict.fromkeys(CARACTERISTICAS_BASE, 0.0)
        base["n_objetos"] = min(len(objetos), 10) / 10
        base["hay_persona"] = float(escena.persona_id is not None)
        base["hay_pedido"] = float(contexto.pedido is not None)
        hora = time.localtime(escena.t)
        angulo = 2 * math.pi * (hora.tm_hour + hora.tm_min / 60) / 24
        base["hora_sin"], base["hora_cos"] = math.sin(angulo), math.cos(angulo)
        aspecto = [0.0] * self.dim_aspecto
        es = [0.0] * (self.max_nombres + 1)
        if obj is None:
            base["sin_objeto"] = 1.0
        else:
            conocido = obj.nombre != DESCONOCIDO
            base["obj_conocido"] = float(conocido)
            base["obj_confianza"] = float(obj.confianza) if conocido else 0.0
            base["dist_persona"] = self._dist(obj.x, obj.y, *self.zona_persona[:2])
            base["dist_apartar"] = self._dist(obj.x, obj.y, *self.zona_apartar[:2])
            base["dist_carrito"] = self._dist(obj.x, obj.y, escena.carrito.x, escena.carrito.y)
            base["en_zona_persona"] = float(self._en_zona(obj.x, obj.y, self.zona_persona))
            base["en_zona_apartar"] = float(self._en_zona(obj.x, obj.y, self.zona_apartar))
            base["aparecio_hace_poco"] = float(conocido and any(
                e.tipo == "aparecio" and e.detalle == obj.nombre and escena.t - e.t <= SEGUNDOS_RECIENTE
                for e in contexto.eventos_recientes))
            base["pedido_es_este_objeto"] = float(contexto.pedido is not None and conocido
                                                  and contexto.pedido.nombre == obj.nombre)
            if obj.embedding is not None:
                aspecto = self._aspecto(obj.embedding)
            if conocido:
                es[self._casilla(obj.nombre)] = 1.0
        hay = [0.0] * (self.max_nombres + 1)
        for o in objetos:
            if o.nombre != DESCONOCIDO:
                hay[self._casilla(o.nombre)] = 1.0
        return list(base.values()) + aspecto + es + hay

    def nombres_caracteristicas(self) -> list[str]:
        casillas = [self.vocabulario[i] if i < len(self.vocabulario) else f"(libre_{i})"
                    for i in range(self.max_nombres)] + ["otro_nombre"]
        return (CARACTERISTICAS_BASE + [f"aspecto_{i}" for i in range(self.dim_aspecto)]
                + [f"es:{c}" for c in casillas] + [f"hay:{c}" for c in casillas])

    def guardar(self, ruta: str) -> None:
        """ruta: archivo .pt (o carpeta -> carpeta/preferencias.pt). Al lado deja
        <nombre>_pesos.npz con la historia de pesos para la visualizacion de P1."""
        ruta = self._ruta_archivo(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        estado = {
            "version": 1,
            "hiper": {"dim_aspecto": self.dim_aspecto, "max_nombres": self.max_nombres,
                      "ocultas": self.ocultas, "pasos_por_actualizacion": self.pasos, "lr": self.lr,
                      "tam_buffer": self.buffer.maxlen, "semilla": self.semilla},
            "vocabulario": list(self.vocabulario),
            "red": self.red.state_dict(),
            "buffer_X": torch.tensor([b[0] for b in self.buffer], dtype=torch.float32),
            "buffer_A": torch.tensor([b[1] for b in self.buffer], dtype=torch.long),
            "buffer_Y": torch.tensor([b[2] for b in self.buffer], dtype=torch.float32),
            "n_actualizaciones": self.n_actualizaciones,
            "conteos": [[n, a, c] for (n, a), c in self.conteos.items()],
        }
        torch.save(estado, ruta)
        if self.historial_pesos:
            self.exportar_historial_pesos(ruta.with_name(ruta.stem + "_pesos.npz"))

    def cargar(self, ruta: str) -> None:
        ruta = self._ruta_archivo(ruta)
        estado = torch.load(ruta, map_location="cpu", weights_only=True)
        h = estado["hiper"]
        self.dim_aspecto, self.max_nombres, self.ocultas = h["dim_aspecto"], h["max_nombres"], h["ocultas"]
        self.pasos, self.lr, self.semilla = h["pasos_por_actualizacion"], h["lr"], h["semilla"]
        self.vocabulario = list(estado["vocabulario"])
        self._proyecciones = {}
        self.red = _red(self.n_entradas, self.ocultas)
        self.red.load_state_dict(estado["red"])
        self.red.eval()
        self.opt = torch.optim.Adam(self.red.parameters(), lr=self.lr, weight_decay=1e-4)
        self.buffer = deque(zip(estado["buffer_X"].tolist(), estado["buffer_A"].tolist(),
                                estado["buffer_Y"].tolist()), maxlen=h["tam_buffer"])
        self.n_actualizaciones = estado["n_actualizaciones"]
        self.conteos = {(n, a): c for n, a, c in estado.get("conteos", [])}
        npz = ruta.with_name(ruta.stem + "_pesos.npz")
        self.historial_pesos = cargar_historial_pesos(npz)["fotos"] if npz.exists() else [self._foto()]

    def clonar(self) -> "PreferenciasRed":
        return copy.deepcopy(self)

    # ---------------------------------------------------------------- extras (P3, P4 y P1)
    def elegir_accion(self, contexto: Contexto, objeto_id: str | None, exploracion: float = 0.1,
                      rng=None) -> Accion:
        """Politica sugerida para P3 y para los experimentos:
          - con probabilidad `exploracion`, una accion al azar;
          - si alguna accion probablemente le guste (>= 0.5), la mejor;
          - si cree que NADA le va a gustar, la que menos ha calificado con ese objeto
            (si no, la red generaliza "mal" y nunca descubre la accion correcta)."""
        import random as _random
        rng = rng or _random
        if rng.random() < exploracion:
            return rng.choice(ACCIONES)
        p = self.puntajes(contexto, objeto_id)
        if max(p.values()) >= 0.5:
            return max(p, key=p.get)
        nombre = self._nombre_de(contexto, objeto_id)
        return min(ACCIONES, key=lambda a: (self.conteos.get((nombre, a.value), 0), -p[a]))

    @property
    def n_entradas(self) -> int:
        return len(CARACTERISTICAS_BASE) + self.dim_aspecto + 2 * (self.max_nombres + 1)

    def predecir_vectores(self, X) -> np.ndarray:
        """Matriz (n, n_entradas) -> probabilidades (n, 4) en el orden de ACCIONES.
        Es la funcion que P4 le pasa a shap.KernelExplainer."""
        with torch.no_grad():
            return torch.sigmoid(self.red(torch.as_tensor(np.asarray(X), dtype=torch.float32))).numpy()

    def fondo(self, n: int = 50) -> np.ndarray:
        """Hasta n vectores del historial (ejemplos de fondo para SHAP)."""
        if not self.buffer:
            return np.zeros((1, self.n_entradas), dtype=np.float32)
        X = np.asarray([b[0] for b in self.buffer], dtype=np.float32)
        return X[-n:]

    def exportar_historial_pesos(self, ruta) -> None:
        """Guarda la historia de pesos en un .npz que se lee solo con numpy:
          W0, b0, W1, b1, ...   apilados: W{j} tiene forma (n_fotos, salidas, entradas)
          n_actualizacion       (n_fotos,) cuantas actualizaciones llevaba en cada foto
          nombres_entrada       nombre de cada entrada (las de la capa 0)
          nombres_salida        nombre de cada salida (las acciones)"""
        fotos = self.historial_pesos
        arreglos = {"n_actualizacion": np.array([f["n"] for f in fotos]),
                    "nombres_entrada": np.array(self.nombres_caracteristicas()),
                    "nombres_salida": np.array([a.value for a in ACCIONES])}
        for j in range(len(fotos[0]["W"])):
            arreglos[f"W{j}"] = np.stack([f["W"][j] for f in fotos])
            arreglos[f"b{j}"] = np.stack([f["b"][j] for f in fotos])
        np.savez_compressed(ruta, **arreglos)

    # ---------------------------------------------------------------- internos
    def _foto(self) -> dict:
        lineales = [m for m in self.red if isinstance(m, nn.Linear)]
        return {"n": self.n_actualizaciones,
                "W": [m.weight.detach().numpy().copy() for m in lineales],
                "b": [m.bias.detach().numpy().copy() for m in lineales]}

    @staticmethod
    def _nombre_de(contexto: Contexto, objeto_id: str | None) -> str:
        obj = next((o for o in contexto.escena.objetos if o.id == objeto_id), None) if objeto_id else None
        return "(sin objeto)" if obj is None else obj.nombre

    def _casilla(self, nombre: str) -> int:
        """Indice de la casilla de ese nombre. Los nombres nuevos toman la siguiente casilla libre."""
        if nombre == DESCONOCIDO:
            return self.max_nombres
        if nombre not in self.vocabulario and len(self.vocabulario) < self.max_nombres:
            self.vocabulario.append(nombre)
        return self.vocabulario.index(nombre) if nombre in self.vocabulario else self.max_nombres

    def _aspecto(self, embedding) -> list[float]:
        """Resumen de largo fijo del embedding (normalizado). Si es largo (512 de la CNN) se
        proyecta con una matriz al azar fija (semilla); si es corto (simulacion) se rellena."""
        v = np.asarray(embedding, dtype=np.float64).ravel()
        v = v / (np.linalg.norm(v) + 1e-9)
        if len(v) <= self.dim_aspecto:
            return list(np.pad(v, (0, self.dim_aspecto - len(v))))
        if len(v) not in self._proyecciones:
            rng = np.random.default_rng(self.semilla + len(v))
            self._proyecciones[len(v)] = rng.normal(0, 1 / math.sqrt(self.dim_aspecto),
                                                    size=(len(v), self.dim_aspecto))
        return list(v @ self._proyecciones[len(v)])

    def _dist(self, x1, y1, x2, y2) -> float:
        return min(math.hypot(x1 - x2, y1 - y2) / self.diagonal, 1.0)

    @staticmethod
    def _en_zona(x, y, zona) -> bool:
        cx, cy, medio = zona
        return abs(x - cx) <= medio and abs(y - cy) <= medio

    @staticmethod
    def _ruta_archivo(ruta) -> Path:
        ruta = Path(ruta)
        return ruta / "preferencias.pt" if ruta.suffix == "" else ruta


def cargar_historial_pesos(ruta) -> dict:
    """Lee el .npz de exportar_historial_pesos(). Devuelve
    {"fotos": [{"n", "W": [...], "b": [...]}, ...], "nombres_entrada": [...], "nombres_salida": [...]}.
    Solo necesita numpy (P1 lo usa para dibujar la red)."""
    d = np.load(ruta)
    capas = sum(1 for k in d.files if k.startswith("W"))
    fotos = [{"n": int(n), "W": [d[f"W{j}"][i] for j in range(capas)], "b": [d[f"b{j}"][i] for j in range(capas)]}
             for i, n in enumerate(d["n_actualizacion"])]
    return {"fotos": fotos, "nombres_entrada": d["nombres_entrada"].tolist(),
            "nombres_salida": d["nombres_salida"].tolist()}
