"""
Reconocedor de objetos con pocos ejemplos, uno por persona (P2 · Parte 1).

Cada objeto se convierte en un EMBEDDING (un vector que describe como se ve). Cuando la persona
dice "esto es mi lapicero rojo", se guarda ese vector con ese nombre SOLO para esa persona.
Despues, un objeto nuevo se compara con lo guardado: si se parece lo suficiente se reconoce;
si no, es "desconocido". Lo que le ensena A no lo sabe B.

Dos versiones (se elige con `modo`, para compararlas en el documento):
  "prototipo"  v1: coseno contra el promedio (prototipo) de los ejemplos de cada nombre.
  "cabeza"     v2: una red pequena por persona (MLP) con una clase extra "desconocido",
               entrenada con los ejemplos de esa persona.

embedding(): resnet18 preentrenada en ImageNet sin su ultima capa -> 512 numeros
(transfer learning). En simulacion no hay imagenes: la percepcion pasa embeddings cortos ya
hechos, y ensenar/reconocer funcionan con cualquier largo de vector.

Uso:
    r = ReconocedorPocosEjemplos()                 # o modo="cabeza"
    e = r.embedding(recorte)                       # recorte: numpy alto x ancho x 3 (BGR)
    r.ensenar("persona_A", "lapicero rojo", e)
    r.reconocer("persona_A", e2)  -> ("lapicero rojo", 0.93)
    r.reconocer("persona_B", e2)  -> ("desconocido", 0.0)
"""
from __future__ import annotations

import json
import warnings
from collections import deque
from pathlib import Path

import numpy as np

from ..configuracion import cargar_config
from ..interfaces import Reconocedor
from ..tipos import DESCONOCIDO

MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)

REENTRENAR_CADA = 50     # v2: reentrenar las cabezas cada tantos embeddings nuevos vistos en la mesa


def coseno(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def limpiar_nombre(nombre: str) -> str:
    """'  Lapicero  Rojo ' -> 'lapicero rojo' (asi la voz y la pantalla coinciden)."""
    return " ".join(str(nombre).lower().split())


# ---------------------------------------------------------------- Embeddings (transfer learning)
class ExtractorEmbeddings:
    """Red preentrenada de torchvision sin su capa de clasificacion. Se carga la primera vez
    que se usa, asi la simulacion (que ya trae embeddings) no necesita torchvision."""

    DIMENSIONES = {"resnet18": 512, "mobilenet_v3_small": 576}

    def __init__(self, arquitectura: str = "resnet18", ruta_pesos: str | None = None,
                 bgr: bool = True, dispositivo: str = "cpu"):
        if arquitectura not in self.DIMENSIONES:
            raise ValueError(f"arquitectura debe ser una de {list(self.DIMENSIONES)}")
        self.arquitectura, self.ruta_pesos = arquitectura, ruta_pesos
        self.bgr, self.dispositivo = bgr, dispositivo      # bgr=True: los recortes vienen de OpenCV
        self.red = None

    def _cargar(self):
        import torch
        import torchvision.models as tvm
        constructor, pesos = {
            "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.DEFAULT),
            "mobilenet_v3_small": (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.DEFAULT),
        }[self.arquitectura]
        try:
            red = constructor(weights=pesos)
        except Exception as e:          # sin internet y sin los pesos en cache
            warnings.warn(f"No se pudieron cargar los pesos preentrenados ({e}); uso pesos al azar.")
            red = constructor(weights=None)
        if self.arquitectura == "resnet18":
            red.fc = torch.nn.Identity()
        else:
            red.classifier = torch.nn.Identity()
        if self.ruta_pesos:             # extractor ajustado por integracion (contrastivo), opcional
            red.load_state_dict(torch.load(self.ruta_pesos, map_location="cpu"), strict=False)
        self.red = red.eval().to(self.dispositivo)

    def preparar(self, recorte) -> np.ndarray:
        """numpy alto x ancho x 3 -> 3 x alto x ancho, normalizado como ImageNet."""
        img = np.asarray(recorte)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        img = img[..., :3]
        if self.bgr:
            img = img[..., ::-1]
        x = img.astype(np.float32)
        if np.issubdtype(img.dtype, np.integer):
            x /= 255.0
        x = (x - MEDIA_IMAGENET) / STD_IMAGENET
        return np.ascontiguousarray(x.transpose(2, 0, 1))

    def __call__(self, recortes: list) -> np.ndarray:
        """Lista de recortes -> matriz (n, 512). Cada recorte se lleva a 224 x 224."""
        import torch
        import torch.nn.functional as F
        if self.red is None:
            self._cargar()
        lote = [F.interpolate(torch.from_numpy(self.preparar(r))[None], size=(224, 224),
                              mode="bilinear", align_corners=False) for r in recortes]
        with torch.no_grad():
            return self.red(torch.cat(lote).to(self.dispositivo)).cpu().numpy()


# ---------------------------------------------------------------- v2: una cabeza por persona
def _distancias(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0.0))


def _radio(clases: list[np.ndarray], fondo: np.ndarray, todos: np.ndarray) -> float:
    """Que tan lejos puede caer otra vista del MISMO objeto (el ruido de la percepcion).
    Se estima con la distancia entre ejemplos del mismo nombre y entre vecinos del fondo."""
    escalas = []
    for X in clases:
        if len(X) >= 2:
            D = _distancias(X, X)
            escalas.append(float(np.median(D[np.triu_indices(len(X), 1)])))
    if len(fondo) >= 10:
        D = _distancias(fondo[-200:], fondo[-200:])
        np.fill_diagonal(D, np.inf)
        escalas.append(float(np.median(D.min(1))))
    tamano = float(np.median(np.linalg.norm(todos, axis=1)))
    if not escalas:
        return 0.25 * tamano
    return max(2.0 * float(np.median(escalas)), 0.1 * tamano)


class CabezaPersona:
    """MLP pequeno de UNA persona. Clases: [desconocido] + los nombres que le enseno.

    Con pocos ejemplos, el entrenamiento se arma asi:
      - positivos: cada ejemplo ensenado, mas copias con un poquito de ruido (aumento de datos);
      - "desconocido": lo que el robot ha visto en la mesa que NO se parece a lo ensenado
        (el fondo), mas puntos inventados alrededor de cada ejemplo, a 2-4 "radios" de distancia.
    """

    def __init__(self, nombres: list[str], dim: int, ocultas: int = 64):
        import torch.nn as nn
        self.nombres, self.dim = list(nombres), dim
        self.media = np.zeros(dim, dtype=np.float32)
        self.escala = np.ones(dim, dtype=np.float32)
        self.red = nn.Sequential(nn.Linear(dim, ocultas), nn.ReLU(), nn.Linear(ocultas, len(nombres) + 1))

    @classmethod
    def entrenar(cls, ejemplos: dict[str, list[np.ndarray]], fondo: list[np.ndarray],
                 semilla: int = 0, pasos: int = 200, copias: int = 20) -> "CabezaPersona":
        import torch
        dim = len(next(iter(ejemplos.values()))[0])
        clases = {n: np.stack([v for v in vs if len(v) == dim]) for n, vs in ejemplos.items()
                  if any(len(v) == dim for v in vs)}
        cabeza = cls(list(clases), dim)
        rng = np.random.default_rng(semilla)
        torch.manual_seed(semilla)

        P = np.concatenate(list(clases.values()))
        yP = np.concatenate([np.full(len(X), i + 1) for i, X in enumerate(clases.values())])
        F = np.stack([f for f in fondo if len(f) == dim]) if any(len(f) == dim for f in fondo) \
            else np.zeros((0, dim), dtype=np.float32)
        radio = _radio(list(clases.values()), F, P)
        if len(F):
            # Lo visto que no se parece a lo ensenado. Margen de 2 radios: con 1 solo ejemplo, otras
            # vistas del MISMO objeto pueden caer lejos y no deben entrenarse como "desconocido".
            F = F[_distancias(F, P).min(1) > 2 * radio]

        u = rng.normal(size=(len(P) * copias, dim))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        S = np.repeat(P, copias, 0) + u * rng.uniform(2, 4, size=(len(u), 1)) * radio
        A = np.repeat(P, copias, 0) + rng.normal(size=(len(u), dim)) * (radio / 2) / np.sqrt(dim)
        X = np.concatenate([P, A, F, S]).astype(np.float32)
        y = np.concatenate([yP, np.repeat(yP, copias), np.zeros(len(F) + len(S))]).astype(np.int64)

        cabeza.media = X.mean(0)
        sd = X.std(0)
        cabeza.escala = np.maximum(sd, 0.1 * sd.mean() + 1e-6).astype(np.float32)
        Xt = torch.from_numpy((X - cabeza.media) / cabeza.escala)
        yt = torch.from_numpy(y)
        cuenta = torch.bincount(yt, minlength=len(clases) + 1).float().clamp(min=1)
        perdida = torch.nn.CrossEntropyLoss(weight=len(yt) / (len(cuenta) * cuenta))   # clases balanceadas
        opt = torch.optim.Adam(cabeza.red.parameters(), lr=1e-2, weight_decay=1e-4)
        cabeza.red.train()
        for _ in range(pasos):
            opt.zero_grad()
            perdida(cabeza.red(Xt), yt).backward()
            opt.step()
        cabeza.red.eval()
        return cabeza

    def probabilidades(self, v: np.ndarray) -> dict[str, float]:
        import torch
        if len(v) != self.dim:
            return {}
        x = torch.from_numpy(((v - self.media) / self.escala).astype(np.float32))[None]
        with torch.no_grad():
            p = torch.softmax(self.red(x), dim=-1)[0].numpy()
        return {DESCONOCIDO: float(p[0]), **{n: float(p[i + 1]) for i, n in enumerate(self.nombres)}}

    def a_dict(self) -> dict:
        import torch
        return {"nombres": self.nombres, "dim": self.dim, "media": torch.from_numpy(self.media),
                "escala": torch.from_numpy(self.escala), "estado": self.red.state_dict()}

    @classmethod
    def desde_dict(cls, d: dict) -> "CabezaPersona":
        cabeza = cls(d["nombres"], d["dim"])
        cabeza.media, cabeza.escala = d["media"].numpy(), d["escala"].numpy()
        cabeza.red.load_state_dict(d["estado"])
        cabeza.red.eval()
        return cabeza


# ---------------------------------------------------------------- El reconocedor
class ReconocedorPocosEjemplos(Reconocedor):
    """Cumple el contrato Reconocedor. Ver la explicacion al inicio del archivo."""

    def __init__(self, modo: str = "prototipo", umbral: float | None = None,
                 arquitectura: str = "resnet18", ruta_pesos_extractor: str | None = None,
                 bgr: bool = True, max_fondo: int = 500, semilla: int = 0):
        if modo not in ("prototipo", "cabeza"):
            raise ValueError("modo debe ser 'prototipo' (v1) o 'cabeza' (v2)")
        cfg = cargar_config().get("reconocimiento", {})
        self.modo = modo
        self.umbral = float(umbral if umbral is not None else cfg.get("umbral_similitud", 0.85))
        self.semilla = semilla
        self.extractor = ExtractorEmbeddings(arquitectura, ruta_pesos_extractor, bgr)
        self.ejemplos: dict[str, dict[str, list[np.ndarray]]] = {}   # persona -> nombre -> vectores
        # Solo v2: lo que el robot ha visto en la mesa (sin nombres), para aprender que es "desconocido"
        self.fondo: deque[np.ndarray] = deque(maxlen=max_fondo)
        self.aprender_del_fondo = True     # en las evaluaciones se apaga, para no "mirar el examen"
        self._cabezas: dict[str, CabezaPersona] = {}
        self._fondo_nuevo = 0

    # ---- contrato
    def embedding(self, recorte) -> list[float]:
        return self.extractor([recorte])[0].tolist()

    def ensenar(self, persona_id, nombre, embedding):
        nombre = limpiar_nombre(nombre)
        if persona_id is None or not nombre or nombre == DESCONOCIDO or embedding is None:
            warnings.warn(f"Ensenanza ignorada (persona={persona_id!r}, nombre={nombre!r}).")
            return
        self.ejemplos.setdefault(persona_id, {}).setdefault(nombre, []).append(self._vector(embedding))
        self._cabezas.pop(persona_id, None)          # su cabeza se reentrena la proxima vez

    def reconocer(self, persona_id, embedding):
        v = self._vector(embedding)
        if self.modo == "cabeza" and self.aprender_del_fondo:
            self._al_fondo(v)
        p = self.puntajes(persona_id, v)
        p_desconocido = p.pop(DESCONOCIDO, 0.0)
        if not p:
            return DESCONOCIDO, 0.0
        nombre, conf = max(p.items(), key=lambda kv: kv[1])
        if conf < self.umbral or p_desconocido > conf:
            return DESCONOCIDO, conf
        return nombre, conf

    def nombres(self, persona_id):
        return list(self.ejemplos.get(persona_id, {}))

    def guardar(self, ruta):
        """Guarda todo en la carpeta `ruta`: reconocedor.json (ejemplos por persona, legible),
        fondo.npy y, en v2, cabezas.pt (la red de cada persona)."""
        carpeta = Path(ruta)
        carpeta.mkdir(parents=True, exist_ok=True)
        datos = {"version": 1, "modo": self.modo, "umbral": self.umbral,
                 "arquitectura": self.extractor.arquitectura,
                 "personas": {p: {n: [v.tolist() for v in vs] for n, vs in por_nombre.items()}
                              for p, por_nombre in self.ejemplos.items()}}
        (carpeta / "reconocedor.json").write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
        if self.fondo:
            dims = {len(f) for f in self.fondo}
            if len(dims) == 1:
                np.save(carpeta / "fondo.npy", np.stack(self.fondo))
        if self._cabezas:
            import torch
            torch.save({p: c.a_dict() for p, c in self._cabezas.items()}, carpeta / "cabezas.pt")

    def cargar(self, ruta):
        """Carga lo guardado por guardar(). El modo y el umbral son los del constructor
        (asi se pueden comparar v1 y v2 con los mismos ejemplos)."""
        carpeta = Path(ruta)
        datos = json.loads((carpeta / "reconocedor.json").read_text(encoding="utf-8"))
        self.ejemplos = {p: {n: [np.asarray(v, dtype=np.float32) for v in vs] for n, vs in por_nombre.items()}
                         for p, por_nombre in datos["personas"].items()}
        self.fondo.clear()
        if (carpeta / "fondo.npy").exists():
            self.fondo.extend(np.load(carpeta / "fondo.npy"))
        self._cabezas = {}
        if (carpeta / "cabezas.pt").exists():
            import torch
            guardadas = torch.load(carpeta / "cabezas.pt", map_location="cpu", weights_only=True)
            self._cabezas = {p: CabezaPersona.desde_dict(d) for p, d in guardadas.items()}

    # ---- extras (para la pantalla, el XAI y los experimentos)
    def embeddings(self, recortes) -> list[list[float]]:
        """Varios recortes de una vez (mas rapido que llamar embedding() uno por uno)."""
        return [v.tolist() for v in self.extractor(list(recortes))] if len(recortes) else []

    def puntajes(self, persona_id, embedding) -> dict[str, float]:
        """Que tanto se parece a cada nombre de esa persona (0 a 1).
        v1: coseno contra cada prototipo. v2: probabilidad de la cabeza (incluye 'desconocido')."""
        v = self._vector(embedding)
        por_nombre = self.ejemplos.get(persona_id) if persona_id is not None else None
        if not por_nombre:
            return {}
        if self.modo == "prototipo":
            return {n: max(0.0, min(1.0, coseno(v, proto)))
                    for n, proto in self.prototipos(persona_id).items() if len(proto) == len(v)}
        return self._cabeza(persona_id).probabilidades(v)

    def prototipos(self, persona_id) -> dict[str, np.ndarray]:
        """El promedio de los ejemplos de cada nombre (lo que 'recuerda' de cada objeto)."""
        return {n: np.mean(vs, axis=0) for n, vs in self.ejemplos.get(persona_id, {}).items()}

    def n_ejemplos(self, persona_id) -> dict[str, int]:
        return {n: len(vs) for n, vs in self.ejemplos.get(persona_id, {}).items()}

    def olvidar(self, persona_id, nombre: str | None = None) -> None:
        """Corregir: borra un nombre de esa persona (o todo lo suyo si nombre es None)."""
        if nombre is None:
            self.ejemplos.pop(persona_id, None)
        else:
            self.ejemplos.get(persona_id, {}).pop(limpiar_nombre(nombre), None)
        self._cabezas.pop(persona_id, None)

    # ---- internos
    @staticmethod
    def _vector(embedding) -> np.ndarray:
        return np.asarray(embedding, dtype=np.float32).ravel()

    def _al_fondo(self, v: np.ndarray):
        self.fondo.append(v)
        self._fondo_nuevo += 1
        if self._fondo_nuevo >= REENTRENAR_CADA:     # la mesa cambio: reentrenar con lo nuevo
            self._fondo_nuevo = 0
            self._cabezas.clear()

    def _cabeza(self, persona_id) -> CabezaPersona:
        if persona_id not in self._cabezas:
            self._cabezas[persona_id] = CabezaPersona.entrenar(self.ejemplos[persona_id], list(self.fondo),
                                                               semilla=self.semilla)
        return self._cabezas[persona_id]
