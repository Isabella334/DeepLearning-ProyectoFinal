"""
Anticipacion (P3): predice el siguiente pedido a partir de la secuencia de eventos recientes.

  AnticipacionFrecuencias  version 1: tabla de que pedido vino despues de cada evento
  AnticipacionSecuencias   version 2: LSTM con embeddings de eventos (la que usa el cerebro)

Cada momento del flujo de eventos es una muestra: la entrada son los ultimos eventos y la
etiqueta es el pedido que vino despues, o NINGUNO si despues no se pidio nada. Asi la
probabilidad que devuelve predecir() es directamente la de que ese pedido sea el siguiente.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import torch
from torch import nn

from ..tipos import Evento
from .contratos import ModeloAnticipacion

NINGUNO = "<ninguno>"
VACIO = "<vacio>"


def token(evento: Evento) -> str:
    return f"{evento.tipo}:{evento.detalle}"


class Vocabulario:
    """Asigna un indice a cada token la primera vez que aparece."""

    def __init__(self, iniciales: list[str] = ()):
        self.tokens: list[str] = []
        self.indices: dict[str, int] = {}
        for t in iniciales:
            self.indice(t)

    def indice(self, t: str, crear: bool = True) -> int:
        if t not in self.indices:
            if not crear:
                return 0
            self.indices[t] = len(self.tokens)
            self.tokens.append(t)
        return self.indices[t]

    def __len__(self):
        return len(self.tokens)


class _AnticipacionBase(ModeloAnticipacion):
    """Lleva el flujo de eventos, el vocabulario y las muestras. Las subclases aprenden de ellas.

    simultaneo_s: eventos separados por menos de esto son el mismo momento (por ejemplo, varios
    objetos que aparecen en el mismo ciclo) y entre ellos no se genera una muestra NINGUNO."""

    def __init__(self, ventana: int = 8, simultaneo_s: float = 0.5, max_muestras: int = 1000):
        self.ventana = ventana
        self.simultaneo_s = simultaneo_s
        self.max_muestras = max_muestras
        self.entradas = Vocabulario([VACIO])
        self.salidas = Vocabulario([NINGUNO])
        self.flujo: list[tuple[str, float]] = []
        self.vistos: set[tuple[float, str, str]] = set()
        self.pendientes: Counter[str] = Counter()
        self.muestras: list[tuple[list[str], str]] = []

    def predecir(self, eventos):
        tokens = [token(e) for e in eventos][-self.ventana:]
        if not tokens or len(self.salidas) <= 1:
            return None, 0.0
        probs = self._probabilidades(tokens)
        mejor = max(range(1, len(self.salidas)), key=lambda i: probs[i])
        if probs[mejor] <= 0.0:
            return None, 0.0
        return self.salidas.tokens[mejor], float(probs[mejor])

    def actualizar(self, eventos, pedido_real):
        eventos = sorted(eventos, key=lambda e: e.t)
        ultimo = eventos[-1] if eventos else None
        realizado = (ultimo is not None and ultimo.tipo == "pedido" and ultimo.detalle == pedido_real
                     and self._clave(ultimo) not in self.vistos)
        nuevas = self._incorporar(eventos[:-1] if realizado else eventos)
        nuevas.append(self._muestra(pedido_real))
        if pedido_real != NINGUNO:
            if realizado:
                self.vistos.add(self._clave(ultimo))
                self._agregar(token(ultimo), ultimo.t)
            else:
                self.pendientes[pedido_real] += 1
                self._agregar(f"pedido:{pedido_real}", self.flujo[-1][1] if self.flujo else 0.0)
        self.muestras = (self.muestras + nuevas)[-self.max_muestras:]
        self._aprender(nuevas)

    def _clave(self, e: Evento):
        return (e.t, e.tipo, e.detalle)

    def _incorporar(self, eventos: list[Evento]) -> list[tuple[list[str], str]]:
        nuevas = []
        for e in eventos:
            if self._clave(e) in self.vistos:
                continue
            self.vistos.add(self._clave(e))
            if e.tipo == "pedido" and self.pendientes[e.detalle] > 0:
                self.pendientes[e.detalle] -= 1      # ya se aprendio cuando llego como pedido_real
                continue
            if e.tipo != "pedido" and self.flujo and e.t - self.flujo[-1][1] > self.simultaneo_s:
                nuevas.append(self._muestra(NINGUNO))
            self._agregar(token(e), e.t)
        return nuevas

    def _agregar(self, tok: str, t: float):
        self.entradas.indice(tok)
        self.flujo = (self.flujo + [(tok, t)])[-self.ventana:]

    def _muestra(self, etiqueta: str) -> tuple[list[str], str]:
        self.salidas.indice(etiqueta)
        return [tok for tok, _ in self.flujo], etiqueta

    def _estado(self) -> dict:
        return dict(ventana=self.ventana, simultaneo_s=self.simultaneo_s,
                    entradas=self.entradas.tokens, salidas=self.salidas.tokens,
                    flujo=[list(f) for f in self.flujo], vistos=[list(v) for v in self.vistos],
                    pendientes=dict(self.pendientes), muestras=[[v, e] for v, e in self.muestras])

    def _restaurar(self, estado: dict):
        self.ventana, self.simultaneo_s = estado["ventana"], estado["simultaneo_s"]
        self.entradas, self.salidas = Vocabulario(estado["entradas"]), Vocabulario(estado["salidas"])
        self.flujo = [tuple(f) for f in estado["flujo"]]
        self.vistos = {tuple(v) for v in estado["vistos"]}
        self.pendientes = Counter(estado["pendientes"])
        self.muestras = [(list(v), e) for v, e in estado["muestras"]]

    def guardar(self, ruta: str) -> None:
        torch.save(self._estado(), ruta)

    def cargar(self, ruta: str) -> None:
        self._restaurar(torch.load(ruta, weights_only=True))

    def _aprender(self, nuevas: list[tuple[list[str], str]]) -> None:
        raise NotImplementedError

    def _probabilidades(self, tokens: list[str]) -> list[float]:
        raise NotImplementedError


class AnticipacionFrecuencias(_AnticipacionBase):
    """Version 1: cuenta que pedido (o NINGUNO) vino despues de cada evento."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conteos: dict[str, Counter[str]] = defaultdict(Counter)

    def _aprender(self, nuevas):
        for ventana, etiqueta in nuevas:
            if ventana:
                self.conteos[ventana[-1]][etiqueta] += 1

    def _probabilidades(self, tokens):
        conteo = self.conteos.get(tokens[-1], Counter())
        total = sum(conteo.values())
        return [conteo[s] / total if total else 0.0 for s in self.salidas.tokens]

    def _estado(self):
        return super()._estado() | {"conteos": {k: dict(v) for k, v in self.conteos.items()}}

    def _restaurar(self, estado):
        super()._restaurar(estado)
        self.conteos = defaultdict(Counter, {k: Counter(v) for k, v in estado["conteos"].items()})


class RedSecuencias(nn.Module):
    """Embedding de eventos, LSTM y una capa lineal con una salida por pedido (mas NINGUNO)."""

    def __init__(self, n_entradas: int, n_salidas: int, dim: int = 16, oculto: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(n_entradas, dim, padding_idx=0)
        self.lstm = nn.LSTM(dim, oculto, batch_first=True)
        self.salida = nn.Linear(oculto, n_salidas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.embedding(x))
        return self.salida(h[:, -1])

    def crecer(self, n_entradas: int, n_salidas: int) -> None:
        """Agranda el embedding y la salida cuando el vocabulario crece, sin perder lo aprendido."""
        viejo = self.embedding
        if n_entradas > viejo.num_embeddings:
            self.embedding = nn.Embedding(n_entradas, viejo.embedding_dim, padding_idx=0)
            with torch.no_grad():
                self.embedding.weight[:viejo.num_embeddings] = viejo.weight
        viejo = self.salida
        if n_salidas > viejo.out_features:
            self.salida = nn.Linear(viejo.in_features, n_salidas)
            with torch.no_grad():
                self.salida.weight[:viejo.out_features] = viejo.weight
                self.salida.bias[:viejo.out_features] = viejo.bias


class AnticipacionSecuencias(_AnticipacionBase):
    """Version 2: LSTM pequeno que recibe los ultimos eventos y predice el siguiente pedido.
    Como hay pocos datos, en cada actualizacion reentrena unas epocas con todas las muestras."""

    def __init__(self, ventana: int = 8, epocas: int = 60, lr: float = 0.02, semilla: int = 0, **kwargs):
        super().__init__(ventana=ventana, **kwargs)
        self.epocas, self.lr, self.semilla = epocas, lr, semilla
        with torch.random.fork_rng():
            torch.manual_seed(semilla)
            self.red = RedSecuencias(len(self.entradas), len(self.salidas))

    def _tensor(self, ventanas: list[list[str]]) -> torch.Tensor:
        filas = []
        for v in ventanas:
            idx = [self.entradas.indice(t, crear=False) for t in v[-self.ventana:]]
            filas.append([0] * (self.ventana - len(idx)) + idx)
        return torch.tensor(filas, dtype=torch.long)

    def _aprender(self, nuevas):
        with torch.random.fork_rng():
            torch.manual_seed(self.semilla + len(self.muestras))
            self.red.crecer(len(self.entradas), len(self.salidas))
        x = self._tensor([v for v, _ in self.muestras])
        y = torch.tensor([self.salidas.indice(e) for _, e in self.muestras], dtype=torch.long)
        opt = torch.optim.Adam(self.red.parameters(), lr=self.lr)
        self.red.train()
        for _ in range(self.epocas):
            opt.zero_grad()
            nn.functional.cross_entropy(self.red(x), y).backward()
            opt.step()

    def _probabilidades(self, tokens):
        self.red.eval()
        with torch.no_grad():
            return torch.softmax(self.red(self._tensor([tokens])), dim=-1)[0].tolist()

    def _estado(self):
        return super()._estado() | {"red": self.red.state_dict(), "epocas": self.epocas,
                                    "lr": self.lr, "semilla": self.semilla}

    def _restaurar(self, estado):
        super()._restaurar(estado)
        self.epocas, self.lr, self.semilla = estado["epocas"], estado["lr"], estado["semilla"]
        self.red = RedSecuencias(len(self.entradas), len(self.salidas))
        self.red.load_state_dict(estado["red"])
