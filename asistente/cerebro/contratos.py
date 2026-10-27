"""
Contrato entre las dos mitades del cerebro, para que P2 y P3 trabajen en paralelo.

  P2 implementa ModeloPreferencias  (que le gusta a cada persona)
  P3 implementa ModeloAnticipacion  (que va a pedir a continuacion)
  P3 arma CerebroAprendiz, que usa los dos y cumple el contrato Cerebro.

Mientras la otra mitad no este lista, cada quien usa la version falsa de abajo.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..tipos import Accion, Contexto, Evento


class ModeloPreferencias(ABC):
    """Una instancia por persona (o por robot)."""

    @abstractmethod
    def puntajes(self, contexto: Contexto, objeto_id: str | None) -> dict[Accion, float]:
        """Probabilidad de 0 a 1 de que a esta persona le guste cada accion sobre ese objeto."""

    @abstractmethod
    def actualizar(self, contexto: Contexto, objeto_id: str | None, accion: Accion,
                   aprobado: bool) -> None:
        """Aprende de un aprobar/corregir."""

    @abstractmethod
    def vector(self, contexto: Contexto, objeto_id: str | None) -> list[float]:
        """Las caracteristicas que ve la red (las usa P4 para SHAP)."""

    @abstractmethod
    def nombres_caracteristicas(self) -> list[str]:
        """Nombre de cada posicion del vector, ej. ['es_basura', 'hay_cuaderno', ...]."""

    @abstractmethod
    def guardar(self, ruta: str) -> None: ...

    @abstractmethod
    def cargar(self, ruta: str) -> None: ...

    @abstractmethod
    def clonar(self) -> "ModeloPreferencias":
        """Copia independiente: el punto de partida de cada robot en el experimento."""


class ModeloAnticipacion(ABC):
    @abstractmethod
    def predecir(self, eventos: list[Evento]) -> tuple[str | None, float]:
        """Siguiente pedido mas probable (ej. 'traer:lapiz_rojo') y su probabilidad."""

    @abstractmethod
    def actualizar(self, eventos: list[Evento], pedido_real: str) -> None:
        """Aprende de un pedido que si ocurrio despues de esos eventos."""


class PreferenciasFalsas(ModeloPreferencias):
    def puntajes(self, contexto, objeto_id):
        return {Accion.APARTAR: 0.4, Accion.TRAER: 0.3, Accion.IGNORAR: 0.2, Accion.ACERCARSE: 0.1}

    def actualizar(self, contexto, objeto_id, accion, aprobado):
        pass

    def vector(self, contexto, objeto_id):
        return [0.0, 1.0]

    def nombres_caracteristicas(self):
        return ["ejemplo_a", "ejemplo_b"]

    def guardar(self, ruta):
        pass

    def cargar(self, ruta):
        pass

    def clonar(self):
        return PreferenciasFalsas()


class AnticipacionFalsa(ModeloAnticipacion):
    def predecir(self, eventos):
        if any(e.detalle == "cuaderno_verde" for e in eventos):
            return "traer:lapiz_rojo", 0.5
        return None, 0.0

    def actualizar(self, eventos, pedido_real):
        pass
