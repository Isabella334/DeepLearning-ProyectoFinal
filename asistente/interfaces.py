"""
CONTRATO 2: lo que cada modulo promete hacer.

Cada persona implementa una o varias de estas clases. Mientras las demas no esten
listas, integracion usa las versiones falsas de mocks.py para probar e integrar.

  Clase           | Responsable | Implementacion real esperada
  ----------------|-------------|--------------------------------------------------
  Cuerpo          | P1          | CuerpoSim (MuJoCo)            · Integracion: CuerpoReal (ESP32)
  Percepcion      | P1          | PercepcionSim (MuJoCo)        · Integracion: PercepcionCamara
  Detector        | Integracion | DetectorYOLO (CNN entrenada con el dataset de la cartulina)
  Reconocedor     | P2          | ReconocedorPocosEjemplos (embeddings + red por persona)
  Cerebro         | P3          | CerebroAprendiz (usa ModeloPreferencias de P2, ver cerebro/contratos.py)
  Voz             | P3          | VozWhisper
  Memoria         | P4          | MemoriaSQLite
  Explicador      | P4          | ExplicadorXAI (SHAP + historial)
  EntradaFeedback | P4          | botones en pantalla (y gestos, si da tiempo)

Las guias de cada persona estan en docs/tareas/.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .tipos import (Accion, Contexto, Decision, Deteccion, Escena, Evento, Explicacion,
                    Feedback, Objeto, Orden, Resultado)


class Cuerpo(ABC):
    """Mueve el carrito. El cerebro NUNCA sabe si es simulado o real."""

    @abstractmethod
    def ejecutar(self, accion: Accion, objeto: Objeto | None, escena: Escena) -> Resultado:
        """Ejecuta la accion completa y regresa al terminar (bloqueante).
        APARTAR: empuja el objeto a la zona de apartar.
        TRAER: lleva el objeto a la zona de la persona.
        ACERCARSE: va a la zona de la persona.
        IGNORAR: no hace nada y devuelve exito=True."""

    def cerrar(self) -> None:
        """Libera recursos (ventana, conexion WiFi)."""


class Reconocedor(ABC):
    """Aprende los NOMBRES de los objetos que le ensena cada persona (pocos ejemplos).
    Cada persona tiene su propio conocimiento: lo que le ensena A no lo sabe B."""

    @abstractmethod
    def embedding(self, recorte) -> list[float]:
        """Convierte la imagen recortada de un objeto (numpy, alto x ancho x 3) en un vector."""

    @abstractmethod
    def ensenar(self, persona_id: str, nombre: str, embedding: list[float]) -> None:
        """'Esto es mi lapicero rojo': guarda un ejemplo con ese nombre para esa persona."""

    @abstractmethod
    def reconocer(self, persona_id: str, embedding: list[float]) -> tuple[str, float]:
        """Devuelve (nombre, confianza). Si no se parece a nada conocido: (DESCONOCIDO, conf)."""

    @abstractmethod
    def nombres(self, persona_id: str) -> list[str]:
        """Los nombres que esa persona le ha ensenado."""

    @abstractmethod
    def guardar(self, ruta: str) -> None: ...

    @abstractmethod
    def cargar(self, ruta: str) -> None: ...


class Percepcion(ABC):
    """Convierte lo que se ve en una Escena, con los objetos ya nombrados por el Reconocedor."""

    @abstractmethod
    def observar(self) -> Escena:
        ...

    @abstractmethod
    def ensenar(self, persona_id: str, objeto_id: str, nombre: str) -> None:
        """Toma el embedding del objeto 'objeto_id' que esta en la mesa y se lo pasa al
        Reconocedor con ese nombre."""


class Detector(ABC):
    """La CNN que encuentra objetos en una imagen de la camara cenital (sin nombrarlos)."""

    @abstractmethod
    def detectar(self, imagen) -> list[Deteccion]:
        """imagen: arreglo de numpy (alto x ancho x 3, BGR como lo da OpenCV)."""


class Cerebro(ABC):
    """Decide que hacer y aprende de la reaccion de la persona."""

    @abstractmethod
    def decidir(self, contexto: Contexto) -> Decision:
        ...

    @abstractmethod
    def aprender(self, contexto: Contexto, decision: Decision, feedback: Feedback) -> None:
        """Actualiza el modelo de esa persona con una interaccion."""

    @abstractmethod
    def guardar(self, ruta: str) -> None:
        ...

    @abstractmethod
    def cargar(self, ruta: str) -> None:
        ...


class Voz(ABC):
    """Escucha a la persona y entiende lo que dice."""

    @abstractmethod
    def escuchar(self, timeout_s: float = 5.0) -> str | None:
        """Espera a que digan el nombre del robot y devuelve el texto de lo que dijeron despues.
        None si no se dijo nada o si las 'orejas' estan apagadas."""

    @abstractmethod
    def interpretar(self, texto: str) -> Orden | None:
        """Convierte el texto en una Orden (ensenar, pedido, regla o feedback)."""


class Memoria(ABC):
    """Todo lo que se guarda para no olvidar al apagar el robot."""

    @abstractmethod
    def registrar_interaccion(self, contexto: Contexto, decision: Decision,
                              resultado: Resultado, feedback: Feedback | None) -> int:
        """Guarda una interaccion completa y devuelve su id."""

    @abstractmethod
    def registrar_evento(self, evento: Evento) -> None:
        ...

    @abstractmethod
    def registrar_ensenanza(self, persona_id: str, nombre: str, objeto_id: str) -> None:
        """Anota que esa persona le enseno ese nombre (para el historial y el XAI)."""

    @abstractmethod
    def eventos_recientes(self, persona_id: str, segundos: float = 300) -> list[Evento]:
        ...

    @abstractmethod
    def historial(self, persona_id: str, limite: int = 200) -> list[dict]:
        """Interacciones pasadas de una persona. Cada dict trae al menos:
        persona_id, accion, objeto_id, objeto_nombre, aprobado (True/False/None), t."""


class Explicador(ABC):
    @abstractmethod
    def explicar(self, contexto: Contexto, decision: Decision, cerebro: Cerebro,
                 memoria: Memoria) -> Explicacion:
        ...


class EntradaFeedback(ABC):
    """De donde llega el 'bien' / 'mal' de la persona."""

    @abstractmethod
    def esperar(self, persona_id: str, timeout_s: float = 10.0) -> Feedback | None:
        """Espera la reaccion de la persona. None si no reacciono a tiempo."""
