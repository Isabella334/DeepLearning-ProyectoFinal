"""
CONTRATO 1: los tipos de datos que se pasan entre modulos.

Este archivo es el "idioma comun" del equipo. Si alguien necesita cambiarlo,
se avisa al grupo ANTES, porque todos dependen de el.

Unidades: posiciones en metros sobre la cartulina, con (0, 0) en el centro.
El eje x va hacia la derecha de la camara cenital y el eje y hacia arriba.

IMPORTANTE: el robot NO trae nombres de objetos escritos en el codigo.
Todo objeto empieza como "desconocido" hasta que una persona se lo ensena.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time

DESCONOCIDO = "desconocido"


class Accion(str, Enum):
    """Las unicas acciones que el cuerpo sabe hacer."""
    APARTAR = "apartar"        # empujar el objeto a la zona de apartar
    TRAER = "traer"            # llevar el objeto a la zona de la persona
    IGNORAR = "ignorar"        # no hacer nada
    ACERCARSE = "acercarse"    # ir hacia la zona de la persona


@dataclass
class Objeto:
    """Un objeto sobre la cartulina, tal como lo entiende el robot."""
    id: str                            # estable mientras el objeto siga en la mesa, ej. "obj_3"
    x: float
    y: float
    nombre: str = DESCONOCIDO          # nombre que le enseno ESTA persona, ej. "lapicero rojo"
    confianza: float = 1.0             # que tan seguro esta del nombre (0 a 1)
    embedding: list[float] | None = None   # vector que describe como se ve (para reconocerlo)


@dataclass
class Deteccion:
    """Salida del detector (CNN): 'aqui hay un objeto', sin nombre. En PIXELES."""
    confianza: float
    caja: tuple[float, float, float, float]   # x1, y1, x2, y2 en pixeles


@dataclass
class PoseCarrito:
    x: float
    y: float
    angulo: float              # radianes; 0 = mirando hacia +x


@dataclass
class Escena:
    """Lo que se ve en la mesa en un instante. La produce Percepcion."""
    objetos: list[Objeto]
    carrito: PoseCarrito
    persona_id: str | None     # quien esta usando el robot (None si nadie)
    t: float = field(default_factory=time.time)


@dataclass
class Evento:
    """Algo que paso, para el modelo de anticipacion.
    tipo: "aparecio", "desaparecio", "pedido", "feedback", "accion"."""
    tipo: str
    detalle: str               # ej. "cuaderno verde", "traer:lapicero rojo"
    persona_id: str | None
    t: float = field(default_factory=time.time)


@dataclass
class Orden:
    """Lo que la persona dijo (voz) o escribio (pantalla), ya interpretado.
    tipo:
      "ensenar"  -> 'esto es mi lapicero rojo'      (nombre="lapicero rojo")
      "pedido"   -> 'traeme el lapicero rojo'       (accion=TRAER, nombre="lapicero rojo")
      "regla"    -> 'si ves basura, apartala'       (accion=APARTAR, nombre="basura")
      "feedback" -> 'bien' / 'no'                   (aprobado=True/False)
    """
    tipo: str
    texto: str                         # lo que se dijo, tal cual
    nombre: str | None = None
    accion: Accion | None = None
    aprobado: bool | None = None


@dataclass
class Contexto:
    """Todo lo que el cerebro necesita para decidir."""
    escena: Escena
    persona_id: str
    eventos_recientes: list[Evento]    # ultimos minutos, el mas reciente al final
    pedido: Orden | None = None        # orden explicita, si la hubo


@dataclass
class Decision:
    """Lo que el cerebro decide hacer."""
    accion: Accion
    objeto_id: str | None              # sobre que objeto (None para IGNORAR/ACERCARSE)
    confianza: float                   # 0 a 1
    preguntar_antes: bool = False      # True si duda: "Te paso el lapicero rojo?"
    motivo: str = ""                   # "pedido", "anticipacion", "preferencia", "regla"
    puntajes: dict[str, float] = field(default_factory=dict)  # puntaje de cada accion, para XAI


@dataclass
class Resultado:
    """Lo que devuelve el cuerpo al terminar una accion."""
    exito: bool
    duracion_s: float
    mensaje: str = ""


@dataclass
class Feedback:
    """La reaccion de la persona: viene de botones, gesto o voz."""
    persona_id: str
    aprobado: bool                     # True = bien, False = mal
    fuente: str = "teclado"            # "teclado", "boton", "gesto", "voz"
    t: float = field(default_factory=time.time)


@dataclass
class Explicacion:
    texto: str                         # frase para la persona
    contribuciones: dict[str, float] = field(default_factory=dict)  # variable -> peso (ej. SHAP)
    evidencia: str = ""                # ej. "aprobado 8 de 10 veces en situaciones parecidas"
