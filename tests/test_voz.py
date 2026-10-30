import sys

import numpy as np
import pytest

from asistente.interfaces import Voz
from asistente.tipos import Accion
from asistente.voz import VozWhisper, interpretar_texto, quitar_palabra_clave

CASOS = [
    # ensenar
    ("esto es mi lapicero rojo", "ensenar", None, "lapicero rojo", None),
    ("Este es el borrador.", "ensenar", None, "borrador", None),
    ("esta es mi lapicera azul", "ensenar", None, "lapicera azul", None),
    ("esto se llama cuaderno verde", "ensenar", None, "cuaderno verde", None),
    ("te presento a mi lápiz", "ensenar", None, "lápiz", None),
    # pedido
    ("tráeme el lapicero rojo", "pedido", Accion.TRAER, "lapicero rojo", None),
    ("traeme el lapicero rojo", "pedido", Accion.TRAER, "lapicero rojo", None),
    ("Pásame el borrador, por favor.", "pedido", Accion.TRAER, "borrador", None),
    ("dame mi cuaderno", "pedido", Accion.TRAER, "cuaderno", None),
    ("¿me pasas el lápiz?", "pedido", Accion.TRAER, "lápiz", None),
    ("¿puedes traerme el lapicero azul?", "pedido", Accion.TRAER, "lapicero azul", None),
    ("necesito el borrador", "pedido", Accion.TRAER, "borrador", None),
    ("quiero que me traigas el lapicero", "pedido", Accion.TRAER, "lapicero", None),
    ("apartá la basura", "pedido", Accion.APARTAR, "basura", None),
    ("quita el borrador de ahí", "pedido", Accion.APARTAR, "borrador", None),
    ("sacá las virutas ya", "pedido", Accion.APARTAR, "virutas", None),
    ("llévate la bolita de papel", "pedido", Accion.APARTAR, "bolita de papel", None),
    ("vení", "pedido", Accion.ACERCARSE, None, None),
    ("acércate", "pedido", Accion.ACERCARSE, None, None),
    ("Ben.", "pedido", Accion.ACERCARSE, None, None),
    # regla
    ("si ves basura, apártala", "regla", Accion.APARTAR, "basura", None),
    ("si ves basura apartala", "regla", Accion.APARTAR, "basura", None),
    ("cuando veas virutas de lápiz, quítalas", "regla", Accion.APARTAR, "virutas de lápiz", None),
    ("siempre que haya un lapicero, tráemelo", "regla", Accion.TRAER, "lapicero", None),
    ("si ves el cuaderno, déjalo", "regla", Accion.IGNORAR, "cuaderno", None),
    # feedback
    ("bien", "feedback", None, None, True),
    ("¡Muy bien, gracias!", "feedback", None, None, True),
    ("sí", "feedback", None, None, True),
    ("perfecto", "feedback", None, None, True),
    ("no", "feedback", None, None, False),
    ("mal", "feedback", None, None, False),
    ("así no", "feedback", None, None, False),
    ("eso no", "feedback", None, None, False),
    ("asino", "feedback", None, None, False),
]


@pytest.mark.parametrize("texto, tipo, accion, nombre, aprobado", CASOS)
def test_interpreta_las_cuatro_clases(texto, tipo, accion, nombre, aprobado):
    o = interpretar_texto(texto)
    assert o is not None, texto
    assert (o.tipo, o.accion, o.nombre, o.aprobado) == (tipo, accion, nombre, aprobado)
    assert o.texto == texto


@pytest.mark.parametrize("texto", ["", "   ", "hola", "qué lindo día", "tráemelo",
                                   "cuando veas el cuaderno verde, tráeme el lapicero rojo"])
def test_frases_sin_orden(texto):
    assert interpretar_texto(texto) is None


def test_quita_el_nombre_del_robot():
    o = interpretar_texto("Robot, tráeme el lapicero rojo.", nombre_robot="robot")
    assert o.tipo == "pedido" and o.nombre == "lapicero rojo"
    assert quitar_palabra_clave("Robot, tráeme el lapicero.", "robot") == "tráeme el lapicero"
    assert quitar_palabra_clave("Oye róbot pásame el borrador", "robot") == "pásame el borrador"
    assert quitar_palabra_clave("Robot.", "robot") == ""
    assert quitar_palabra_clave("tráeme el lapicero", "robot") is None
    assert quitar_palabra_clave("esto es mi robot", "robot") is None


def test_nombre_del_robot_mal_transcrito():
    assert quitar_palabra_clave("Pobot, si ves basura, apártala.", "robot") == "si ves basura, apártala"
    assert quitar_palabra_clave("Bobots, aparta la bolita", "robot") == "aparta la bolita"
    assert quitar_palabra_clave("Oye, pobot, ven", "robot") == "ven"
    assert quitar_palabra_clave("Todo bien, gracias", "robot") is None
    assert quitar_palabra_clave("Toma el borrador", "robot") is None
    assert quitar_palabra_clave("Nova, tráeme eso", "Nova") == "tráeme eso"


def test_ayuda_para_whisper_sin_nombres_escritos():
    v = VozWhisper(nombre_robot="robot")
    assert v._ayuda().startswith("Robot,") and "Objetos" not in v._ayuda()
    v.vocabulario = ["objeto a", "objeto b"]
    assert v._ayuda().endswith("Objetos: objeto a, objeto b.")


class VozFalsa(VozWhisper):
    """VozWhisper sin microfono ni Whisper: devuelve frases en orden."""

    def __init__(self, frases):
        super().__init__(nombre_robot="robot")
        self.frases = list(frases)

    def _grabar(self, segundos):
        return np.full(10, 0.5 if self.frases else 0.0, dtype=np.float32)

    def transcribir(self, audio):
        return self.frases.pop(0)


def test_escuchar_solo_despues_del_nombre():
    assert VozFalsa(["Robot, tráeme el lapicero rojo."]).escuchar() == "tráeme el lapicero rojo"
    assert VozFalsa(["Robot.", "Pásame el borrador"]).escuchar() == "Pásame el borrador"
    assert VozFalsa(["tráeme el lapicero rojo"]).escuchar(timeout_s=0.05) is None


def test_orejas_apagadas():
    v = VozFalsa(["Robot, tráeme el lapicero rojo."])
    v.orejas = False
    assert v.escuchar() is None


def test_sin_microfono_ni_whisper_devuelve_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    v = VozWhisper(nombre_robot="robot")
    assert isinstance(v, Voz)
    assert v.escuchar(timeout_s=0.1) is None
    assert v.interpretar("robot, tráeme el lapicero rojo").accion == Accion.TRAER
