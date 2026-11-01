import time

import pytest

from asistente.interfaces import Explicador
from asistente.memoria import MemoriaSQLite
from asistente.tipos import (DESCONOCIDO, Accion, Contexto, Decision, Escena, Evento, Explicacion,
                             Feedback, Objeto, Orden, PoseCarrito, Resultado)
from asistente.xai.explicador import ExplicadorXAI

P = "persona_A"


def _contexto(nombre="objeto x", eventos=None, pedido=None):
    escena = Escena([Objeto("obj_1", 0.1, 0.0, nombre=nombre), Objeto("obj_2", -0.1, 0.0, nombre="senal y")],
                    PoseCarrito(0, 0, 0), P)
    return Contexto(escena, P, eventos or [], pedido)


def _decision(motivo, accion=Accion.TRAER, objeto_id="obj_1", **kw):
    return Decision(accion, objeto_id, confianza=0.8, motivo=motivo, **kw)


def test_cumple_contrato():
    assert isinstance(ExplicadorXAI(), Explicador)


@pytest.mark.parametrize("motivo", ["pedido", "anticipacion", "preferencia", "regla", "otro"])
def test_texto_por_motivo(motivo):
    e = ExplicadorXAI().explicar(_contexto(), _decision(motivo), None, MemoriaSQLite())
    assert isinstance(e, Explicacion)
    assert "objeto x" in e.texto


def test_texto_pedido_no_encontrado():
    ctx = _contexto(pedido=Orden("pedido", "traeme objeto z", nombre="objeto z", accion=Accion.TRAER))
    d = Decision(Accion.IGNORAR, None, confianza=0.0, preguntar_antes=True, motivo="pedido")
    assert "objeto z" in ExplicadorXAI().explicar(ctx, d, None, MemoriaSQLite()).texto


def test_texto_anticipacion_nombra_la_senal():
    ctx = _contexto(eventos=[Evento("aparecio", "senal y", P)])
    assert "senal y" in ExplicadorXAI().explicar(ctx, _decision("anticipacion"), None, MemoriaSQLite()).texto


def test_objeto_desconocido_no_muestra_desconocido():
    e = ExplicadorXAI().explicar(_contexto(nombre=DESCONOCIDO), _decision("preferencia"), None, MemoriaSQLite())
    assert DESCONOCIDO not in e.texto


def test_evidencia_preferencia():
    m = MemoriaSQLite()
    ctx = _contexto()
    for aprobado in (True, True, True, False, None):
        fb = None if aprobado is None else Feedback(P, aprobado)
        m.registrar_interaccion(ctx, _decision("preferencia"), Resultado(True, 0.0), fb)
    m.registrar_interaccion(ctx, _decision("preferencia", Accion.APARTAR), Resultado(True, 0.0), Feedback(P, True))
    e = ExplicadorXAI().explicar(ctx, _decision("preferencia"), None, m)
    assert "3 de 4" in e.evidencia


def test_evidencia_primera_vez():
    e = ExplicadorXAI().explicar(_contexto(), _decision("preferencia"), None, MemoriaSQLite())
    assert "primera vez" in e.evidencia


def test_evidencia_anticipacion():
    m = MemoriaSQLite()
    t0 = time.time() - 3000
    for i, pidio in enumerate((True, False, True)):
        t = t0 + i * 600
        m.registrar_evento(Evento("aparecio", "senal y", P, t=t))
        if pidio:
            m.registrar_evento(Evento("pedido", "traer:objeto x", P, t=t + 30))
    actual = Evento("aparecio", "senal y", P)
    m.registrar_evento(actual)
    e = ExplicadorXAI().explicar(_contexto(eventos=[actual]), _decision("anticipacion"), None, m)
    assert "2 de las 3" in e.evidencia


class _MemoriaRota(MemoriaSQLite):
    def historial(self, persona_id, limite=200):
        raise RuntimeError("base caida")

    def eventos_recientes(self, persona_id, segundos=300):
        raise RuntimeError("base caida")


@pytest.mark.parametrize("motivo", ["preferencia", "anticipacion"])
def test_nunca_falla(motivo):
    ctx = _contexto(eventos=[Evento("aparecio", "senal y", P)])
    e = ExplicadorXAI().explicar(ctx, _decision(motivo), None, _MemoriaRota())
    assert isinstance(e, Explicacion) and e.texto
