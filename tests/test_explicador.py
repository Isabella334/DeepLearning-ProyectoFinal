import time

import numpy as np
import pytest

from asistente.cerebro.contratos import ModeloPreferencias, PreferenciasFalsas
from asistente.interfaces import Explicador
from asistente.memoria import MemoriaSQLite
from asistente.tipos import (DESCONOCIDO, Accion, Contexto, Decision, Escena, Evento, Explicacion,
                             Feedback, Objeto, Orden, PoseCarrito, Resultado)
from asistente.xai.explicador import ExplicadorXAI

P = "persona_A"
ACCIONES = list(Accion)


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



class _PreferenciasLineales(ModeloPreferencias):
    """Solo 'es_objeto_x' importa de verdad para TRAER; 'hay_senal' casi nada."""

    def vector(self, contexto, objeto_id):
        obj = next((o for o in contexto.escena.objetos if o.id == objeto_id), None)
        return [float(obj is not None and obj.nombre == "objeto x"), 1.0]

    def nombres_caracteristicas(self):
        return ["es_objeto_x", "hay_senal"]

    def predecir_vectores(self, X):
        X = np.asarray(X, dtype=float)
        salida = np.full((len(X), len(ACCIONES)), 0.1)
        salida[:, ACCIONES.index(Accion.TRAER)] = 0.8 * X[:, 0] + 0.1 * X[:, 1]
        return salida

    def fondo(self, n=50):
        return np.zeros((1, 2))

    def puntajes(self, contexto, objeto_id):
        return dict(zip(ACCIONES, self.predecir_vectores([self.vector(contexto, objeto_id)])[0]))

    def actualizar(self, contexto, objeto_id, accion, aprobado):
        pass

    def guardar(self, ruta):
        pass

    def cargar(self, ruta):
        pass

    def clonar(self):
        return self


class _Cerebro:
    def __init__(self, modelo):
        self.modelo = modelo

    def preferencias_de(self, persona_id):
        return self.modelo


class _PreferenciasRotas(_PreferenciasLineales):
    def vector(self, contexto, objeto_id):
        raise RuntimeError("red rota")


def test_shap_con_modelo():
    e = ExplicadorXAI().explicar(_contexto(), _decision("preferencia"), _Cerebro(_PreferenciasLineales()),
                                 MemoriaSQLite())
    c = e.contribuciones
    assert c["es_objeto_x"] > 0
    assert abs(c["es_objeto_x"]) > abs(c["hay_senal"])


def test_shap_sin_predecir_vectores():
    d = _decision("preferencia", puntajes={"traer": 0.3, "apartar": 0.4})
    e = ExplicadorXAI().explicar(_contexto(), d, _Cerebro(PreferenciasFalsas()), MemoriaSQLite())
    assert e.contribuciones == d.puntajes


@pytest.mark.parametrize("cerebro", [None, object(), _Cerebro(_PreferenciasRotas())])
def test_shap_nunca_falla(cerebro):
    d = _decision("preferencia", puntajes={"traer": 0.3})
    e = ExplicadorXAI().explicar(_contexto(), d, cerebro, MemoriaSQLite())
    assert e.texto and e.contribuciones == d.puntajes


def test_shap_con_preferencias_red():
    preferencias = pytest.importorskip("asistente.cerebro.preferencias")
    from asistente import mocks
    from asistente.cerebro.cerebro import CerebroAprendiz

    cerebro = CerebroAprendiz(preferencias=preferencias.PreferenciasRed(pasos_por_actualizacion=5))
    percepcion = mocks.PercepcionFalsa(semilla=3)
    for _ in range(3):
        ctx = Contexto(percepcion.observar(), P, [])
        d = cerebro.decidir(ctx)
        cerebro.aprender(ctx, d, Feedback(P, True))
    ctx = Contexto(percepcion.observar(), P, [])
    d = cerebro.decidir(ctx)
    e = ExplicadorXAI().explicar(ctx, d, cerebro, MemoriaSQLite())
    nombres = set(cerebro.preferencias_de(P).nombres_caracteristicas())
    assert e.contribuciones and set(e.contribuciones) <= nombres
