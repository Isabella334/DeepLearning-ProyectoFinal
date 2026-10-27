"""
Prueba que cada implementacion cumpla su contrato. En la integracion se agrega aqui cada clase
real cuando llega y corre:   python -m pytest tests -q
Si pasa, la pieza se puede conectar con las demas.
"""
import numpy as np

from asistente import mocks
from asistente.interfaces import (Cerebro, Cuerpo, Detector, Explicador, Memoria, Percepcion,
                                  Reconocedor, Voz)
from asistente.tipos import (DESCONOCIDO, Accion, Contexto, Decision, Deteccion, Escena,
                             Explicacion, Feedback, Orden, Resultado)

# Agregar aqui cada clase real cuando llegue.
RECONOCEDORES = [lambda: mocks.ReconocedorFalso()]
PERCEPCIONES = [lambda: mocks.PercepcionFalsa()]
CUERPOS = [lambda: mocks.CuerpoFalso()]
CEREBROS = [lambda: mocks.CerebroAlAzar()]
VOCES = [lambda: mocks.VozTeclado()]
MEMORIAS = [lambda: mocks.MemoriaEnRAM()]
DETECTORES = [lambda: mocks.DetectorFalso()]
EXPLICADORES = [lambda: mocks.ExplicadorSimple()]


def _contexto(semilla=1):
    escena = mocks.PercepcionFalsa(semilla=semilla).observar()
    return Contexto(escena=escena, persona_id="persona_A", eventos_recientes=[])


def test_reconocedor_aprende_por_persona():
    for crear in RECONOCEDORES:
        r = crear()
        assert isinstance(r, Reconocedor)
        rojo, azul = [1.0, 0.1, 0.1, 0.2], [0.1, 0.1, 1.0, 0.2]
        assert r.reconocer("A", rojo)[0] == DESCONOCIDO          # al inicio no sabe nada
        r.ensenar("A", "lapicero rojo", rojo)
        r.ensenar("A", "lapicero azul", azul)
        assert r.reconocer("A", rojo)[0] == "lapicero rojo"
        assert r.reconocer("B", rojo)[0] == DESCONOCIDO          # lo de A no lo sabe B
        emb = r.embedding(np.zeros((32, 32, 3), dtype=np.uint8))
        assert isinstance(emb, list) and len(emb) > 0


def test_percepcion():
    for crear in PERCEPCIONES:
        p = crear()
        assert isinstance(p, Percepcion)
        e = p.observar()
        assert isinstance(e, Escena)
        for o in e.objetos:
            assert 0.0 <= o.confianza <= 1.0 + 1e-6


def test_cuerpo():
    escena = mocks.PercepcionFalsa(semilla=1).observar()
    for crear in CUERPOS:
        c = crear()
        assert isinstance(c, Cuerpo)
        assert isinstance(c.ejecutar(Accion.IGNORAR, None, escena), Resultado)


def test_cerebro_decide_y_aprende():
    for crear in CEREBROS:
        cb = crear()
        assert isinstance(cb, Cerebro)
        for semilla in range(5):                     # incluye mesas vacias y personas None
            ctx = _contexto(semilla)
            d = cb.decidir(ctx)
            assert isinstance(d, Decision) and 0.0 <= d.confianza <= 1.0
            assert d.objeto_id is None or any(o.id == d.objeto_id for o in ctx.escena.objetos)
            cb.aprender(ctx, d, Feedback("persona_A", True))


def test_voz_interpreta():
    for crear in VOCES:
        v = crear()
        assert isinstance(v, Voz)
        o = v.interpretar("esto es mi lapicero rojo")
        assert isinstance(o, Orden) and o.tipo == "ensenar" and o.nombre == "lapicero rojo"
        o = v.interpretar("traeme el lapicero rojo")
        assert o.tipo == "pedido" and o.accion == Accion.TRAER


def test_memoria_y_explicador():
    for crear_m in MEMORIAS:
        m = crear_m()
        assert isinstance(m, Memoria)
        ctx = _contexto()
        d = mocks.CerebroAlAzar().decidir(ctx)
        m.registrar_interaccion(ctx, d, Resultado(True, 1.0), Feedback("persona_A", True))
        m.registrar_ensenanza("persona_A", "lapicero rojo", "obj_0")
        h = m.historial("persona_A")
        assert len(h) == 1 and {"persona_id", "accion", "objeto_id", "objeto_nombre", "aprobado", "t"} <= set(h[0])
        for crear_e in EXPLICADORES:
            e = crear_e()
            assert isinstance(e, Explicador)
            assert isinstance(e.explicar(ctx, d, mocks.CerebroAlAzar(), m), Explicacion)


def test_detector():
    imagen = np.zeros((480, 640, 3), dtype=np.uint8)
    for crear in DETECTORES:
        d = crear()
        assert isinstance(d, Detector)
        for det in d.detectar(imagen):
            assert isinstance(det, Deteccion) and len(det.caja) == 4


def test_mitades_del_cerebro():
    from asistente.cerebro.contratos import (AnticipacionFalsa, ModeloAnticipacion,
                                             ModeloPreferencias, PreferenciasFalsas)
    ctx = _contexto()
    for p in [PreferenciasFalsas()]:          # agregar aqui el ModeloPreferencias de P2
        assert isinstance(p, ModeloPreferencias)
        assert set(p.puntajes(ctx, None)) <= set(Accion)
        assert len(p.vector(ctx, None)) == len(p.nombres_caracteristicas())
        assert isinstance(p.clonar(), ModeloPreferencias)
    for a in [AnticipacionFalsa()]:           # agregar aqui el ModeloAnticipacion de P3
        assert isinstance(a, ModeloAnticipacion)
        pedido, prob = a.predecir([])
        assert 0.0 <= prob <= 1.0
