from app.feedback_pantalla import FeedbackPantalla
from app.sesion import Sesion
from asistente.interfaces import Cerebro, EntradaFeedback
from asistente.tipos import DESCONOCIDO, Accion, Decision

P = "persona_A"


class _CerebroFijo(Cerebro):
    def __init__(self, duda=True):
        self.duda = duda

    def decidir(self, contexto):
        obj = contexto.escena.objetos[0]
        return Decision(Accion.TRAER, obj.id, confianza=0.4 if self.duda else 0.9,
                        preguntar_antes=self.duda, motivo="preferencia")

    def aprender(self, contexto, decision, feedback):
        self.ultimo = feedback

    def guardar(self, ruta):
        pass

    def cargar(self, ruta):
        pass


def _hasta_ver(sesion, condicion, intentos=20):
    for _ in range(intentos):
        if condicion(sesion.observar()):
            return True
    return False


def _ensenar_uno(sesion, nombre="objeto x"):
    _hasta_ver(sesion, lambda e: any(o.nombre == DESCONOCIDO for o in e.objetos))
    obj = next(o for o in sesion.escena.objetos if o.nombre == DESCONOCIDO)
    assert sesion.ensenar(obj.id, nombre)
    return obj.id


def test_feedback_pantalla():
    f = FeedbackPantalla()
    assert isinstance(f, EntradaFeedback)
    f.pulsar(P, True)
    fb = f.esperar(P)
    assert fb.aprobado is True and fb.fuente == "boton" and fb.persona_id == P
    assert f.esperar(P) is None


def test_feedback_pantalla_por_persona():
    f = FeedbackPantalla()
    f.pulsar("B", False)
    assert f.esperar(P) is None
    assert f.esperar("B").aprobado is False


def test_ciclo_completo():
    s = Sesion(P)
    s.observar()
    s.decidir()
    assert s.decision is not None and s.explicacion.texto
    s.ejecutar(confirmado=True)
    s.responder(True)
    h = s.historial()
    assert len(h) == 1 and h[0]["aprobado"] is True
    assert s.decision is None


def test_ensenar_y_reconocer():
    s = Sesion(P)
    oid = _ensenar_uno(s)
    assert _hasta_ver(s, lambda e: any(o.id == oid and o.nombre == "objeto x" for o in e.objetos))
    assert len(s.memoria.ensenanzas(P)) == 1


def test_ensenar_invalido():
    s = Sesion(P)
    s.observar()
    assert not s.ensenar("obj_que_no_existe", "objeto x")
    if s.escena.objetos:
        assert not s.ensenar(s.escena.objetos[0].id, "   ")
    assert s.memoria.ensenanzas(P) == []


def test_pedido_llega_al_cerebro():
    s = Sesion(P)
    oid = _ensenar_uno(s)
    _hasta_ver(s, lambda e: any(o.id == oid and o.nombre == "objeto x" for o in e.objetos))
    s.pedir("objeto x")
    decision, explicacion = s.decidir()
    assert decision.motivo == "pedido" and decision.objeto_id == oid
    assert any(e.tipo == "pedido" for e in s.memoria.eventos_recientes(P))
    assert s.decidir()[0].motivo != "pedido"       # el pedido se consume una sola vez


def test_no_ejecuta_sin_confirmar():
    cerebro = _CerebroFijo(duda=True)
    s = Sesion(P, cerebro=cerebro)
    _hasta_ver(s, lambda e: bool(e.objetos))
    s.decidir()
    assert s.necesita_confirmacion
    assert s.ejecutar(confirmado=False).exito is False
    s.responder()
    assert cerebro.ultimo.aprobado is False          # el "no" a la pregunta es el feedback
    assert s.historial()[0]["aprobado"] is False


def test_responder_sin_reaccion():
    s = Sesion(P, cerebro=_CerebroFijo(duda=False))
    _hasta_ver(s, lambda e: bool(e.objetos))
    s.decidir()
    assert not s.necesita_confirmacion
    assert s.ejecutar().exito
    s.responder(None)
    assert s.historial()[0]["aprobado"] is None


def test_estimulo_llama_gancho():
    recibidos = []
    s = Sesion(P, ganchos={"aparecer": recibidos.append})
    s.estimulo("aparecer", "obj_1")
    s.estimulo("quitar", "obj_1")                    # sin gancho: no hace nada
    assert recibidos == ["obj_1"]


def test_persiste_entre_sesiones(tmp_path):
    ruta = str(tmp_path / "robot.db")
    s = Sesion(P, ruta_db=ruta)
    _hasta_ver(s, lambda e: bool(e.objetos))
    s.decidir()
    s.ejecutar(confirmado=True)
    s.responder(True)
    s.cerrar()
    assert len(Sesion(P, ruta_db=ruta).historial()) == 1
