from pathlib import Path

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


def test_lo_aprendido_sobrevive_al_reabrir(tmp_path):
    ruta_db, modelos = str(tmp_path / "robot.db"), str(tmp_path / "modelos")
    s = Sesion(P, ruta_db=ruta_db, carpeta_modelos=modelos)
    oid = _ensenar_uno(s)
    _hasta_ver(s, lambda e: bool(e.objetos))
    s.decidir()
    s.ejecutar(confirmado=True)
    s.responder(True)
    s.cerrar()

    otra = Sesion(P, ruta_db=ruta_db, carpeta_modelos=modelos)
    assert _hasta_ver(otra, lambda e: any(o.id == oid and o.nombre == "objeto x" for o in e.objetos))
    assert otra.cerebro.preferencias_de(P).n_actualizaciones == 1
    assert otra.memoria.modelos(P)


def test_sin_carpeta_no_guarda_modelos():
    s = Sesion(P)
    _ensenar_uno(s)
    assert s.memoria.modelos(P) == []


def test_pantalla_ciclo_con_botones():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("../app/pantalla.py", default_timeout=120)
    at.session_state["ruta_db"] = ":memory:"
    at.run()
    assert not at.exception

    def boton(texto):
        return next(b for b in at.button if b.label == texto)

    boton("Mirar la mesa").click().run()
    boton("Decidir qué hacer").click().run()
    assert not at.exception
    sesion = at.session_state["sesion"]
    assert sesion.decision is not None
    if sesion.necesita_confirmacion:
        boton("Sí, hacelo").click().run()
        boton("Siguiente").click().run()
    else:
        boton("Ejecutar").click().run()
        boton("Aprobado").click().run()
    assert not at.exception
    assert len(sesion.historial()) == 1


def test_pantalla_sin_emojis():
    import re

    texto = (Path(__file__).resolve().parents[1] / "app" / "pantalla.py").read_text(encoding="utf-8")
    assert not re.search("[\U0001F300-\U0001FAFF\u2600-\u27BF]", texto)


def test_sesion_simulada_mesa_estable():
    s = Sesion.simulada(P)
    ids = [(o.id, round(o.x, 3), round(o.y, 3)) for o in s.observar().objetos]
    assert ids and ids == [(o.id, round(o.x, 3), round(o.y, 3)) for o in s.observar().objetos]


def test_sesion_simulada_poner_y_quitar():
    s = Sesion.simulada(P)
    antes = {o.id for o in s.observar().objetos}
    presentes = list(s.mundo.presentes().values())
    tipo = next(t for t, d in s.mundo.tipos.items()
                if d.get("copias", 1) > presentes.count(t) and d.get("empujable", True))
    s.estimulo("poner", tipo)
    nuevos = {o.id for o in s.observar().objetos} - antes
    assert len(nuevos) == 1
    s.estimulo("quitar", nuevos.pop())
    assert {o.id for o in s.observar().objetos} == antes


def test_sesion_simulada_mueve_objetos():
    s = Sesion.simulada(P, cerebro=_CerebroFijo(duda=False))
    obj = s.observar().objetos[0]
    s.decidir()
    assert s.ejecutar().exito
    despues = next(o for o in s.observar().objetos if o.id == obj.id)
    assert (round(despues.x, 2), round(despues.y, 2)) != (round(obj.x, 2), round(obj.y, 2))


def test_etiquetas_legibles():
    from app.etiquetas import legibles

    c = legibles({"hora_sin": 0.25, "hora_cos": 0.5, "aspecto_0": 0.5, "aspecto_3": -0.25,
                  "es:objeto x": 1.0, "hay:senal y": -0.5, "obj_confianza": 0.25, "algo_nuevo": 0.1})
    assert c == {"hora del día": 0.75, "cómo se ve": 0.25, "es 'objeto x'": 1.0,
                 "hay 'senal y' en la mesa": -0.5, "qué tan seguro está del nombre": 0.25, "algo_nuevo": 0.1}
