import csv
import threading
import time

from asistente import mocks
from asistente.interfaces import Memoria
from asistente.memoria import MemoriaSQLite
from asistente.tipos import Accion, Contexto, Decision, Evento, Feedback, Resultado


def _contexto(persona="persona_A"):
    escena = mocks.PercepcionFalsa(semilla=1).observar()
    return Contexto(escena=escena, persona_id=persona, eventos_recientes=[])


def _interaccion(m, persona="persona_A", aprobado=True):
    ctx = _contexto(persona)
    obj = ctx.escena.objetos[0]
    decision = Decision(Accion.TRAER, obj.id, confianza=0.8, motivo="preferencia")
    fb = None if aprobado is None else Feedback(persona, aprobado, fuente="boton")
    return m.registrar_interaccion(ctx, decision, Resultado(True, 1.5, "ok"), fb), obj


def test_cumple_contrato():
    assert isinstance(MemoriaSQLite(), Memoria)


def test_interaccion_completa():
    m = MemoriaSQLite()
    assert _interaccion(m)[0] == 1
    id2, obj = _interaccion(m)
    assert id2 == 2
    h = m.historial("persona_A")[-1]
    assert h["accion"] == "traer"
    assert h["objeto_id"] == obj.id
    assert h["objeto_nombre"] == obj.nombre
    assert h["aprobado"] is True
    assert h["exito"] is True
    assert h["motivo"] == "preferencia"
    assert h["fuente"] == "boton"


def test_interaccion_sin_feedback_ni_objeto():
    m = MemoriaSQLite()
    m.registrar_interaccion(_contexto(), Decision(Accion.IGNORAR, None, 1.0),
                            Resultado(True, 0.0), None)
    h = m.historial("persona_A")[0]
    assert h["aprobado"] is None
    assert h["objeto_nombre"] is None


def test_historial_por_persona_y_limite():
    m = MemoriaSQLite()
    for _ in range(5):
        _interaccion(m, "A")
    for _ in range(2):
        _interaccion(m, "B")
    assert len(m.historial("A", limite=3)) == 3
    assert all(h["persona_id"] == "A" for h in m.historial("A"))
    ids = [h["id"] for h in m.historial("A")]
    assert ids == sorted(ids)


def test_eventos_recientes_filtra_tiempo():
    m = MemoriaSQLite()
    m.registrar_evento(Evento("aparecio", "objeto x", "A", t=time.time() - 1000))
    m.registrar_evento(Evento("pedido", "traer:objeto x", "A"))
    m.registrar_evento(Evento("aparecio", "objeto x", "B"))
    recientes = m.eventos_recientes("A", segundos=300)
    assert [e.tipo for e in recientes] == ["pedido"]
    assert isinstance(recientes[0], Evento)


def test_ensenanzas_y_modelos():
    m = MemoriaSQLite()
    m.registrar_ensenanza("A", "objeto x", "obj_2")
    m.registrar_modelo("A", 1, "modelos/a.pt")
    assert m.ensenanzas("A")[0]["nombre"] == "objeto x"
    assert m.ensenanzas("B") == []


def test_persiste_al_reabrir(tmp_path):
    ruta = str(tmp_path / "robot.db")
    m = MemoriaSQLite(ruta)
    _interaccion(m)
    m.registrar_evento(Evento("aparecio", "objeto x", "persona_A"))
    m.registrar_ensenanza("persona_A", "objeto x", "obj_2")
    m.cerrar()

    otra = MemoriaSQLite(ruta)
    assert len(otra.historial("persona_A")) == 1
    assert len(otra.eventos_recientes("persona_A")) == 1
    assert len(otra.ensenanzas("persona_A")) == 1
    otra.cerrar()


def test_uso_desde_otro_hilo():
    m = MemoriaSQLite()
    errores = []

    def registrar():
        try:
            m.registrar_evento(Evento("aparecio", "objeto x", "A"))
        except Exception as e:
            errores.append(e)

    hilo = threading.Thread(target=registrar)
    hilo.start()
    hilo.join()
    assert not errores
    assert len(m.eventos_recientes("A")) == 1


def test_exportar_csv(tmp_path):
    m = MemoriaSQLite()
    _interaccion(m)
    _interaccion(m, aprobado=None)
    rutas = m.exportar_csv(str(tmp_path / "csv"))
    assert len(rutas) == 5 and all(r.exists() for r in rutas)
    interacciones = next(r for r in rutas if r.stem == "interacciones")
    with open(interacciones, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    assert len(filas) == 2 and "aprobado" in filas[0]
