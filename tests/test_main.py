import time

import main
from asistente import mocks
from asistente.cerebro.cerebro import CerebroAprendiz
from asistente.interfaces import Cuerpo, EntradaFeedback, Percepcion
from asistente.tipos import Escena, Feedback, Objeto, PoseCarrito, Resultado

PERSONA = "persona_A"


class MesaGuionada(Percepcion):
    def __init__(self):
        self.objetos: list[Objeto] = []

    def observar(self):
        return Escena(objetos=list(self.objetos), carrito=PoseCarrito(0, 0, 0), persona_id=PERSONA)

    def ensenar(self, persona_id, objeto_id, nombre):
        pass


class CuerpoQueAnota(Cuerpo):
    def __init__(self):
        self.hechas = []

    def ejecutar(self, accion, objeto, escena):
        self.hechas.append((accion.value, objeto.nombre if objeto else None))
        return Resultado(exito=True, duracion_s=0.0)


class PersonaGuionada(EntradaFeedback):
    def __init__(self):
        self.respuesta: bool | None = True
        self.preguntas = 0

    def esperar(self, persona_id, timeout_s=10.0):
        self.preguntas += 1
        return None if self.respuesta is None else Feedback(persona_id, self.respuesta)


class VozGuionada(mocks.VozTeclado):
    def __init__(self):
        self.frase: str | None = None

    def escuchar(self, timeout_s=5.0):
        frase, self.frase = self.frase, None
        return frase


class MemoriaConReloj(mocks.MemoriaEnRAM):
    """Cada ciclo pasa 10 s, para que los eventos de ciclos distintos no parezcan simultaneos."""

    def __init__(self):
        super().__init__()
        self.reloj = time.time()

    def registrar_evento(self, evento):
        self.reloj += 0.01
        evento.t = self.reloj
        super().registrar_evento(evento)


class Robot:
    def __init__(self, cerebro=None):
        self.mesa, self.cuerpo, self.persona = MesaGuionada(), CuerpoQueAnota(), PersonaGuionada()
        self.voz, self.memoria = VozGuionada(), MemoriaConReloj()
        self.cerebro = cerebro or CerebroAprendiz()
        self.anteriores = set()

    def ciclo(self, frase=None):
        self.voz.frase = frase
        self.memoria.reloj += 10.0
        self.anteriores = main.un_ciclo(self.mesa, self.cuerpo, self.cerebro, self.memoria,
                                        mocks.ExplicadorSimple(), self.persona, self.voz, PERSONA,
                                        self.anteriores)


def _objeto(oid, nombre):
    return Objeto(id=oid, x=0.0, y=0.0, nombre=nombre, confianza=0.95)


def test_pedido_por_voz_se_ejecuta():
    r = Robot()
    r.mesa.objetos = [_objeto("obj_1", "objeto a"), _objeto("obj_2", "objeto b")]
    r.ciclo("traeme el objeto b")
    assert r.cuerpo.hechas == [("traer", "objeto b")]
    assert [e.detalle for e in r.memoria.eventos if e.tipo == "pedido"] == ["traer:objeto b"]


def test_regla_no_se_registra_como_pedido():
    r = Robot()
    r.mesa.objetos = [_objeto("obj_1", "basura")]
    r.ciclo("si ves basura, apartala")
    assert r.cuerpo.hechas == [("apartar", "basura")]
    assert not [e for e in r.memoria.eventos if e.tipo == "pedido"]


def test_si_duda_pregunta_y_no_actua_sin_confirmacion():
    r = Robot()
    r.mesa.objetos = [_objeto("obj_1", "objeto a")]
    for respuesta in (False, None):
        r.persona.respuesta = respuesta
        r.ciclo()
        assert r.cuerpo.hechas == []
    r.persona.respuesta = True
    r.ciclo()
    assert len(r.cuerpo.hechas) == 1
    assert r.persona.preguntas == 3          # la confirmacion ya es el feedback: no pregunta dos veces
    assert [h["aprobado"] for h in r.memoria.historial(PERSONA)] == [False, None, True]


def test_pedido_de_algo_que_no_esta():
    r = Robot()
    r.mesa.objetos = [_objeto("obj_1", "objeto a")]
    r.ciclo("traeme el objeto z")
    assert r.cuerpo.hechas == [("ignorar", None)] and r.persona.preguntas == 0


def test_aprende_el_habito_y_se_adelanta():
    r = Robot()
    r.persona.respuesta = False                 # rechaza lo que hace por su cuenta
    pedido = _objeto("obj_2", "objeto pedido")
    for _ in range(5):
        r.mesa.objetos = [pedido]
        r.ciclo()
        r.mesa.objetos = [pedido, _objeto("obj_1", "objeto senal")]
        r.ciclo("traeme el objeto pedido")
    r.mesa.objetos = [pedido]
    r.ciclo()
    r.cuerpo.hechas.clear()
    r.persona.respuesta = True
    r.mesa.objetos = [pedido, _objeto("obj_1", "objeto senal")]
    r.ciclo()
    assert r.cuerpo.hechas == [("traer", "objeto pedido")]
    assert r.memoria.historial(PERSONA)[-1]["aprobado"] is True


def test_vocabulario_de_la_voz():
    r = Robot()
    r.voz.vocabulario = []
    r.mesa.objetos = [_objeto("obj_1", "objeto b"), _objeto("obj_2", "desconocido"),
                      _objeto("obj_3", "objeto a")]
    r.ciclo()
    assert r.voz.vocabulario == ["objeto a", "objeto b"]
