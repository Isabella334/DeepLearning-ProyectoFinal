import random

import pytest

from asistente.cerebro.anticipacion import (NINGUNO, AnticipacionFrecuencias,
                                            AnticipacionSecuencias, Vocabulario)
from asistente.cerebro.contratos import ModeloAnticipacion
from asistente.tipos import Evento

MODELOS = [AnticipacionFrecuencias, AnticipacionSecuencias]
PEDIDO = "traer:objeto_pedido"


class Flujo:
    """Eventos sinteticos con tiempos explicitos, un segundo entre cada uno."""

    def __init__(self, semilla=0):
        self.rng = random.Random(semilla)
        self.eventos: list[Evento] = []
        self.t = 0.0

    def agregar(self, tipo, detalle):
        self.t += 1.0
        self.eventos.append(Evento(tipo, detalle, "persona_A", self.t))
        return self.eventos[-1]

    def ruido(self):
        for _ in range(self.rng.randint(2, 4)):
            self.agregar("aparecio", f"ruido_{self.rng.randint(1, 5)}")

    def patron(self, modelo, pedido=PEDIDO, senal="objeto_senal"):
        self.ruido()
        self.agregar("aparecio", senal)
        self.agregar("pedido", pedido)
        modelo.actualizar(self.eventos[-12:], pedido)


@pytest.mark.parametrize("Modelo", MODELOS)
def test_sin_datos_no_predice(Modelo):
    a = Modelo()
    assert isinstance(a, ModeloAnticipacion)
    assert a.predecir([]) == (None, 0.0)
    assert a.predecir([Evento("aparecio", "objeto_senal", "persona_A")]) == (None, 0.0)


@pytest.mark.parametrize("Modelo", MODELOS)
def test_aprende_patron_en_pocas_repeticiones(Modelo):
    a, f = Modelo(), Flujo()
    for _ in range(10):
        f.patron(a)
    f.ruido()
    f.agregar("aparecio", "objeto_senal")
    pedido, prob = a.predecir(f.eventos)
    assert pedido == PEDIDO and prob > 0.7


@pytest.mark.parametrize("Modelo", MODELOS)
def test_sin_disparador_no_se_anticipa(Modelo):
    a, f = Modelo(), Flujo()
    for _ in range(10):
        f.patron(a)
    f.ruido()
    assert a.predecir(f.eventos)[1] < 0.5


def test_vocabulario_crece_solo():
    v = Vocabulario()
    assert v.indice("aparecio:a") == 0 and v.indice("aparecio:b") == 1 and v.indice("aparecio:a") == 0
    assert v.indice("aparecio:c", crear=False) == 0 and len(v) == 2


def test_lstm_aprende_un_segundo_patron():
    a, f = AnticipacionSecuencias(), Flujo()
    for _ in range(8):
        f.patron(a)
    for _ in range(8):
        f.patron(a, pedido="apartar:otro_objeto", senal="otra_senal")
    f.ruido()
    f.agregar("aparecio", "otra_senal")
    assert a.predecir(f.eventos)[0] == "apartar:otro_objeto"
    f.ruido()
    f.agregar("aparecio", "objeto_senal")
    assert a.predecir(f.eventos)[0] == PEDIDO


@pytest.mark.parametrize("Modelo", MODELOS)
def test_ventanas_solapadas_no_duplican_muestras(Modelo):
    a, f = Modelo(), Flujo()
    f.ruido()
    f.agregar("aparecio", "objeto_senal")
    f.agregar("pedido", PEDIDO)
    a.actualizar(f.eventos, PEDIDO)
    n = len(a.muestras)
    a.actualizar(f.eventos, NINGUNO)
    assert len(a.muestras) == n + 1


@pytest.mark.parametrize("Modelo", MODELOS)
def test_pedido_sin_evento_no_se_cuenta_dos_veces(Modelo):
    a, f = Modelo(), Flujo()
    f.agregar("aparecio", "objeto_senal")
    a.actualizar(f.eventos, PEDIDO)
    f.agregar("pedido", PEDIDO)
    f.agregar("aparecio", "ruido_1")
    a.actualizar(f.eventos, NINGUNO)
    assert [tok for tok, _ in a.flujo].count(f"pedido:{PEDIDO}") == 1


@pytest.mark.parametrize("Modelo", MODELOS)
def test_rechazos_bajan_la_confianza(Modelo):
    a, f = Modelo(), Flujo()
    for _ in range(6):
        f.patron(a)
    f.agregar("aparecio", "objeto_senal")
    antes = a.predecir(f.eventos)[1]
    for _ in range(6):
        a.actualizar(f.eventos, NINGUNO)
    assert a.predecir(f.eventos)[1] < antes


@pytest.mark.parametrize("Modelo", MODELOS)
def test_guardar_y_cargar(Modelo, tmp_path):
    a, f = Modelo(), Flujo()
    for _ in range(6):
        f.patron(a)
    f.agregar("aparecio", "objeto_senal")
    ruta = str(tmp_path / "anticipacion.pt")
    a.guardar(ruta)
    b = Modelo()
    b.cargar(ruta)
    pa, qa = a.predecir(f.eventos)
    pb, qb = b.predecir(f.eventos)
    assert pa == pb and qa == pytest.approx(qb)
