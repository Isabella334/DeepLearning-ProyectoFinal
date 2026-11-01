import itertools

import numpy as np
import pytest

pytest.importorskip("mujoco")

from asistente.cuerpo_sim import CuerpoSim, MundoSim, PercepcionSim   # noqa: E402
from asistente.interfaces import Cuerpo, Percepcion                    # noqa: E402
from asistente.mocks import ReconocedorFalso                           # noqa: E402
from asistente.sim_mujoco import cargar_config, cargar_tipos           # noqa: E402
from asistente.tipos import DESCONOCIDO, Accion, Escena, Objeto        # noqa: E402

POSICIONES = [(-0.2, -0.15), (0.0, -0.1), (0.2, -0.15), (-0.15, 0.05), (0.15, 0.05), (0.0, 0.12)]


def _tipo(cfg, empujable=True):
    """Elige un tipo sin escribir su nombre en el codigo."""
    tipos = cargar_tipos(cfg)
    return next(n for n, t in tipos.items() if t.get("empujable", True) == empujable
                and t["forma"] == "box")


def _mundo_con_uno(tipo, x, y):
    mundo = MundoSim(iniciales=[])
    mundo.colocar("obj_1", tipo, x, y)
    escena = PercepcionSim(mundo, ReconocedorFalso()).observar()
    return mundo, CuerpoSim(mundo, limite_s=60), escena, escena.objetos[0]


def test_formas_y_aspectos():
    cfg = cargar_config()
    tipos = cargar_tipos(cfg)
    assert set(tipos) == set(cfg["objetos_del_experimento"])
    for (a, ta), (b, tb) in itertools.combinations(tipos.items(), 2):
        va, vb = np.array(ta["aspecto"]), np.array(tb["aspecto"])
        coseno = va @ vb / np.linalg.norm(va) / np.linalg.norm(vb)
        assert coseno < cfg["reconocimiento"]["umbral_similitud"] - 0.05, f"{a} y {b} se confunden"


def test_mesa_con_las_medidas_de_la_cartulina():
    mundo = MundoSim()
    ancho, alto = mundo.cfg["cartulina"]["ancho"], mundo.cfg["cartulina"]["alto"]
    assert np.allclose(mundo.model.geom("mesa").size[:2], [ancho / 2, alto / 2])
    for oid in mundo.presentes():
        assert mundo.en_mesa(mundo.pos(oid))
        assert not mundo.en_zona(mundo.pos(oid), "apartar") and not mundo.en_zona(mundo.pos(oid), "persona")


def test_colocar_y_quitar():
    cfg = cargar_config()
    mundo = MundoSim(iniciales=[])
    percepcion = PercepcionSim(mundo, ReconocedorFalso())
    assert isinstance(percepcion, Percepcion)
    assert percepcion.observar().objetos == []
    senal = _tipo(cfg, empujable=False)
    mundo.colocar("obj_9", senal, 0.0, 0.0)
    escena = percepcion.observar()
    assert isinstance(escena, Escena) and [o.id for o in escena.objetos] == ["obj_9"]
    assert escena.objetos[0].nombre == DESCONOCIDO
    mundo.quitar("obj_9")
    assert percepcion.observar().objetos == []


def test_ensenar_un_nombre():
    mundo = MundoSim()
    percepcion = PercepcionSim(mundo, ReconocedorFalso())
    escena = percepcion.observar()
    primero = escena.objetos[0]
    percepcion.ensenar("persona_A", primero.id, "mi cosa")
    nombres = {o.id: o.nombre for o in percepcion.observar().objetos}
    assert nombres[primero.id] == "mi cosa"
    assert list(nombres.values()).count("mi cosa") == 1


@pytest.mark.parametrize("accion,zona", [(Accion.APARTAR, "apartar"), (Accion.TRAER, "persona")])
@pytest.mark.parametrize("x,y", POSICIONES)
def test_empujar_a_la_zona(accion, zona, x, y):
    mundo, cuerpo, escena, obj = _mundo_con_uno(_tipo(cargar_config()), x, y)
    r = cuerpo.ejecutar(accion, obj, escena)
    assert r.exito, r.mensaje
    assert mundo.en_zona(mundo.pos(obj.id), zona, margen=0.0)


def test_acercarse_e_ignorar():
    mundo, cuerpo, escena, obj = _mundo_con_uno(_tipo(cargar_config()), 0.15, -0.1)
    assert isinstance(cuerpo, Cuerpo)
    antes = mundo.pos(obj.id)
    r = cuerpo.ejecutar(Accion.IGNORAR, None, escena)
    assert r.exito and np.allclose(mundo.pos(obj.id), antes)
    r = cuerpo.ejecutar(Accion.ACERCARSE, None, escena)
    assert r.exito, r.mensaje
    p = PercepcionSim(mundo, ReconocedorFalso()).observar().carrito
    centro, _ = mundo.zona("persona")
    assert np.linalg.norm([p.x - centro[0], p.y - centro[1]]) < 0.05


def test_nunca_se_cuelga():
    cfg = cargar_config()
    mundo, cuerpo, escena, obj = _mundo_con_uno(_tipo(cfg), 0.2, -0.15)
    cuerpo.limite_s = 0.5
    r = cuerpo.ejecutar(Accion.TRAER, obj, escena)
    assert not r.exito and r.duracion_s <= 0.6
    # Pedidos imposibles: fallan sin simular
    assert not cuerpo.ejecutar(Accion.APARTAR, None, escena).exito
    fantasma = Objeto(id="no_existe", x=0, y=0)
    assert not cuerpo.ejecutar(Accion.TRAER, fantasma, escena).exito
    mundo.colocar("obj_2", _tipo(cfg, empujable=False), -0.1, -0.05)
    r = cuerpo.ejecutar(Accion.APARTAR, Objeto(id="obj_2", x=-0.1, y=-0.05), escena)
    assert not r.exito and r.duracion_s == 0.0


def test_ciclos_con_varios_objetos():
    mundo = MundoSim(semilla=3)
    percepcion, cuerpo = PercepcionSim(mundo, ReconocedorFalso()), CuerpoSim(mundo, limite_s=30)
    rng = np.random.default_rng(0)
    for _ in range(4):
        escena = percepcion.observar()
        if not escena.objetos:
            break
        obj = escena.objetos[rng.integers(len(escena.objetos))]
        accion = [Accion.APARTAR, Accion.TRAER, Accion.ACERCARSE][rng.integers(3)]
        r = cuerpo.ejecutar(accion, obj, escena)
        assert r.duracion_s <= 30.1
