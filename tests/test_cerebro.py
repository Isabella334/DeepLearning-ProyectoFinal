from asistente import mocks
from asistente.cerebro.cerebro import CerebroAprendiz, cargar_zonas
from asistente.cerebro.contratos import ModeloPreferencias, PreferenciasFalsas
from asistente.tipos import (Accion, Contexto, Decision, Escena, Evento, Feedback, Objeto, Orden,
                             PoseCarrito)

PERSONA = "persona_A"
ZONAS = {Accion.APARTAR: (0.25, 0.17, 0.07), Accion.TRAER: (-0.25, 0.17, 0.07)}


def _objeto(oid, nombre, x=0.0, y=0.0, confianza=0.95):
    return Objeto(id=oid, x=x, y=y, nombre=nombre, confianza=confianza)


def _contexto(objetos, eventos=(), pedido=None, persona=PERSONA):
    escena = Escena(objetos=list(objetos), carrito=PoseCarrito(0, 0, 0), persona_id=persona)
    return Contexto(escena=escena, persona_id=PERSONA, eventos_recientes=list(eventos), pedido=pedido)


class PreferenciasContador(ModeloPreferencias):
    """Puntaje de cada accion = aprobaciones / intentos (0.5 sin datos)."""

    def __init__(self):
        self.conteo = {a: [0, 0] for a in Accion}

    def puntajes(self, contexto, objeto_id):
        return {a: (s + 1) / (n + 2) for a, (s, n) in self.conteo.items()}

    def actualizar(self, contexto, objeto_id, accion, aprobado):
        self.conteo[accion][0] += int(aprobado)
        self.conteo[accion][1] += 1

    def vector(self, contexto, objeto_id):
        return [0.0]

    def nombres_caracteristicas(self):
        return ["nada"]

    def guardar(self, ruta):
        pass

    def cargar(self, ruta):
        pass

    def clonar(self):
        return PreferenciasContador()


def _ensenar_habito(cerebro, objetos, repeticiones=6):
    """Cada vez que aparece la senal, la persona pide traer el objeto pedido."""
    eventos, t = [], 0.0
    for i in range(repeticiones):
        t += 10.0
        eventos.append(Evento("aparecio", f"ruido_{i % 3}", PERSONA, t))
        t += 10.0
        eventos.append(Evento("aparecio", "objeto senal", PERSONA, t))
        eventos.append(Evento("pedido", "traer:objeto pedido", PERSONA, t + 2))
        pedido = Orden("pedido", "traeme el objeto pedido", nombre="objeto pedido", accion=Accion.TRAER)
        cerebro.decidir(_contexto(objetos, eventos, pedido))
        t += 10.0
        eventos.append(Evento("desaparecio", "objeto senal", PERSONA, t))
    t += 10.0
    eventos.append(Evento("aparecio", "objeto senal", PERSONA, t))
    return eventos


def test_zonas_desde_config():
    assert cargar_zonas() == ZONAS


def test_nunca_elige_objetos_fuera_de_la_escena():
    cerebro = CerebroAprendiz(zonas=ZONAS)
    for semilla in range(30):
        ctx = _contexto(mocks.PercepcionFalsa(semilla=semilla).observar().objetos)
        d = cerebro.decidir(ctx)
        assert 0.0 <= d.confianza <= 1.0
        assert d.objeto_id is None or any(o.id == d.objeto_id for o in ctx.escena.objetos)
        cerebro.aprender(ctx, d, Feedback(PERSONA, semilla % 2 == 0))


def test_mesa_vacia_y_sin_persona():
    cerebro = CerebroAprendiz(zonas=ZONAS)
    d = cerebro.decidir(_contexto([]))
    assert d.accion == Accion.IGNORAR and d.objeto_id is None
    d = cerebro.decidir(_contexto([_objeto("obj_1", "objeto a")], persona=None))
    assert d.accion == Accion.IGNORAR and d.objeto_id is None


def test_pedido_gana_y_no_pregunta():
    cerebro = CerebroAprendiz(zonas=ZONAS)
    objetos = [_objeto("obj_1", "objeto a"), _objeto("obj_2", "Lápiz Rojo")]
    pedido = Orden("pedido", "traeme el lapiz rojo", nombre="lapiz rojo", accion=Accion.TRAER)
    d = cerebro.decidir(_contexto(objetos, pedido=pedido))
    assert (d.accion, d.objeto_id, d.motivo) == (Accion.TRAER, "obj_2", "pedido")
    assert not d.preguntar_antes and d.puntajes["pedido"] == 1.0 and "traer" in d.puntajes


def test_pedido_de_algo_que_no_esta():
    cerebro = CerebroAprendiz(zonas=ZONAS)
    pedido = Orden("pedido", "traeme el objeto z", nombre="objeto z", accion=Accion.TRAER)
    d = cerebro.decidir(_contexto([_objeto("obj_1", "objeto a")], pedido=pedido))
    assert d.accion == Accion.IGNORAR and d.objeto_id is None and d.preguntar_antes


def test_pedido_ambiguo_pregunta():
    cerebro = CerebroAprendiz(zonas=ZONAS)
    objetos = [_objeto("obj_1", "lapicero rojo"), _objeto("obj_2", "lapicero azul")]
    pedido = Orden("pedido", "traeme el lapicero", nombre="lapicero", accion=Accion.TRAER)
    d = cerebro.decidir(_contexto(objetos, pedido=pedido))
    assert d.accion == Accion.TRAER and d.objeto_id in ("obj_1", "obj_2") and d.preguntar_antes


def test_regla_se_guarda_y_se_aplica():
    cerebro = CerebroAprendiz(zonas=ZONAS)
    basura = _objeto("obj_2", "basura")
    regla = Orden("regla", "si ves basura apartala", nombre="basura", accion=Accion.APARTAR)
    d = cerebro.decidir(_contexto([_objeto("obj_1", "objeto a"), basura], pedido=regla))
    assert (d.accion, d.objeto_id, d.motivo) == (Accion.APARTAR, "obj_2", "regla")
    d = cerebro.decidir(_contexto([_objeto("obj_1", "objeto a"), basura]))
    assert d.motivo == "regla" and d.objeto_id == "obj_2"
    apartada = _objeto("obj_2", "basura", x=0.25, y=0.17)
    assert cerebro.decidir(_contexto([apartada])).motivo != "regla"


def test_anticipa_el_habito_una_vez_por_momento():
    cerebro = CerebroAprendiz(zonas=ZONAS)
    objetos = [_objeto("obj_1", "objeto senal"), _objeto("obj_2", "objeto pedido")]
    eventos = _ensenar_habito(cerebro, objetos)
    d = cerebro.decidir(_contexto(objetos, eventos))
    assert (d.accion, d.objeto_id, d.motivo) == (Accion.TRAER, "obj_2", "anticipacion")
    assert d.puntajes["anticipacion"] > 0.7 and not d.preguntar_antes
    assert cerebro.decidir(_contexto(objetos, eventos)).motivo != "anticipacion"


def test_no_anticipa_lo_que_ya_esta_en_la_zona_de_la_persona():
    cerebro = CerebroAprendiz(zonas=ZONAS)
    objetos = [_objeto("obj_1", "objeto senal"), _objeto("obj_2", "objeto pedido", x=-0.25, y=0.17)]
    eventos = _ensenar_habito(cerebro, objetos)
    assert cerebro.decidir(_contexto(objetos, eventos)).motivo != "anticipacion"


def test_rechazar_la_anticipacion_baja_la_confianza():
    cerebro = CerebroAprendiz(zonas=ZONAS)
    objetos = [_objeto("obj_1", "objeto senal"), _objeto("obj_2", "objeto pedido")]
    eventos = _ensenar_habito(cerebro, objetos, repeticiones=3)
    ctx = _contexto(objetos, eventos)
    d = cerebro.decidir(ctx)
    antes = cerebro.anticipacion_de(PERSONA).predecir(eventos)[1]
    for _ in range(3):
        cerebro.aprender(ctx, d, Feedback(PERSONA, aprobado=False))
    assert cerebro.anticipacion_de(PERSONA).predecir(eventos)[1] < antes


def test_preferencias_separadas_por_persona():
    cerebro = CerebroAprendiz(preferencias=PreferenciasContador(), zonas=ZONAS)
    objetos = [_objeto("obj_1", "objeto a")]
    ctx_a = _contexto(objetos)
    ctx_b = Contexto(escena=ctx_a.escena, persona_id="persona_B", eventos_recientes=[])
    for _ in range(5):
        cerebro.aprender(ctx_a, Decision(Accion.APARTAR, "obj_1", 1.0), Feedback(PERSONA, True))
    assert cerebro.decidir(ctx_a).accion == Accion.APARTAR
    assert cerebro.preferencias_de("persona_B").conteo[Accion.APARTAR] == [0, 0]
    assert cerebro.decidir(ctx_b).confianza == 0.5


def test_confianza_baja_pregunta_antes():
    cerebro = CerebroAprendiz(preferencias=PreferenciasFalsas(), zonas=ZONAS)
    d = cerebro.decidir(_contexto([_objeto("obj_1", "objeto a")]))
    assert d.motivo == "preferencia" and d.confianza < 0.7 and d.preguntar_antes
    assert set(d.puntajes) == {a.value for a in Accion}


def test_guardar_y_cargar(tmp_path):
    cerebro = CerebroAprendiz(zonas=ZONAS)
    objetos = [_objeto("obj_1", "objeto senal"), _objeto("obj_2", "objeto pedido"),
               _objeto("obj_3", "basura")]
    regla = Orden("regla", "si ves basura apartala", nombre="basura", accion=Accion.APARTAR)
    cerebro.decidir(_contexto([], pedido=regla))
    eventos = _ensenar_habito(cerebro, objetos)
    cerebro.guardar(str(tmp_path / "cerebro"))

    otro = CerebroAprendiz(zonas=ZONAS)
    otro.cargar(str(tmp_path / "cerebro"))
    assert otro.reglas == cerebro.reglas
    assert otro.anticipacion_de(PERSONA).predecir(eventos) == cerebro.anticipacion_de(PERSONA).predecir(eventos)
    assert otro.decidir(_contexto(objetos, eventos)).motivo == "regla"
