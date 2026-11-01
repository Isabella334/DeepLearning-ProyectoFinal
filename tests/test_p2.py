"""
Pruebas de P2: reconocedor con pocos ejemplos y modelo de preferencias.
    python -m pytest tests/test_p2.py -q
"""
import random
from collections import Counter

import numpy as np
import pytest

from asistente import mocks
from asistente.cerebro.preferencias import ACCIONES, PreferenciasRed, cargar_historial_pesos
from asistente.tipos import DESCONOCIDO, Accion, Contexto, Escena, Objeto, PoseCarrito
from asistente.vision.reconocedor import ReconocedorPocosEjemplos

TIPOS = mocks.PercepcionFalsa.TIPOS


def _vista(tipo, rng, ruido=0.03):
    return list(np.asarray(TIPOS[tipo]) + rng.normal(0, ruido, len(TIPOS[tipo])))


# ---------------------------------------------------------------- Reconocedor
@pytest.mark.parametrize("modo", ["prototipo", "cabeza"])
def test_reconoce_con_3_ejemplos_en_percepcion_falsa(modo):
    """'Listo cuando': con 3 ejemplos por objeto reconoce los objetos de la percepcion falsa."""
    r = ReconocedorPocosEjemplos(modo=modo)
    perc = mocks.PercepcionFalsa(r, persona_id="A", semilla=3)
    vistos = Counter()
    while min(vistos[t] for t in TIPOS) < 3:
        for o in perc.observar().objetos:
            tipo = perc.tipo_oculto[o.id]
            if vistos[tipo] < 3:
                perc.ensenar("A", o.id, tipo)
                vistos[tipo] += 1
    aciertos = [o.nombre == perc.tipo_oculto[o.id] for _ in range(30) for o in perc.observar().objetos]
    assert np.mean(aciertos) >= 0.95


@pytest.mark.parametrize("modo", ["prototipo", "cabeza"])
def test_lo_de_una_persona_no_lo_sabe_otra(modo):
    rng = np.random.default_rng(0)
    r = ReconocedorPocosEjemplos(modo=modo)
    for t in TIPOS:
        for _ in range(3):
            r.ensenar("A", t, _vista(t, rng))
    assert r.nombres("B") == []
    assert all(r.reconocer("B", _vista(t, rng))[0] == DESCONOCIDO for t in TIPOS)
    assert sorted(r.nombres("A")) == sorted(TIPOS)


def test_v2_rechaza_lo_que_no_le_ensenaron():
    """La v1 confunde objetos parecidos no ensenados; la v2 (con clase 'desconocido') no."""
    rng = np.random.default_rng(1)
    r = ReconocedorPocosEjemplos(modo="cabeza")
    for _ in range(20):                       # el robot mira la mesa
        for t in TIPOS:
            r.reconocer("A", _vista(t, rng))
    ensenados, otros = list(TIPOS)[:3], list(TIPOS)[3:]
    for t in ensenados:
        for _ in range(3):
            r.ensenar("A", t, _vista(t, rng))
    r.aprender_del_fondo = False
    assert np.mean([r.reconocer("A", _vista(t, rng))[0] == DESCONOCIDO for t in otros for _ in range(30)]) >= 0.9
    assert np.mean([r.reconocer("A", _vista(t, rng))[0] == t for t in ensenados for _ in range(30)]) >= 0.9


@pytest.mark.parametrize("modo", ["prototipo", "cabeza"])
def test_guardar_y_cargar(modo, tmp_path):
    rng = np.random.default_rng(2)
    r = ReconocedorPocosEjemplos(modo=modo)
    for t in TIPOS:
        r.ensenar("A", t, _vista(t, rng))
    pruebas = [_vista(t, rng) for t in TIPOS]
    antes = [r.reconocer("A", v) for v in pruebas]
    r.guardar(tmp_path / "rec")
    r2 = ReconocedorPocosEjemplos(modo=modo)
    r2.cargar(tmp_path / "rec")
    r2.aprender_del_fondo = r.aprender_del_fondo = False
    assert r2.nombres("A") == r.nombres("A")
    for (n1, c1), (n2, c2) in zip(antes, [r2.reconocer("A", v) for v in pruebas]):
        assert n1 == n2 and abs(c1 - c2) < 1e-4


def test_nombres_se_normalizan_y_se_pueden_olvidar():
    r = ReconocedorPocosEjemplos()
    r.ensenar("A", "  Lapicero   Rojo ", [1.0, 0.1, 0.1, 0.2])
    assert r.nombres("A") == ["lapicero rojo"]
    with pytest.warns(UserWarning):
        r.ensenar("A", DESCONOCIDO, [0.1, 0.1, 1.0, 0.2])      # no es un nombre: se ignora
    assert r.nombres("A") == ["lapicero rojo"]
    r.olvidar("A", "lapicero rojo")
    assert r.reconocer("A", [1.0, 0.1, 0.1, 0.2])[0] == DESCONOCIDO


def test_embedding_real_con_resnet():
    pytest.importorskip("torchvision")
    rng = np.random.default_rng(0)

    def recorte(bgr):                         # un objeto de color sobre la cartulina, con ruido
        img = np.full((60, 80, 3), 200, dtype=np.uint8)
        img[15:45, 20:60] = bgr
        return np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)

    r = ReconocedorPocosEjemplos()
    rojo, azul = (30, 30, 220), (220, 40, 30)          # BGR, como OpenCV
    e = r.embedding(recorte(rojo))
    assert isinstance(e, list) and len(e) == 512
    r.ensenar("A", "rojo", e)
    r.ensenar("A", "azul", r.embedding(recorte(azul)))
    p = r.puntajes("A", r.embedding(recorte(rojo)))
    assert p["rojo"] > p["azul"]
    assert len(r.embeddings([recorte(rojo), recorte(azul)])) == 2


# ---------------------------------------------------------------- Preferencias
def _contexto(objetos, persona="A"):
    return Contexto(escena=Escena(objetos=objetos, carrito=PoseCarrito(0, 0, 0), persona_id=persona),
                    persona_id=persona, eventos_recientes=[])


def test_preferencias_contrato_y_casos_borde():
    p = PreferenciasRed()
    vacio = _contexto([])
    for oid in (None, "no_existe"):
        s = p.puntajes(vacio, oid)
        assert set(s) == set(Accion) and all(0.0 <= v <= 1.0 for v in s.values())
    obj = Objeto("obj_0", 0.1, 0.0, nombre="lapicero rojo", confianza=0.9, embedding=[0.0] * 512)
    ctx = _contexto([obj])
    assert len(p.vector(ctx, "obj_0")) == len(p.nombres_caracteristicas()) == p.n_entradas
    assert "es:lapicero rojo" in p.nombres_caracteristicas()      # los nombres llegan de los datos
    assert p.predecir_vectores(p.fondo()).shape[1] == len(ACCIONES)
    p.actualizar(vacio, None, Accion.ACERCARSE, aprobado=True)


def test_preferencias_aprende_un_gusto_simple():
    """Con un usuario simulado (3 acciones distintas) supera el 80% de aciertos.
    La velocidad (~30 interacciones) la mide experimentos/usuarios_simulados.py."""
    rng = random.Random(0)
    rec = ReconocedorPocosEjemplos()
    perc = mocks.PercepcionFalsa(rec, persona_id="A", semilla=0)
    for _ in range(10):                                   # le ensena los nombres
        for o in perc.observar().objetos:
            perc.ensenar("A", o.id, perc.tipo_oculto[o.id])
    regla = {t: ACCIONES[i % 3] for i, t in enumerate(TIPOS)}       # apartar / traer / ignorar
    p = PreferenciasRed(semilla=0)

    def situacion():
        e = perc.observar()
        while not e.objetos:
            e = perc.observar()
        e.t = 1_790_000_000.0                             # hora fija: el resultado no depende del reloj
        o = rng.choice(e.objetos)
        return Contexto(escena=e, persona_id="A", eventos_recientes=[]), o.id, perc.tipo_oculto[o.id]

    prueba = [situacion() for _ in range(100)]
    for _ in range(60):
        ctx, oid, tipo = situacion()
        a = p.elegir_accion(ctx, oid, exploracion=0.1, rng=rng)
        p.actualizar(ctx, oid, a, aprobado=(a == regla[tipo]))
    aciertos = np.mean([max(s := p.puntajes(c, o), key=s.get) == regla[t] for c, o, t in prueba])
    assert aciertos > 0.8


def test_clonar_guardar_cargar_e_historial_de_pesos(tmp_path):
    obj = Objeto("obj_0", 0.2, 0.1, nombre="borrador", embedding=[0.9, 0.9, 0.9, 0.6])
    ctx = _contexto([obj])
    p = PreferenciasRed()
    copia = p.clonar()
    for _ in range(3):
        copia.actualizar(ctx, "obj_0", Accion.APARTAR, aprobado=True)
    assert p.n_actualizaciones == 0 and copia.n_actualizaciones == 3      # copias independientes
    assert copia.puntajes(ctx, "obj_0")[Accion.APARTAR] > p.puntajes(ctx, "obj_0")[Accion.APARTAR]

    copia.guardar(tmp_path / "pref_A.pt")
    otra = PreferenciasRed()
    otra.cargar(tmp_path / "pref_A.pt")
    assert otra.puntajes(ctx, "obj_0") == pytest.approx(copia.puntajes(ctx, "obj_0"))
    assert otra.nombres_caracteristicas() == copia.nombres_caracteristicas()

    h = cargar_historial_pesos(tmp_path / "pref_A_pesos.npz")      # lo que usa P1
    assert [f["n"] for f in h["fotos"]] == [0, 1, 2, 3]
    assert h["fotos"][0]["W"][0].shape == (copia.ocultas, copia.n_entradas)
    assert h["nombres_salida"] == [a.value for a in ACCIONES]
