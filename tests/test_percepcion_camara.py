"""Percepcion con camara, sin camara ni YOLO: una imagen dibujada con los 4 marcadores y
un detector falso. Revisa la conversion a metros, los ids estables, ensenar y el gemelo."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from asistente.configuracion import cargar_config
from asistente.mocks import ReconocedorFalso
from asistente.tipos import DESCONOCIDO, Deteccion
from asistente.vision.percepcion_camara import PercepcionCamara

ESCALA = 1000          # pixeles por metro en la imagen de prueba (vista desde arriba)


def _px(xy, alto_img):
    return int(xy[0] * ESCALA + 500), int(alto_img / 2 - xy[1] * ESCALA)


def _imagen():
    cfg = cargar_config()
    img = np.full((700, 1000, 3), 230, np.uint8)
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i, c in cfg["aruco"]["centros"].items():
        m = cv2.aruco.generateImageMarker(dic, int(i), 80)
        x, y = _px(c, 700)
        img[y - 40:y + 40, x - 40:x + 40] = m[..., None]
    return img


class Fuente:
    def __init__(self, img): self.img = img
    def read(self): return True, self.img.copy()


class DetectorFalso:
    """Devuelve cajas en las posiciones (en metros) que se le digan."""
    def __init__(self): self.objetos = []
    def detectar(self, imagen):
        out = []
        for (x, y), color in self.objetos:
            px, py = _px((x, y), imagen.shape[0])
            imagen[py - 20:py + 5, px - 15:px + 15] = color
            out.append(Deteccion(0.9, (px - 15, py - 20, px + 15, py + 5)))
        return out


def _percepcion(mundo=None):
    det = DetectorFalso()
    return PercepcionCamara(ReconocedorFalso(), detector=det, camara=Fuente(_imagen()), mundo=mundo), det


def test_metros_ids_estables_y_ensenar():
    p, det = _percepcion()
    det.objetos = [((0.1, 0.0), (0, 0, 255)), ((-0.2, -0.1), (255, 0, 0))]
    e = p.observar()
    assert len(e.objetos) == 2
    ids = {o.id for o in e.objetos}
    assert all(o.nombre == DESCONOCIDO for o in e.objetos)
    rojo = min(e.objetos, key=lambda o: abs(o.x - 0.1) + abs(o.y))
    assert abs(rojo.x - 0.1) < 0.02 and abs(rojo.y) < 0.02          # pixeles -> metros
    det.objetos = [((0.11, 0.01), (0, 0, 255)), ((-0.2, -0.1), (255, 0, 0))]   # se movio 1 cm
    assert {o.id for o in p.observar().objetos} == ids                # mismos ids
    p.ensenar("persona_A", rojo.id, "lapicero rojo")
    e = p.observar()
    assert next(o for o in e.objetos if o.id == rojo.id).nombre == "lapicero rojo"


def test_objeto_que_se_va_se_olvida():
    p, det = _percepcion()
    det.objetos = [((0.0, 0.0), (0, 255, 0))]
    p.observar()
    det.objetos = []
    for _ in range(p.olvidar_tras + 1):
        e = p.observar()
    assert e.objetos == [] and p.rastros == []


def test_gemelo_digital():
    from asistente.cuerpo_sim import MundoSim
    mundo = MundoSim(con_camara=True, iniciales=[])
    p, det = _percepcion(mundo)
    det.objetos = [((0.05, -0.05), (0, 0, 255))]
    e = p.observar()
    oid = e.objetos[0].id
    assert oid in mundo.presentes()
    assert np.allclose(mundo.pos(oid), [0.05, -0.05], atol=0.02)
