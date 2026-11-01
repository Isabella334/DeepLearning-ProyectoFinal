import numpy as np

from app.imaginacion import dibujar_imaginacion
from asistente import mocks
from asistente.tipos import Accion, Decision


def test_dibuja_cada_accion():
    escena = mocks.PercepcionFalsa(semilla=2).observar()
    oid = escena.objetos[0].id if escena.objetos else None
    for decision in [None, Decision(Accion.TRAER, oid, 0.8), Decision(Accion.APARTAR, oid, 0.5, True),
                     Decision(Accion.ACERCARSE, None, 0.6), Decision(Accion.IGNORAR, None, 1.0)]:
        imagen = dibujar_imaginacion(escena, decision)
        assert imagen.ndim == 3 and imagen.shape[2] == 3 and imagen.dtype == np.uint8
