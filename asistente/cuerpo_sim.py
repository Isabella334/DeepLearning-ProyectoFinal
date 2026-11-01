from __future__ import annotations

import numpy as np

from . import sim_mujoco as sim
from .interfaces import Cuerpo, Percepcion
from .sim_mujoco import MundoSim
from .tipos import Accion, Escena, Objeto, PoseCarrito, Resultado

__all__ = ["MundoSim", "PercepcionSim", "CuerpoSim"]


class PercepcionSim(Percepcion):
    """Los nombres los pone el Reconocedor, no el mundo."""

    def __init__(self, mundo: MundoSim, reconocedor, persona_id: str = "persona_A", ruido: float = 0.02):
        self.m, self.reconocedor, self.persona_id, self.ruido = mundo, reconocedor, persona_id, ruido
        self.ultima = None

    def observar(self) -> Escena:
        p, ang = sim.pose_carrito(self.m.model, self.m.data)
        objetos = []
        for oid, tipo in self.m.presentes().items():
            x, y = self.m.pos(oid)
            if not self.m.en_mesa((x, y), margen=0.02):
                continue          # se cayo de la cartulina: la camara no lo veria
            emb = [v + float(np.random.normal(0, self.ruido)) for v in self.m.tipos[tipo]["aspecto"]]
            nombre, conf = self.reconocedor.reconocer(self.persona_id, emb)
            objetos.append(Objeto(id=oid, x=float(x), y=float(y), nombre=nombre,
                                  confianza=float(np.clip(conf, 0.0, 1.0)), embedding=emb))
        self.ultima = Escena(objetos=objetos, carrito=PoseCarrito(float(p[0]), float(p[1]), float(ang)),
                             persona_id=self.persona_id)
        return self.ultima

    def ensenar(self, persona_id, objeto_id, nombre):
        obj = next((o for o in (self.ultima.objetos if self.ultima else []) if o.id == objeto_id), None)
        if obj is not None:
            self.reconocedor.ensenar(persona_id, nombre, obj.embedding)


class CuerpoSim(Cuerpo):
    ZONA_DE = {Accion.APARTAR: "apartar", Accion.TRAER: "persona"}
    HECHO = {Accion.APARTAR: "objeto apartado", Accion.TRAER: "objeto traido"}

    def __init__(self, mundo: MundoSim, limite_s: float = 60.0):
        self.m, self.limite_s = mundo, limite_s

    def ejecutar(self, accion, objeto, escena):
        if accion == Accion.IGNORAR:
            return Resultado(True, 0.0, "no hice nada")
        if accion == Accion.ACERCARSE:
            return self._acercarse()
        if accion in self.ZONA_DE:
            return self._empujar(accion, objeto)
        return Resultado(False, 0.0, f"accion desconocida: {accion}")

    def _empujar(self, accion, objeto):
        if objeto is None:
            return Resultado(False, 0.0, f"{accion.value} necesita un objeto")
        if objeto.id not in self.m.presentes() or not self.m.en_mesa(self.m.pos(objeto.id), margen=0.02):
            return Resultado(False, 0.0, f"{objeto.id} ya no esta en la mesa")
        if not self.m.empujable(objeto.id):
            return Resultado(False, 0.0, f"{objeto.id} no se puede empujar")
        zona = self.ZONA_DE[accion]
        if self.m.en_zona(self.m.pos(objeto.id), zona):
            return Resultado(True, 0.0, f"ya estaba en la zona de {zona}")

        d, t0 = self.m.data, self.m.data.time
        emp = self.m.empujador(objeto.id, zona)
        try:
            while d.time - t0 < self.limite_s:
                if emp.paso() == "listo":
                    return Resultado(True, d.time - t0, self.HECHO[accion])
                self.m.paso()
                xy = self.m.pos(objeto.id)
                if not self.m.en_mesa(xy, margen=0.02) or self.m.altura_actual(objeto.id) < -0.01:
                    return Resultado(False, d.time - t0, "el objeto se salio de la cartulina")
            return Resultado(False, d.time - t0, "se acabo el tiempo")
        finally:
            self.m.frenar()

    def _acercarse(self, tolerancia: float = 0.03):
        d, t0 = self.m.data, self.m.data.time
        meta, _ = self.m.zona("persona")
        try:
            while d.time - t0 < self.limite_s:
                if sim.ir_hacia(self.m.model, d, meta, vel=10.0) < tolerancia:
                    return Resultado(True, d.time - t0, "llegue a la persona")
                self.m.paso()
            return Resultado(False, d.time - t0, "se acabo el tiempo")
        finally:
            self.m.frenar()

    def cerrar(self):
        if self.m.visor is not None:
            self.m.visor.close()
