"""
Cuerpo y percepcion SIMULADOS con MuJoCo (punto de partida de la Persona 1).

Ya funciona: APARTAR el borrador (usa sim_mujoco.py).
Falta (tarea de P1, ver docs/tareas/P1_simulacion.md): varios objetos, TRAER,
ACERCARSE, medidas y zonas desde config.yaml, vista de la imaginacion.

En simulacion la percepcion es "perfecta": se leen las posiciones directo de
MuJoCo. Asi la simulacion NO depende de la camara ni de la CNN.
"""
from __future__ import annotations

import numpy as np
import mujoco

from . import sim_mujoco as sim
from .interfaces import Cuerpo, Percepcion
from .tipos import Accion, Escena, Objeto, PoseCarrito, Resultado

ZONA_APARTAR = sim.ZONA                      # esquina donde se apartan las cosas
ZONA_PERSONA = np.array([-0.45, 0.25])       # TODO P1: zona frente a la persona


class MundoSim:
    """Comparte un solo modelo de MuJoCo entre el cuerpo y la percepcion."""

    def __init__(self, ventana: bool = False):
        self.model = mujoco.MjModel.from_xml_string(sim.XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.visor = None
        if ventana:
            from mujoco import viewer
            self.visor = viewer.launch_passive(self.model, self.data)

    def paso(self):
        mujoco.mj_step(self.model, self.data)
        if self.visor is not None:
            self.visor.sync()


class PercepcionSim(Percepcion):
    """Percepcion perfecta de POSICIONES, pero los NOMBRES los pone el Reconocedor.
    El mundo simulado sabe que es cada objeto (tipo oculto); el robot no:
    solo recibe un embedding simulado y lo tiene que aprender. TODO P1: varios objetos."""

    # Aspecto simulado de cada tipo de objeto de la simulacion (el robot nunca ve estos nombres)
    ASPECTO = {"borrador": [0.9, 0.9, 0.9, 0.6]}

    def __init__(self, mundo: MundoSim, reconocedor, persona_id: str = "persona_A"):
        self.m, self.reconocedor, self.persona_id = mundo, reconocedor, persona_id
        self.ultima = None

    def observar(self) -> Escena:
        p, ang = sim.pose_carrito(self.m.model, self.m.data)
        bx, by = sim.pos_objeto(self.m.model, self.m.data, "borrador")
        emb = [v + float(np.random.normal(0, 0.02)) for v in self.ASPECTO["borrador"]]
        nombre, conf = self.reconocedor.reconocer(self.persona_id, emb)
        objetos = [Objeto(id="borrador", x=float(bx), y=float(by), nombre=nombre, confianza=conf, embedding=emb)]
        self.ultima = Escena(objetos=objetos, carrito=PoseCarrito(float(p[0]), float(p[1]), float(ang)),
                             persona_id=self.persona_id)
        return self.ultima

    def ensenar(self, persona_id, objeto_id, nombre):
        obj = next((o for o in (self.ultima.objetos if self.ultima else []) if o.id == objeto_id), None)
        if obj is not None:
            self.reconocedor.ensenar(persona_id, nombre, obj.embedding)


class CuerpoSim(Cuerpo):
    def __init__(self, mundo: MundoSim, limite_s: float = 60.0):
        self.m, self.limite_s = mundo, limite_s

    def ejecutar(self, accion, objeto, escena):
        d = self.m.data
        t0 = d.time
        if accion == Accion.IGNORAR:
            return Resultado(True, 0.0, "no hice nada")
        if accion == Accion.APARTAR and objeto is not None and objeto.id == "borrador":
            sim.accion_apartar.fase = "aproximar"
            while d.time - t0 < self.limite_s:
                fase = sim.accion_apartar(self.m.model, d, objeto="borrador")
                self.m.paso()
                if fase == "listo":
                    return Resultado(True, d.time - t0, "objeto apartado")
            d.ctrl[:] = 0
            return Resultado(False, d.time - t0, "se acabo el tiempo")
        # TODO P1: TRAER y ACERCARSE (reusar ir_hacia y la logica de empujar hacia ZONA_PERSONA)
        return Resultado(False, 0.0, f"{accion.value} todavia no esta implementado en simulacion")

    def cerrar(self):
        if self.m.visor is not None:
            self.m.visor.close()
