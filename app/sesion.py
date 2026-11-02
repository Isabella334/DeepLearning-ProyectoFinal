from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from asistente.cerebro.cerebro import CerebroAprendiz
from asistente.cerebro.preferencias import PreferenciasRed
from asistente.interfaces import Cerebro, Cuerpo, Explicador, Percepcion
from asistente.memoria import MemoriaSQLite
from asistente.mocks import CuerpoFalso, PercepcionFalsa
from asistente.tipos import (DESCONOCIDO, Accion, Contexto, Decision, Escena, Evento, Explicacion,
                             Orden, Resultado)
from asistente.vision.reconocedor import ReconocedorPocosEjemplos
from asistente.xai.explicador import ExplicadorXAI

from .feedback_pantalla import FeedbackPantalla


class Sesion:

    def __init__(self, persona_id: str = "persona_A", ruta_db: str = ":memory:",
                 percepcion: Percepcion | None = None, cuerpo: Cuerpo | None = None,
                 cerebro: Cerebro | None = None, explicador: Explicador | None = None,
                 ganchos: dict[str, Callable[[str | None], None]] | None = None,
                 carpeta_modelos: str | None = None):
        self.persona_id = persona_id
        self.carpeta_modelos = Path(carpeta_modelos) if carpeta_modelos else None
        self.percepcion = percepcion or PercepcionFalsa(ReconocedorPocosEjemplos(modo="cabeza"), persona_id)
        self.cuerpo = cuerpo or CuerpoFalso()
        self.cerebro = cerebro or CerebroAprendiz(preferencias=PreferenciasRed())
        self.memoria = MemoriaSQLite(ruta_db)
        self.explicador = explicador or ExplicadorXAI()
        self.feedback = FeedbackPantalla()
        self.ganchos = ganchos or {}      # integracion los conecta con la simulacion
        self.mundo = None
        self.escena: Escena | None = None
        self._pedido: Orden | None = None
        self._anteriores: set[str] = set()
        self._limpiar()
        self._cargar_modelos()

    def _limpiar(self) -> None:
        self.contexto: Contexto | None = None
        self.decision: Decision | None = None
        self.explicacion: Explicacion | None = None
        self.resultado: Resultado | None = None

    @classmethod
    def simulada(cls, persona_id: str = "persona_A", ruta_db: str = ":memory:",
                 carpeta_modelos: str | None = None, semilla: int = 0, **kwargs) -> "Sesion":
        from asistente.cuerpo_sim import CuerpoSim, MundoSim, PercepcionSim

        mundo = MundoSim(semilla=semilla)

        def poner(tipo):
            if not any(t == tipo and b not in mundo.ocupante for b, t in mundo.casillas):
                raise ValueError(f"no quedan copias libres de '{tipo}'")
            x, y = mundo.lugar_libre(tipo)
            mundo.colocar(mundo.nuevo_id(), tipo, x, y)

        sesion = cls(persona_id, ruta_db,
                     percepcion=PercepcionSim(mundo, ReconocedorPocosEjemplos(modo="cabeza"), persona_id),
                     cuerpo=CuerpoSim(mundo), ganchos={"poner": poner, "quitar": mundo.quitar},
                     carpeta_modelos=carpeta_modelos, **kwargs)
        sesion.mundo = mundo
        return sesion

    @classmethod
    def camara(cls, persona_id: str = "persona_A", ruta_db: str = ":memory:",
               carpeta_modelos: str | None = None, camara=0, **kwargs) -> "Sesion":
        """Version B (gemelo digital): la camara ve la cartulina real y el carrito virtual actua.
        Los objetos se ponen y se quitan con la mano en la mesa de verdad."""
        from asistente.cuerpo_sim import CuerpoSim, MundoSim
        from asistente.vision.percepcion_camara import PercepcionCamara

        mundo = MundoSim(con_camara=True, iniciales=[])
        percepcion = PercepcionCamara(ReconocedorPocosEjemplos(modo="cabeza"), camara=camara,
                                      persona_id=persona_id, mundo=mundo)
        sesion = cls(persona_id, ruta_db, percepcion=percepcion, cuerpo=CuerpoSim(mundo),
                     carpeta_modelos=carpeta_modelos, **kwargs)
        sesion.mundo, sesion.con_camara = mundo, True
        return sesion

    con_camara = False

    @property
    def vista_camara(self):
        """Lo que ve la camara con cajas e ids (RGB), o None si no hay camara."""
        vista = getattr(self.percepcion, "vista", None)
        return None if vista is None else vista[..., ::-1]

    @property
    def reconocedor(self):
        return getattr(self.percepcion, "reconocedor", None)

    def _cargar_modelos(self) -> None:
        if self.carpeta_modelos is None:
            return
        if (self.carpeta_modelos / "cerebro" / "cerebro.json").exists():
            self.cerebro.cargar(str(self.carpeta_modelos / "cerebro"))
        if self.reconocedor is not None and (self.carpeta_modelos / "reconocedor" / "reconocedor.json").exists():
            self.reconocedor.cargar(str(self.carpeta_modelos / "reconocedor"))

    def guardar(self) -> bool:
        if self.carpeta_modelos is None:
            return False
        self.cerebro.guardar(str(self.carpeta_modelos / "cerebro"))
        if self.reconocedor is not None:
            self.reconocedor.guardar(str(self.carpeta_modelos / "reconocedor"))
        version = len(self.memoria.modelos(self.persona_id)) + 1
        self.memoria.registrar_modelo(self.persona_id, version, str(self.carpeta_modelos))
        return True

    @property
    def objeto(self):
        if self.decision is None or self.escena is None:
            return None
        return next((o for o in self.escena.objetos if o.id == self.decision.objeto_id), None)

    @property
    def necesita_confirmacion(self) -> bool:
        return (self.decision is not None and self.decision.preguntar_antes
                and self.decision.accion != Accion.IGNORAR)

    def tipos_para_poner(self) -> list[str]:
        if self.mundo is None or self.con_camara:
            return []
        libres = {t for b, t in self.mundo.casillas if b not in self.mundo.ocupante}
        return [t for t in self.mundo.tipos if t in libres]

    def _retirar_caidos(self) -> None:
        # Lo que se cae de la cartulina se levanta del piso: si no, ocupa su copia y no se ve
        for oid in list(self.mundo.presentes()):
            if not self.mundo.en_mesa(self.mundo.pos(oid), margen=0.02):
                self.mundo.quitar(oid)

    def observar(self) -> Escena:
        if self.mundo is not None:
            self._retirar_caidos()
        self.escena = self.percepcion.observar()
        actuales = {o.nombre for o in self.escena.objetos if o.nombre != DESCONOCIDO}
        for nombre in actuales - self._anteriores:
            self.memoria.registrar_evento(Evento("aparecio", nombre, self.persona_id))
        for nombre in self._anteriores - actuales:
            self.memoria.registrar_evento(Evento("desaparecio", nombre, self.persona_id))
        self._anteriores = actuales
        self._limpiar()
        return self.escena

    def ensenar(self, objeto_id: str, nombre: str) -> bool:
        nombre = nombre.strip()
        presente = self.escena is not None and any(o.id == objeto_id for o in self.escena.objetos)
        if not nombre or not presente:
            return False
        self.percepcion.ensenar(self.persona_id, objeto_id, nombre)
        self.memoria.registrar_ensenanza(self.persona_id, nombre, objeto_id)
        self.guardar()
        return True

    def pedir(self, nombre: str, accion: Accion = Accion.TRAER) -> None:
        self._pedido = Orden("pedido", f"{accion.value} {nombre}", nombre=nombre, accion=accion)
        self.memoria.registrar_evento(Evento("pedido", f"{accion.value}:{nombre}", self.persona_id))

    def estimulo(self, tipo: str, dato: str | None = None) -> None:
        gancho = self.ganchos.get(tipo)
        if gancho is not None:
            gancho(dato)

    def decidir(self) -> tuple[Decision, Explicacion]:
        if self.escena is None:
            self.observar()
        self.contexto = Contexto(escena=self.escena, persona_id=self.persona_id, pedido=self._pedido,
                                 eventos_recientes=self.memoria.eventos_recientes(self.persona_id))
        self._pedido = None
        self.decision = self.cerebro.decidir(self.contexto)
        self.explicacion = self.explicador.explicar(self.contexto, self.decision, self.cerebro, self.memoria)
        self.resultado = None
        return self.decision, self.explicacion

    def ejecutar(self, confirmado: bool | None = None) -> Resultado:
        if self.decision is None:
            return Resultado(False, 0.0, "todavia no hay decision")
        if self.necesita_confirmacion:
            if confirmado is not None:
                self.feedback.pulsar(self.persona_id, confirmado)   # como en main: la respuesta es el feedback
            if not confirmado:
                self.resultado = Resultado(False, 0.0, "no lo hice: no me lo confirmaste")
                return self.resultado
        self.resultado = self.cuerpo.ejecutar(self.decision.accion, self.objeto, self.escena)
        return self.resultado

    def responder(self, aprobado: bool | None = None) -> int | None:
        if self.decision is None:
            return None
        if aprobado is not None:
            self.feedback.pulsar(self.persona_id, aprobado)
        fb = self.feedback.esperar(self.persona_id)
        if fb is not None:
            self.cerebro.aprender(self.contexto, self.decision, fb)
            self.memoria.registrar_evento(Evento("feedback", "si" if fb.aprobado else "no", self.persona_id))
        resultado = self.resultado or Resultado(False, 0.0, "no se ejecuto")
        id_ = self.memoria.registrar_interaccion(self.contexto, self.decision, resultado, fb)
        if fb is not None:
            self.guardar()
        self._limpiar()
        return id_

    def reiniciar(self) -> None:
        self.memoria.olvidar(self.persona_id)
        if self.carpeta_modelos is not None and self.carpeta_modelos.exists():
            shutil.rmtree(self.carpeta_modelos)

    def historial(self, limite: int = 10) -> list[dict]:
        return self.memoria.historial(self.persona_id, limite)[::-1]

    def cerrar(self) -> None:
        self.cuerpo.cerrar()
        if hasattr(self.percepcion, "cerrar"):
            self.percepcion.cerrar()
        self.memoria.cerrar()
