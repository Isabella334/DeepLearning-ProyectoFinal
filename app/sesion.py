from __future__ import annotations

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
                 ganchos: dict[str, Callable[[str | None], None]] | None = None):
        self.persona_id = persona_id
        self.percepcion = percepcion or PercepcionFalsa(ReconocedorPocosEjemplos(modo="cabeza"), persona_id)
        self.cuerpo = cuerpo or CuerpoFalso()
        self.cerebro = cerebro or CerebroAprendiz(preferencias=PreferenciasRed())
        self.memoria = MemoriaSQLite(ruta_db)
        self.explicador = explicador or ExplicadorXAI()
        self.feedback = FeedbackPantalla()
        self.ganchos = ganchos or {}      # integracion los conecta con la simulacion
        self.escena: Escena | None = None
        self._pedido: Orden | None = None
        self._anteriores: set[str] = set()
        self._limpiar()

    def _limpiar(self) -> None:
        self.contexto: Contexto | None = None
        self.decision: Decision | None = None
        self.explicacion: Explicacion | None = None
        self.resultado: Resultado | None = None

    @property
    def objeto(self):
        if self.decision is None or self.escena is None:
            return None
        return next((o for o in self.escena.objetos if o.id == self.decision.objeto_id), None)

    @property
    def necesita_confirmacion(self) -> bool:
        return (self.decision is not None and self.decision.preguntar_antes
                and self.decision.accion != Accion.IGNORAR)

    def observar(self) -> Escena:
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
        return True

    def pedir(self, nombre: str, accion: Accion = Accion.TRAER) -> None:
        self._pedido = Orden("pedido", f"{accion.value} {nombre}", nombre=nombre, accion=accion)
        self.memoria.registrar_evento(Evento("pedido", f"{accion.value}:{nombre}", self.persona_id))

    def estimulo(self, tipo: str, objeto_id: str | None = None) -> None:
        gancho = self.ganchos.get(tipo)
        if gancho is not None:
            gancho(objeto_id)

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
        self._limpiar()
        return id_

    def historial(self, limite: int = 10) -> list[dict]:
        return self.memoria.historial(self.persona_id, limite)[::-1]

    def cerrar(self) -> None:
        self.cuerpo.cerrar()
        self.memoria.cerrar()
