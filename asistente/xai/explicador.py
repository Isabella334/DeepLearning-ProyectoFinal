from __future__ import annotations

import numpy as np
import shap

from ..cerebro.cerebro import normalizar
from ..interfaces import Cerebro, Explicador, Memoria
from ..tipos import DESCONOCIDO, Accion, Contexto, Decision, Explicacion

VERBO = {Accion.TRAER: "te traigo", Accion.APARTAR: "aparto", Accion.ACERCARSE: "me acerco",
         Accion.IGNORAR: "no hago nada"}
TODO_EL_HISTORIAL_S = 10 * 365 * 24 * 3600


class ExplicadorXAI(Explicador):

    def __init__(self, n_fondo: int = 50, nsamples: int = 100, ventana_s: float = 120.0):
        self.n_fondo, self.nsamples, self.ventana_s = n_fondo, nsamples, ventana_s

    def explicar(self, contexto: Contexto, decision: Decision, cerebro: Cerebro | None,
                 memoria: Memoria) -> Explicacion:
        obj = next((o for o in contexto.escena.objetos if o.id == decision.objeto_id), None)
        nombre = obj.nombre if obj else None
        try:
            texto = self._texto(contexto, decision, nombre)
        except Exception:
            texto = f"Decidí {decision.accion.value} ({decision.motivo or 'sin motivo'})."
        try:
            evidencia = self._evidencia(contexto, decision, nombre, memoria)
        except Exception:
            evidencia = ""
        try:
            contribuciones = self._contribuciones(contexto, decision, cerebro)
        except Exception:
            contribuciones = dict(decision.puntajes)
        return Explicacion(texto=texto, evidencia=evidencia, contribuciones=contribuciones)

    def _contribuciones(self, contexto: Contexto, decision: Decision, cerebro: Cerebro | None) -> dict:
        """SHAP sobre la red de preferencias para la accion elegida; sin red, los puntajes de la decision."""
        modelo = cerebro.preferencias_de(contexto.persona_id) if hasattr(cerebro, "preferencias_de") else None
        if not (hasattr(modelo, "predecir_vectores") and hasattr(modelo, "fondo")):
            return dict(decision.puntajes)
        x = np.asarray([modelo.vector(contexto, decision.objeto_id)], dtype=float)
        nombres = modelo.nombres_caracteristicas()      # despues de vector(): puede sumar nombres nuevos
        j = list(Accion).index(decision.accion)
        # En log-odds: la red satura cerca de 0 y 1 y en probabilidad todo se ve como ~0
        explicador = shap.KernelExplainer(lambda X: np.clip(modelo.predecir_vectores(X)[:, j], 1e-6, 1 - 1e-6),
                                          np.asarray(modelo.fondo(self.n_fondo), dtype=float), link="logit")
        valores = np.asarray(explicador.shap_values(x, nsamples=self.nsamples, silent=True)).reshape(-1)
        return {n: float(v) for n, v in zip(nombres, valores) if abs(v) > 1e-6}

    def _texto(self, contexto: Contexto, decision: Decision, nombre: str | None) -> str:
        motivo, pedido = decision.motivo, contexto.pedido
        if motivo == "pedido" and decision.objeto_id is None and decision.preguntar_antes and pedido:
            return f"No encontré '{pedido.nombre}' en la mesa."
        que = VERBO[decision.accion] + (f" {self._nombre(nombre)}" if decision.objeto_id else "")
        if motivo == "pedido":
            porque = "porque me lo pediste"
        elif motivo == "anticipacion":
            senal = self._senal(contexto)
            porque = (f"porque sueles pedirlo cuando aparece '{senal}'" if senal
                      else "porque sueles pedirlo en momentos como este")
        elif motivo == "preferencia":
            porque = "porque es lo que más te ha gustado"
        elif motivo == "regla":
            porque = "porque me dijiste que lo hiciera cuando lo viera"
        else:
            porque = f"({motivo})" if motivo else ""
        frase = f"{que} {porque}".strip()
        return frase[0].upper() + frase[1:] + "."

    @staticmethod
    def _nombre(nombre: str | None) -> str:
        return "ese objeto" if nombre in (None, DESCONOCIDO) else f"'{nombre}'"

    @staticmethod
    def _senal(contexto: Contexto) -> str | None:
        apariciones = [e for e in contexto.eventos_recientes if e.tipo == "aparecio"]
        return max(apariciones, key=lambda e: e.t).detalle if apariciones else None

    def _evidencia(self, contexto: Contexto, decision: Decision, nombre: str | None,
                   memoria: Memoria) -> str:
        if decision.motivo == "anticipacion" and nombre:
            evidencia = self._evidencia_habito(contexto, decision, nombre, memoria)
            if evidencia:
                return evidencia
        calificadas = [h for h in memoria.historial(contexto.persona_id)
                       if h["accion"] == decision.accion.value and h["objeto_nombre"] == nombre
                       and h["aprobado"] is not None]
        if not calificadas:
            return "Es la primera vez que lo hago."
        aprobadas = sum(1 for h in calificadas if h["aprobado"])
        return f"Aprobaste esto {aprobadas} de {len(calificadas)} {'vez' if len(calificadas) == 1 else 'veces'}."

    def _evidencia_habito(self, contexto: Contexto, decision: Decision, nombre: str,
                          memoria: Memoria) -> str | None:
        """Cuantas de las apariciones pasadas de la senal terminaron en este mismo pedido."""
        senal = self._senal(contexto)
        if senal is None:
            return None
        ahora = max(e.t for e in contexto.eventos_recientes if e.tipo == "aparecio")
        eventos = memoria.eventos_recientes(contexto.persona_id, segundos=TODO_EL_HISTORIAL_S)
        buscado = normalizar(f"{decision.accion.value}:{nombre}")
        pedidos = [e.t for e in eventos if e.tipo == "pedido" and normalizar(e.detalle) == buscado]
        apariciones = [e.t for e in eventos if e.tipo == "aparecio" and e.detalle == senal and e.t < ahora]
        if not apariciones:
            return None
        seguidas = sum(1 for t in apariciones if any(t <= tp <= t + self.ventana_s for tp in pedidos))
        if len(apariciones) == 1:
            pidio = "me pediste" if seguidas else "no me pediste"
            return f"La única vez que apareció '{senal}', {pidio} '{nombre}'."
        return f"En {seguidas} de las {len(apariciones)} veces que apareció '{senal}', me pediste '{nombre}'."
