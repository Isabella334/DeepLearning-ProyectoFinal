"""
Versiones FALSAS de cada modulo. Cumplen el contrato pero son muy simples.

Se usan en la integracion para probar e integrar: cuando llega una pieza real, se prueba
junto a las falsas de los demas. Tambien sirven para que cualquiera corra
main.py de punta a punta desde el primer dia.
"""
from __future__ import annotations

import math
import random
import re
import time

from .interfaces import (Cerebro, Cuerpo, Detector, EntradaFeedback, Explicador, Memoria,
                         Percepcion, Reconocedor, Voz)
from .tipos import (DESCONOCIDO, Accion, Decision, Deteccion, Escena, Evento, Explicacion,
                    Feedback, Objeto, Orden, PoseCarrito, Resultado)


def _coseno(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb + 1e-9)


# ---------------------------------------------------------------- Reconocedor
class ReconocedorFalso(Reconocedor):
    """Guarda el promedio de los ejemplos de cada nombre y compara con el coseno.
    (El real de P2 usa una red preentrenada y una red pequena por persona.)"""

    def __init__(self, umbral: float = 0.85):
        self.umbral = umbral
        self.ejemplos: dict[str, dict[str, list[list[float]]]] = {}   # persona -> nombre -> vectores

    def embedding(self, recorte):
        return [float(v) for v in recorte.reshape(-1, recorte.shape[-1]).mean(axis=0)]  # color promedio

    def ensenar(self, persona_id, nombre, embedding):
        self.ejemplos.setdefault(persona_id, {}).setdefault(nombre, []).append(list(embedding))

    def reconocer(self, persona_id, embedding):
        mejor, conf = DESCONOCIDO, 0.0
        for nombre, vecs in self.ejemplos.get(persona_id, {}).items():
            proto = [sum(c) / len(vecs) for c in zip(*vecs)]
            s = _coseno(embedding, proto)
            if s > conf:
                mejor, conf = nombre, s
        return (mejor, conf) if conf >= self.umbral else (DESCONOCIDO, conf)

    def nombres(self, persona_id):
        return list(self.ejemplos.get(persona_id, {}))

    def guardar(self, ruta):
        pass

    def cargar(self, ruta):
        pass


# ---------------------------------------------------------------- Percepcion
class PercepcionFalsa(Percepcion):
    """Un mundo inventado. El MUNDO sabe que es cada objeto (tipo oculto),
    pero el ROBOT solo ve su embedding y lo nombra con el Reconocedor."""

    # Cada tipo oculto tiene un "aspecto" (embedding base). El robot nunca ve estos nombres.
    TIPOS = {"lapicero rojo": [1.0, 0.1, 0.1, 0.2], "lapicero azul": [0.1, 0.1, 1.0, 0.2],
             "borrador": [0.9, 0.9, 0.9, 0.6], "basura": [0.7, 0.7, 0.6, 1.0],
             "cuaderno verde": [0.1, 1.0, 0.2, 0.9]}

    def __init__(self, reconocedor: Reconocedor | None = None, persona_id="persona_A", semilla=0):
        self.reconocedor = reconocedor or ReconocedorFalso()
        self.persona_id = persona_id
        self.rng = random.Random(semilla)
        self.ultima: Escena | None = None
        self.tipo_oculto: dict[str, str] = {}   # objeto_id -> tipo (solo para evaluar)

    def observar(self):
        objetos = []
        for i, (tipo, base) in enumerate(self.TIPOS.items()):
            if self.rng.random() < 0.6:
                emb = [v + self.rng.gauss(0, 0.03) for v in base]
                nombre, conf = self.reconocedor.reconocer(self.persona_id, emb)
                oid = f"obj_{i}"
                self.tipo_oculto[oid] = tipo
                objetos.append(Objeto(id=oid, x=self.rng.uniform(-0.25, 0.25), y=self.rng.uniform(-0.2, 0.2),
                                      nombre=nombre, confianza=conf, embedding=emb))
        persona = self.persona_id if self.rng.random() > 0.05 else None   # a veces no hay nadie
        self.ultima = Escena(objetos=objetos, carrito=PoseCarrito(0, 0, 0), persona_id=persona)
        return self.ultima

    def ensenar(self, persona_id, objeto_id, nombre):
        obj = next((o for o in (self.ultima.objetos if self.ultima else []) if o.id == objeto_id), None)
        if obj is not None and obj.embedding is not None:
            self.reconocedor.ensenar(persona_id, nombre, obj.embedding)


# ---------------------------------------------------------------- Detector
class DetectorFalso(Detector):
    """Devuelve dos detecciones fijas (sin nombre, como el detector real)."""
    def detectar(self, imagen):
        alto, ancho = imagen.shape[:2]
        return [Deteccion(0.9, (0.30 * ancho, 0.40 * alto, 0.40 * ancho, 0.45 * alto)),
                Deteccion(0.8, (0.60 * ancho, 0.60 * alto, 0.68 * ancho, 0.66 * alto))]


# ---------------------------------------------------------------- Cuerpo
class CuerpoFalso(Cuerpo):
    def ejecutar(self, accion, objeto, escena):
        print(f"   [cuerpo falso] {accion.value} {objeto.nombre if objeto else ''}")
        return Resultado(exito=True, duracion_s=0.0, mensaje="sin fisica")


# ---------------------------------------------------------------- Cerebro
class CerebroAlAzar(Cerebro):
    def decidir(self, contexto):
        objetos = contexto.escena.objetos
        if not objetos:
            return Decision(Accion.IGNORAR, None, confianza=1.0, motivo="mesa vacia")
        obj = random.choice(objetos)
        accion = random.choice([Accion.APARTAR, Accion.TRAER, Accion.IGNORAR])
        return Decision(accion, obj.id if accion != Accion.IGNORAR else None, confianza=0.33,
                        preguntar_antes=True, motivo="azar",
                        puntajes={a.value: 1 / 3 for a in (Accion.APARTAR, Accion.TRAER, Accion.IGNORAR)})

    def aprender(self, contexto, decision, feedback):
        pass

    def guardar(self, ruta):
        pass

    def cargar(self, ruta):
        pass


# ---------------------------------------------------------------- Voz
class VozTeclado(Voz):
    """Escribís en la terminal en vez de hablar."""
    def escuchar(self, timeout_s=5.0):
        t = input("   (voz) Decile algo al robot [Enter = nada]: ").strip()
        return t or None

    def interpretar(self, texto):
        t = texto.lower().strip()
        if t in ("bien", "si", "sí", "gracias"):
            return Orden("feedback", texto, aprobado=True)
        if t in ("no", "mal"):
            return Orden("feedback", texto, aprobado=False)
        m = re.match(r"esto es (?:mi |un |una |el |la )?(.+)", t)
        if m:
            return Orden("ensenar", texto, nombre=m.group(1))
        m = re.match(r"(?:traeme|tr[aá]eme|pasame|p[aá]same) (?:el |la |mi )?(.+)", t)
        if m:
            return Orden("pedido", texto, nombre=m.group(1), accion=Accion.TRAER)
        m = re.match(r"aparta (?:el |la )?(.+)", t)
        if m:
            return Orden("pedido", texto, nombre=m.group(1), accion=Accion.APARTAR)
        return None


# ---------------------------------------------------------------- Memoria
class MemoriaEnRAM(Memoria):
    def __init__(self):
        self.interacciones, self.eventos, self.ensenanzas = [], [], []

    def registrar_interaccion(self, contexto, decision, resultado, feedback):
        obj = next((o for o in contexto.escena.objetos if o.id == decision.objeto_id), None)
        self.interacciones.append(dict(persona_id=contexto.persona_id, accion=decision.accion.value,
                                       objeto_id=decision.objeto_id,
                                       objeto_nombre=obj.nombre if obj else None,
                                       aprobado=None if feedback is None else feedback.aprobado,
                                       t=time.time()))
        return len(self.interacciones)

    def registrar_evento(self, evento):
        self.eventos.append(evento)

    def registrar_ensenanza(self, persona_id, nombre, objeto_id):
        self.ensenanzas.append(dict(persona_id=persona_id, nombre=nombre, objeto_id=objeto_id, t=time.time()))

    def eventos_recientes(self, persona_id, segundos=300):
        limite = time.time() - segundos
        return [e for e in self.eventos if e.persona_id == persona_id and e.t >= limite]

    def historial(self, persona_id, limite=200):
        return [i for i in self.interacciones if i["persona_id"] == persona_id][-limite:]


# ---------------------------------------------------------------- Explicador
class ExplicadorSimple(Explicador):
    def explicar(self, contexto, decision, cerebro, memoria):
        hist = [h for h in memoria.historial(contexto.persona_id) if h["accion"] == decision.accion.value]
        aprobadas = sum(1 for h in hist if h["aprobado"])
        return Explicacion(texto=f"Elegi '{decision.accion.value}' ({decision.motivo}) "
                                 f"con confianza {decision.confianza:.0%}.",
                           evidencia=f"Aprobaste esta accion {aprobadas} de {len(hist)} veces.")


# ---------------------------------------------------------------- Feedback
class FeedbackAlAzar(EntradaFeedback):
    def esperar(self, persona_id, timeout_s=10.0):
        return Feedback(persona_id=persona_id, aprobado=random.random() < 0.5, fuente="azar")


class FeedbackTeclado(EntradaFeedback):
    def esperar(self, persona_id, timeout_s=10.0):
        r = input("   Te parecio bien? [s/n/Enter=sin respuesta]: ").strip().lower()
        if r not in ("s", "n"):
            return None
        return Feedback(persona_id=persona_id, aprobado=(r == "s"), fuente="teclado")
