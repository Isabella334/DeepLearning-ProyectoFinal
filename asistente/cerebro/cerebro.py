"""
CerebroAprendiz (P3): junta pedidos, reglas, anticipacion y preferencias en UNA decision.

Prioridad: pedido explicito > regla dicha por voz > anticipacion > preferencias > ignorar.
Si la confianza queda bajo el umbral, la decision sale con preguntar_antes=True.
Cada persona tiene su propio modelo de preferencias, su propia anticipacion y sus reglas.
"""
from __future__ import annotations

import copy
import json
import unicodedata
from pathlib import Path

import yaml

from ..interfaces import Cerebro
from ..tipos import Accion, Contexto, Decision, Feedback, Objeto
from .anticipacion import NINGUNO, AnticipacionFrecuencias
from .contratos import ModeloAnticipacion, ModeloPreferencias, PreferenciasFalsas

CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
ACCIONES_CON_OBJETO = (Accion.APARTAR, Accion.TRAER)


def normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return " ".join(sin_tildes.lower().split())


def cargar_zonas(ruta: Path = CONFIG) -> dict[Accion, tuple[float, float, float]]:
    """Zona destino de cada accion, como (x, y, medio_lado) en metros."""
    try:
        zonas = yaml.safe_load(ruta.read_text())["zonas"]
        return {Accion.APARTAR: tuple(zonas["apartar"]), Accion.TRAER: tuple(zonas["persona"])}
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return {}


class CerebroAprendiz(Cerebro):

    def __init__(self, preferencias: ModeloPreferencias | None = None,
                 anticipacion: ModeloAnticipacion | None = None, umbral: float = 0.7,
                 umbral_anticipacion: float = 0.5,
                 zonas: dict[Accion, tuple[float, float, float]] | None = None):
        self.plantilla_preferencias = preferencias or PreferenciasFalsas()
        self.plantilla_anticipacion = anticipacion or AnticipacionFrecuencias()
        self.umbral = umbral
        self.umbral_anticipacion = umbral_anticipacion
        self.zonas = cargar_zonas() if zonas is None else zonas
        self.preferencias: dict[str, ModeloPreferencias] = {}
        self.anticipaciones: dict[str, ModeloAnticipacion] = {}
        self.reglas: dict[str, dict[str, Accion]] = {}
        self.anticipado: dict[str, tuple[str, float]] = {}   # persona -> (pedido, t del momento)

    def preferencias_de(self, persona_id: str) -> ModeloPreferencias:
        if persona_id not in self.preferencias:
            self.preferencias[persona_id] = self.plantilla_preferencias.clonar()
        return self.preferencias[persona_id]

    def anticipacion_de(self, persona_id: str) -> ModeloAnticipacion:
        if persona_id not in self.anticipaciones:
            self.anticipaciones[persona_id] = copy.deepcopy(self.plantilla_anticipacion)
        return self.anticipaciones[persona_id]

    def decidir(self, contexto: Contexto) -> Decision:
        persona, pedido = contexto.persona_id, contexto.pedido
        if pedido is not None and pedido.tipo == "regla" and pedido.nombre and pedido.accion:
            self.reglas.setdefault(persona, {})[normalizar(pedido.nombre)] = pedido.accion
        if pedido is not None and pedido.tipo == "pedido" and pedido.accion:
            return self._por_pedido(contexto)
        if not contexto.escena.objetos:
            return Decision(Accion.IGNORAR, None, confianza=1.0, motivo="sin objetos")
        if contexto.escena.persona_id is None:
            return Decision(Accion.IGNORAR, None, confianza=1.0, motivo="sin persona")
        return (self._por_regla(contexto) or self._por_anticipacion(contexto)
                or self._por_preferencia(contexto))

    def _por_pedido(self, contexto: Contexto) -> Decision:
        pedido = contexto.pedido
        if pedido.nombre:
            self.anticipacion_de(contexto.persona_id).actualizar(
                contexto.eventos_recientes, f"{pedido.accion.value}:{pedido.nombre}")
        if pedido.accion in (Accion.ACERCARSE, Accion.IGNORAR):
            return Decision(pedido.accion, None, confianza=1.0, motivo="pedido", puntajes={"pedido": 1.0})
        candidatos = self._buscar(contexto.escena.objetos, pedido.nombre or "")
        if not candidatos:
            return Decision(Accion.IGNORAR, None, confianza=0.0, preguntar_antes=True, motivo="pedido",
                            puntajes={"pedido": 0.0})
        obj = candidatos[0]
        confianza = min(1.0, obj.confianza) if len(candidatos) == 1 else 0.5
        return self._decision(contexto, pedido.accion, obj, confianza, "pedido", {"pedido": 1.0})

    def _por_regla(self, contexto: Contexto) -> Decision | None:
        for regla, accion in self.reglas.get(contexto.persona_id, {}).items():
            for obj in self._buscar(contexto.escena.objetos, regla):
                if accion not in ACCIONES_CON_OBJETO or not self._en_destino(obj, accion):
                    return self._decision(contexto, accion, obj, min(1.0, obj.confianza), "regla",
                                          {"regla": 1.0})
        return None

    def _por_anticipacion(self, contexto: Contexto) -> Decision | None:
        eventos = contexto.eventos_recientes
        if not eventos:
            return None
        momento = max(e.t for e in eventos)
        persona = contexto.persona_id
        if self.anticipado.get(persona, (None, None))[1] == momento:
            return None         # ya se anticipo (o se pregunto) en este mismo momento
        prediccion, prob = self.anticipacion_de(persona).predecir(eventos)
        if prediccion is None or prob < self.umbral_anticipacion:
            return None
        valor, _, nombre = prediccion.partition(":")
        try:
            accion = Accion(valor)
        except ValueError:
            return None
        for obj in self._buscar(contexto.escena.objetos, nombre):
            if accion not in ACCIONES_CON_OBJETO or not self._en_destino(obj, accion):
                self.anticipado[persona] = (prediccion, momento)
                return self._decision(contexto, accion, obj, prob * min(1.0, obj.confianza),
                                      "anticipacion", {"anticipacion": prob})
        return None

    def _por_preferencia(self, contexto: Contexto) -> Decision:
        modelo = self.preferencias_de(contexto.persona_id)
        mejor: tuple[float, Accion, Objeto | None, dict] = (-1.0, Accion.IGNORAR, None, {})
        for obj in contexto.escena.objetos:
            puntajes = modelo.puntajes(contexto, obj.id)
            for accion, p in puntajes.items():
                if accion in ACCIONES_CON_OBJETO and self._en_destino(obj, accion):
                    continue
                if p > mejor[0]:
                    mejor = (p, accion, obj if accion in ACCIONES_CON_OBJETO else None, puntajes)
        p, accion, obj, puntajes = mejor
        confianza = min(1.0, max(0.0, p))
        return Decision(accion, obj.id if obj else None, confianza=confianza,
                        preguntar_antes=accion != Accion.IGNORAR and confianza < self.umbral,
                        motivo="preferencia", puntajes={a.value: float(v) for a, v in puntajes.items()})

    def _decision(self, contexto: Contexto, accion: Accion, obj: Objeto, confianza: float,
                  motivo: str, extra: dict[str, float]) -> Decision:
        puntajes = self.preferencias_de(contexto.persona_id).puntajes(contexto, obj.id)
        objeto_id = obj.id if accion in ACCIONES_CON_OBJETO else None
        return Decision(accion, objeto_id, confianza=confianza, preguntar_antes=confianza < self.umbral,
                        motivo=motivo, puntajes={a.value: float(v) for a, v in puntajes.items()} | extra)

    def _buscar(self, objetos: list[Objeto], nombre: str) -> list[Objeto]:
        """Objetos con ese nombre exacto; si no hay, los que lo contienen ('lapicero' -> 'lapicero rojo')."""
        buscado = normalizar(nombre)
        if not buscado:
            return []
        exactos = [o for o in objetos if normalizar(o.nombre) == buscado]
        parecidos = [o for o in objetos if f" {buscado} " in f" {normalizar(o.nombre)} "]
        return sorted(exactos or parecidos, key=lambda o: -o.confianza)

    def _en_destino(self, obj: Objeto, accion: Accion) -> bool:
        zona = self.zonas.get(accion)
        if zona is None:
            return False
        x, y, medio = zona
        return abs(obj.x - x) <= medio and abs(obj.y - y) <= medio

    def aprender(self, contexto: Contexto, decision: Decision, feedback: Feedback) -> None:
        persona = contexto.persona_id
        self.preferencias_de(persona).actualizar(contexto, decision.objeto_id, decision.accion,
                                                 feedback.aprobado)
        if decision.motivo == "anticipacion" and persona in self.anticipado:
            pedido = self.anticipado[persona][0]
            self.anticipacion_de(persona).actualizar(contexto.eventos_recientes,
                                                     pedido if feedback.aprobado else NINGUNO)

    def guardar(self, ruta: str) -> None:
        carpeta = Path(ruta)
        carpeta.mkdir(parents=True, exist_ok=True)
        personas = sorted(set(self.preferencias) | set(self.anticipaciones) | set(self.reglas))
        for i, persona in enumerate(personas):
            sub = carpeta / f"persona_{i}"
            sub.mkdir(exist_ok=True)
            self.preferencias_de(persona).guardar(str(sub / "preferencias.pt"))
            anticipacion = self.anticipacion_de(persona)
            if hasattr(anticipacion, "guardar"):
                anticipacion.guardar(str(sub / "anticipacion.pt"))
        indice = {"personas": personas,
                  "reglas": {p: {n: a.value for n, a in r.items()} for p, r in self.reglas.items()}}
        (carpeta / "cerebro.json").write_text(json.dumps(indice, ensure_ascii=False, indent=2))

    def cargar(self, ruta: str) -> None:
        carpeta = Path(ruta)
        indice = json.loads((carpeta / "cerebro.json").read_text())
        self.preferencias, self.anticipaciones, self.anticipado = {}, {}, {}
        self.reglas = {p: {n: Accion(a) for n, a in r.items()} for p, r in indice["reglas"].items()}
        for i, persona in enumerate(indice["personas"]):
            sub = carpeta / f"persona_{i}"
            if (sub / "preferencias.pt").exists():
                self.preferencias_de(persona).cargar(str(sub / "preferencias.pt"))
            anticipacion = self.anticipacion_de(persona)
            if hasattr(anticipacion, "cargar") and (sub / "anticipacion.pt").exists():
                anticipacion.cargar(str(sub / "anticipacion.pt"))
