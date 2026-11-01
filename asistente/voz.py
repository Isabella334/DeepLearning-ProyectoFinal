"""
Voz (P3): escucha a la persona con Whisper y convierte lo que dice en una Orden.

    v = VozWhisper()
    v.escuchar()                          -> "traeme el lapicero rojo"  (solo si antes dijo el nombre del robot)
    v.interpretar("si ves basura, apartala") -> Orden(tipo="regla", accion=APARTAR, nombre="basura")

interpretar_texto() es una funcion pura con reglas para el espanol: tambien la puede usar
la pantalla o el teclado. Los verbos se comparan sin tildes y sin distinguir b de v (suenan
igual y Whisper las confunde), pero el nombre del objeto se devuelve tal como se dijo.

Probar el microfono:  python -m asistente.voz
"""
from __future__ import annotations

import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import yaml

from .interfaces import Voz
from .tipos import Accion, Orden

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def _comparable(palabra: str) -> str:
    return sin_tildes(palabra.lower()).replace("v", "b")


def _conjunto(*palabras: str) -> set[str]:
    return {_comparable(p) for p in palabras}


POSITIVAS = _conjunto("bien", "si", "gracias", "perfecto", "exacto", "correcto", "genial", "excelente",
                      "ok", "okay", "vale", "dale", "bueno", "super", "eso", "claro")
NEGATIVAS = _conjunto("no", "mal", "incorrecto", "equivocado", "error", "detente", "para",
                      "asino")   # 'asi no' como lo suele escribir Whisper
RELLENO_FEEDBACK = _conjunto("muy", "es", "asi", "esta", "estuvo", "buen", "trabajo", "era", "eso")

MULETILLAS = _conjunto("oye", "hey", "hola", "porfa", "porfavor")
ARTICULOS = _conjunto("el", "la", "los", "las", "un", "una", "unos", "unas", "mi", "mis", "tu", "tus",
                      "su", "sus", "al", "a", "ese", "esa", "este", "esta")
FINALES = _conjunto("ahora", "aqui", "aca", "ahi", "ya", "rapido")
DEMOSTRATIVOS = _conjunto("esto", "este", "esta", "eso", "ese", "esa")
VER = _conjunto("ves", "veas", "hay", "haya", "aparece", "aparezca", "encuentras", "encuentres",
                "miras", "mires", "esta", "este")
CORTESIA = _conjunto("me", "puedes", "podrias", "podes", "puede", "podria", "que")
ACERCARSE = _conjunto("ven", "veni", "vente", "acercate", "acerquese", "aproximate", "arrimate")
POR_FAVOR = [_comparable("por"), _comparable("favor")]

_VERBOS = {
    Accion.TRAER: ["trae", "traer", "traes", "traiga", "traigas", "pasa", "pasar", "pasas", "pase",
                   "alcanza", "alcanzar", "alcanzas", "acerca", "acercar", "acercas", "dame",
                   "darme", "das"],
    Accion.APARTAR: ["aparta", "apartar", "apartas", "aparte", "quita", "quitar", "quitas", "quite",
                     "saca", "sacar", "sacas", "saque", "retira", "retirar", "retire", "mueve",
                     "mover", "mueva", "tira", "tirar", "tire", "empuja", "empujar", "empuje",
                     "llevate"],
    Accion.IGNORAR: ["ignora", "ignorar", "ignore", "deja", "dejar", "deje"],
}
_PATRONES = {a: re.compile(rf"^(?:{'|'.join(map(_comparable, v))})(?:me)?(?:te)?(?:lo|la|los|las)?$")
             for a, v in _VERBOS.items()}


def _palabras(texto: str) -> tuple[list[str], list[str]]:
    """(palabras tal cual en minusculas, las mismas en forma comparable), sin puntuacion."""
    originales = re.sub(r"[^\w\s]", " ", texto.lower()).split()
    return originales, [_comparable(p) for p in originales]


def _accion(palabra: str) -> Accion | None:
    return next((a for a, patron in _PATRONES.items() if patron.match(palabra)), None)


def _nombre(originales: list[str], palabras: list[str]) -> str | None:
    i, j = 0, len(palabras)
    while i < j and palabras[i] in ARTICULOS:
        i += 1
    while j > i and palabras[j - 1] in FINALES:
        j -= 1
        if j > i and palabras[j - 1] == "de":
            j -= 1
    return " ".join(originales[i:j]) or None


def _feedback(palabras: list[str]) -> bool | None:
    if not set(palabras) <= POSITIVAS | NEGATIVAS | RELLENO_FEEDBACK:
        return None
    if NEGATIVAS & set(palabras):
        return False
    return True if POSITIVAS & set(palabras) else None


def quitar_palabra_clave(texto: str, nombre_robot: str) -> str | None:
    """Si el nombre del robot aparece entre las primeras palabras, devuelve lo que sigue.
    'Robot, traeme el lapicero.' -> 'traeme el lapicero'   Sin el nombre -> None.
    La comparacion es aproximada porque Whisper suele escribir mal el nombre ('Pobot', 'Bobot'),
    y mas tolerante en la primera palabra, que es donde casi siempre se dice."""
    clave = " ".join(_comparable(p) for p in nombre_robot.split())
    n = len(nombre_robot.split())
    trozos = texto.split()
    limpias = [_comparable(re.sub(r"[^\w]", "", t)) for t in trozos]
    for i in range(min(3, len(limpias))):
        candidato = " ".join(limpias[i:i + n])
        if SequenceMatcher(None, candidato, clave).ratio() >= (0.6 if i == 0 else 0.75):
            return " ".join(trozos[i + n:]).strip(" ,.;:!¡?¿")
    return None


def interpretar_texto(texto: str, nombre_robot: str | None = None) -> Orden | None:
    """Convierte una frase en espanol en una Orden: ensenar, pedido, regla o feedback."""
    originales, palabras = _palabras(texto)
    while len(palabras) > 1 and palabras[-2:] == POR_FAVOR:
        originales, palabras = originales[:-2], palabras[:-2]
    if nombre_robot:
        resto = quitar_palabra_clave(" ".join(originales), nombre_robot)
        if resto is not None:
            originales, palabras = _palabras(resto)
    while palabras and palabras[0] in MULETILLAS:
        originales, palabras = originales[1:], palabras[1:]
    if not palabras:
        return None

    aprobado = _feedback(palabras)
    if aprobado is not None:
        return Orden("feedback", texto, aprobado=aprobado)

    if palabras[:2] == ["aprende", "que"]:
        originales, palabras = originales[2:], palabras[2:]
    inicio = None
    if len(palabras) > 2 and palabras[0] in DEMOSTRATIVOS and palabras[1] == "es":
        inicio = 2
    elif len(palabras) > 3 and palabras[0] in DEMOSTRATIVOS and palabras[1:3] == ["se", "llama"]:
        inicio = 3
    elif palabras[:2] in (["se", "llama"], ["te", "presento"]):
        inicio = 2
    if inicio is not None:
        nombre = _nombre(originales[inicio:], palabras[inicio:])
        return Orden("ensenar", texto, nombre=nombre) if nombre else None

    condicion = 2 if palabras[:2] == ["siempre", "que"] else 1 if palabras[0] in ("si", "cuando") else 0
    if condicion and len(palabras) > condicion and palabras[condicion] in VER:
        for k in range(condicion + 1, len(palabras)):
            accion = _accion(palabras[k])
            if accion is not None:
                nombre = _nombre(originales[condicion + 1:k], palabras[condicion + 1:k])
                otro = _nombre(originales[k + 1:], palabras[k + 1:])
                # 'cuando veas X, traeme Y' habla de dos objetos: una regla solo tiene uno
                return Orden("regla", texto, nombre=nombre, accion=accion) if nombre and not otro else None
        return None

    if palabras[0] in ACERCARSE:
        return Orden("pedido", texto, accion=Accion.ACERCARSE)
    if len(palabras) > 1 and palabras[0] in ("necesito", "quiero") and palabras[1] == "que":
        originales, palabras = originales[1:], palabras[1:]
    while palabras and palabras[0] in CORTESIA:
        originales, palabras = originales[1:], palabras[1:]
    if palabras and palabras[0] in ("necesito", "quiero"):
        nombre = _nombre(originales[1:], palabras[1:])
        return Orden("pedido", texto, nombre=nombre, accion=Accion.TRAER) if nombre else None
    accion = _accion(palabras[0]) if palabras else None
    if accion is not None:
        nombre = _nombre(originales[1:], palabras[1:])
        return Orden("pedido", texto, nombre=nombre, accion=accion) if nombre else None
    return None


def _nombre_desde_config(ruta: Path = CONFIG) -> str:
    try:
        return str(yaml.safe_load(ruta.read_text())["voz"]["nombre_del_robot"])
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return "robot"


class VozWhisper(Voz):
    """Graba del microfono por pedazos, transcribe con Whisper y solo hace caso si empiezan
    con el nombre del robot. Si falta el microfono o la biblioteca, escuchar() devuelve None."""

    def __init__(self, modelo: str = "base", idioma: str = "es", nombre_robot: str | None = None,
                 duracion_s: float = 4.0, frecuencia: int = 16000, umbral_silencio: float = 0.01,
                 dispositivo: int | str | None = None):
        self.modelo, self.idioma = modelo, idioma
        self.nombre_robot = nombre_robot or _nombre_desde_config()
        self.duracion_s, self.frecuencia = duracion_s, frecuencia
        self.umbral_silencio, self.dispositivo = umbral_silencio, dispositivo
        self.orejas = True          # el boton de "orejas" lo apaga y enciende
        self.vocabulario: list[str] = []   # nombres que la persona le enseno, ayudan a Whisper
        self._whisper = None
        self._error_avisado = False

    def escuchar(self, timeout_s: float = 5.0) -> str | None:
        if not self.orejas:
            return None
        try:
            fin = time.monotonic() + timeout_s
            while time.monotonic() < fin:
                texto = self._oir(min(self.duracion_s, max(0.5, fin - time.monotonic())))
                if texto is None:
                    continue
                resto = quitar_palabra_clave(texto, self.nombre_robot)
                if resto:
                    return resto
                if resto == "":     # solo dijo el nombre: lo que diga ahora es la orden
                    return self._oir(self.duracion_s) or None
            return None
        except Exception as e:
            if not self._error_avisado:
                print(f"   [voz] no disponible: {e}")
                self._error_avisado = True
            return None

    def interpretar(self, texto: str) -> Orden | None:
        return interpretar_texto(texto, self.nombre_robot)

    def _oir(self, segundos: float) -> str | None:
        audio = self._grabar(segundos)
        if np.abs(audio).max(initial=0.0) < self.umbral_silencio:
            return None
        return self.transcribir(audio) or None

    def _grabar(self, segundos: float) -> np.ndarray:
        import sounddevice as sd
        audio = sd.rec(int(segundos * self.frecuencia), samplerate=self.frecuencia, channels=1,
                       dtype="float32", device=self.dispositivo)
        sd.wait()
        return audio[:, 0]

    def transcribir(self, audio: np.ndarray) -> str:
        """Audio mono float32 a 16 kHz -> texto."""
        if self._whisper is None:
            import whisper
            self._whisper = whisper.load_model(self.modelo)
        resultado = self._whisper.transcribe(audio, language=self.idioma, fp16=False,
                                             initial_prompt=self._ayuda())
        return resultado["text"].strip()

    def _ayuda(self) -> str:
        """Texto de ejemplo para Whisper: el estilo de las ordenes y los nombres conocidos."""
        n = self.nombre_robot.capitalize()
        ayuda = f"{n}, tráeme eso. {n}, ven. {n}, si ves eso, apártalo."
        if self.vocabulario:
            ayuda += f" Objetos: {', '.join(self.vocabulario)}."
        return ayuda


if __name__ == "__main__":
    voz = VozWhisper()
    print(f"Decí '{voz.nombre_robot}' y una orden (Ctrl+C para salir).")
    while True:
        texto = voz.escuchar(timeout_s=8.0)
        if texto:
            print(f"  oí: {texto!r}  ->  {voz.interpretar(texto)}")
