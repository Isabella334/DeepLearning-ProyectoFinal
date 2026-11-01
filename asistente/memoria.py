from __future__ import annotations

import csv
import sqlite3
import threading
import time
from pathlib import Path

from .interfaces import Memoria
from .tipos import Contexto, Decision, Evento, Feedback, Resultado

ESQUEMA = """
CREATE TABLE IF NOT EXISTS personas (
    id TEXT PRIMARY KEY, creada REAL NOT NULL);
CREATE TABLE IF NOT EXISTS ensenanzas (
    id INTEGER PRIMARY KEY, persona_id TEXT NOT NULL, nombre TEXT NOT NULL,
    objeto_id TEXT, t REAL NOT NULL);
CREATE TABLE IF NOT EXISTS eventos (
    id INTEGER PRIMARY KEY, tipo TEXT NOT NULL, detalle TEXT NOT NULL,
    persona_id TEXT, t REAL NOT NULL);
CREATE TABLE IF NOT EXISTS interacciones (
    id INTEGER PRIMARY KEY, persona_id TEXT NOT NULL, accion TEXT NOT NULL, objeto_id TEXT,
    objeto_nombre TEXT, confianza REAL, motivo TEXT, exito INTEGER, aprobado INTEGER,
    fuente TEXT, t REAL NOT NULL);
CREATE TABLE IF NOT EXISTS modelos (
    id INTEGER PRIMARY KEY, persona_id TEXT NOT NULL, version TEXT NOT NULL,
    ruta TEXT NOT NULL, t REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_eventos ON eventos (persona_id, t);
CREATE INDEX IF NOT EXISTS idx_interacciones ON interacciones (persona_id, id);
"""
TABLAS = ("personas", "ensenanzas", "eventos", "interacciones", "modelos")


def _bool(v):
    return None if v is None else bool(v)


class MemoriaSQLite(Memoria):

    def __init__(self, ruta: str = ":memory:"):
        # Streamlit atiende cada clic en otro hilo: una conexion compartida protegida con un candado
        self.con = sqlite3.connect(ruta, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.candado = threading.Lock()
        with self.candado, self.con:
            self.con.executescript(ESQUEMA)

    def _escribir(self, sql: str, params: tuple, persona_id: str | None) -> int:
        with self.candado, self.con:
            if persona_id is not None:
                self.con.execute("INSERT OR IGNORE INTO personas VALUES (?, ?)", (persona_id, time.time()))
            return self.con.execute(sql, params).lastrowid

    def _leer(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.candado:
            return self.con.execute(sql, params).fetchall()

    def registrar_interaccion(self, contexto: Contexto, decision: Decision,
                              resultado: Resultado, feedback: Feedback | None) -> int:
        obj = next((o for o in contexto.escena.objetos if o.id == decision.objeto_id), None)
        return self._escribir(
            "INSERT INTO interacciones (persona_id, accion, objeto_id, objeto_nombre, confianza, motivo,"
            " exito, aprobado, fuente, t) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (contexto.persona_id, decision.accion.value, decision.objeto_id, obj.nombre if obj else None,
             decision.confianza, decision.motivo, int(resultado.exito),
             None if feedback is None else int(feedback.aprobado),
             None if feedback is None else feedback.fuente, time.time()),
            contexto.persona_id)

    def registrar_evento(self, evento: Evento) -> None:
        self._escribir("INSERT INTO eventos (tipo, detalle, persona_id, t) VALUES (?, ?, ?, ?)",
                       (evento.tipo, evento.detalle, evento.persona_id, evento.t), evento.persona_id)

    def registrar_ensenanza(self, persona_id: str, nombre: str, objeto_id: str) -> None:
        self._escribir("INSERT INTO ensenanzas (persona_id, nombre, objeto_id, t) VALUES (?, ?, ?, ?)",
                       (persona_id, nombre, objeto_id, time.time()), persona_id)

    def registrar_modelo(self, persona_id: str, version, ruta: str) -> None:
        self._escribir("INSERT INTO modelos (persona_id, version, ruta, t) VALUES (?, ?, ?, ?)",
                       (persona_id, str(version), ruta, time.time()), persona_id)

    def eventos_recientes(self, persona_id: str, segundos: float = 300) -> list[Evento]:
        filas = self._leer("SELECT tipo, detalle, persona_id, t FROM eventos"
                           " WHERE persona_id = ? AND t >= ? ORDER BY t, id",
                           (persona_id, time.time() - segundos))
        return [Evento(f["tipo"], f["detalle"], f["persona_id"], f["t"]) for f in filas]

    def historial(self, persona_id: str, limite: int = 200) -> list[dict]:
        filas = self._leer("SELECT * FROM (SELECT * FROM interacciones WHERE persona_id = ?"
                           " ORDER BY id DESC LIMIT ?) ORDER BY id", (persona_id, limite))
        return [dict(f) | {"exito": _bool(f["exito"]), "aprobado": _bool(f["aprobado"])} for f in filas]

    def ensenanzas(self, persona_id: str) -> list[dict]:
        return [dict(f) for f in self._leer("SELECT * FROM ensenanzas WHERE persona_id = ? ORDER BY id",
                                            (persona_id,))]

    def exportar_csv(self, carpeta: str) -> list[Path]:
        destino = Path(carpeta)
        destino.mkdir(parents=True, exist_ok=True)
        rutas = []
        for tabla in TABLAS:
            with self.candado:
                cursor = self.con.execute(f"SELECT * FROM {tabla} ORDER BY rowid")
                columnas = [c[0] for c in cursor.description]
                filas = cursor.fetchall()
            ruta = destino / f"{tabla}.csv"
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                escritor = csv.writer(f)
                escritor.writerow(columnas)
                escritor.writerows(filas)
            rutas.append(ruta)
        return rutas

    def cerrar(self) -> None:
        with self.candado:
            self.con.close()
