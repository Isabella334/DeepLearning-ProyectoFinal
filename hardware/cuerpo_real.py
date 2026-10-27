"""
Cuerpo REAL (tarea de integracion). Mismo contrato que CuerpoSim: el cerebro no nota la diferencia.

Idea: repetir el mismo control que en simulacion (ir_hacia, apartar), pero:
  - la pose del carrito viene de la camara (ArUco) en vez de MuJoCo;
  - las velocidades se mandan por UDP al ESP32 (ver PROTOCOLO.md).
"""
from __future__ import annotations

import json
import socket
import time

from asistente.interfaces import Cuerpo, Percepcion
from asistente.tipos import Accion, Resultado


class EnlaceESP32:
    def __init__(self, ip: str, puerto: int = 4210):
        self.dest = (ip, puerto)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def velocidades(self, izq: float, der: float):
        izq, der = max(-1.0, min(1.0, izq)), max(-1.0, min(1.0, der))
        self.sock.sendto(json.dumps({"izq": izq, "der": der}).encode(), self.dest)

    def parar(self):
        self.velocidades(0.0, 0.0)


class CuerpoReal(Cuerpo):
    def __init__(self, ip_esp32: str, percepcion: Percepcion, limite_s: float = 60.0):
        self.enlace, self.percepcion, self.limite_s = EnlaceESP32(ip_esp32), percepcion, limite_s

    def ejecutar(self, accion, objeto, escena):
        if accion == Accion.IGNORAR:
            return Resultado(True, 0.0, "no hice nada")
        t0 = time.time()
        # TODO integracion: bucle a ~20 Hz:
        #   escena = self.percepcion.observar()          # pose del carrito y del objeto
        #   calcular velocidades con la misma logica de asistente/sim_mujoco.py
        #   self.enlace.velocidades(izq, der)
        #   terminar cuando el objeto llegue a la zona o se acabe el tiempo
        self.enlace.parar()
        return Resultado(False, time.time() - t0, "CuerpoReal todavia no implementado")

    def cerrar(self):
        self.enlace.parar()
