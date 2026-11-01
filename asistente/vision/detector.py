"""
Detector de objetos con YOLO (tarea de Isabella). Cumple el contrato Detector de interfaces.py:
encuentra DONDE hay objetos, sin nombrarlos (una sola clase: "objeto").
Los nombres los pone despues el Reconocedor de cada persona.

    det = DetectorYOLO("modelos/detector/best.pt")
    det.detectar(imagen_bgr) -> [Deteccion(confianza=0.91, caja=(x1, y1, x2, y2)), ...]
"""
from __future__ import annotations

from ..interfaces import Detector
from ..tipos import Deteccion


class DetectorYOLO(Detector):
    def __init__(self, ruta_pesos: str = "modelos/detector/best.pt", confianza_min: float = 0.35,
                 imgsz: int = 640):
        from ultralytics import YOLO          # import aqui: solo lo necesita quien use el detector
        self.modelo = YOLO(ruta_pesos)
        self.confianza_min, self.imgsz = confianza_min, imgsz

    def detectar(self, imagen) -> list[Deteccion]:
        r = self.modelo.predict(imagen, conf=self.confianza_min, imgsz=self.imgsz, verbose=False)[0]
        return [Deteccion(confianza=float(c), caja=tuple(float(v) for v in xyxy))
                for xyxy, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
