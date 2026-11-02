"""
Percepcion con CAMARA (tarea de Isabella). Es la pieza de la version B, el gemelo digital:
la camara ve la cartulina real y el carrito virtual actua sobre una copia de esa mesa.

En cada observar():
  1. Los 4 marcadores ArUco de las esquinas -> homografia (pixeles -> metros de la cartulina).
     Se guarda la ultima buena: si un marcador "parpadea", se sigue usando.
  2. DetectorYOLO -> cajas de "objeto" (sin nombre).
  3. Cada caja -> punto de apoyo en metros, recorte -> embedding -> Reconocedor de la persona.
  4. Ids estables: cada objeto se empareja con el mas cercano del cuadro anterior.
  5. Si hay un mundo simulado, se copia ahi lo que ve la camara (sincronizar).

Probar solo la camara (sin el resto del robot):
    python -m asistente.vision.percepcion_camara            # ventana con cajas, ids y metros
    python -m asistente.vision.percepcion_camara --camara 1
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from ..configuracion import cargar_config
from ..interfaces import Percepcion
from ..tipos import DESCONOCIDO, Escena, Objeto, PoseCarrito

# La ultima calibracion buena (homografia) se guarda aqui: si la camara no se mueve, sirve
# aunque despues algun marcador no se vea (por reflejo, una mano encima, etc.)
CALIBRACION = Path(__file__).resolve().parents[2] / "modelos" / "calibracion_camara.npy"

# Tamano del objeto simulado segun lo ancho que se ve en la camara (metros)
TAMANOS = [(0.06, "real pequeno"), (0.13, "real mediano"), (float("inf"), "real grande")]


class Rastro:
    """Un objeto que se viene siguiendo entre cuadros."""

    def __init__(self, oid, xy, ancho, emb):
        self.id, self.xy, self.ancho = oid, np.asarray(xy, float), ancho
        self.emb = np.asarray(emb, float)
        self.xy_camara = self.xy.copy()       # ultima posicion REAL (para saber si la persona lo movio)
        self.perdido = 0                      # cuadros seguidos sin verlo
        self.nombre, self.confianza = DESCONOCIDO, 0.0

    def actualizar(self, xy, ancho, emb, alfa=0.5):
        self.xy, self.ancho, self.perdido = np.asarray(xy, float), ancho, 0
        # Promedio movil del embedding: el nombre "parpadea" menos y se ensena con un mejor ejemplo
        self.emb = alfa * np.asarray(emb, float) + (1 - alfa) * self.emb


class PercepcionCamara(Percepcion):
    def __init__(self, reconocedor, detector=None, camara=0, persona_id: str = "persona_A",
                 mundo=None, ventana: bool = False, cfg: dict | None = None,
                 distancia_max: float = 0.07, olvidar_tras: int = 3, ancho: int = 1280, alto: int = 720,
                 guardar_calibracion: bool | None = None):
        """camara: numero de webcam, ruta de un video, o cualquier objeto con .read() como cv2.
        mundo: MundoSim(con_camara=True) para el gemelo digital (opcional)."""
        import cv2
        self.cv2 = cv2
        self.cfg = cfg or cargar_config()
        self.reconocedor, self.persona_id, self.mundo, self.ventana = reconocedor, persona_id, mundo, ventana
        if guardar_calibracion is None:       # por defecto, solo con una webcam de verdad
            guardar_calibracion = isinstance(camara, int)
        if detector is None:
            from .detector import DetectorYOLO
            detector = DetectorYOLO()
        self.detector = detector
        self.distancia_max, self.olvidar_tras = distancia_max, olvidar_tras

        self._vaciar = 1                      # cuadros a descartar para tener uno fresco
        if hasattr(camara, "read"):
            self.cap = camara
        else:
            self._vaciar = 4 if isinstance(camara, int) else 1
            if isinstance(camara, int) and sys.platform == "win32":
                # En Windows, DirectShow abre la webcam al instante (el modo por defecto puede trabarse)
                self.cap = cv2.VideoCapture(camara, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(camara)
            if not self.cap.isOpened():
                raise RuntimeError(f"No se pudo abrir la camara {camara!r}. Proba con --camara 1 (o 2), "
                                   "o revisa que ningun otro programa la este usando.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ar = self.cfg.get("aruco", {})
        dic = getattr(cv2.aruco, ar.get("diccionario", "DICT_4X4_50"))
        self.aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dic),
                                             cv2.aruco.DetectorParameters())
        self.centros = {int(k): np.array(v, float) for k, v in ar["centros"].items()}
        self.id_carrito = int(ar.get("carrito", 10))
        cart = self.cfg["cartulina"]
        self.medio = np.array([cart["ancho"] / 2, cart["alto"] / 2])

        self.H = None                         # homografia pixeles -> metros (la ultima buena)
        self.ruta_calibracion = CALIBRACION if guardar_calibracion else None
        if self.ruta_calibracion is not None and self.ruta_calibracion.exists():
            self.H = np.load(self.ruta_calibracion)
            print(f"   (camara) Usando la calibracion guardada ({self.ruta_calibracion.name}). "
                  "Si moviste la camara, se actualiza sola cuando vea las 4 esquinas.")
        self.rastros: list[Rastro] = []
        self._n = 0
        self.ultima: Escena | None = None
        self.ultimo_cuadro = None
        self.vista = None                     # imagen con cajas, ids y nombres (BGR), para la pantalla

    # ------------------------------------------------------------------ geometria
    def _leer(self):
        cuadro = None
        for _ in range(self._vaciar):          # la webcam guarda cuadros viejos: se vacia el buffer
            ok, c = self.cap.read()
            if ok:
                cuadro = c
        if cuadro is None:
            if self.ultimo_cuadro is not None and self._vaciar == 1:
                return self.ultimo_cuadro     # un video o una foto que se acabo: se repite el ultimo cuadro
            raise RuntimeError("No llega imagen de la camara.")
        return cuadro

    def _homografia(self, cuadro):
        esquinas, ids, _ = self.aruco.detectMarkers(cuadro)
        vistos = {}
        if ids is not None:
            vistos = {int(i): c[0] for i, c in zip(ids.flatten(), esquinas)}
        comunes = [i for i in self.centros if i in vistos]
        if len(comunes) == 4:
            src = np.array([vistos[i].mean(axis=0) for i in comunes], np.float32)
            dst = np.array([self.centros[i] for i in comunes], np.float32)
            self.H = self.cv2.getPerspectiveTransform(src, dst)
            if self.ruta_calibracion is not None:
                self.ruta_calibracion.parent.mkdir(parents=True, exist_ok=True)
                np.save(self.ruta_calibracion, self.H)
        return vistos

    def a_metros(self, puntos_px) -> np.ndarray:
        p = np.asarray(puntos_px, np.float32).reshape(-1, 1, 2)
        return self.cv2.perspectiveTransform(p, self.H).reshape(-1, 2)

    def _en_marcador(self, xy) -> bool:
        return any(np.all(np.abs(xy - c) < 0.065) for c in self.centros.values())

    # ------------------------------------------------------------------ contrato
    def observar(self) -> Escena:
        cuadro = self._leer()
        self.ultimo_cuadro = cuadro
        vistos = self._homografia(cuadro)
        if self.H is None:
            print("   (camara) Todavia no veo las 4 esquinas: acomoda la camara.")
            self.vista = cuadro.copy()
            for i, c in vistos.items():           # los que si ve, en verde con su numero
                self.cv2.polylines(self.vista, [c.astype(np.int32)], True, (0, 200, 0), 3)
                self.cv2.putText(self.vista, str(i), tuple(int(v) for v in c.mean(axis=0)),
                                 self.cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
            faltan = sorted(set(self.centros) - set(vistos))
            self.cv2.putText(self.vista, f"Necesito ver las 4 esquinas UNA vez. Faltan: {faltan}", (20, 35),
                             self.cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if self.ventana:
                self.cv2.imshow("Lo que ve el robot", self.vista)
                self.cv2.waitKey(1)
            return self._escena([], vistos)

        cajas, puntos, anchos = [], [], []
        alto_img, ancho_img = cuadro.shape[:2]
        for d in self.detector.detectar(cuadro):
            x1, y1, x2, y2 = d.caja
            # Punto de apoyo: cerca de la base de la caja (la camara esta inclinada)
            base = self.a_metros([[(x1 + x2) / 2, y1 + 0.85 * (y2 - y1)], [x1, y2], [x2, y2]])
            xy = base[0]
            if np.any(np.abs(xy) > self.medio + 0.02) or self._en_marcador(xy):
                continue                       # fuera de la cartulina o sobre un marcador
            cajas.append((max(0, int(x1)), max(0, int(y1)), min(ancho_img, int(x2)), min(alto_img, int(y2))))
            puntos.append(xy)
            anchos.append(float(np.linalg.norm(base[1] - base[2])))

        recortes = [cuadro[y1:y2, x1:x2] for x1, y1, x2, y2 in cajas]
        if hasattr(self.reconocedor, "embeddings"):
            embs = self.reconocedor.embeddings(recortes)
        else:
            embs = [self.reconocedor.embedding(r) for r in recortes]

        self._emparejar(puntos, anchos, embs)
        for r in self.rastros:
            if r.perdido == 0:
                r.nombre, r.confianza = self.reconocedor.reconocer(self.persona_id, r.emb.tolist())
        if self.mundo is not None:
            self.sincronizar()
        activos = [r for r in self.rastros if r.perdido == 0]
        self.vista = self._dibujar(cuadro, activos, cajas, vistos)
        if self.ventana:
            self.cv2.imshow("Lo que ve el robot", self.vista)
            self.cv2.waitKey(1)
        return self._escena(activos, vistos)

    def ensenar(self, persona_id, objeto_id, nombre):
        r = next((r for r in self.rastros if r.id == objeto_id), None)
        if r is not None:
            self.reconocedor.ensenar(persona_id, nombre, r.emb.tolist())
            r.nombre, r.confianza = nombre, 1.0

    def cerrar(self):
        if hasattr(self.cap, "release"):
            self.cap.release()
        if self.ventana:
            self.cv2.destroyAllWindows()

    # ------------------------------------------------------------------ seguimiento
    def _emparejar(self, puntos, anchos, embs):
        libres = list(range(len(puntos)))
        pares = sorted(((np.linalg.norm(r.xy_camara - puntos[j]), i, j)
                        for i, r in enumerate(self.rastros) for j in libres))
        usados_r, usados_d = set(), set()
        for dist, i, j in pares:
            if dist > self.distancia_max or i in usados_r or j in usados_d:
                continue
            r = self.rastros[i]
            r.actualizar(puntos[j], anchos[j], embs[j])
            usados_r.add(i)
            usados_d.add(j)
        for i, r in enumerate(self.rastros):
            if i not in usados_r:
                r.perdido += 1
        for j in libres:
            if j not in usados_d:                 # objeto nuevo en la mesa
                self._n += 1
                self.rastros.append(Rastro(f"cam_{self._n}", puntos[j], anchos[j], embs[j]))
        olvidados = [r for r in self.rastros if r.perdido > self.olvidar_tras]
        if self.mundo is not None:
            for r in olvidados:
                self.mundo.quitar(r.id)
        self.rastros = [r for r in self.rastros if r.perdido <= self.olvidar_tras]

    def sincronizar(self, umbral_movido: float = 0.02):
        """Copia la mesa real al mundo simulado.
        - Objeto nuevo: se coloca en la simulacion.
        - Objeto que la PERSONA movio (cambio en la camara): se mueve en la simulacion.
        - Si solo lo movio el carrito virtual, la simulacion conserva su posicion
          (la camara sigue viendo el objeto real donde estaba: eso no es un cambio)."""
        presentes = self.mundo.presentes()
        for r in self.rastros:
            if r.perdido:
                continue
            tipo = next(t for limite, t in TAMANOS if r.ancho < limite)
            if r.id not in presentes:
                try:
                    self.mundo.colocar(r.id, tipo, *r.xy)
                except ValueError:
                    continue                      # no quedan copias de ese tamano en la simulacion
                r.xy_camara = r.xy.copy()
            elif np.linalg.norm(r.xy - r.xy_camara) > umbral_movido:
                self.mundo.colocar(r.id, presentes[r.id], *r.xy)
                r.xy_camara = r.xy.copy()

    # ------------------------------------------------------------------ salida
    def _escena(self, rastros, vistos) -> Escena:
        objetos = []
        for r in rastros:
            xy = r.xy
            if self.mundo is not None and r.id in self.mundo.presentes():
                xy = self.mundo.pos(r.id)          # el gemelo: donde lo dejo el carrito virtual
            objetos.append(Objeto(id=r.id, x=float(xy[0]), y=float(xy[1]), nombre=r.nombre,
                                  confianza=float(np.clip(r.confianza, 0, 1)), embedding=r.emb.tolist()))
        self.ultima = Escena(objetos=objetos, carrito=self._carrito(vistos),
                             persona_id=self.persona_id, t=time.time())
        return self.ultima

    def _carrito(self, vistos) -> PoseCarrito:
        if self.mundo is not None:
            from ..sim_mujoco import pose_carrito
            p, ang = pose_carrito(self.mundo.model, self.mundo.data)
            return PoseCarrito(float(p[0]), float(p[1]), float(ang))
        if self.H is not None and self.id_carrito in vistos:      # carrito fisico (version C)
            c = self.a_metros(vistos[self.id_carrito])
            frente = (c[0] + c[1]) / 2 - c.mean(axis=0)
            return PoseCarrito(float(c.mean(axis=0)[0]), float(c.mean(axis=0)[1]),
                               float(np.arctan2(frente[1], frente[0])))
        return PoseCarrito(0.0, -self.medio[1] + 0.07, np.pi / 2)

    def _dibujar(self, cuadro, rastros, cajas, vistos):
        """Devuelve la imagen de la camara con lo que entendio el robot encima."""
        cv2 = self.cv2
        vista = cuadro.copy()
        for c in vistos.values():
            cv2.polylines(vista, [c.astype(np.int32)], True, (0, 200, 0), 2)
        for r in rastros:
            px = cv2.perspectiveTransform(np.float32([[r.xy]]), np.linalg.inv(self.H))[0, 0]
            color = (0, 200, 0) if r.nombre != DESCONOCIDO else (0, 165, 255)
            cv2.circle(vista, (int(px[0]), int(px[1])), 6, color, -1)
            cv2.putText(vista, f"{r.id}: {r.nombre} ({r.confianza:.0%})", (int(px[0]) + 8, int(px[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        for x1, y1, x2, y2 in cajas:
            cv2.rectangle(vista, (x1, y1), (x2, y2), (255, 200, 0), 1)
        estado = ("Esquinas OK (calibracion guardada)" if len([i for i in self.centros if i in vistos]) == 4
                  else "Faltan esquinas: usando la calibracion guardada")
        cv2.putText(vista, estado, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return vista


def main():
    import argparse

    from ..mocks import ReconocedorFalso
    ap = argparse.ArgumentParser(description="Probar la camara: cajas, ids y posicion en metros")
    ap.add_argument("--camara", default="0", help="numero de camara o ruta de un video")
    ap.add_argument("--reconocedor", choices=["p2", "simple"], default="simple")
    args = ap.parse_args()
    cam = int(args.camara) if args.camara.isdigit() else args.camara
    if args.reconocedor == "p2":
        from .reconocedor import ReconocedorPocosEjemplos
        rec = ReconocedorPocosEjemplos()
    else:
        rec = ReconocedorFalso()
    p = PercepcionCamara(rec, camara=cam, ventana=True)
    print("Mostrando lo que ve el robot. Ctrl+C para terminar.")
    try:
        while True:
            e = p.observar()
            print("  ", [(o.id, round(o.x, 2), round(o.y, 2)) for o in e.objetos])
    except KeyboardInterrupt:
        pass
    finally:
        p.cerrar()


if __name__ == "__main__":
    main()
