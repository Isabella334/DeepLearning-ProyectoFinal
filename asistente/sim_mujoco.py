# En macOS la ventana necesita mjpython: mjpython -m asistente.sim_mujoco
from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CONFIG = RAIZ / "config.yaml"
RUTA_TIPOS = Path(__file__).resolve().parent / "sim_objetos.yaml"

ESTACIONAMIENTO = np.array([3.0, 3.0])   # donde esperan los objetos que no estan en la mesa
FRICCION_DEFECTO = [0.5, 0.005, 0.0001]


def cargar_config(ruta=RUTA_CONFIG) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_tipos(cfg: dict, ruta=RUTA_TIPOS) -> dict:
    with open(ruta, encoding="utf-8") as f:
        tipos = yaml.safe_load(f)
    faltan = [n for n in cfg["objetos_del_experimento"] if n not in tipos]
    if faltan:
        raise ValueError(f"Faltan en {ruta.name}: {faltan}")
    return {n: tipos[n] for n in cfg["objetos_del_experimento"]}


def radio(tipo: dict) -> float:
    t, forma = tipo["tamano"], tipo["forma"]
    if forma == "box":
        return max(t[0], t[1])
    return t[1] if forma == "cylinder" else t[0]


def altura(tipo: dict) -> float:
    t = tipo["tamano"]
    return t[2] if tipo["forma"] == "box" else t[0]


def _geom_objeto(tipo: dict) -> str:
    t, forma = tipo["tamano"], tipo["forma"]
    fr = " ".join(str(v) for v in tipo.get("friccion", FRICCION_DEFECTO))
    rgba = " ".join(str(v) for v in tipo["rgba"])
    extra = ' euler="0 90 0"' if forma == "cylinder" else ""   # acostado
    condim = ' condim="6"' if forma in ("cylinder", "sphere") else ""   # para que no ruede
    return (f'<geom type="{forma}" size="{" ".join(str(v) for v in t)}"{extra}{condim} '
            f'mass="{tipo["masa"]}" rgba="{rgba}" friction="{fr}"/>')


def construir_xml(cfg: dict, tipos: dict) -> tuple[str, list[tuple[str, str]]]:
    # Un body por cada copia de cada objeto: MuJoCo no deja crear bodies en caliente
    ancho, alto = cfg["cartulina"]["ancho"], cfg["cartulina"]["alto"]
    zonas = ""
    for nombre, rgba in (("apartar", "0.2 0.7 0.3 0.4"), ("persona", "0.2 0.4 0.9 0.4")):
        x, y, m = cfg["zonas"][nombre]
        zonas += (f'<geom name="zona_{nombre}" type="box" pos="{x} {y} 0.0005" size="{m} {m} 0.0005" '
                  f'rgba="{rgba}" contype="0" conaffinity="0"/>\n')
    bordes = ""
    for i, (x, y, sx, sy) in enumerate([(0, alto / 2, ancho / 2, 0.003), (0, -alto / 2, ancho / 2, 0.003),
                                        (ancho / 2, 0, 0.003, alto / 2), (-ancho / 2, 0, 0.003, alto / 2)]):
        bordes += (f'<geom name="borde_{i}" type="box" pos="{x} {y} 0.0006" size="{sx} {sy} 0.0006" '
                   f'rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>\n')

    casillas, cuerpos = [], ""
    for nombre, tipo in tipos.items():
        for _ in range(tipo.get("copias", 1)):
            body = f"objeto_{len(casillas)}"
            px, py = ESTACIONAMIENTO + [0.1 * len(casillas), 0]
            cuerpos += (f'<body name="{body}" pos="{px} {py} {altura(tipo)}"><freejoint/>'
                        f'{_geom_objeto(tipo)}</body>\n')
            casillas.append((body, nombre))

    cx, cy = 0.0, -alto / 2 + 0.07
    xml = f"""
<mujoco model="carrito_ayudante">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><headlight diffuse="0.7 0.7 0.7"/><global offwidth="1280" offheight="1024"/></visual>
  <asset>
    <texture name="carton" type="2d" builtin="checker" rgb1="0.95 0.93 0.86" rgb2="0.92 0.90 0.83" width="256" height="256"/>
    <material name="cartulina" texture="carton" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <!-- El piso es solo visual; la que choca es la cartulina -->
    <geom name="piso" type="plane" size="2 2 0.01" pos="0 0 -0.001" rgba="0.45 0.4 0.35 1" contype="0" conaffinity="0"/>
    <geom name="mesa" type="plane" size="{ancho / 2} {alto / 2} 0.01" material="cartulina" friction="0.8 0.005 0.0001"/>
    {bordes}
    {zonas}
    <camera name="cenital" pos="0 0 {max(ancho, alto) * 1.3}" xyaxes="1 0 0 0 1 0"/>

    <body name="carrito" pos="{cx} {cy} 0.03">
      <freejoint/>
      <geom name="chasis" type="box" size="0.06 0.045 0.015" mass="0.35" rgba="0.15 0.35 0.75 1"/>
      <!-- Pala a 3 mm del piso: mas alta pasa por encima de lapiceros y virutas -->
      <geom name="pala_fondo" type="box" pos="0.068 0 -0.015" size="0.004 0.062 0.012" mass="0.02" rgba="0.95 0.6 0.1 1"/>
      <geom name="pala_izq" type="box" pos="0.094 0.064 -0.015" size="0.026 0.003 0.012" euler="0 0 12" mass="0.01" rgba="0.95 0.6 0.1 1"/>
      <geom name="pala_der" type="box" pos="0.094 -0.064 -0.015" size="0.026 0.003 0.012" euler="0 0 -12" mass="0.01" rgba="0.95 0.6 0.1 1"/>
      <geom name="loca" type="sphere" pos="0.04 0 -0.02" size="0.01" friction="0.001 0.001 0.001" mass="0.01" rgba="0.3 0.3 0.3 1"/>
      <body name="rueda_izq" pos="-0.02 0.055 -0.002">
        <joint name="j_izq" type="hinge" axis="0 1 0" damping="0.001" armature="0.0005"/>
        <geom type="cylinder" size="0.028 0.006" euler="90 0 0" friction="1.2 0.01 0.001" mass="0.03" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="rueda_der" pos="-0.02 -0.055 -0.002">
        <joint name="j_der" type="hinge" axis="0 1 0" damping="0.001" armature="0.0005"/>
        <geom type="cylinder" size="0.028 0.006" euler="90 0 0" friction="1.2 0.01 0.001" mass="0.03" rgba="0.1 0.1 0.1 1"/>
      </body>
    </body>

    {cuerpos}
  </worldbody>

  <actuator>
    <velocity name="m_izq" joint="j_izq" kv="0.05" ctrlrange="-20 20"/>
    <velocity name="m_der" joint="j_der" kv="0.05" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""
    return xml, casillas


def pose_carrito(model, data):
    bid = model.body("carrito").id
    x, y = data.xpos[bid][:2]
    R = data.xmat[bid].reshape(3, 3)
    ang = np.arctan2(R[1, 0], R[0, 0])
    return np.array([x, y]), ang


def pos_body(model, data, body):
    return data.xpos[model.body(body).id][:2].copy()


def ir_hacia(model, data, meta, vel=12.0):
    p, ang = pose_carrito(model, data)
    d = meta - p
    dist = np.linalg.norm(d)
    err = np.arctan2(d[1], d[0]) - ang
    err = (err + np.pi) % (2 * np.pi) - np.pi
    giro = 14.0 * err
    avance = vel if abs(err) < 0.5 else 0.0   # primero alinear, luego avanzar
    data.ctrl[0] = np.clip(avance - giro, -20, 20)   # rueda izquierda
    data.ctrl[1] = np.clip(avance + giro, -20, 20)   # rueda derecha
    return dist


def en_pala(p, ang, q) -> bool:
    rel = q - p
    adelante = np.cos(ang) * rel[0] + np.sin(ang) * rel[1]
    costado = -np.sin(ang) * rel[0] + np.cos(ang) * rel[1]
    return 0.06 < adelante < 0.14 and abs(costado) < 0.07


def esquivar(p, meta, obstaculos):
    """obstaculos: lista de (posicion, alcance). Devuelve una meta intermedia."""
    d = meta - p
    dist = np.linalg.norm(d)
    if dist < 1e-6:
        return meta
    rumbo = d / dist
    for q, a in obstaculos:
        rel = p - q
        r = np.linalg.norm(rel)
        if r < a and np.dot(-rel, rumbo) > -0.02:      # cerca y adelante
            rumbo = rumbo + 1.5 * (a - r) / a * rel / max(r, 1e-6)
    return p + rumbo / np.linalg.norm(rumbo) * min(dist, 0.10)


class Empujador:
    """aproximar -> girar -> empujar -> listo"""

    MAX_TRABAS = 6

    def __init__(self, model, data, body, zona, medio_lado, margen=0.01, otros=None, limites=None):
        self.model, self.data, self.body = model, data, body
        self.zona, self.tolerancia = np.asarray(zona, dtype=float), medio_lado - margen
        self.otros, self.limites = otros or (lambda: []), limites
        self.fase, self.rumbo, self.t_giro, self.t_fase = "aproximar", 0.0, 0.0, 0.0
        self.trabas, self.atorado, self.desvio = 0, None, None

    def paso(self) -> str:
        model, data = self.model, self.data
        obj = pos_body(model, data, self.body)
        if np.all(np.abs(obj - self.zona) < self.tolerancia):
            data.ctrl[:] = 0
            self.fase = "listo"
            return "listo"
        hacia_zona = self.zona - obj
        u = hacia_zona / np.linalg.norm(hacia_zona)   # direccion de empuje
        p, ang = pose_carrito(model, data)

        if self.fase == "aproximar":
            detras = obj - 0.17 * u
            rel = p - obj
            delante = np.dot(rel, u) > -0.06       # el carrito esta del lado equivocado
            if delante:
                # Rodearlo para no empujarlo sin querer
                perp = np.array([-u[1], u[0]])
                lado = 1.0 if np.dot(rel, perp) >= 0 else -1.0
                meta = obj + lado * 0.16 * perp - 0.05 * u
            else:
                meta = detras
            if self.limites is not None:
                meta = np.clip(meta, -np.asarray(self.limites), self.limites)
            # Si se trabo muchas veces deja de esquivar, para no quedarse quieto
            otros = self.otros() if self.trabas < self.MAX_TRABAS else []
            atorado = next((q for q in otros + [obj] if en_pala(p, ang, q)), None)
            if atorado is not None:
                self.fase, self.t_fase, self.atorado = "retroceder", data.time, atorado
                self.trabas += 1
                return self.fase
            obstaculos = [(q, 0.15) for q in otros] + [(obj, 0.12)]
            ir_hacia(model, data, esquivar(p, meta, obstaculos), vel=10.0)
            if np.linalg.norm(meta - p) < 0.015 and not delante:
                self.fase = "girar"
                self.rumbo = np.arctan2(u[1], u[0])
                self.t_giro = data.time
                self.trabas = 0
        elif self.fase == "retroceder":
            data.ctrl[0] = data.ctrl[1] = -8.0
            if data.time - self.t_fase > 0.5:
                # Hacerse a un lado, lejos de lo que estorbaba
                izq = np.array([-np.sin(ang), np.cos(ang)])
                lado = -1.0 if np.dot(self.atorado - p, izq) > 0 else 1.0
                self.desvio = p + lado * 0.12 * izq
                if self.limites is not None:
                    self.desvio = np.clip(self.desvio, -np.asarray(self.limites), self.limites)
                self.fase, self.t_fase = "desviar", data.time
        elif self.fase == "desviar":
            if ir_hacia(model, data, self.desvio, vel=10.0) < 0.02 or data.time - self.t_fase > 2.0:
                self.fase = "aproximar"
        elif self.fase == "girar":
            err = self.rumbo - ang
            err = (err + np.pi) % (2 * np.pi) - np.pi
            data.ctrl[0], data.ctrl[1] = -10 * err, 10 * err
            if abs(err) < 0.08 or data.time - self.t_giro > 3.0:
                self.fase = "empujar"
        elif self.fase == "empujar":
            ir_hacia(model, data, obj + 0.10 * u, vel=7.0)   # mantiene el objeto centrado en la pala
            # Se le salio de la pala: volver a acomodarse
            frente = np.array([np.cos(ang), np.sin(ang)])
            rel = obj - p
            lateral = abs(frente[0] * rel[1] - frente[1] * rel[0])
            if lateral > 0.045 or np.dot(rel, frente) < 0 or np.linalg.norm(rel) > 0.20:
                self.fase = "aproximar"
        return self.fase


class MundoSim:
    """El mundo sabe que es cada objeto (tipo oculto); el robot no."""

    def __init__(self, ventana: bool = False, cfg: dict | None = None, semilla: int = 0,
                 iniciales: list[str] | None = None):
        self.cfg = cfg or cargar_config()
        self.tipos = cargar_tipos(self.cfg)
        xml, self.casillas = construir_xml(self.cfg, self.tipos)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.ocupante: dict[str, str] = {}      # body -> id del objeto
        self.rng = np.random.default_rng(semilla)
        self._siguiente_id = 0
        mujoco.mj_forward(self.model, self.data)
        if iniciales is None:
            iniciales = [n for n, t in self.tipos.items() if t.get("al_inicio", True)]
        for tipo in iniciales:
            self.colocar(self.nuevo_id(), tipo, *self.lugar_libre(tipo))
        self.visor = None
        if ventana:
            from mujoco import viewer
            self.visor = viewer.launch_passive(self.model, self.data)

    def nuevo_id(self) -> str:
        self._siguiente_id += 1
        return f"obj_{self._siguiente_id}"

    def presentes(self) -> dict[str, str]:
        """{id: tipo oculto}"""
        tipo_de = dict(self.casillas)
        return {oid: tipo_de[body] for body, oid in self.ocupante.items()}

    def _body(self, oid: str) -> str | None:
        return next((b for b, o in self.ocupante.items() if o == oid), None)

    def _poner(self, body, x, y, ang=0.0, z=None):
        jnt = self.model.body(body).jntadr[0]
        qa, va = self.model.jnt_qposadr[jnt], self.model.jnt_dofadr[jnt]
        tipo = self.tipos[dict(self.casillas)[body]]
        self.data.qpos[qa:qa + 7] = [x, y, altura(tipo) if z is None else z,
                                     np.cos(ang / 2), 0, 0, np.sin(ang / 2)]
        self.data.qvel[va:va + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def colocar(self, oid: str, tipo: str, x: float, y: float, ang: float = 0.0) -> None:
        body = self._body(oid)
        if body is None:
            libres = [b for b, t in self.casillas if t == tipo and b not in self.ocupante]
            if not libres:
                raise ValueError(f"No quedan copias libres de '{tipo}' (ver 'copias' en sim_objetos.yaml)")
            body = libres[0]
            self.ocupante[body] = oid
        self._poner(body, x, y, ang)

    def quitar(self, oid: str) -> None:
        body = self._body(oid)
        if body is None:
            return
        del self.ocupante[body]
        i = [b for b, _ in self.casillas].index(body)
        self._poner(body, *(ESTACIONAMIENTO + [0.1 * i, 0]))

    def lugar_libre(self, tipo: str, intentos: int = 500) -> tuple[float, float]:
        ancho, alto = self.cfg["cartulina"]["ancho"], self.cfg["cartulina"]["alto"]
        r = radio(self.tipos[tipo])
        ocupados = [(self.pos(o), radio(self.tipos[t])) for o, t in self.presentes().items()]
        ocupados.append((pose_carrito(self.model, self.data)[0], 0.11))
        ocupados += [(np.array(self.cfg["zonas"][z][:2]), self.cfg["zonas"][z][2] * 1.2)
                     for z in ("apartar", "persona")]
        for _ in range(intentos):
            p = self.rng.uniform([-ancho / 2 + r + 0.03, -alto / 2 + r + 0.03],
                                 [ancho / 2 - r - 0.03, alto / 2 - r - 0.03])
            if all(np.linalg.norm(p - q) > r + rq + 0.03 for q, rq in ocupados):
                return float(p[0]), float(p[1])
        raise RuntimeError(f"No hay lugar libre en la mesa para '{tipo}'")

    def pos(self, oid: str) -> np.ndarray:
        return pos_body(self.model, self.data, self._body(oid))

    def altura_actual(self, oid: str) -> float:
        return float(self.data.xpos[self.model.body(self._body(oid)).id][2])

    def empujable(self, oid: str) -> bool:
        return self.tipos[self.presentes()[oid]].get("empujable", True)

    def zona(self, nombre: str) -> tuple[np.ndarray, float]:
        x, y, m = self.cfg["zonas"][nombre]
        return np.array([x, y]), m

    def en_zona(self, xy, nombre: str, margen: float = 0.01) -> bool:
        centro, m = self.zona(nombre)
        return bool(np.all(np.abs(np.asarray(xy) - centro) < m - margen))

    def en_mesa(self, xy, margen: float = 0.0) -> bool:
        ancho, alto = self.cfg["cartulina"]["ancho"], self.cfg["cartulina"]["alto"]
        return abs(xy[0]) <= ancho / 2 + margen and abs(xy[1]) <= alto / 2 + margen

    def empujador(self, oid: str, zona: str) -> Empujador:
        def otros():
            return [q for o in self.presentes() if o != oid and self.en_mesa(q := self.pos(o), 0.02)]
        # El centro del carrito puede asomarse 6 cm del borde (su parte trasera sobresale)
        ancho, alto = self.cfg["cartulina"]["ancho"], self.cfg["cartulina"]["alto"]
        return Empujador(self.model, self.data, self._body(oid), *self.zona(zona),
                         otros=otros, limites=(ancho / 2 + 0.06, alto / 2 + 0.06))

    def paso(self):
        t0 = time.time()
        mujoco.mj_step(self.model, self.data)
        if self.visor is not None and self.visor.is_running():
            self.visor.sync()
            time.sleep(max(0.0, self.model.opt.timestep - (time.time() - t0)))   # tiempo real

    def frenar(self):
        self.data.ctrl[:] = 0


def correr(con_ventana=True, segundos=40.0):
    mundo = MundoSim(ventana=con_ventana)
    logrados = 0
    try:
        for oid, tipo in mundo.presentes().items():
            if not mundo.empujable(oid):
                continue
            inicio, t0 = mundo.pos(oid), mundo.data.time
            emp = mundo.empujador(oid, "apartar")
            while mundo.data.time - t0 < segundos and emp.paso() != "listo":
                mundo.paso()
            mundo.frenar()
            ok = emp.fase == "listo"
            logrados += ok
            print(f"{oid} ({tipo}): {inicio.round(3)} -> {mundo.pos(oid).round(3)} en "
                  f"{mundo.data.time - t0:.1f} s  ->  {'LOGRADO' if ok else 'NO LOGRADO'}")
            mundo.quitar(oid)   # para que no estorbe al siguiente
    finally:
        if mundo.visor is not None:
            mundo.visor.close()
    print(f"Apartados: {logrados}")
    return logrados


if __name__ == "__main__":
    correr(con_ventana="--test" not in sys.argv)
