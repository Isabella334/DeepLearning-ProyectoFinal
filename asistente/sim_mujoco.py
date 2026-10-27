"""
Carrito ayudante - primera simulacion en MuJoCo
-----------------------------------------------
Una mesa, un carrito de traccion diferencial con una pala en forma de U
al frente y un borrador. El carrito se pone detras del borrador y lo empuja
hasta una zona objetivo (accion "apartar").

Uso directo (para probar solo la fisica):
    python -m asistente.sim_mujoco          # abre la ventana 3D
    python -m asistente.sim_mujoco --test   # sin ventana

El "cerebro" (decidir QUE hacer) ira despues en otro archivo. Este archivo
solo es el "cuerpo": como moverse para cumplir una accion.
"""
import sys
import time
import numpy as np
import mujoco

XML = """
<mujoco model="carrito_ayudante">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><headlight diffuse="0.7 0.7 0.7"/></visual>
  <asset>
    <texture name="madera" type="2d" builtin="checker" rgb1="0.80 0.68 0.50" rgb2="0.76 0.64 0.46" width="256" height="256"/>
    <material name="mesa" texture="madera" texrepeat="6 6"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <!-- Mesa de 1.2 m x 0.8 m -->
    <geom name="mesa" type="plane" size="0.6 0.4 0.01" material="mesa" friction="0.8 0.005 0.0001"/>
    <!-- Zona objetivo (solo visual): adonde se apartan las cosas -->
    <geom name="zona" type="box" pos="0.45 0.25 0.0005" size="0.08 0.08 0.0005" rgba="0.2 0.7 0.3 0.4" contype="0" conaffinity="0"/>
    <!-- Camara cenital, como la webcam real -->
    <camera name="cenital" pos="0 0 1.3" xyaxes="1 0 0 0 1 0"/>

    <!-- Carrito -->
    <body name="carrito" pos="-0.35 -0.2 0.03">
      <freejoint/>
      <geom name="chasis" type="box" size="0.06 0.045 0.015" mass="0.35" rgba="0.15 0.35 0.75 1"/>
      <!-- Pala en forma de U: placa de fondo y dos alas hacia adelante -->
      <geom name="pala_fondo" type="box" pos="0.068 0 -0.008" size="0.004 0.062 0.012" mass="0.02" rgba="0.95 0.6 0.1 1"/>
      <geom name="pala_izq" type="box" pos="0.094 0.064 -0.008" size="0.026 0.003 0.012" euler="0 0 12" mass="0.01" rgba="0.95 0.6 0.1 1"/>
      <geom name="pala_der" type="box" pos="0.094 -0.064 -0.008" size="0.026 0.003 0.012" euler="0 0 -12" mass="0.01" rgba="0.95 0.6 0.1 1"/>
      <!-- Rueda loca adelante (esfera sin friccion) -->
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

    <!-- Borrador: el primer objeto de prueba -->
    <body name="borrador" pos="0.05 0.0 0.008">
      <freejoint/>
      <geom name="borrador" type="box" size="0.025 0.012 0.008" mass="0.02" rgba="0.95 0.95 0.95 1" friction="0.5 0.005 0.0001"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Motores de velocidad, como los motores DC con su driver -->
    <velocity name="m_izq" joint="j_izq" kv="0.05" ctrlrange="-20 20"/>
    <velocity name="m_der" joint="j_der" kv="0.05" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""

ZONA = np.array([0.45, 0.25])


def pose_carrito(model, data):
    """Posicion (x, y) y angulo del carrito. En el robot real esto lo da el ArUco."""
    bid = model.body("carrito").id
    x, y = data.xpos[bid][:2]
    R = data.xmat[bid].reshape(3, 3)
    ang = np.arctan2(R[1, 0], R[0, 0])
    return np.array([x, y]), ang


def pos_objeto(model, data, nombre):
    return data.xpos[model.body(nombre).id][:2].copy()


def ir_hacia(model, data, meta, vel=12.0):
    """Control simple: girar hacia la meta y avanzar. Devuelve la distancia."""
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


def accion_apartar(model, data, objeto="borrador"):
    """Accion 'apartar': ponerse detras del objeto y empujarlo hasta la zona.
    Maquina de estados simple: aproximar -> girar -> empujar -> listo."""
    obj = pos_objeto(model, data, objeto)
    hacia_zona = ZONA - obj
    dist_zona = np.linalg.norm(hacia_zona)
    if np.all(np.abs(obj - ZONA) < 0.06):     # el objeto ya esta dentro de la zona
        data.ctrl[:] = 0
        accion_apartar.fase = "listo"
        return "listo"
    u = hacia_zona / dist_zona                 # direccion en la que hay que empujar
    p, ang = pose_carrito(model, data)
    fase = accion_apartar.fase

    if fase == "aproximar":
        detras = obj - 0.17 * u                # punto detras del objeto
        rel = p - obj
        delante = np.dot(rel, u) > -0.06       # el carrito esta del lado equivocado
        if delante:
            # Rodear el objeto por el costado mas cercano para no empujarlo sin querer
            perp = np.array([-u[1], u[0]])
            lado = 1.0 if np.dot(rel, perp) >= 0 else -1.0
            meta = obj + lado * 0.16 * perp - 0.05 * u
        else:
            meta = detras
        if ir_hacia(model, data, meta, vel=10.0) < 0.015 and not delante:
            fase = "girar"
            accion_apartar.rumbo = np.arctan2(u[1], u[0])   # rumbo fijo para girar
            accion_apartar.t_giro = data.time
    elif fase == "girar":
        err = accion_apartar.rumbo - ang
        err = (err + np.pi) % (2 * np.pi) - np.pi
        data.ctrl[0], data.ctrl[1] = -10 * err, 10 * err
        if abs(err) < 0.08 or data.time - accion_apartar.t_giro > 3.0:
            fase = "empujar"
    elif fase == "empujar":
        # Apuntar a la zona, corrigiendo para mantener el objeto centrado en la pala
        ir_hacia(model, data, obj + 0.10 * u, vel=7.0)
        # Si el objeto se salio de la pala, volver a acomodarse
        frente = np.array([np.cos(ang), np.sin(ang)])
        rel = obj - p
        lateral = abs(frente[0] * rel[1] - frente[1] * rel[0])
        if lateral > 0.045 or np.dot(rel, frente) < 0 or np.linalg.norm(rel) > 0.20:
            fase = "aproximar"
    accion_apartar.fase = fase
    return fase


accion_apartar.fase = "aproximar"
accion_apartar.rumbo = 0.0
accion_apartar.t_giro = 0.0


def correr(con_ventana=True, segundos=40.0):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    inicio = pos_objeto(model, data, "borrador")

    if con_ventana:
        from mujoco import viewer
        with viewer.launch_passive(model, data) as v:
            while v.is_running() and data.time < segundos:
                t0 = time.time()
                estado = accion_apartar(model, data)
                mujoco.mj_step(model, data)
                v.sync()
                time.sleep(max(0, model.opt.timestep - (time.time() - t0)))
                if estado == "listo":
                    break
    else:
        while data.time < segundos:
            estado = accion_apartar(model, data)
            mujoco.mj_step(model, data)
            if estado == "listo":
                break

    final = pos_objeto(model, data, "borrador")
    d = np.linalg.norm(final - ZONA)
    print(f"Tiempo simulado: {data.time:.1f} s")
    print(f"Borrador: inicio {inicio.round(3)} -> final {final.round(3)}")
    dentro = bool(np.all(np.abs(final - ZONA) < 0.08))
    print(f"Distancia al centro de la zona: {d:.3f} m  ->  {'LOGRADO' if dentro else 'NO LOGRADO'}")
    return d


if __name__ == "__main__":
    correr(con_ventana="--test" not in sys.argv)
