# P1 · Simulación, imaginación y visualizaciones

> Antes, leé `docs/VISION.md` (5 minutos).

## Tu misión en una frase

Construir **la mesa virtual** donde actúa el robot: una copia en MuJoCo de la cartulina real,
con sus objetos y un carrito que sabe hacer las 4 acciones, más las **vistas** que muestran
qué “imagina” el robot y cómo cambian sus cerebros.

## Por qué importa

En la versión B (el plan principal), **toda la acción del robot pasa aquí**: la cámara ve la
mesa real, integración copia las posiciones a tu mundo, y tu carrito virtual es el que se mueve.
Si el hardware falla, tu simulación **es** el robot.

## Lo que ya existe (no empezás de cero)

- `asistente/sim_mujoco.py`: mesa, carrito con pala en U y **un** borrador. La acción
  **apartar** ya funciona: el carrito rodea el objeto, se acomoda detrás y lo empuja a la zona
  (probada con 8 posiciones iniciales). Corré `python -m asistente.sim_mujoco` para verla.
- `asistente/cuerpo_sim.py`: `CuerpoSim` y `PercepcionSim`, que conectan esa física con el
  resto del robot. Hoy solo manejan el borrador.

## Qué tenés que construir

### 1. La mesa con las medidas reales
- Leer `config.yaml` → `cartulina.ancho`, `cartulina.alto` y `zonas` (apartar y persona).
  Integración pone los números reales cuando mida la cartulina.
- La mesa de MuJoCo mide exactamente eso, con (0, 0) en el centro.
- Dibujá las dos zonas en el piso (como el cuadro verde que ya existe).

### 2. Varios objetos
- Crear un objeto de MuJoCo por cada elemento de `config.yaml → objetos_del_experimento`
  (formas simples: cilindros finos para lapiceros, cajas para borrador y cuaderno, una esfera
  para la bolita de papel). Que sean livianos y que no rueden.
- Una función para **poner y quitar objetos** durante la simulación (la pantalla tendrá botones
  como “sacar el cuaderno verde”).
- `MundoSim.colocar(id, tipo, x, y)` y `MundoSim.quitar(id)`.
- **Importante:** el mundo sabe qué es cada objeto (su “tipo oculto”), **el robot no**.
  `PercepcionSim.observar()` devuelve cada objeto con su posición y un **embedding simulado**
  (un vector fijo por tipo + un poquito de ruido, como ya está en `ASPECTO`), y el nombre lo pone
  el `Reconocedor`. Así el robot tiene que aprender los nombres, igual que con la cámara.

### 3. Las acciones que faltan (en `CuerpoSim.ejecutar`)
- **TRAER:** igual que apartar, pero empujando el objeto hacia la zona de la persona.
- **ACERCARSE:** ir a la zona de la persona sin empujar nada.
- **IGNORAR:** ya está.
- Reutilizá `ir_hacia()` y la lógica de fases de `accion_apartar()` (aproximar → girar → empujar).
- Que nunca se quede trabado: si pasa el tiempo límite, devuelve `Resultado(exito=False, ...)`.

### 4. Vista de la “imaginación”
- Una ventana o imagen vista desde arriba (como `vista_cenital.py` del GIF) que muestre la mesa,
  los objetos con **el nombre que el robot cree** (o “¿?” si es desconocido), y **el plan**:
  el objetivo marcado y una flecha del camino.
- Función sugerida: `dibujar_imaginacion(escena, decision) -> imagen` (numpy o matplotlib).

### 5. Visualización de los cerebros (después de lo anterior)
- Con los pesos que guarda P2 (`ModeloPreferencias.guardar`), dibujar la red: neuronas y
  conexiones con color y grosor según el peso.
- Un deslizador o una serie de imágenes para ver cómo cambia después de cada interacción.
- Dos cerebros lado a lado (las dos copias del experimento).
- Coordiná con P2 el formato de los archivos que guarda.

## Qué recibís y qué entregás

```python
# Recibís (del cerebro, vía main.py):
accion: Accion          # Accion.TRAER
objeto: Objeto | None   # Objeto(id="obj_2", x=0.12, y=-0.05, nombre="lapicero rojo", ...)
escena: Escena

# Entregás:
CuerpoSim.ejecutar(...)  -> Resultado(exito=True, duracion_s=8.3, mensaje="objeto traido")
PercepcionSim.observar() -> Escena(objetos=[Objeto(...), ...], carrito=PoseCarrito(...), persona_id="persona_A")
```

## Cómo empezar (sugerencia)

1. **Lunes:** leer `sim_mujoco.py` y correrlo. Pasar las medidas a `config.yaml`. Agregar varios objetos.
2. **Martes:** TRAER y ACERCARSE. Probar con `python main.py --modo sim --pasos 10`.
3. **Miércoles:** vista de la imaginación. Commit y pull request.
4. **Jueves:** visualización de cerebros con lo que tenga P2.

## Bibliotecas

`mujoco`, `numpy`, `matplotlib`, `pyyaml`.

## Listo cuando

- `python main.py --modo sim --pasos 10` corre sin errores con varios objetos, y las 4 acciones
  terminan (con éxito o con un `Resultado` de fallo, pero nunca colgado).
- La vista de la imaginación muestra objetos, nombres y plan.

## No hagas

- No escribas nombres de objetos en el código fuera de lo que leés de `config.yaml`.
- No cambies `tipos.py` ni `interfaces.py`; si necesitás algo, avisá al grupo.

## Documento

Sección **Descripción de la solución** (`docs/documento/04_descripcion_solucion.md`) y las
grabaciones de pantalla para el video final.
