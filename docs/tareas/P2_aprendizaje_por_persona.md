# P2 · Aprendizaje por persona: reconocer objetos y aprender gustos

> Antes, leé `docs/VISION.md` (5 minutos).

## Tu misión en una frase

Construir **lo que el robot aprende de cada persona**: (1) los **nombres** de sus objetos con
pocos ejemplos, y (2) sus **gustos** (qué acciones aprueba). Son dos redes neuronales pequeñas,
**una por persona**, que empiezan sin saber nada.

## Por qué importa

Es el corazón de la idea: *el robot no viene programado, se le enseña*. Tus dos modelos son los
que hacen que dos personas terminen con dos robots distintos, y tus curvas de aprendizaje son
los **resultados** del documento.

---

## Parte 1 · Reconocedor de objetos (contrato `Reconocedor` en `interfaces.py`)

**Archivo:** `asistente/vision/reconocedor.py`, clase `ReconocedorPocosEjemplos(Reconocedor)`.

**Idea:** cada objeto se convierte en un **embedding** (un vector que describe cómo se ve). Cuando
la persona dice “esto es mi lapicero rojo”, se guarda ese vector con ese nombre. Después, un
objeto nuevo se compara con lo guardado: si se parece lo suficiente, se reconoce; si no, es
`"desconocido"`.

```python
r = ReconocedorPocosEjemplos()
e = r.embedding(recorte)                  # recorte: imagen numpy del objeto -> list[float]
r.ensenar("persona_a", "lapicero rojo", e)      # guarda un ejemplo para persona_a
r.reconocer("persona_a", e2)  -> ("lapicero rojo", 0.93)
r.reconocer("juan", e2) -> ("desconocido", 0.10)   # Juan no se lo ha enseñado
r.nombres("persona_a")        -> ["lapicero rojo"]
```

**Cómo hacerlo, por pasos:**
1. `embedding()`: una red **preentrenada** de `torchvision` (por ejemplo `resnet18` o
   `mobilenet_v3_small`) sin su última capa. El recorte se redimensiona a 224×224 y sale un vector
   de 512 números (o los que dé la red). Esto es transfer learning (semana 9).
2. `ensenar()` y `reconocer()`, versión 1: guardar los vectores por persona y por nombre, y comparar
   con el **coseno** contra el promedio de cada nombre (prototipo). Debajo de
   `config.yaml → reconocimiento.umbral_similitud`, devolver `"desconocido"`.
3. Versión 2 (la de deep learning “de verdad”): **una cabeza clasificadora pequeña por persona**
   (una capa densa o dos) entrenada con los ejemplos de esa persona, más una clase “desconocido”.
   Comparala contra la versión 1: esa comparación es un resultado para el documento.
4. `guardar()` / `cargar()`: los ejemplos y las cabezas de cada persona en archivos.

**Ojo:** en la simulación no hay imágenes. La percepción simulada te pasa embeddings ya hechos
(vectores cortos). Tu `ensenar` y `reconocer` tienen que funcionar con **cualquier** largo de vector.

## Parte 2 · Modelo de preferencias (contrato en `asistente/cerebro/contratos.py`)

**Archivo:** `asistente/cerebro/preferencias.py`, clase `PreferenciasRed(ModeloPreferencias)`.

**Idea:** dada la situación y un objeto, la red da una probabilidad de que a **esta persona** le
guste cada acción. Cada “bien” o “mal” es un ejemplo de entrenamiento.

```python
p = PreferenciasRed()
p.puntajes(contexto, "obj_2") -> {Accion.APARTAR: 0.81, Accion.TRAER: 0.10, Accion.IGNORAR: 0.07, Accion.ACERCARSE: 0.02}
p.actualizar(contexto, "obj_2", Accion.APARTAR, aprobado=True)
p.vector(contexto, "obj_2") -> [1.0, 0.0, 0.3, ...]          # lo que ve la red
p.nombres_caracteristicas() -> ["obj_es_basura", "hay_cuaderno", ...]   # mismo largo que vector
nuevo = p.clonar()                                            # copia independiente
```

**Cómo hacerlo, por pasos:**
1. **Características** (`vector`): convertir el contexto en números. Ideas: el embedding del objeto
   (o un resumen de él), si el objeto es conocido, distancia del objeto a la persona, cuántos objetos
   hay, si apareció algo en los últimos segundos, la hora. **Sin nombres escritos a mano**: usá el
   embedding o el nombre aprendido convertido a número (por ejemplo, su posición en `nombres()`).
2. **Red:** una MLP pequeña (PyTorch) que recibe el vector y da 4 salidas con sigmoide
   (una por acción). Pérdida: entropía cruzada binaria (semana 8).
3. **`actualizar`:** un paso (o unos pocos) de optimización con ese ejemplo. Como hay pocos datos,
   conviene guardar los últimos ejemplos y reentrenar con todos (un “buffer”).
4. **`guardar`, `cargar`, `clonar`:** con `torch.save` y `copy.deepcopy`. Guardá también una copia
   de los pesos después de cada actualización (P1 los usa para la visualización; acuerden el formato).

## Parte 3 · Usuarios simulados y curvas (tus resultados)

**Archivo:** `experimentos/usuarios_simulados.py`.
- Un “usuario” es una regla oculta, por ejemplo: *“le gusta que aparten lo que tenga aspecto de
  basura y que le traigan lo que parezca lapicero”*. El usuario simulado aprueba o corrige según esa regla.
- Correr 100 interacciones con `mocks.PercepcionFalsa` y graficar el **porcentaje de aciertos**
  contra el número de interacciones. Con dos usuarios con reglas distintas, mostrar que dos copias
  del mismo cerebro terminan distintas.
- Para el reconocedor: aciertos con 1, 3 y 5 ejemplos por objeto.

## Cómo empezar (sugerencia)

1. **Lunes:** Parte 1, versión 1 (embeddings preentrenados + coseno).
2. **Martes:** Parte 2 (preferencias) con usuarios simulados.
3. **Miércoles:** curvas de aprendizaje; Parte 1, versión 2 si da tiempo. Commit y pull request.

## Bibliotecas

`torch`, `torchvision`, `numpy`, `matplotlib`.

## Listo cuando

- Reconocedor: con 3 ejemplos por objeto, reconoce bien los objetos de la percepción falsa, y lo
  que aprende una persona no lo sabe otra.
- Preferencias: con un usuario simulado, acierta su gusto en más del 80% de los casos después de
  unas 30 interacciones.
- Las gráficas están en `docs/figuras/`.

## No hagas

- No escribas nombres de objetos en el código.
- No cambies los contratos sin avisar.

## Documento

Sección **Resultados (métricas)** (`docs/documento/06_resultados.md`).
