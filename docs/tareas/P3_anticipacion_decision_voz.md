# P3 · Anticipación, decisión y voz

> Antes, leé `docs/VISION.md` (5 minutos).

## Tu misión en una frase

Construir **la parte del cerebro que decide qué hacer**, la que **aprende hábitos para
adelantarse**, y **el oído** del robot (entender lo que se le dice).

## Por qué importa

Es lo que hace que el robot parezca inteligente: no solo obedece, también se anticipa
(*“sacaste el cuaderno verde, ¿te paso el lapicero rojo?”*) y pregunta cuando no está seguro.

---

## Parte 1 · Modelo de anticipación (contrato en `asistente/cerebro/contratos.py`)

**Archivo:** `asistente/cerebro/anticipacion.py`, clase `AnticipacionSecuencias(ModeloAnticipacion)`.

**Idea:** mirar la secuencia de eventos recientes y predecir el siguiente pedido. Es como GPT
(predecir lo siguiente), pero con eventos en vez de palabras.

```python
a = AnticipacionSecuencias()
eventos = [Evento("aparecio", "cuaderno verde", "persona_a"), ...]
a.predecir(eventos)  -> ("traer:lapicero rojo", 0.86)     # o (None, 0.0) si no hay patrón
a.actualizar(eventos, pedido_real="traer:lapicero rojo")  # aprende de lo que sí pasó
```

**Cómo hacerlo, por pasos:**
1. **Vocabulario dinámico:** cada evento distinto (“aparecio:cuaderno verde”, “traer:lapicero rojo”)
   recibe un índice la primera vez que aparece. **Nada escrito a mano:** el vocabulario crece solo.
2. **Versión 1 (base para comparar):** contar qué pedido vino después de cada evento
   (una tabla de frecuencias). Sirve para tener algo funcionando el lunes.
3. **Versión 2:** un **LSTM** pequeño (semana 3) o un Transformer pequeño (semana 6) con embeddings
   de los eventos (semana 4), que recibe los últimos N eventos y predice el siguiente pedido con softmax.
4. Comparar versión 1 y 2: es un resultado para el documento.

## Parte 2 · El cerebro completo (contrato `Cerebro` en `interfaces.py`)

**Archivo:** `asistente/cerebro/cerebro.py`, clase `CerebroAprendiz(Cerebro)`.

**Idea:** junta todo para tomar **una** decisión:

1. Si hay un **pedido** explícito (`contexto.pedido`, ej. “tráeme el lapicero rojo”): buscar en la
   escena el objeto con ese nombre y hacer esa acción. `motivo="pedido"`, confianza alta.
2. Si no, preguntar a la **anticipación**: si predice algo con probabilidad alta y ese objeto está en
   la mesa, hacerlo. `motivo="anticipacion"`.
3. Si no, preguntar a las **preferencias** (de P2) por cada objeto y elegir la mejor acción.
   `motivo="preferencia"`.
4. Si la confianza es baja (por ejemplo, menor a 0.7), `preguntar_antes=True`.
5. Si no hay nada que hacer, `IGNORAR`.

```python
cerebro = CerebroAprendiz(preferencias=PreferenciasFalsas(), anticipacion=AnticipacionSecuencias())
d = cerebro.decidir(contexto)
# Decision(accion=Accion.TRAER, objeto_id="obj_2", confianza=0.86, preguntar_antes=False,
#          motivo="anticipacion", puntajes={"traer": 0.86, ...})
cerebro.aprender(contexto, d, Feedback("persona_a", aprobado=True))   # le pasa el feedback a las preferencias
```

- Mientras P2 termina, usá `PreferenciasFalsas` (ya existe en `cerebro/contratos.py`).
- `puntajes` lo usa P4 para las explicaciones: llenalo siempre.
- **Una preferencia por persona:** guardá un `ModeloPreferencias` por `persona_id` (con `clonar()`
  para crear el de una persona nueva).
- `guardar()` / `cargar()`: todo el cerebro de un robot en una carpeta.

## Parte 3 · Voz (contrato `Voz` en `interfaces.py`)

**Archivo:** `asistente/voz.py`, clase `VozWhisper(Voz)`.

```python
v = VozWhisper()
v.escuchar() -> "esto es mi lapicero rojo"   # solo después de oír el nombre del robot
v.interpretar("esto es mi lapicero rojo")  -> Orden(tipo="ensenar", nombre="lapicero rojo", ...)
v.interpretar("tráeme el lapicero rojo")   -> Orden(tipo="pedido", accion=Accion.TRAER, nombre="lapicero rojo")
v.interpretar("si ves basura, apartala")   -> Orden(tipo="regla", accion=Accion.APARTAR, nombre="basura")
v.interpretar("bien") -> Orden(tipo="feedback", aprobado=True)
```

**Cómo hacerlo, por pasos:**
1. Grabar audio del micrófono de la laptop con `sounddevice` en pedazos de unos segundos.
2. Transcribir con **Whisper** (modelo `base` o `small`, en español). Es un Transformer preentrenado.
3. Palabra clave: solo hacer caso si el texto empieza con el nombre del robot
   (`config.yaml → voz.nombre_del_robot`).
4. `interpretar`: empezá con reglas sencillas (como las de `mocks.VozTeclado`), con variantes en
   español (tráeme, pásame, dame, quitá, apartá…). Si da tiempo, un modelo de lenguaje pequeño.
5. Si no hay micrófono o falla, que devuelva `None` sin romper el programa.

## Cómo empezar (sugerencia)

1. **Lunes:** Parte 2 con la anticipación versión 1. Probar con `python main.py --modo falso --voz teclado`.
2. **Martes:** anticipación versión 2 (LSTM).
3. **Miércoles:** voz con Whisper. Commit y pull request.

## Cómo probar sin los demás

- Generá secuencias de eventos de mentira: repetí 10 veces “aparecio:cuaderno verde” →
  “traer:lapicero rojo” mezclado con eventos al azar, y medí desde qué repetición la predice bien.

## Bibliotecas

`torch`, `numpy`, `openai-whisper`, `sounddevice`.

## Listo cuando

- Después de 5 a 10 repeticiones de un patrón, la anticipación lo predice con más de 70% de confianza.
- `CerebroAprendiz` nunca elige un objeto que no está en la escena y funciona con mesas vacías.
- La voz entiende las 4 clases de orden en español.

## No hagas

- No escribas nombres de objetos en el código: los nombres llegan por la voz o la pantalla.
- No cambies los contratos sin avisar.

## Documento

Sección **Herramientas aplicadas** (`docs/documento/05_herramientas_aplicadas.md`) y los
**diagramas** de arquitectura.
