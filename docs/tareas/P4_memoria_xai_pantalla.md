# P4 · Memoria, explicaciones y pantalla

> Antes, leé `docs/VISION.md` (5 minutos).

## Tu misión en una frase

Construir **la memoria** del robot (que no olvide nada al apagarse), **las explicaciones** de sus
decisiones (XAI) y **la pantalla** desde donde la persona lo usa.

## Por qué importa

El curso pide XAI, y es lo que hace que la persona confíe en el robot. La pantalla es lo que se
va a ver en la presentación y en el video. Y la memoria guarda los datos del experimento, de donde
salen todos los resultados.

---

## Parte 1 · Memoria (contrato `Memoria` en `interfaces.py`)

**Archivo:** `asistente/memoria.py`, clase `MemoriaSQLite(Memoria)`.

```python
m = MemoriaSQLite("robot.db")          # ":memory:" para pruebas
m.registrar_interaccion(contexto, decision, resultado, feedback) -> 17   # id
m.registrar_evento(Evento("aparecio", "cuaderno verde", "persona_a"))
m.registrar_ensenanza("persona_a", "lapicero rojo", "obj_2")
m.eventos_recientes("persona_a", segundos=300) -> [Evento, ...]
m.historial("persona_a") -> [{"persona_id": "persona_a", "accion": "traer", "objeto_id": "obj_2",
                        "objeto_nombre": "lapicero rojo", "aprobado": True, "t": 1730000000.0, ...}]
```

**Tablas:** `personas`, `ensenanzas` (persona, nombre, objeto, fecha), `eventos`,
`interacciones` (persona, acción, objeto, nombre del objeto, confianza, motivo, éxito, aprobado,
fecha) y `modelos` (persona, versión, ruta del archivo, fecha). Usá `sqlite3`, que ya viene con Python.

**Extra útil:** una función `exportar_csv()` para que P2 e integración saquen métricas del experimento.

## Parte 2 · Explicaciones (contrato `Explicador` en `interfaces.py`)

**Archivo:** `asistente/xai/explicador.py`, clase `ExplicadorXAI(Explicador)`.

```python
e = ExplicadorXAI()
e.explicar(contexto, decision, cerebro, memoria) -> Explicacion(
    texto="Te traje el lapicero rojo porque sacaste el cuaderno verde.",
    contribuciones={"hay_cuaderno": 0.42, "obj_cerca_persona": 0.11, ...},   # SHAP
    evidencia="Las últimas 6 de 7 veces que apareció el cuaderno verde, me pediste el lapicero rojo.")
```

**Cómo hacerlo, por pasos:**
1. **Texto según el motivo** (`decision.motivo`): “pedido” → “Me lo pediste”; “anticipacion” →
   “Sueles pedirlo cuando…”; “preferencia” → “Aprobaste esto X de Y veces”.
2. **Evidencia** con el historial de la memoria: cuántas veces se aprobó esa acción con ese objeto.
3. **SHAP** sobre el modelo de preferencias de P2: usá `vector()` y `nombres_caracteristicas()`
   (del contrato `ModeloPreferencias`) y `shap.KernelExplainer` con unos 50 ejemplos de fondo del
   historial. Mientras P2 termina, probalo con `PreferenciasFalsas`.
4. Que nunca falle: si no hay historial o no hay modelo, devolver una explicación simple.

## Parte 3 · Pantalla

**Archivo:** `app/pantalla.py` (Streamlit, Gradio o Tkinter; Streamlit es lo más rápido).

**Qué tiene que tener:**
- **Lo que ve el robot:** la lista de objetos con su nombre (o “desconocido”). Más adelante, la
  imagen de la “imaginación” que hace P1.
- **Enseñar por escrito:** elegir un objeto desconocido y escribir su nombre (respaldo de la voz).
- **Botones de estímulos** (para la versión todo digital): “sacar cuaderno verde”, “tirar basura”,
  “pedir un objeto”. Por ahora, que llamen a funciones vacías que integración conecta con la simulación.
- **La decisión y su explicación**, con un gráfico de barras de las contribuciones de SHAP.
- **Botones 👍 / 👎** como `EntradaFeedback` (clase `FeedbackPantalla` en `app/`).
- **Historial:** las últimas interacciones de la persona.

Para que no dependa de nadie, construila con todos los mocks de `asistente/mocks.py`
(mirá `main.py`: la pantalla es otra forma de correr el mismo ciclo).

## Parte 4 (extra) · Gestos

`asistente/vision/gestos.py`: pulgar arriba / abajo con la cámara de la laptop usando **MediaPipe
Hands** como otra `EntradaFeedback`. Solo si te sobra tiempo.

## Cómo empezar (sugerencia)

1. **Lunes:** memoria SQLite completa.
2. **Martes:** explicador (texto + evidencia, después SHAP).
3. **Miércoles:** pantalla con mocks. Commit y pull request.

## Bibliotecas

`sqlite3` (viene con Python), `shap`, `streamlit` (o gradio), `matplotlib`.

## Listo cuando

- La memoria guarda y recupera todo, y el archivo `.db` sobrevive a cerrar el programa.
- Cada decisión trae una explicación con texto, evidencia y contribuciones.
- La pantalla corre con los mocks y permite enseñar, aprobar y corregir.

## No hagas

- No escribas nombres de objetos en el código.
- No guardes fotos ni audio en la base de datos: solo texto, números y embeddings (privacidad).
- No cambies los contratos sin avisar.

## Documento y presentación

Secciones **Descripción del problema**, **Análisis** y **Bibliografía** (`docs/documento/`),
y las **slides** (máximo 10).
