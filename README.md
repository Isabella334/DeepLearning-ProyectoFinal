# Asistente robótico que aprende de cada persona

Un robot de escritorio que **no viene programado: se le enseña**. Aprende los nombres de tus
objetos, tus gustos y tus hábitos, se adelanta a lo que vas a pedir y explica por qué hace cada cosa.

👉 **Empezá por [`docs/VISION.md`](docs/VISION.md)** (5 minutos) y después por la guía de tu tarea.

## Versiones

| Versión | Percepción | Cuerpo | Estado |
|---|---|---|---|
| A · Todo digital | MuJoCo (la pantalla pone los objetos) | Carrito virtual | Respaldo |
| **B · Gemelo digital** | Webcam + CNN sobre la cartulina | Carrito virtual | **Plan principal** |
| C · Físico | Webcam + CNN sobre la cartulina | Carrito real (ESP32) | Si el hardware llega |

## Quién hace qué

| Persona | Tarea | Guía |
|---|---|---|
| **Integración** | Mundo físico, visión (dataset y CNN), integración, experimento y video | [`INTEGRACION_mundo_fisico_e_integracion.md`](docs/tareas/INTEGRACION_mundo_fisico_e_integracion.md) |
| **P1** | Simulación (mesa virtual, objetos, acciones), imaginación y visualizaciones | [`P1_simulacion.md`](docs/tareas/P1_simulacion.md) |
| **P2** | Aprendizaje por persona: reconocer objetos y aprender gustos | [`P2_aprendizaje_por_persona.md`](docs/tareas/P2_aprendizaje_por_persona.md) |
| **P3** | Anticipación de hábitos, decisión y voz | [`P3_anticipacion_decision_voz.md`](docs/tareas/P3_anticipacion_decision_voz.md) |
| **P4** | Memoria, explicaciones (XAI) y pantalla | [`P4_memoria_xai_pantalla.md`](docs/tareas/P4_memoria_xai_pantalla.md) |

**Todos:** su historial de prompts en `docs/prompts/<nombre>.md` (hay una plantilla) y su
sección del documento en `docs/documento/`.

## Correr el proyecto

```bash
pip install -r requirements.txt
python main.py --modo falso --voz teclado --feedback teclado --pasos 3   # probalo enseñándole algo
python main.py --modo sim --pasos 3                                      # en MuJoCo
python -m pytest tests -q                                                # revisa los contratos
```

## Reglas del repositorio

1. `asistente/tipos.py`, `asistente/interfaces.py` y `config.yaml` **no se cambian sin avisar al grupo**.
2. **Ningún nombre de objeto escrito en el código.** Todo objeto empieza como `"desconocido"`.
3. Cada quien con **su propia cuenta de GitHub** configurada (`git config user.name` y `user.email`).
4. **Una rama por persona** y pull requests hacia `main`. Quien lleva la integración los revisa y los une.
5. **Commits pequeños y frecuentes**, al menos uno por día de trabajo.
6. Si dos trabajan en la misma compu, el commit lleva `Co-authored-by:` con el otro.

## Calendario

| Día | Qué pasa |
|---|---|
| Dom 25 | Reunión de 30 min: leer `VISION.md` juntos y crear las ramas |
| Lun 26 – Mié 28 | Cada quien en su tarea · Integración: cartulina, dataset, CNN y cámara |
| Mié 28 (noche) | Pull requests de todos |
| Jue 29 | Integración de la versión B |
| Vie 30 | Correcciones · **congelar el código a las 8 p. m.** |
| Sáb 31 – Dom 1 | Experimento con dos personas · métricas, video y slides |
| Lun 2 | Entrega |

## Estructura

```
config.yaml             medidas de la cartulina, zonas, ArUco, objetos del experimento
asistente/
  tipos.py              contrato de datos
  interfaces.py         contrato de módulos
  mocks.py              versiones simples de todo (para probar e integrar)
  sim_mujoco.py         física de la simulación (P1)
  cuerpo_sim.py         Cuerpo y Percepcion simulados (P1)
  vision/               detector y cámara (integración), reconocedor (P2)
  cerebro/              contratos internos, preferencias (P2), anticipación y cerebro (P3)
  voz.py                (P3)
  memoria.py  xai/      (P4)
app/                    pantalla (P4) y visualizaciones (P1)
hardware/               protocolo y carrito real (integración)
herramientas/           marcadores y captura del dataset (integración)
experimentos/           usuarios simulados y curvas (P2)
docs/
  VISION.md             el proyecto en 5 minutos
  tareas/               una guía por persona
  documento/            una sección del documento por persona
  prompts/              historial de prompts de cada persona
tests/
main.py                 integración (integración)
```
