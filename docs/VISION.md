# El proyecto en 5 minutos

> Leé esto primero. Después leé **solo** la guía de tu tarea en `docs/tareas/`.

## Qué estamos construyendo

Un **robot asistente de escritorio que aprende de cada persona**. No viene programado:
se le enseña.

- **No sabe qué hay en la mesa hasta que se lo enseñás.** Ponés tu lapicero y le decís
  *“esto es mi lapicero rojo”*. Desde ahí lo reconoce. Lo que le enseña una persona no lo sabe otra.
- **Aprende tus gustos.** Hace algo (apartar la basura, traerte un objeto) y vos le decís
  “bien” o “mal”. Con el tiempo hace lo que te gusta a vos.
- **Aprende tus hábitos y se adelanta.** Si cada vez que sacás tu cuaderno verde le pedís el
  lapicero rojo, un día te lo trae sin que se lo pidas (o te pregunta primero si no está seguro).
- **Explica por qué hizo cada cosa.** *“Te traje el lapicero rojo porque las últimas 6 de 7 veces
  que sacaste el cuaderno verde me lo pediste.”*

## Cómo se ve la demo (versión B, el plan principal)

```
  CARTULINA REAL (mesa)                     PANTALLA
  ┌──────────────────────┐   webcam    ┌─────────────────────────────┐
  │  ✏️ lapicero  📓 cuaderno│ ─────────▶ │ Gemelo digital (MuJoCo):     │
  │       🧽 borrador       │            │  la misma mesa, con un       │
  │ [ArUco en las esquinas]│            │  carrito virtual que actúa   │
  └──────────────────────┘             │  = la “imaginación” del robot│
                                        │ Explicación + botones        │
                                        └─────────────────────────────┘
```

La persona actúa en el mundo real (pone objetos, habla, aprueba o corrige) y el robot
actúa en la simulación. Si el carrito físico funciona a tiempo, se conecta al final.

## Las piezas (cada una es una clase de `asistente/interfaces.py`)

| Pieza | Qué hace | Quién |
|---|---|---|
| Percepción | Ve la mesa y arma la `Escena` (qué objetos hay y dónde) | P1 (simulada) · Integración (cámara) |
| Detector | CNN que encuentra objetos en la imagen, sin nombrarlos | Integración |
| Reconocedor | Aprende el **nombre** de cada objeto, por persona, con pocos ejemplos | P2 |
| Cerebro | Decide qué hacer y aprende de “bien/mal” y de los hábitos | P2 (preferencias) · P3 (anticipación y decisión) |
| Voz | Escucha el nombre del robot y entiende la orden | P3 |
| Cuerpo | Mueve el carrito para cumplir la acción | P1 (simulado) · Integración (real) |
| Memoria | Guarda todo en SQLite | P4 |
| Explicador | Explica cada decisión (XAI) | P4 |
| Pantalla | Botones, explicación y visualización | P4 (pantalla) · P1 (visualizaciones) |

## Un ciclo del robot (lo que hace `main.py`)

1. **Percepción** observa la mesa → `Escena` (los objetos que no conoce salen como `"desconocido"`).
2. **Voz** escucha: puede ser “esto es mi lapicero rojo” (enseñar), “tráeme el lapicero rojo”
   (pedido) o “bien”/“no” (feedback).
3. **Cerebro** decide → `Decision` (acción, objeto, confianza, si pregunta antes).
4. **Explicador** explica la decisión.
5. **Cuerpo** ejecuta la acción.
6. La persona aprueba o corrige → el **Cerebro** aprende y la **Memoria** lo guarda todo.

## Las 4 acciones que existen

`APARTAR` (empujar a la zona de apartar) · `TRAER` (llevar a la zona de la persona) ·
`ACERCARSE` · `IGNORAR`. El cerebro solo elige **cuál** y **sobre qué objeto**; el cuerpo sabe **cómo**.

## Reglas de oro

1. **Ningún nombre de objeto escrito en el código.** Todo empieza como `"desconocido"`.
   La lista de `config.yaml` es solo para crear la simulación y para medir resultados.
2. **Los datos se pasan con los tipos de `asistente/tipos.py`.** Posiciones en **metros**,
   con (0, 0) en el centro de la cartulina.
3. **No se cambian `tipos.py`, `interfaces.py` ni `config.yaml` sin avisar al grupo.**
4. **Commits pequeños y seguidos, con tu propia cuenta,** y tu historial de prompts en
   `docs/prompts/<tu-nombre>.md` (lo piden en el reporte).

## Para probar todo de punta a punta

```bash
pip install -r requirements.txt
python main.py --modo falso --voz teclado --feedback teclado --pasos 3
```

Escribí *“esto es mi lapicero rojo”* cuando te pida la voz, y fijate cómo en el siguiente
ciclo el robot ya lo reconoce. Eso, en chiquito, es todo el proyecto.
