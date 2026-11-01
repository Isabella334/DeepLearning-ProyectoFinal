# 2. Análisis

## 2.1 Requisitos y lo que implican

| Requisito | Consecuencia de diseño |
|---|---|
| Reconocer objetos con uno o pocos ejemplos | No se entrena un clasificador desde cero: se usan *embeddings* de una red preentrenada (ResNet-18 sobre ImageNet, 512 valores) y se compara o se entrena una cabeza pequeña sobre ellos. |
| Lo que aprende una persona no lo sabe otra | Un modelo de reconocimiento, uno de preferencias y uno de anticipación **por persona**, clonados de una plantilla común. |
| Aprender mientras se usa | Aprendizaje en línea: cada "bien" o "mal" es un ejemplo nuevo que se guarda en un *buffer* y dispara unos pocos pasos de entrenamiento. |
| Solo se califica la acción que se hizo | La pérdida (entropía cruzada binaria) se calcula únicamente sobre la salida de esa acción; las demás no reciben gradiente. |
| No actuar a ciegas | Si la confianza de la decisión queda bajo un umbral (0.7), el robot pregunta antes de actuar. |
| Confianza de la persona | Cada decisión llega con una explicación: frase, evidencia del historial y contribución de cada variable. |
| Recordar y respetar la privacidad | Memoria persistente en SQLite con solo texto, números y *embeddings*; nada de fotos ni audio. |
| Correr en una laptop, en tiempo real | Modelos pequeños (MLP y LSTM de pocas capas) que entrenan en CPU en milisegundos. |

## 2.2 Datos disponibles

No existe un conjunto de datos de gustos y hábitos de personas sobre una mesa, así que el proyecto
genera los suyos:

- **Usuarios simulados** (`experimentos/usuarios.yaml`): personas con reglas de gusto distintas
  (por ejemplo, una que aparta la basura y pide los lapiceros, y otra cuyos gustos dependen de qué
  otro objeto hay en la mesa). Permiten repetir el experimento con varias semillas y medir curvas
  de aprendizaje.
- **Simulación física** (MuJoCo): la misma mesa y los mismos objetos que la cartulina real. Cada
  tipo de objeto tiene un "aspecto" (un *embedding* corto) que es lo único que el robot ve de él.
- **Dataset propio de la cartulina** para el detector de objetos (YOLO) en la versión con cámara.
- **El uso real**: la memoria registra cada interacción y se exporta a CSV para el experimento con
  dos personas.

## 2.3 Alternativas evaluadas

| Subproblema | Alternativas | Elegida | Por qué |
|---|---|---|---|
| Reconocer objetos | Clasificador entrenado desde cero · prototipo por coseno (v1) · cabeza MLP por persona con clase "desconocido" (v2) | v1 y v2 se comparan; el robot usa v2 | Desde cero no aprende con 1 ejemplo. El prototipo acierta con un solo ejemplo pero rechaza peor lo desconocido; la cabeza rechaza mejor pero necesita más de un ejemplo cuando hay mucho ruido (ver Resultados). |
| Aprender gustos | Tabla de conteos · MLP con buffer · aprendizaje por refuerzo completo | MLP de 4 salidas sigmoide | La tabla no generaliza a objetos o situaciones nuevas; el refuerzo completo necesita muchas más interacciones de las que una persona da. |
| Anticipar hábitos | Frecuencias de "qué pedido siguió a cada evento" (v1) · LSTM sobre la secuencia de eventos (v2) | v2 | La LSTM tiene en cuenta el orden y el contexto de varios eventos; los eventos casi simultáneos se agrupan en "momentos" porque dentro de un ciclo su orden es arbitrario. |
| Decidir | Una sola red para todo · prioridades explícitas | Pedido > regla > anticipación > preferencia | Las órdenes explícitas siempre ganan; cada fuente de decisión se puede explicar por separado. |
| Explicar | Mapas de atención o gradientes · LIME · SHAP | SHAP (KernelExplainer) + frase + evidencia | SHAP no depende de la arquitectura y tiene base teórica (valores de Shapley); LIME ajusta un modelo local que cambia entre corridas. La frase y la evidencia cubren las decisiones que no salen de la red (pedidos, reglas, hábitos). |
| Recordar | Archivos JSON sueltos · base SQLite | SQLite | Consultas por persona y por tiempo, una sola fuente de verdad y exportación directa a CSV para las métricas. |
| Usarlo | Terminal · pantalla web | Pantalla en Streamlit sobre la simulación | Permite enseñar, aprobar y ver la explicación junto a lo que el robot planea, sin hardware. |

## 2.4 Cómo se explica una decisión

1. **Frase según el motivo** de la decisión (pedido, regla, anticipación o preferencia), con los
   nombres que enseñó la persona.
2. **Evidencia** contada en la memoria: cuántas veces aprobó esa acción con ese objeto o, para un
   hábito, cuántas de las veces que apareció la señal terminaron en ese mismo pedido.
3. **Contribuciones SHAP** sobre la red de preferencias para la acción elegida, usando como fondo
   hasta 50 situaciones del historial de esa persona. Se explican los *log-odds* y no la
   probabilidad: la red llega a probabilidades cercanas a 0 o 1, donde la sigmoide es casi plana y
   todas las contribuciones en probabilidad se ven como cero aunque una variable pese mucho.

## 2.5 Cómo se va a medir

| Pregunta | Métrica | De dónde sale |
|---|---|---|
| ¿Reconoce con pocos ejemplos? | Acierto y tasa de rechazo de desconocidos según el número de ejemplos | Experimento del reconocedor |
| ¿Aprende los gustos? | Aciertos de preferencia por número de interacciones y aprobaciones en línea | Usuarios simulados y CSV de la memoria |
| ¿Es personal? | Aciertos de un robot evaluado con los gustos de la otra persona | Usuarios simulados |
| ¿Anticipa? | Confianza en la predicción cuando aparece la señal y falsas alarmas sin ella | Experimento de anticipación |
| ¿Explica bien? | Coherencia entre la variable con mayor contribución SHAP y la regla real del usuario simulado | Explicador sobre usuarios simulados |

## 2.6 Riesgos y limitaciones

- **Simulación y realidad.** En la simulación, el aspecto de cada objeto es un vector corto con
  ruido, mucho más fácil de separar que un recorte de cámara. Los resultados de reconocimiento en
  simulación son una cota optimista.
- **Usuarios simulados.** Siguen reglas fijas; una persona real es menos consistente.
- **Alcance de SHAP.** Solo explica la red de preferencias. Las decisiones por pedido, regla o
  hábito se explican con la frase y la evidencia del historial.
- **Explicador lento si crece el modelo.** KernelExplainer escala con el número de variables y de
  ejemplos de fondo; con unas 50 variables y 50 ejemplos responde en menos de un segundo.
- **Nombres repetidos.** Si una persona le da el mismo nombre a dos objetos distintos, el robot
  aprende dos aspectos para ese nombre y puede reconocer ambos con él.
- **Física de la mesa.** La zona de la persona está cerca del borde de la cartulina y un empujón
  puede tirar el objeto; los objetos caídos se retiran de la escena.
