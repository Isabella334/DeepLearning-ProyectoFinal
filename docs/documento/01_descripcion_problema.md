# 1. Descripción del problema

## Contexto

Los robots de servicio y los asistentes de escritorio suelen llegar con su comportamiento
definido de fábrica: reconocen un catálogo fijo de objetos y ejecutan reglas escritas por quien los
programó. Ese enfoque falla justo donde un asistente personal debería ser útil. Cada persona tiene
objetos distintos (su lapicero, su cuaderno), los llama de forma distinta y tiene gustos y hábitos
propios: a una le gusta que le aparten la basura, otra prefiere que no le muevan nada, y una
tercera siempre pide el mismo lapicero cuando saca cierto cuaderno.

Programar a mano cada uno de esos casos no escala. El robot tendría que **aprender de la persona
que lo usa**, con muy pocos ejemplos, mientras lo usa, y sin mezclar lo que aprende de una persona
con lo que aprende de otra.

## El problema

Construir un robot asistente de escritorio que, partiendo de cero:

1. **Reconozca los objetos que cada persona le enseña** a partir de uno o pocos ejemplos
   ("esto es mi lapicero rojo"). Todo objeto empieza como *desconocido*; lo que le enseña una
   persona no lo sabe otra.
2. **Aprenda los gustos de cada persona** a partir de su reacción a lo que hace ("bien" / "mal"),
   eligiendo entre cuatro acciones: apartar un objeto, traerlo, acercarse o no hacer nada.
3. **Aprenda hábitos y se adelante** a los pedidos: si cada vez que aparece cierto objeto la
   persona pide otro, el robot debería ofrecerlo sin que se lo pidan, y preguntar antes cuando no
   está seguro.
4. **Explique por qué hizo cada cosa**, en palabras de la persona y con evidencia de su propio
   historial, para que pueda confiar en él o corregirlo.
5. **Recuerde lo aprendido** entre sesiones, guardando solo texto y números (sin fotos ni audio).

## Por qué es un problema de aprendizaje profundo

- **Percepción con pocos ejemplos:** reconocer un objeto nuevo con una sola foto no se resuelve
  entrenando un clasificador desde cero. Requiere representaciones aprendidas de antemano
  (*transfer learning*) sobre las que se pueda comparar o entrenar una cabeza pequeña.
- **Aprendizaje en línea por persona:** las preferencias llegan de a una, con etiquetas parciales
  (solo se califica la acción que se hizo) y cambian de persona a persona.
- **Secuencias:** anticipar un pedido depende del orden de los eventos recientes en la mesa, lo que
  lleva a modelos de secuencias.
- **Confianza:** una red que decide sobre los objetos de una persona necesita explicar sus
  decisiones (XAI); de lo contrario, sus errores no se pueden entender ni corregir.

## Alcance

- La mesa es una cartulina con marcadores ArUco en las esquinas; las posiciones se miden en metros
  desde su centro.
- El robot solo sabe hacer cuatro acciones (apartar, traer, acercarse, ignorar). El cerebro decide
  **cuál** y **sobre qué objeto**; el cuerpo resuelve **cómo**.
- El plan principal es un gemelo digital: la persona actúa en el mundo real (pone objetos, habla,
  aprueba o corrige) y el robot actúa en una simulación física de la misma mesa (MuJoCo). Un
  carrito físico es una extensión opcional.
- Ningún nombre de objeto está escrito en el código; la lista de objetos de la configuración solo
  se usa para construir la simulación y para medir resultados.

## Preguntas que guían el proyecto

1. ¿Con cuántos ejemplos aprende el robot a reconocer un objeto nuevo de una persona, y qué tan
   bien rechaza los que no conoce?
2. ¿Después de cuántas interacciones acierta lo que le gusta a cada persona, y cuánto se diferencian
   los robots entrenados por personas distintas?
3. ¿Desde cuándo anticipa un hábito, y con qué tasa de falsas alarmas?
4. ¿Las explicaciones reflejan de verdad lo que pesó en la decisión del modelo?
