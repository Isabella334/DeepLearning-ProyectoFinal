# Integración · Mundo físico, visión e integración

> Guía del rol de integración. Resume lo que te toca y en qué orden.

## Tu misión

Que el robot **vea el mundo real** (cartulina, cámara y CNN), **unir** el trabajo de todos,
y correr el **experimento**. El carrito físico, solo cuando el gemelo digital funcione completo.

## 1. Cartulina y marcadores (lunes)
- Cartulina de color claro y **mate** (no blanca: el borrador y el papel son blancos).
- `herramientas/marcadores.py`: generar con OpenCV los ArUco `DICT_4X4_50` (0–3 esquinas, 10 carrito),
  de 6 cm con borde blanco, en una hoja para imprimir en papel mate.
- Medir la cartulina y poner `ancho`, `alto` y las `zonas` en `config.yaml`. **Avisar a P1.**

## 2. Dataset (lunes y martes)
- `herramientas/capturar_dataset.py`: toma fotos desde la webcam (o el celular como webcam) con
  un nombre que incluya la fecha, la sesión y la condición de luz.
- 200–300 fotos desde arriba, varios objetos por foto; variar posición, ángulo, altura, luz,
  sombras, manos en la imagen y fotos “vacías” o con objetos que no son del experimento.
- Etiquetar con cajas (Roboflow, CVAT o LabelImg):
  - **Para el detector:** todas las cajas con la etiqueta `objeto`.
  - **Para el ajuste contrastivo (opcional):** guardar también qué objeto es cada caja. Esas
    etiquetas solo se usan para entrenar; **el robot no las conoce**.
- Reservar **una sesión completa** (otra luz u otro ángulo) solo para prueba. Y algunos objetos
  que **nunca** aparezcan en el entrenamiento (por ejemplo, crayones) para el experimento de
  “aprender objetos nuevos”.

## 3. Detector CNN (martes y miércoles)
- `asistente/vision/entrenar_detector.py`: ajuste fino de YOLO (`ultralytics`, modelo `yolov8n`)
  con una sola clase. Métricas: mAP, precisión, recall y ejemplos de errores.
- `asistente/vision/detector.py`: `DetectorYOLO(Detector)`, que devuelve `Deteccion(confianza, caja)`.
- **Opcional:** ajuste contrastivo del extractor de embeddings (pérdida contrastiva o triplet, S8)
  con los recortes etiquetados, para que P2 lo use en vez de la red preentrenada tal cual.

## 4. Percepción con cámara (miércoles)
- `asistente/vision/percepcion_camara.py`: `PercepcionCamara(Percepcion)`.
  1. Detectar los 4 ArUco de las esquinas → homografía (píxeles → metros de la cartulina).
  2. `DetectorYOLO` → cajas → centro de cada caja convertido a metros.
  3. Recortar cada caja → `reconocedor.embedding(recorte)` → `reconocedor.reconocer(persona, emb)`.
  4. **Ids estables:** emparejar cada objeto con el más cercano del cuadro anterior.
  5. `ensenar(persona, objeto_id, nombre)`: pasar el embedding de ese objeto al reconocedor.
- **Decidir qué objeto se está enseñando** cuando dicen “esto es…”: el más reciente que apareció,
  o el que está en una “zona de enseñar” marcada en la cartulina (lo más confiable para la demo).
- `MundoSim.sincronizar(escena)`: con las funciones `colocar` y `quitar` de P1.

## 5. Integración (jueves y viernes)
- En `main.py → construir()`, cambiar cada mock por la pieza real **una a la vez**.
- Agregar cada clase real a `tests/test_contratos.py` y correr `python -m pytest tests -q`.
- Revisar los pull requests de todos.
- **Viernes 8 p. m.:** congelar el código.

## 6. Carrito físico (solo si la versión B ya funciona)
- Chasis y pala impresos, ESP32 con el firmware de `hardware/PROTOCOLO.md`,
  `hardware/aruco.py → pose_desde_imagen()` y `CuerpoReal`.

## 7. Experimento, video y documento (sábado y domingo)
- Dos personas entrenan cada una su copia del robot: le enseñan sus objetos, sus gustos y un hábito.
- Exportar la memoria a CSV y sacar con P2 las curvas: aciertos de reconocimiento, aciertos de
  preferencias y desde cuándo se anticipa.
- Video de demostración con las grabaciones de P1.
- Armar el documento final y juntar los historiales de prompts.

## Documento
**Propuesta de solución** y **Conclusión**.
