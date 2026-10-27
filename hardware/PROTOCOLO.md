# Protocolo laptop ↔ carrito (contrato de integración)

La laptop hace todo el trabajo pesado (visión y decisiones). El carrito solo recibe
velocidades de rueda. Así el firmware del ESP32 es pequeño y no depende de nadie.

## Mensaje laptop → ESP32

- **Transporte:** UDP por WiFi, puerto `4210`.
- **Frecuencia:** unas 20 veces por segundo.
- **Formato:** texto JSON en una línea.

```json
{"izq": 0.6, "der": 0.4}
```

`izq` y `der` van de `-1.0` (reversa a fondo) a `1.0` (adelante a fondo).

## Reglas de seguridad del firmware

1. Si no llega ningún mensaje en **0.5 s**, el carrito se detiene (por si se cae el WiFi).
2. Los valores fuera de rango se recortan a `[-1, 1]`.

## Mensaje ESP32 → laptop (opcional)

Cada segundo, el estado del botón de las “orejas” y el voltaje de la batería:

```json
{"orejas": true, "bateria_v": 7.6}
```

## Posición del carrito (ArUco)

Integración entrega la función `pose_desde_imagen(frame) -> PoseCarrito | None` en
`hardware/aruco.py`, en metros y con los mismos ejes que la simulación
(ver `asistente/tipos.py`). Se usa dentro de la percepción con cámara.
Mientras no exista, se usa una pose fija.
