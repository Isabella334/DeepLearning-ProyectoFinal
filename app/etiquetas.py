from __future__ import annotations

# Nombres de las caracteristicas de PreferenciasRed (P2) en palabras de la persona
ETIQUETAS = {
    "sin_objeto": "no es sobre un objeto",
    "obj_conocido": "ya sabe su nombre",
    "obj_confianza": "qué tan seguro está del nombre",
    "dist_persona": "distancia a tu zona",
    "dist_apartar": "distancia a la zona de apartar",
    "dist_carrito": "distancia al carrito",
    "en_zona_persona": "ya está frente a vos",
    "en_zona_apartar": "ya está apartado",
    "n_objetos": "cuántos objetos hay",
    "hay_persona": "hay alguien usando el robot",
    "aparecio_hace_poco": "apareció hace poco",
    "hay_pedido": "le pediste algo",
    "pedido_es_este_objeto": "le pediste este objeto",
    "hora_sin": "hora del día",
    "hora_cos": "hora del día",
}


def etiqueta(nombre: str) -> str:
    if nombre.startswith("aspecto_"):
        return "cómo se ve"
    if nombre.startswith("es:"):
        return f"es '{nombre[3:]}'"
    if nombre.startswith("hay:"):
        return f"hay '{nombre[4:]}' en la mesa"
    return ETIQUETAS.get(nombre, nombre)


def legibles(contribuciones: dict[str, float]) -> dict[str, float]:
    suma: dict[str, float] = {}
    for nombre, valor in contribuciones.items():
        clave = etiqueta(nombre)
        suma[clave] = suma.get(clave, 0.0) + valor
    return suma
