"""
Entrena el DETECTOR de objetos (YOLO, una sola clase: "objeto") con el dataset de Roboflow.
Tarea de Isabella. Pensado para Google Colab con GPU, pero tambien corre en la laptop (lento).

Uso (Colab):
    !pip install -q ultralytics roboflow
    # 1. Descargar el dataset con el codigo que da Roboflow (Export -> YOLOv8 -> Show download code)
    # 2. Entrenar:
    !python asistente/vision/entrenar_detector.py --datos {dataset.location}/data.yaml

Resultado: carpeta resultados_detector/ (y resultados_detector.zip) con
    best.pt                 los pesos entrenados (van a modelos/detector/best.pt)
    metricas.json           mAP, precision y recall en validacion y en prueba
    figuras/                curvas de entrenamiento, matriz de confusion, ejemplos (para el documento)
"""
import argparse
import json
import os
import shutil

import yaml
from ultralytics import YOLO


def arreglar_rutas(ruta_yaml):
    """Roboflow escribe rutas relativas ('../train/images') que a veces no se encuentran.
    Las cambia por rutas absolutas que si existan y guarda un data_fijo.yaml al lado."""
    base = os.path.dirname(os.path.abspath(ruta_yaml))
    with open(ruta_yaml) as f:
        datos = yaml.safe_load(f)
    datos.pop("path", None)
    for clave in ("train", "val", "test"):
        p = datos.get(clave)
        if not p:
            continue
        candidatos = [p, os.path.join(base, p), os.path.join(base, p.replace("../", "").lstrip("./"))]
        existentes = [os.path.abspath(c) for c in candidatos if os.path.isdir(c)]
        if existentes:
            datos[clave] = existentes[0]
        else:
            print(f"AVISO: no encontre la carpeta de '{clave}' ({p}); se omite.")
            datos.pop(clave)
    salida = os.path.join(base, "data_fijo.yaml")
    with open(salida, "w") as f:
        yaml.safe_dump(datos, f)
    return salida, datos


def resumen(m):
    return {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datos", required=True, help="data.yaml exportado de Roboflow (formato YOLOv8)")
    p.add_argument("--modelo", default="yolov8n.pt", help="modelo preentrenado (transfer learning)")
    p.add_argument("--epocas", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--salida", default="resultados_detector")
    args = p.parse_args()

    ruta_yaml, datos = arreglar_rutas(args.datos)
    print("Clases:", datos.get("names"), "| splits:", [k for k in ("train", "val", "test") if k in datos])
    salida = os.path.abspath(args.salida)
    os.makedirs(os.path.join(salida, "figuras"), exist_ok=True)

    # 1. Ajuste fino de YOLO preentrenado en COCO
    modelo = YOLO(args.modelo)
    modelo.train(data=ruta_yaml, epochs=args.epocas, imgsz=args.imgsz, batch=args.batch,
                 patience=20, seed=0, project=salida, name="entrenamiento", exist_ok=True, plots=True)
    carpeta_ent = os.path.join(salida, "entrenamiento")
    best = os.path.join(carpeta_ent, "weights", "best.pt")
    shutil.copy2(best, os.path.join(salida, "best.pt"))

    # 2. Evaluar con el mejor modelo
    mejor = YOLO(best)
    metricas = {"modelo_base": args.modelo, "epocas_max": args.epocas, "imgsz": args.imgsz,
                "validacion": resumen(mejor.val(data=ruta_yaml, split="val", project=salida,
                                                 name="validacion", exist_ok=True, plots=True))}
    carpetas = [("entrenamiento", carpeta_ent), ("validacion", os.path.join(salida, "validacion"))]
    if "test" in datos:
        metricas["prueba"] = resumen(mejor.val(data=ruta_yaml, split="test", project=salida,
                                               name="prueba", exist_ok=True, plots=True))
        carpetas.append(("prueba", os.path.join(salida, "prueba")))
    with open(os.path.join(salida, "metricas.json"), "w") as f:
        json.dump(metricas, f, indent=2)

    # 3. Juntar las figuras para el documento
    for nombre, carpeta in carpetas:
        for archivo in sorted(os.listdir(carpeta)):
            if archivo.lower().endswith((".png", ".jpg")):
                shutil.copy2(os.path.join(carpeta, archivo), os.path.join(salida, "figuras", f"{nombre}_{archivo}"))

    shutil.make_archive(salida, "zip", salida)
    print("\n=== METRICAS ===")
    print(json.dumps(metricas, indent=2))
    print(f"\nListo. Pesos: {os.path.join(salida, 'best.pt')}   Todo junto: {salida}.zip")


if __name__ == "__main__":
    main()
