"""
Graba un video del dataset con la webcam, sin efecto espejo y sin compresion de apps.
Muestra en pantalla cuantos marcadores ArUco de las esquinas se ven (lo que se guarda va limpio).

Ademas del video, guarda directamente 1 foto por segundo en dataset/fotos/<nombre>/,
asi el dataset no depende de que el video se pueda abrir despues.

Uso:
    python herramientas/grabar_dataset.py sesion1_dia
    python herramientas/grabar_dataset.py sesion2_lampara --camara 1
    python herramientas/grabar_dataset.py sesion3_desorden --carpeta C:/dataset   # fuera de OneDrive

Teclas:  q = terminar y guardar (con la ventana de la camara seleccionada)
"""
import argparse
import os
import time

import cv2


def abrir_salida(base, fps, w, h):
    """Prueba varios formatos hasta encontrar uno que este computador pueda escribir."""
    for ext, codec in [(".mp4", "mp4v"), (".avi", "MJPG"), (".avi", "XVID")]:
        ruta = base + ext
        salida = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if salida.isOpened():
            return salida, ruta
        salida.release()
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("nombre", help="ej. sesion1_dia, sesion2_lampara, sesion5_prueba")
    p.add_argument("--camara", type=int, default=0, help="0 = camara de la laptop")
    p.add_argument("--ancho", type=int, default=1280)
    p.add_argument("--alto", type=int, default=720)
    p.add_argument("--carpeta", default="dataset", help="donde guardar (por defecto ./dataset)")
    p.add_argument("--fotos_por_segundo", type=float, default=1.0, help="0 = no guardar fotos")
    args = p.parse_args()

    carpeta = os.path.abspath(args.carpeta)
    dir_videos = os.path.join(carpeta, "videos")
    dir_fotos = os.path.join(carpeta, "fotos", args.nombre)
    os.makedirs(dir_videos, exist_ok=True)
    os.makedirs(dir_fotos, exist_ok=True)

    cap = cv2.VideoCapture(args.camara)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.alto)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("No se pudo abrir la camara. Proba con --camara 1")

    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 120 else 30
    salida, ruta = abrir_salida(os.path.join(dir_videos, args.nombre), fps, w, h)
    if salida is None:
        print("AVISO: este computador no pudo crear el archivo de video. Se guardan solo las fotos.")

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                       cv2.aruco.DetectorParameters())
    print(f"Grabando {w}x{h} a {fps:.0f} fps")
    print(f"  Video: {ruta}")
    print(f"  Fotos: {dir_fotos}")
    print("Presiona q (con la ventana de la camara seleccionada) para terminar.")

    cuadros, fotos = 0, 0
    intervalo = 1.0 / args.fotos_por_segundo if args.fotos_por_segundo > 0 else None
    proxima_foto = time.time()
    try:
        while ok:
            if salida is not None:
                salida.write(frame)                       # se guarda SIN dibujos
            cuadros += 1
            if intervalo and time.time() >= proxima_foto:
                fotos += 1
                cv2.imwrite(os.path.join(dir_fotos, f"{args.nombre}_{fotos:04d}.jpg"), frame)
                proxima_foto += intervalo

            vista = frame.copy()
            esquinas, ids, _ = detector.detectMarkers(frame)
            vistos = set() if ids is None else {int(i) for i in ids.flatten()} & {0, 1, 2, 3}
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vista, esquinas, ids)
            color = (0, 200, 0) if len(vistos) == 4 else (0, 0, 255)
            cv2.putText(vista, f"Esquinas: {len(vistos)}/4  {sorted(vistos)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(vista, f"Fotos guardadas: {fotos}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Grabando (q para terminar)", vista)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            ok, frame = cap.read()
    finally:                                              # tambien si se corta con Ctrl+C
        cap.release()
        if salida is not None:
            salida.release()
        cv2.destroyAllWindows()

    print(f"\nListo: {cuadros} cuadros grabados, {fotos} fotos guardadas.")
    if ruta and os.path.exists(ruta) and os.path.getsize(ruta) > 0:
        print(f"  Video OK ({os.path.getsize(ruta) / 1e6:.1f} MB): {ruta}")
    elif salida is not None:
        print(f"  AVISO: el video no aparece en {ruta}. Las fotos si estan en {dir_fotos}")
    print(f"  Fotos en: {dir_fotos}")


if __name__ == "__main__":
    main()