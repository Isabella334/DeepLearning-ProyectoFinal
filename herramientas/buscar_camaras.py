"""Busca las camaras conectadas y muestra cual es cual.  Uso: python herramientas/buscar_camaras.py"""
import sys

import cv2

backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
encontradas = []
for i in range(5):
    cap = cv2.VideoCapture(i, backend)
    ok, cuadro = cap.read() if cap.isOpened() else (False, None)
    if ok:
        encontradas.append(i)
        print(f"Camara {i}: OK ({cuadro.shape[1]}x{cuadro.shape[0]}). Mirala en la ventana y presiona una tecla.")
        cv2.putText(cuadro, f"Camara {i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.imshow("Camara", cuadro)
        cv2.waitKey(0)
    cap.release()
cv2.destroyAllWindows()
print("Camaras que funcionan:", encontradas or "ninguna")
if encontradas:
    print(f"Usa:  python -m asistente.vision.percepcion_camara --camara {encontradas[-1]}")
