from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from asistente.interfaces import EntradaFeedback
from asistente.tipos import Feedback


class FeedbackPantalla(EntradaFeedback):
    def __init__(self):
        self.colas: dict[str, deque[Feedback]] = defaultdict(deque)
        self.candado = threading.Lock()

    def pulsar(self, persona_id: str, aprobado: bool) -> None:
        with self.candado:
            self.colas[persona_id].append(Feedback(persona_id, aprobado, fuente="boton"))

    def esperar(self, persona_id: str, timeout_s: float = 0.0) -> Feedback | None:
        # La pantalla no puede bloquearse: por defecto solo mira si ya pulsaron
        limite = time.time() + timeout_s
        while True:
            with self.candado:
                if self.colas[persona_id]:
                    return self.colas[persona_id].popleft()
            if time.time() >= limite:
                return None
            time.sleep(0.05)
