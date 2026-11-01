"""
Bucle principal: une todos los modulos. Lo mantiene el rol de integracion.

Ejemplos:
    python main.py --modo falso --pasos 5                    # todo falso, corre en cualquier compu
    python main.py --modo falso --voz teclado --feedback teclado   # vos le hablas (escribiendo)
    python main.py --modo sim --pasos 3                      # cuerpo y percepcion en MuJoCo
    python main.py --modo falso --voz whisper --feedback teclado   # le hablas por el microfono
    python main.py --modo real --voz teclado --feedback teclado --ventana   # version B: camara + carrito virtual

Para usar un modulo real, se cambia SOLO la funcion construir(). Nada mas.
"""
import argparse

from asistente.cerebro.cerebro import CerebroAprendiz
from asistente.cerebro.preferencias import PreferenciasRed
from asistente.mocks import (CuerpoFalso, ExplicadorSimple, FeedbackAlAzar, FeedbackTeclado,
                             MemoriaEnRAM, PercepcionFalsa, VozTeclado)
from asistente.tipos import DESCONOCIDO, Accion, Contexto, Evento, Feedback, Resultado
from asistente.vision.reconocedor import ReconocedorPocosEjemplos

PREGUNTAS = {Accion.TRAER: "Te paso '{}'?", Accion.APARTAR: "Aparto '{}'?", Accion.ACERCARSE: "Me acerco?"}


def construir(args):
    reconocedor = ReconocedorPocosEjemplos(modo="cabeza")   # P2 (v2; modo="prototipo" para la v1)
    if args.modo == "sim":
        from asistente.cuerpo_sim import CuerpoSim, MundoSim, PercepcionSim
        mundo = MundoSim(ventana=args.ventana)
        percepcion, cuerpo = PercepcionSim(mundo, reconocedor, args.persona), CuerpoSim(mundo)
    elif args.modo == "real":
        # Version B, gemelo digital: la camara ve la cartulina real y el carrito virtual actua
        from asistente.cuerpo_sim import CuerpoSim, MundoSim
        from asistente.vision.percepcion_camara import PercepcionCamara
        mundo = MundoSim(ventana=args.ventana, con_camara=True, iniciales=[])
        camara = int(args.camara) if str(args.camara).isdigit() else args.camara
        percepcion = PercepcionCamara(reconocedor, camara=camara, persona_id=args.persona,
                                      mundo=mundo, ventana=not args.sin_vista)
        cuerpo = CuerpoSim(mundo)
    else:
        percepcion, cuerpo = PercepcionFalsa(reconocedor, args.persona), CuerpoFalso()

    cerebro = CerebroAprendiz(preferencias=PreferenciasRed())   # P2
    memoria = MemoriaEnRAM()                  # P4 lo cambia por MemoriaSQLite
    explicador = ExplicadorSimple()           # P4 lo cambia por ExplicadorXAI
    feedback = FeedbackTeclado() if args.feedback == "teclado" else FeedbackAlAzar()
    voz = None
    if args.voz == "teclado":
        voz = VozTeclado()
    elif args.voz == "whisper":
        from asistente.voz import VozWhisper
        voz = VozWhisper()
    return percepcion, cuerpo, cerebro, memoria, explicador, feedback, voz


def atender_voz(voz, percepcion, memoria, persona_id, escena):
    """Escucha una orden. Devuelve un pedido (para el cerebro) o un Feedback, o None."""
    if voz is None:
        return None, None
    if hasattr(voz, "vocabulario"):       # los nombres que ya conoce ayudan a Whisper a entenderlos
        voz.vocabulario = sorted({o.nombre for o in escena.objetos if o.nombre != DESCONOCIDO})
    texto = voz.escuchar()
    orden = voz.interpretar(texto) if texto else None
    if orden is None:
        return None, None
    if orden.tipo == "ensenar":
        # Se le pone el nombre al objeto desconocido mas reciente de la mesa
        desconocidos = [o for o in escena.objetos if o.nombre == DESCONOCIDO]
        if desconocidos:
            obj = desconocidos[-1]
            percepcion.ensenar(persona_id, obj.id, orden.nombre)
            memoria.registrar_ensenanza(persona_id, orden.nombre, obj.id)
            print(f"   Aprendido: {obj.id} es '{orden.nombre}'")
        return None, None
    if orden.tipo == "feedback":
        return None, Feedback(persona_id, orden.aprobado, fuente="voz")
    if orden.tipo == "pedido" and orden.nombre:      # las reglas no son pedidos para la anticipacion
        memoria.registrar_evento(Evento("pedido", f"{orden.accion.value}:{orden.nombre}", persona_id))
    return orden, None


def un_ciclo(percepcion, cuerpo, cerebro, memoria, explicador, entrada, voz, persona_id, anteriores):
    escena = percepcion.observar()

    # Eventos: que aparecio o desaparecio desde la ultima vez (para la anticipacion)
    actuales = {o.nombre for o in escena.objetos if o.nombre != DESCONOCIDO}
    for nombre in actuales - anteriores:
        memoria.registrar_evento(Evento("aparecio", nombre, persona_id))
    for nombre in anteriores - actuales:
        memoria.registrar_evento(Evento("desaparecio", nombre, persona_id))

    pedido, _ = atender_voz(voz, percepcion, memoria, persona_id, escena)
    contexto = Contexto(escena=escena, persona_id=persona_id, pedido=pedido,
                        eventos_recientes=memoria.eventos_recientes(persona_id))
    decision = cerebro.decidir(contexto)
    explicacion = explicador.explicar(contexto, decision, cerebro, memoria)
    objeto = next((o for o in escena.objetos if o.id == decision.objeto_id), None)

    print(f"   Veo: {[o.nombre for o in escena.objetos]}")
    print(f"-> Decision: {decision.accion.value} {objeto.nombre if objeto else ''} "
          f"(confianza {decision.confianza:.0%})")
    print(f"   Explicacion: {explicacion.texto} {explicacion.evidencia}")

    fb = None
    confirmar = decision.preguntar_antes and decision.accion != Accion.IGNORAR
    if confirmar:
        # Si duda, pregunta antes de actuar; la respuesta es el feedback de esta decision
        print(f"   Robot: {PREGUNTAS[decision.accion].format(objeto.nombre if objeto else '')}")
        fb = entrada.esperar(persona_id)
    elif decision.preguntar_antes and contexto.pedido is not None:
        print(f"   Robot: No encuentro '{contexto.pedido.nombre}' en la mesa.")

    if confirmar and (fb is None or not fb.aprobado):
        resultado = Resultado(exito=False, duracion_s=0.0, mensaje="no lo hice: no me lo confirmaste")
    else:
        resultado = cuerpo.ejecutar(decision.accion, objeto, escena)
    print(f"   Resultado: {'ok' if resultado.exito else 'fallo'} - {resultado.mensaje}")

    if not confirmar and decision.accion != Accion.IGNORAR:
        fb = entrada.esperar(persona_id)
    if fb is not None:
        cerebro.aprender(contexto, decision, fb)
        memoria.registrar_evento(Evento("feedback", "si" if fb.aprobado else "no", persona_id))
    memoria.registrar_interaccion(contexto, decision, resultado, fb)
    return actuales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["falso", "sim", "real"], default="falso")
    ap.add_argument("--feedback", choices=["azar", "teclado"], default="azar")
    ap.add_argument("--voz", choices=["no", "teclado", "whisper"], default="no")
    ap.add_argument("--persona", default="persona_A")
    ap.add_argument("--pasos", type=int, default=5)
    ap.add_argument("--ventana", action="store_true")
    ap.add_argument("--camara", default="0", help="modo real: numero de webcam o ruta de un video")
    ap.add_argument("--sin-vista", dest="sin_vista", action="store_true",
                    help="modo real: no abrir la ventana con lo que ve la camara")
    args = ap.parse_args()

    percepcion, cuerpo, cerebro, memoria, explicador, entrada, voz = construir(args)
    anteriores = set()
    try:
        for i in range(args.pasos):
            print(f"\nCiclo {i + 1}")
            anteriores = un_ciclo(percepcion, cuerpo, cerebro, memoria, explicador, entrada, voz,
                                  args.persona, anteriores)
    finally:
        cuerpo.cerrar()
        if hasattr(percepcion, "cerrar"):
            percepcion.cerrar()
    print(f"\nInteracciones guardadas: {len(memoria.historial(args.persona))}")


if __name__ == "__main__":
    main()
