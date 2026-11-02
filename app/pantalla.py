import html
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:      # streamlit solo agrega app/ al path
    sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt   # noqa: E402
import streamlit as st            # noqa: E402

from app.etiquetas import legibles                           # noqa: E402
from app.imaginacion import dibujar_imaginacion, vista_real   # noqa: E402
from app.sesion import Sesion                                 # noqa: E402
from asistente.tipos import DESCONOCIDO, Accion               # noqa: E402

# Version B: streamlit run app/pantalla.py -- --camara 0   (sin "--camara": mesa simulada)
CAMARA = None
if "--camara" in sys.argv:
    _i = sys.argv.index("--camara")
    CAMARA = sys.argv[_i + 1] if _i + 1 < len(sys.argv) else "0"
    CAMARA = int(CAMARA) if CAMARA.isdigit() else CAMARA

PREGUNTAS = {Accion.TRAER: "¿Te paso {}?", Accion.APARTAR: "¿Aparto {}?", Accion.ACERCARSE: "¿Me acerco?"}
PASOS = ("Mirar la mesa", "Decidir qué hacer", "Responder")
RESPUESTA = {True: "aprobado", False: "desaprobado", None: "sin respuesta"}
VERDE, ROJO = "#3A9D4F", "#C94A3A"
MAX_BARRAS = 10

ESTILO = f"""
<style>
.pasos {{ display: flex; gap: .75rem; list-style: none; padding: 0 !important; margin: .25rem 0 1rem;
          counter-reset: paso; }}
.pasos li {{ flex: 1; margin: 0; padding: .5rem 0 0; border-top: 4px solid #B8C4CA; color: #5B6A72;
             counter-increment: paso; }}
.pasos li::before {{ content: counter(paso) ". "; font-weight: 700; }}
.pasos li.hecho {{ border-color: #1E2830; color: #1E2830; }}
.pasos li.actual {{ border-color: #E8710A; color: #1E2830; font-weight: 700; }}
.voz {{ font-family: "Atkinson Hyperlegible Mono", monospace; font-size: clamp(1.25rem, 2.2vw, 1.7rem);
        line-height: 1.35; max-width: 38ch; margin: .25rem 0 .75rem; }}
.evidencia {{ font-size: 1.05rem; color: #3D4B53; max-width: 60ch; margin: 0 0 .5rem; }}
.datos {{ font-size: .95rem; color: #5B6A72; margin: 0 0 1rem; }}
.st-key-aprobar button {{ background: {VERDE}; border-color: {VERDE}; color: #FFFFFF; }}
.st-key-desaprobar button {{ background: {ROJO}; border-color: {ROJO}; color: #FFFFFF; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 2.5rem; }}
@media (max-width: 640px) {{ .pasos {{ flex-direction: column; gap: .25rem; }} }}
</style>
"""


def _sesion() -> Sesion:
    persona, ruta = st.session_state.persona.strip() or "persona_A", st.session_state.ruta_db.strip()
    clave = (persona, ruta)
    if st.session_state.get("clave") != clave:
        if "sesion" in st.session_state:
            st.session_state.sesion.cerrar()
        en_disco = ruta != ":memory:"
        crear = Sesion.simulada if CAMARA is None else (lambda *a, **k: Sesion.camara(*a, camara=CAMARA, **k))
        st.session_state.sesion = crear(
            persona, ruta_db=str(RAIZ / ruta) if en_disco else ruta,
            carpeta_modelos=str(RAIZ / "modelos" / persona) if en_disco else None)
        st.session_state.clave = clave
    return st.session_state.sesion


def _avisar(texto: str, tipo: str = "info") -> None:
    st.session_state.aviso = (tipo, texto)


def _nombre(obj) -> str:
    return "ese objeto" if obj is None or obj.nombre == DESCONOCIDO else f"'{obj.nombre}'"


def _ubicacion(obj) -> str:
    vertical = "arriba" if obj.y > 0.08 else "abajo" if obj.y < -0.08 else "en medio"
    lado = "a la izquierda" if obj.x < -0.1 else "a la derecha" if obj.x > 0.1 else "al centro"
    return f"{obj.id} ({vertical}, {lado})"


def _etiqueta(obj) -> str:
    return _ubicacion(obj) if obj.nombre == DESCONOCIDO else f"{obj.nombre} ({obj.id})"


def mirar():
    _sesion().observar()


def decidir():
    s = _sesion()
    if s.escena is None:
        s.observar()
    s.decidir()


def ejecutar(confirmado):
    _sesion().ejecutar(confirmado)


def responder(aprobado):
    s = _sesion()
    s.responder(aprobado)
    s.observar()
    _avisar("Respuesta guardada: el robot aprendió de ella." if aprobado is not None
            else "Guardado sin respuesta.", "success")


def poner():
    s, tipo = _sesion(), st.session_state.get("poner_tipo")
    try:
        s.estimulo("poner", tipo)
    except (ValueError, RuntimeError):
        _avisar(f"No hay espacio en la mesa para '{tipo}'. Quitá algún objeto y probá de nuevo.", "warning")
    s.observar()


def reiniciar():
    s = _sesion()
    s.reiniciar()
    s.cerrar()
    for clave in ("sesion", "clave", "confirmar_reinicio"):
        st.session_state.pop(clave, None)
    _avisar(f"Listo: el robot olvidó todo lo que aprendió de {s.persona_id} y la mesa volvió a empezar.", "success")


def quitar():
    s = _sesion()
    s.estimulo("quitar", st.session_state.get("quitar_obj"))
    s.observar()


def ensenar():
    s, objeto_id = _sesion(), st.session_state.get("ensenar_obj") or ""
    nombre = st.session_state.ensenar_nombre.strip()
    if s.ensenar(objeto_id, nombre):
        _avisar(f"Aprendido: {objeto_id} es '{nombre}'.", "success")
        st.session_state.ensenar_nombre = ""
        s.observar()
    else:
        _avisar("Elegí un objeto de la mesa y escribí su nombre.", "warning")


def pedir():
    nombre = st.session_state.get("pedir_nombre")
    if not nombre:
        _avisar("Primero enseñale el nombre de algún objeto.", "warning")
        return
    s = _sesion()
    s.pedir(nombre, Accion(st.session_state.pedir_accion))
    s.decidir()


def barras_contribuciones(contribuciones: dict[str, float]):
    top = sorted(contribuciones.items(), key=lambda kv: abs(kv[1]), reverse=True)[:MAX_BARRAS][::-1]
    fig, ax = plt.subplots(figsize=(4.6, 0.34 * len(top) + 0.9))
    ax.barh([n for n, _ in top], [v for _, v in top], color=[VERDE if v >= 0 else ROJO for _, v in top])
    ax.axvline(0, color="#1E2830", lw=0.8)
    ax.set_xlabel("en contra  |  a favor", color="#3D4B53")
    ax.tick_params(labelsize=9, colors="#1E2830")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    fig.tight_layout()
    return fig


def etapa(s: Sesion) -> int:
    return 0 if s.escena is None else 1 if s.decision is None else 2


def vista_pasos(actual: int):
    items = "".join(f'<li class="{"actual" if i == actual else "hecho" if i < actual else ""}">{p}</li>'
                    for i, p in enumerate(PASOS))
    st.html(f'<ol class="pasos">{items}</ol>')


def vista_mesa(s: Sesion):
    if s.con_camara and s.vista_camara is not None:
        st.subheader("Lo que ve la cámara")
        st.image(s.vista_camara, width="stretch")
    if s.mundo is not None:
        st.subheader("La mesa virtual" if s.con_camara else "La mesa")
        st.image(vista_real(s.mundo, 640), width="stretch")


def vista_imaginacion(s: Sesion):
    if s.escena is not None:
        st.image(dibujar_imaginacion(s.escena, s.decision), width="stretch")
    if s.decision is not None and s.explicacion.contribuciones:
        st.subheader("Qué pesó en la decisión")
        st.caption("Cuánto empujó cada dato a la red de preferencias hacia esta acción (SHAP).")
        fig = barras_contribuciones(legibles(s.explicacion.contribuciones))
        st.pyplot(fig)
        plt.close(fig)


def _voz(texto: str):
    st.html(f'<p class="voz">{html.escape(texto)}</p>')


def vista_robot(s: Sesion):
    d = s.decision
    if s.escena is None:
        _voz("Todavía no miré la mesa.")
        st.button("Mirar la mesa", on_click=mirar, type="primary")
        return
    if d is None:
        _voz("Ya miré la mesa. ¿Decido qué hacer?")
        izq, der = st.columns(2)
        izq.button("Decidir qué hacer", on_click=decidir, type="primary", width="stretch")
        der.button("Mirar la mesa de nuevo", on_click=mirar, width="stretch")
        return

    _voz(s.explicacion.texto)
    if s.explicacion.evidencia:
        st.html(f'<p class="evidencia">{html.escape(s.explicacion.evidencia)}</p>')
    st.html(f'<p class="datos">Acción: {d.accion.value}. Confianza: {d.confianza:.0%}. '
            f'Motivo: {html.escape(d.motivo or "sin motivo")}.</p>')
    if s.resultado is None:
        if s.necesita_confirmacion:
            st.write(PREGUNTAS[d.accion].format(_nombre(s.objeto)))
            si, no = st.columns(2)
            si.button("Sí, hacelo", on_click=ejecutar, args=(True,), type="primary", width="stretch")
            no.button("No", on_click=ejecutar, args=(False,), width="stretch")
        else:
            st.button("Ejecutar", on_click=ejecutar, args=(None,), type="primary")
    else:
        st.write(("Listo: " if s.resultado.exito else "No lo hice: ") + s.resultado.mensaje + ".")
        if s.necesita_confirmacion:            # la respuesta a la pregunta ya fue el feedback
            st.button("Siguiente", on_click=responder, args=(None,), type="primary")
        else:
            st.write("¿Estuvo bien?")
            bien, mal, nada = st.columns(3)
            bien.button("Aprobado", key="aprobar", on_click=responder, args=(True,), width="stretch")
            mal.button("Desaprobado", key="desaprobar", on_click=responder, args=(False,), width="stretch")
            nada.button("Sin opinión", on_click=responder, args=(None,), width="stretch")


def vista_historial(s: Sesion):
    st.subheader("Historial")
    filas = s.historial(10)
    if not filas:
        st.write("Todavía no hay interacciones. Pedile al robot que decida algo y respondele.")
        return
    st.dataframe([{"Hora": datetime.fromtimestamp(h["t"]).strftime("%H:%M:%S"), "Acción": h["accion"],
                   "Objeto": h["objeto_nombre"] or "", "Motivo": h["motivo"],
                   "Confianza": f"{h['confianza']:.0%}", "Resultado": "hecho" if h["exito"] else "no hecho",
                   "Tu respuesta": RESPUESTA[h["aprobado"]]} for h in filas], hide_index=True)


def barra_lateral(s: Sesion):
    with st.sidebar:
        st.subheader("Quién usa el robot")
        st.text_input("Persona", key="persona")
        with st.expander("Avanzado"):
            st.text_input("Base de datos", key="ruta_db", help="Escribí :memory: para no guardar nada.")
            st.checkbox("Quiero que el robot olvide todo lo que aprendió de esta persona",
                        key="confirmar_reinicio")
            st.button("Reiniciar desde cero", on_click=reiniciar,
                      disabled=not st.session_state.get("confirmar_reinicio"))

        objetos = s.escena.objetos if s.escena else []
        st.subheader("La mesa")
        if s.con_camara:
            st.caption("Poné y quitá objetos con la mano en la cartulina; después tocá «Mirar la mesa».")
        elif s.mundo is not None:
            disponibles = s.tipos_para_poner()
            st.selectbox("Objeto para poner", disponibles, key="poner_tipo",
                         help="Cada objeto existe una sola vez: si ya está en la mesa, no aparece aquí.")
            st.button("Poner en la mesa", on_click=poner, disabled=not disponibles)
        if not s.con_camara:
            st.selectbox("Objeto para quitar", [o.id for o in objetos], key="quitar_obj",
                         format_func=lambda oid: next((_etiqueta(o) for o in objetos if o.id == oid), oid))
            st.button("Quitar de la mesa", on_click=quitar, disabled=not objetos)

        st.subheader("Enseñar un nombre")
        # Primero los que no tienen nombre; tambien se puede elegir uno con nombre para CORREGIRLO
        candidatos = sorted(objetos, key=lambda o: o.nombre != DESCONOCIDO)
        st.selectbox("Objeto (sin nombre, o uno mal reconocido para corregirlo)", [o.id for o in candidatos],
                     key="ensenar_obj",
                     format_func=lambda oid: next((_etiqueta(o) for o in candidatos if o.id == oid), oid))
        st.text_input("Su nombre", key="ensenar_nombre", placeholder="mi lapicero")
        st.button("Enseñar", on_click=ensenar, disabled=not objetos)

        st.subheader("Pedir algo")
        conocidos = s.reconocedor.nombres(s.persona_id) if s.reconocedor else []
        st.selectbox("Objeto", conocidos, key="pedir_nombre")
        st.radio("Qué hacer", [Accion.TRAER.value, Accion.APARTAR.value], key="pedir_accion",
                 format_func=str.capitalize, horizontal=True)
        st.button("Pedir", on_click=pedir, disabled=not conocidos)


def main():
    st.set_page_config(page_title="Robot asistente", layout="wide")
    st.session_state.setdefault("persona", "persona_A")
    st.session_state.setdefault("ruta_db", "robot.db")
    st.html(ESTILO)
    s = _sesion()
    barra_lateral(s)

    st.title("Robot asistente")
    st.write("Enseñale el nombre de tus objetos, pedile cosas y decile si lo que hizo estuvo bien. "
             "Así aprende lo que te gusta a vos.")
    if "aviso" in st.session_state:
        tipo, texto = st.session_state.pop("aviso")
        getattr(st, tipo)(texto)
    vista_pasos(etapa(s))
    robot, imaginacion = st.columns([5, 4], gap="large")
    with robot:
        vista_robot(s)
        vista_mesa(s)
    with imaginacion:
        vista_imaginacion(s)
    vista_historial(s)


main()
