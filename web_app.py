"""Demo web del prototipo de LSP del Grupo 05.

Se ejecuta con: python -m streamlit run web_app.py
Reutiliza motor.py, entrenar.py, data/ y models/ de la COPIA LSP_WEB.
"""
from __future__ import annotations

from collections import Counter, deque
from datetime import datetime
from pathlib import Path
import threading
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from motor import (
    SEQUENCE_LEN, EVIDENCE, MODELS, counts, draw_hands, ensure_hand_model,
    normalize_label, prepare_dirs, save_sequence, save_static, vectorize,
)
from entrenar import (
    load_dynamic_model, load_static_model, train_dynamic, train_static,
)

st.set_page_config(page_title="Grupo 05 | Reconocimiento LSP", page_icon="🤟", layout="wide")
prepare_dirs()

st.markdown("""
<style>
.stApp {background-color: #f3f5f9;}
h1,h2,h3 {color: #152b47;}
div.stButton > button[kind="primary"] {background: #da7539; border: 0;}
</style>
""", unsafe_allow_html=True)


class SesionCamara:
    """Datos de una sola visita; compartidos con el hilo de la cámara mediante un candado."""

    def __init__(self):
        self.lock = threading.RLock()
        self.detector = None
        self.inicio = time.monotonic()
        self.ultimo_ms = -1
        self.modo = "Estática (Random Forest)"
        self.modelo = None
        self.etiquetas = None
        self.reconociendo = False
        self.ultimo_vector = None
        self.ultimo_fotograma = None
        self.mano_tiempo = 0.0
        self.contador = 0
        self.historial = deque(maxlen=5)
        self.secuencia = deque(maxlen=SEQUENCE_LEN)
        self.grabando = False
        self.grabados = []
        self.grabar_etiqueta = ""
        self.grabar_fuente = ""
        self.preparado_en = 0.0
        self.estado = "Presiona START y permite el acceso a la cámara."
        self.resultado = "Reconocimiento desactivado."
        self.puntuacion = ""

    def cambiar_modo(self, nuevo):
        with self.lock:
            if nuevo != self.modo:
                self.modo = nuevo
                self.reconociendo = False
                self.modelo = None
                self.etiquetas = None
                self.historial.clear()
                self.secuencia.clear()
                self.grabando = False
                self.resultado = "Activa el reconocimiento para empezar."

    def activar_reconocimiento(self):
        with self.lock:
            if self.modo.startswith("Movimiento"):
                modelo, etiquetas = load_dynamic_model()
            else:
                modelo, etiquetas = load_static_model(), None
            self.modelo, self.etiquetas = modelo, etiquetas
            self.reconociendo = True
            self.historial.clear()
            self.secuencia.clear()
            self.estado = "Reconocimiento activado."

    def desactivar_reconocimiento(self):
        with self.lock:
            self.reconociendo = False
            self.resultado = "Reconocimiento desactivado."
            self.puntuacion = ""

    def iniciar_secuencia(self, etiqueta, fuente):
        with self.lock:
            self.grabando = True
            self.grabados = []
            self.grabar_etiqueta = etiqueta
            self.grabar_fuente = fuente
            self.preparado_en = time.monotonic() + 1.5
            self.estado = "Prepárate. La secuencia empezará en 1,5 segundos."

    def cerrar_detector(self):
        with self.lock:
            if self.detector is not None:
                self.detector.close()
                self.detector = None
            self.ultimo_vector = None
            self.grabando = False
            self.secuencia.clear()
            self.estado = "Cámara apagada."

    def procesar(self, frame):
        imagen = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        with self.lock:
            try:
                import mediapipe as mp

                if self.detector is None:
                    archivo = ensure_hand_model()
                    opciones = mp.tasks.vision.HandLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(model_asset_path=str(archivo)),
                        running_mode=mp.tasks.vision.RunningMode.VIDEO,
                        num_hands=2,
                        min_hand_detection_confidence=0.5,
                        min_hand_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self.detector = mp.tasks.vision.HandLandmarker.create_from_options(opciones)
                    self.inicio = time.monotonic()
                    self.ultimo_ms = -1

                rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                entrada = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                ms = max(self.ultimo_ms + 1, int((time.monotonic() - self.inicio) * 1000))
                self.ultimo_ms = ms
                manos = self.detector.detect_for_video(entrada, ms)
                rasgos = vectorize(manos)
                self.ultimo_vector = None if rasgos is None else rasgos.copy()
                self.ultimo_fotograma = imagen.copy()
                self.contador += 1
                if rasgos is not None:
                    self.mano_tiempo = time.monotonic()
                cantidad = draw_hands(imagen, manos, cv2)

                if self.grabando and time.monotonic() >= self.preparado_en:
                    if rasgos is None:
                        self.grabando = False
                        self.grabados = []
                        self.estado = "Grabación cancelada: no se detecta la mano."
                    else:
                        self.grabados.append(rasgos.copy())
                        self.estado = f"Grabando movimiento: {len(self.grabados)}/{SEQUENCE_LEN}"
                        if len(self.grabados) >= SEQUENCE_LEN:
                            archivo = save_sequence(self.grabar_etiqueta, self.grabados,
                                                    source=self.grabar_fuente)
                            self.grabados = []
                            self.grabando = False
                            self.estado = f"Secuencia guardada: {archivo.name}"

                if self.reconociendo and self.modelo is not None:
                    self.predecir(rasgos)

                mensaje = f"MANOS DETECTADAS: {cantidad}" if cantidad else "MUESTRA TU MANO"
                cv2.rectangle(imagen, (0, 0), (imagen.shape[1], 42), (21, 43, 71), -1)
                cv2.putText(imagen, mensaje, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as error:
                self.estado = f"Error de cámara o detector: {error}"
                cv2.putText(imagen, "ERROR: REVISA EL ESTADO", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(imagen, format="bgr24")

    def predecir(self, rasgos):
        if rasgos is None:
            self.historial.clear()
            self.secuencia.clear()
            self.resultado = "Sin manos detectadas."
            self.puntuacion = ""
            return
        if self.modo.startswith("Movimiento"):
            self.secuencia.append(rasgos.copy())
            if len(self.secuencia) < SEQUENCE_LEN:
                self.resultado = f"Analizando movimiento ({len(self.secuencia)}/{SEQUENCE_LEN})"
                return
            if self.contador % 5:
                return
            probabilidades = self.modelo.predict(
                np.stack(self.secuencia)[None, ...], verbose=0
            )[0]
            etiquetas = self.etiquetas
            umbral = 0.70
        else:
            if self.contador % 3:
                return
            probabilidades = self.modelo.predict_proba(rasgos.reshape(1, -1))[0]
            etiquetas = self.modelo.classes_
            umbral = 0.65

        indice = int(np.argmax(probabilidades))
        valor = float(probabilidades[indice])
        propuesta = str(etiquetas[indice]) if valor >= umbral else "NO SEGURO"
        self.historial.append(propuesta)
        estable, repeticiones = Counter(self.historial).most_common(1)[0]
        if repeticiones < 3 and len(self.historial) >= 3:
            estable = "ESTABILIZANDO..."
        self.resultado = f"Resultado: {estable}"
        self.puntuacion = f"Score del modelo: {valor:.2f} (no es certeza calibrada)"


if "camara_lsp" not in st.session_state:
    st.session_state.camara_lsp = SesionCamara()
camara = st.session_state.camara_lsp

st.markdown("### GRUPO 05 | PROTOTIPO DE VISIÓN COMPUTACIONAL")
st.title("Reconocimiento de señas de la Lengua de Señas Peruana")
st.caption("Demo académica: reconoce únicamente las etiquetas entrenadas; no traduce frases completas.")

izquierda, derecha = st.columns([1.2, 1], gap="large")
with derecha:
    st.subheader("1. Tipo de reconocimiento")
    modo = st.selectbox("Selecciona el modelo", ["Estática (Random Forest)", "Movimiento (LSTM)"],
                        label_visibility="collapsed")
    camara.cambiar_modo(modo)

    st.subheader("2. Capturar ejemplos")
    etiqueta = st.text_input("Etiqueta de la seña o gesto", value="GESTO_A")
    fuente = st.text_input("Referencia de la seña (diccionario / validación)", value="")

    if st.button("Guardar 1 ejemplo", use_container_width=True, type="primary"):
        try:
            normalizada = normalize_label(etiqueta)
            if modo.startswith("Movimiento"):
                camara.iniciar_secuencia(normalizada, fuente)
                st.info("Mantén la mano visible hasta completar los 30 fotogramas.")
            else:
                with camara.lock:
                    vector = None if camara.ultimo_vector is None else camara.ultimo_vector.copy()
                    reciente = time.monotonic() - camara.mano_tiempo < 1
                if vector is None or not reciente:
                    st.warning("Enciende la cámara y muestra una mano antes de guardar.")
                else:
                    save_static(normalizada, vector, source=fuente)
                    st.success(f"Muestra de {normalizada} guardada.")
        except Exception as error:
            st.error(str(error))

    st.caption("Estáticas: una posición | Movimiento: 30 fotogramas")
    st.markdown("**Muestras actuales:**")
    try:
        resumen = counts("dynamic" if modo.startswith("Movimiento") else "static")
        st.caption(" | ".join(f"{k}: {v}" for k, v in resumen.items()) or "Sin muestras.")
    except Exception as error:
        st.warning(f"No se pudieron leer las muestras: {error}")

    st.subheader("3. Entrenar y reconocer")
    if st.button("ENTRENAR CON MIS MUESTRAS", use_container_width=True):
        camara.desactivar_reconocimiento()
        try:
            with st.spinner("Entrenando con las muestras de LSP_WEB..."):
                informe = train_dynamic() if modo.startswith("Movimiento") else train_static()
            st.success("Modelo entrenado en la copia LSP_WEB.")
        except Exception as error:
            st.error(f"No se pudo entrenar: {error}")

    with camara.lock:
        activo = camara.reconociendo
    if st.button("Desactivar reconocimiento" if activo else "Activar reconocimiento en vivo",
                 use_container_width=True):
        if activo:
            camara.desactivar_reconocimiento()
        else:
            try:
                camara.activar_reconocimiento()
            except Exception as error:
                st.error(f"No se pudo cargar el modelo: {error}")

    tipo_reporte = "reporte_lstm.txt" if modo.startswith("Movimiento") else "reporte_estatico.txt"
    reporte = MODELS / tipo_reporte
    if reporte.is_file():
        with st.expander("Abrir reporte de evaluación"):
            st.text(reporte.read_text(encoding="utf-8"))

    @st.fragment(run_every="1s")
    def estado_actual():
        with camara.lock:
            texto, valor, estado = camara.resultado, camara.puntuacion, camara.estado
        st.markdown(f"**{texto}**")
        if valor:
            st.caption(valor)
        st.caption(f"Estado: {estado}")

    estado_actual()

with izquierda:
    st.subheader("CÁMARA EN VIVO")
    st.caption("Pulsa START para encender la cámara y STOP para apagarla. Autoriza el permiso del navegador.")
    webrtc_streamer(
        key="lsp-camara-visitante",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=camara.procesar,
        on_video_ended=camara.cerrar_detector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
    if st.button("Guardar evidencia de cámara"):
        with camara.lock:
            imagen = None if camara.ultimo_fotograma is None else camara.ultimo_fotograma.copy()
        if imagen is None:
            st.warning("Primero enciende la cámara.")
        else:
            EVIDENCE.mkdir(parents=True, exist_ok=True)
            archivo = EVIDENCE / ("captura_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
            cv2.imwrite(str(archivo), imagen)
            st.success(f"Evidencia guardada: {archivo.name}")

st.caption("Advertencia: las etiquetas deben validarse con fuentes de LSP. En hosting gratuito, las nuevas muestras pueden perderse al reiniciar el servidor; el uso público requiere almacenamiento persistente y controles de acceso.")
