# =============================================================================
#  Controlador del vehiculo - Webots
#  Actividad 4.1 - Deteccion de Señales de Trafico con CNN  (Equipo 24)
#
#  Este controlador parte del esquema teclado-solo de la Actividad 2.1
#  (21siguelalinea.py) y le agrega:
#
#    1. Carga del CNN entrenado en training/gtsrb_cnn.ipynb (modelo_gtsrb.keras)
#       a traves de la clase TrafficSignDetector definida en este mismo dir.
#
#    2. En cada paso de simulacion:
#         - lee el frame BGRA de la camara
#         - convierte a BGR y extrae una ROI fija del cuadrante superior-derecho
#           (donde aparecen las señales del lado derecho del camino segun la
#           configuracion del world city_2025a_activity_3_1.wbt)
#         - infiere con el CNN; si confianza > threshold imprime y dibuja
#         - muestra el frame con la caja y el nombre de la señal en el Display
#
#    3. Control manual estandar por teclado:
#         - ↑/↓   acelerar/frenar
#         - ←/→   girar
#         - A     screenshot
#         - P     pausa/reanuda la deteccion (no la simulacion)
#
#  IMPORTANTE: el vehiculo arranca en el carril DERECHO, sin tocar la linea
#  amarilla central — la posicion inicial del BmwX5 en
#  SDC_webots/worlds/city_2025a_activity_3_1.wbt ya esta ajustada para esto.
# =============================================================================

import os
import sys
import platform
import time
from datetime import datetime


# -----------------------------------------------------------------------------
# Bootstrap portable de los bindings de Webots (igual al de Act 3.1)
# -----------------------------------------------------------------------------
def _bootstrap_webots():
    system = platform.system()
    explicit = os.environ.get("WEBOTS_HOME")
    candidates = []
    if explicit:
        candidates.append(explicit)
    if system == "Darwin":
        candidates += [
            os.path.expanduser("~/Applications/Webots.app"),
            "/Applications/Webots.app",
        ]
    elif system == "Linux":
        candidates += ["/usr/local/webots", "/snap/webots/current/usr/share/webots"]
    elif system == "Windows":
        candidates += [r"C:\Program Files\Webots"]

    for base in candidates:
        if system == "Darwin":
            py_path = os.path.join(base, "Contents", "lib", "controller", "python")
        else:
            py_path = os.path.join(base, "lib", "controller", "python")
        if os.path.isdir(py_path):
            os.environ["WEBOTS_HOME"] = base
            if py_path in sys.path:
                sys.path.remove(py_path)
            sys.path.insert(0, py_path)
            return base
    return None


_bootstrap_webots()

import numpy as np                                              # noqa: E402
import cv2                                                      # noqa: E402

from controller import Display, Keyboard                        # noqa: E402
from vehicle import Car, Driver                                 # noqa: E402

from traffic_sign_detector import TrafficSignDetector, SIGN_NAMES  # noqa: E402


# =============================================================================
# CONFIGURACION
# =============================================================================
DEBOUNCE_TIME = 0.1
MAX_ANGLE = 0.5
MAX_SPEED = 80
SPEED_INCR = 5
ANGLE_INCR = 0.05

# Umbral de confianza de la CNN para reportar una señal.
# Se elige alto (0.90) porque las señales del simulador son menos
# variables que las del GTSRB y un umbral bajo genera falsos positivos
# (asfalto/cielo clasificados como alguna señal).
CONFIDENCE_THRESHOLD = 0.90

# Cada cuantos frames del controlador corremos la inferencia CNN. Con basicTimeStep=50ms
# 1 = ~20Hz, 3 = ~6.6Hz. La inferencia toma ~10-15ms en Apple GPU, asi que podemos
# correrla en cada frame, pero 2 da margen para que el resto del controlador no se atrase.
INFERENCE_EVERY_N_FRAMES = 2


# =============================================================================
# RESOLUCION DEL ARCHIVO .keras
# =============================================================================
# El .keras vive en ../../models/ relativo a controllers/vehicle/.
# Pero Webots ejecuta este script con CWD = controllers/vehicle/, asi que
# resolvemos absoluto para evitar sorpresas.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "models", "modelo_gtsrb.keras"))


# =============================================================================
# CAPTURA DE IMAGEN DE LA CAMARA (BGRA -> BGR)
# =============================================================================
def get_image(camera):
    raw = camera.getImage()
    img = np.frombuffer(raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


# =============================================================================
# REGION OF INTEREST - cuadrante superior derecho
# =============================================================================
# Las señales del lado del conductor (derecha en Mexico/Alemania, donde el GTSRB
# fue capturado) aparecen siempre en la mitad superior-derecha del frame de la
# camara onboard. Limitamos la inferencia a esa region para:
#   - mejorar latencia (menos pixeles)
#   - reducir falsos positivos (asfalto, cielo, edificios)
#
# Los limites son fracciones del frame, validados visualmente contra el world
# city_2025a_activity_3_1.wbt.
ROI_X1_FRAC = 0.55
ROI_X2_FRAC = 0.99
ROI_Y1_FRAC = 0.05
ROI_Y2_FRAC = 0.65


def extract_roi(frame_bgr):
    h, w = frame_bgr.shape[:2]
    x1 = int(w * ROI_X1_FRAC)
    x2 = int(w * ROI_X2_FRAC)
    y1 = int(h * ROI_Y1_FRAC)
    y2 = int(h * ROI_Y2_FRAC)
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


# =============================================================================
# HUD - dibujar caja y texto sobre el frame
# =============================================================================
def draw_hud(frame_bgr, roi_box, sign_name, confidence, paused):
    x1, y1, x2, y2 = roi_box

    color = (0, 200, 0) if sign_name else (160, 160, 160)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

    # Etiqueta con la prediccion (o mensaje "no detection")
    if sign_name:
        label = f"{sign_name}  ({confidence*100:.0f}%)"
        cv2.putText(frame_bgr, label, (x1, max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame_bgr, f"sin senal  (conf={confidence*100:.0f}%)",
                    (x1, max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if paused:
        cv2.putText(frame_bgr, "DETECCION PAUSADA  [P para reanudar]",
                    (10, frame_bgr.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    return frame_bgr


# =============================================================================
# DISPLAY DE WEBOTS - pasa una imagen RGB de cualquier tamaño al panel
# =============================================================================
def push_to_display(display, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # El Display de Webots usa width/height fijos; los tomamos del propio device
    target = (display.getWidth(), display.getHeight())
    if (image_rgb.shape[1], image_rgb.shape[0]) != target:
        image_rgb = cv2.resize(image_rgb, target)
    ref = display.imageNew(image_rgb.tobytes(), Display.RGB, width=target[0], height=target[1])
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)


# =============================================================================
# MAIN
# =============================================================================
def main():
    speed = 0
    angle = 0.0
    last_press = {}
    detection_paused = False

    driver = Car()
    timestep = int(driver.getBasicTimeStep())

    # Devices
    camera = driver.getDevice("camera")
    camera.enable(timestep)

    display = Display("display_image")

    keyboard = driver.getKeyboard()
    keyboard.enable(timestep)

    # Detector CNN (lazy load: el primer predict tarda ~3-5s)
    print(f"[vehicle] Modelo esperado en: {MODEL_PATH}")
    detector = TrafficSignDetector(
        model_path=MODEL_PATH,
        img_size=32,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    print(f"[vehicle] Controlador iniciado. Confianza min = {CONFIDENCE_THRESHOLD}.")
    print(f"[vehicle] {len(SIGN_NAMES)} clases registradas en el detector.")
    print("[vehicle] Teclas:  UP/DOWN  acelerar/frenar   LEFT/RIGHT  giro   A  screenshot   P  pausa deteccion")

    # Historial de señales detectadas durante la corrida.
    detected_history = []  # list of (timestamp, class_id, name, confidence)
    last_logged_class = None  # evita spammear consola con el mismo class_id frame tras frame
    frame_counter = 0
    last_pred = (None, None, 0.0)  # se mantiene entre frames cuando saltamos inferencia

    while driver.step() != -1:
        frame_counter += 1

        # 1) Captura
        frame = get_image(camera)

        # 2) ROI
        roi, roi_box = extract_roi(frame)

        # 3) Inferencia cada N frames (a menos que este pausada)
        if not detection_paused and frame_counter % INFERENCE_EVERY_N_FRAMES == 0:
            class_id, name, confidence = detector.predict(roi)
            last_pred = (class_id, name, confidence)
            if class_id is not None and class_id != last_logged_class:
                stamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{stamp}] Senal detectada: {class_id:>2}  {name:<25}  conf={confidence:.2f}")
                detected_history.append((stamp, class_id, name, confidence))
                last_logged_class = class_id
            elif class_id is None:
                last_logged_class = None
        else:
            class_id, name, confidence = last_pred

        # 4) HUD + display
        annotated = draw_hud(frame.copy(), roi_box, name, confidence, detection_paused)
        push_to_display(display, annotated)

        # 5) Teclado (con debounce y debounce-per-key)
        now = time.time()
        key = keyboard.getKey()
        if key != -1 and (key not in last_press or now - last_press[key] >= DEBOUNCE_TIME):
            last_press[key] = now

            if key == keyboard.UP:
                if speed < MAX_SPEED:
                    speed += SPEED_INCR
                    print(f"[vehicle] speed -> {speed}")
            elif key == keyboard.DOWN:
                if speed > 0:
                    speed -= SPEED_INCR
                    print(f"[vehicle] speed -> {speed}")
            elif key == keyboard.RIGHT:
                angle = min(angle + ANGLE_INCR, MAX_ANGLE)
                print(f"[vehicle] angle -> {angle:+.2f}")
            elif key == keyboard.LEFT:
                angle = max(angle - ANGLE_INCR, -MAX_ANGLE)
                print(f"[vehicle] angle -> {angle:+.2f}")
            elif key == ord('A'):
                stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                fname = os.path.join(os.getcwd(), f"screenshot_{stamp}.png")
                camera.saveImage(fname, 1)
                print(f"[vehicle] Screenshot guardado: {fname}")
            elif key == ord('P'):
                detection_paused = not detection_paused
                print(f"[vehicle] Deteccion {'PAUSADA' if detection_paused else 'REANUDADA'}")

        # 6) Aplicar al vehiculo
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

    # Salida del bucle: imprimir resumen
    print("\n[vehicle] ===== Resumen de la corrida =====")
    print(f"[vehicle] Total detecciones registradas: {len(detected_history)}")
    unique_classes = sorted({c for _, c, _, _ in detected_history})
    print(f"[vehicle] Clases unicas detectadas    : {len(unique_classes)}")
    for cid in unique_classes:
        print(f"[vehicle]   - {cid:>2}  {SIGN_NAMES[cid]}")


if __name__ == "__main__":
    main()
