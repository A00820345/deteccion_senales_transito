# =============================================================================
#  Traffic Sign Detector  -  Actividad 4.1 (Equipo 24)
#
#  Encapsula la inferencia del modelo Keras entrenado en gtsrb_cnn.ipynb sobre
#  una Region Of Interest (ROI) recortada del frame de la camara del vehiculo.
#
#  Diseño:
#    - Carga perezosa del modelo (solo la primera vez que se llama a predict),
#      asi el constructor no penaliza el arranque del controlador.
#    - Mapeo class_id -> nombre legible (43 clases del GTSRB) en SIGN_NAMES.
#    - Umbral de confianza configurable; debajo se devuelve (None, None).
#
#  La integracion en Webots se hace desde vehicle.py: ese script extrae la ROI
#  del cuadrante superior derecho del frame (donde aparecen las señales del
#  lado del conductor) y llama a TrafficSignDetector.predict().
# =============================================================================
import os
import sys

import numpy as np
import cv2

# Silenciar logs ruidosos de TensorFlow al cargar el modelo.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# Nombres oficiales del GTSRB (43 clases) — mismo orden que el dataset Kaggle.
SIGN_NAMES = {
    0: "Speed limit 20",        1: "Speed limit 30",        2: "Speed limit 50",
    3: "Speed limit 60",        4: "Speed limit 70",        5: "Speed limit 80",
    6: "End speed limit 80",    7: "Speed limit 100",       8: "Speed limit 120",
    9: "No passing",           10: "No passing trucks",    11: "Priority intersection",
    12: "Priority road",       13: "Yield",                14: "Stop",
    15: "No vehicles",         16: "No trucks",            17: "No entry",
    18: "General caution",     19: "Dangerous curve left", 20: "Dangerous curve right",
    21: "Double curve",        22: "Bumpy road",           23: "Slippery road",
    24: "Road narrows",        25: "Road work",            26: "Traffic signals",
    27: "Pedestrians",         28: "Children crossing",    29: "Bicycles crossing",
    30: "Beware ice",          31: "Wild animals",         32: "End restrictions",
    33: "Turn right",          34: "Turn left",            35: "Ahead only",
    36: "Go straight/right",   37: "Go straight/left",     38: "Keep right",
    39: "Keep left",           40: "Roundabout",           41: "End no passing",
    42: "End no passing trucks",
}


class TrafficSignDetector:
    """Inferencia de señales de tránsito con el CNN GTSRB del Equipo 24.

    Parámetros
    ----------
    model_path : str
        Ruta absoluta al .keras exportado por gtsrb_cnn.ipynb.
    img_size : int, default 32
        Tamaño de entrada del modelo (se redimensiona la ROI a este tamaño).
    confidence_threshold : float, default 0.85
        Probabilidad mínima de la clase más probable para reportar detección.
        El umbral relativamente alto evita falsos positivos en frames donde
        la ROI captura asfalto/cielo en vez de una señal.
    """

    def __init__(self, model_path, img_size=32, confidence_threshold=0.85):
        self.model_path = model_path
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self._model = None  # se carga al primer predict (lazy)

    # ------------------------------------------------------------------
    # Carga perezosa del modelo Keras
    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        # Import diferido: TensorFlow pesa ~700 MB y tarda en cargar; lo evitamos
        # mientras el usuario solo edita parámetros del controlador.
        from tensorflow.keras.models import load_model  # noqa: WPS433
        print(f"[TrafficSignDetector] Cargando modelo desde {self.model_path}")
        self._model = load_model(self.model_path)
        print("[TrafficSignDetector] Modelo cargado.")

    # ------------------------------------------------------------------
    # Preprocesamiento de la ROI exactamente como en el entrenamiento
    # ------------------------------------------------------------------
    def _preprocess(self, roi_bgr):
        """ROI BGR (cualquier tamaño) -> tensor (1, img_size, img_size, 3) float32 [0,1]."""
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi_rgb, (self.img_size, self.img_size))
        roi = roi.astype(np.float32) / 255.0
        return np.expand_dims(roi, axis=0)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def predict(self, roi_bgr):
        """Predice la clase de señal en una ROI.

        Devuelve
        --------
        (class_id, name, confidence) si confidence >= threshold,
        (None, None, confidence) en caso contrario.

        confidence siempre se retorna para que el caller pueda mostrarlo en
        el HUD aun cuando esté por debajo del umbral.
        """
        self._ensure_loaded()
        x = self._preprocess(roi_bgr)
        probs = self._model.predict(x, verbose=0)[0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        if confidence >= self.confidence_threshold:
            return class_id, SIGN_NAMES[class_id], confidence
        return None, None, confidence

    def predict_top_k(self, roi_bgr, k=3):
        """Útil para debug: top-k clases con su probabilidad."""
        self._ensure_loaded()
        x = self._preprocess(roi_bgr)
        probs = self._model.predict(x, verbose=0)[0]
        top = np.argsort(probs)[::-1][:k]
        return [(int(i), SIGN_NAMES[int(i)], float(probs[i])) for i in top]
