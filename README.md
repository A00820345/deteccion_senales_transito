# Actividad 4.1 — Detección de Señales de Tránsito con CNN

Equipo 24 — Maestría en Inteligencia Artificial Aplicada / Navegación Autónoma
Tecnológico de Monterrey

| Integrante                  | Matrícula |
|:----------------------------|:---------:|
| Rafael Sánchez Marmolejo    | A00820345 |
| Alonso Rojas Loreto         | A01706755 |
| Tonatiuh Salas Ortiz        | A01080251 |
| Mario Sánchez Valtierra     | A01797169 |

## Estructura del proyecto

```
Actividad 4.1 - Detección de Señales de Tránsito/
├── README.md                            ← este archivo
├── 21siguelalinea.py                    ← controlador base de la Actividad 2.1 (referencia)
├── SDC_webots/
│   ├── controllers/
│   │   ├── supervisor_controller/       ← supervisor del mundo (heredado de Act 3.1)
│   │   └── vehicle → ../../controllers/vehicle   (symlink)
│   └── worlds/
│       └── city_2025a_activity_3_1.wbt  ← mismo mundo de la Actividad 3.1, 16 señales
├── controllers/
│   └── vehicle/
│       ├── vehicle.py                   ← controlador con teclado + inferencia CNN
│       ├── traffic_sign_detector.py     ← clase que envuelve el modelo .keras
│       └── runtime.ini                  ← apunta a la Python del .venv del proyecto
├── models/
│   └── modelo_gtsrb.keras               ← modelo CNN entrenado (98.38% accuracy)
├── training/
│   ├── gtsrb_cnn.ipynb                  ← notebook entregable paso a paso
│   ├── gtsrb_cnn.pdf                    ← PDF exportado del notebook (rúbrica)
│   ├── assets/tec_logo.png
│   └── dataset/GTSRB → /Users/.../Downloads/GTSRB   (symlink, gitignored)
└── reporte/
```

## 1. Dataset GTSRB

El *German Traffic Sign Recognition Benchmark* (GTSRB) es el dataset clásico
para clasificación de señales de tránsito. Se usa la versión Kaggle, que ya
viene partida en `Train/{0..42}/`, `Test/`, `Meta/` con sus respectivos CSVs.

- **Origen:** Stallkamp et al. (2011), *IJCNN*.
- **Tamaño:** ~50 000 imágenes, 43 clases.
- **Distribución:** 39 209 imágenes de entrenamiento + 12 630 de test oficial.

En este proyecto el dataset NO se incluye en git (pesa ~394 MB). El archivo
`training/dataset/GTSRB` es un *symlink* a la copia local en
`~/Downloads/GTSRB`. Para reproducirlo en otra máquina:

1. Bajar el dataset desde la fuente oficial o un mirror público (Kaggle, OpenSLR, etc.).
2. Colocar la carpeta en cualquier ruta del disco.
3. Crear el symlink: `ln -s /ruta/a/GTSRB training/dataset/GTSRB`.

## 2. Entrenamiento del modelo (CNN)

### 2.1 Entorno Python

Se usa un virtualenv con **Python 3.12** y el plugin `tensorflow-metal` para
aprovechar la GPU integrada de Apple Silicon (la del Mac del equipo es Apple M4):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install "tensorflow==2.16.*" "tensorflow-metal==1.2.0" "opencv-python<4.10" \
            scikit-learn matplotlib pandas pillow tqdm jupyter "nbconvert[webpdf]" playwright
playwright install chromium    # para exportar el notebook a PDF
```

> ⚠️ La combinación TF 2.16 + tensorflow-metal 1.2.0 es la única estable a la
> fecha del proyecto (junio 2026). TF 2.21 + tensorflow-metal 1.2.0 falla al
> cargar `libmetal_plugin.dylib`.

### 2.2 Reproducir el entrenamiento

```bash
source .venv/bin/activate
cd training
jupyter nbconvert --to notebook --execute --inplace gtsrb_cnn.ipynb
```

Tiempo aproximado: **~8 min en Apple M4** (21 épocas × 20s c/u + carga de datos).

### 2.3 Resultados obtenidos

| Métrica                       | Valor       |
|:------------------------------|:-----------:|
| Accuracy en validación (15%)  | **99.91%**  |
| Accuracy en test oficial GTSRB | **98.38%** |
| Loss en test oficial          | 0.0573      |
| Épocas hasta EarlyStopping    | 21          |
| Tamaño del modelo             | 16 MB       |

### 2.4 Arquitectura de la red

3 bloques convolucionales + cabeza densa, todo con BatchNormalization y Dropout:

```
Bloque 1:  Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)   # 32×32 → 16×16
Bloque 2:  Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)   # 16×16 → 8×8
Bloque 3:  Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25) # 8×8 → 4×4
Cabeza :   Flatten → Dense(512) → BN → Dropout(0.5) → Dense(43, softmax)
```

- Optimizador: **Adam** (lr inicial = 1e-3, ReduceLROnPlateau)
- Pérdida: `categorical_crossentropy`
- Callbacks: `EarlyStopping(patience=5)` + `ReduceLROnPlateau(patience=3, factor=0.5)`
- Batch size: 64, máximo 25 épocas

## 3. Integración con Webots

### 3.1 Apertura del mundo

1. Abrir Webots (versión R2023b o superior).
2. `File → Open World…` → seleccionar
   `SDC_webots/worlds/city_2025a_activity_3_1.wbt`.
3. Verificar que el controlador del BmwX5 esté en `vehicle` (debe estar por
   defecto). Si no, click derecho sobre el `BmwX5` → `Edit Controller`.

### 3.2 Controlador `vehicle.py`

El archivo `runtime.ini` fuerza a Webots a usar la Python del virtualenv del
proyecto, donde está instalado TensorFlow. La primera ejecución tarda
~5-10 segundos cargando el modelo `.keras`; las siguientes son instantáneas.

**Teclas:**

| Tecla   | Acción                              |
|:--------|:------------------------------------|
| ↑       | Acelerar (+5 km/h, máx 80)          |
| ↓       | Frenar (−5 km/h)                    |
| ←       | Girar a la izquierda                |
| →       | Girar a la derecha                  |
| `A`     | Screenshot de la cámara             |
| `P`     | Pausa/reanuda la detección CNN      |

### 3.3 ¿Qué hace el controlador?

Cada paso de simulación:

1. Captura el frame BGRA 256×128 de la cámara onboard.
2. Convierte a BGR y extrae una **ROI fija** del cuadrante superior-derecho del frame
   (x: 55%-99%, y: 5%-65% del frame total). Las señales del lado del conductor
   aparecen siempre ahí en este mundo.
3. Redimensiona la ROI a 32×32 y normaliza a `[0, 1]`.
4. Llama a `TrafficSignDetector.predict()` → CNN → softmax sobre 43 clases.
5. Si confianza ≥ **0.90**, imprime la señal en consola, la dibuja con bounding
   box verde sobre el frame, y la muestra en el display secundario del coche.
6. Al final del recorrido (al cerrar Webots) imprime el resumen de clases únicas
   detectadas.

### 3.4 Carril derecho

La posición inicial del `BmwX5` en `city_2025a_activity_3_1.wbt`
(`translation 5.60911 45.14 0.26702`, `rotation -π Z`) lo deja en el carril
derecho mirando hacia el lado del recorrido, **fuera de la línea amarilla central**.
Esto es necesario para que la ROI fija capte las señales del lado del conductor.

## 4. Validación del requisito 50% de señales

El mundo contiene 16 señales reales (5 Caution, 4 Order, 1 Stop, 1 Yield, 5 SpeedLimit).
La actividad pide detectar al menos el 50% (8 señales) durante un recorrido manual.

Durante el recorrido manual se imprime en consola cada vez que se detecta una clase
NUEVA con alta confianza. Al cerrar Webots se imprime el resumen final.

## 5. Entregables

- **Notebook ejecutado:** [training/gtsrb_cnn.ipynb](training/gtsrb_cnn.ipynb)
- **PDF del notebook:** [training/gtsrb_cnn.pdf](training/gtsrb_cnn.pdf)
- **Modelo entrenado:** [models/modelo_gtsrb.keras](models/modelo_gtsrb.keras)
- **Controlador Webots:** [controllers/vehicle/vehicle.py](controllers/vehicle/vehicle.py)
- **Video YouTube:** _por subir tras la validación en Webots_

## 6. Referencias

- Stallkamp, J., Schlipsing, M., Salmen, J. & Igel, C. (2011). *The German
  Traffic Sign Recognition Benchmark: A Multiclass Classification
  Competition*. The 2011 International Joint Conference on Neural
  Networks (IJCNN). https://www.researchgate.net/publication/224260296
- Ranjan, S. & Senthamilarasu, S. (2020). *Applied Deep Learning and Computer
  Vision for Self-Driving Cars*. Capítulo 7. Packt Publishing.
- Cyberbotics. *Webots Driver Library (Python)*. https://cyberbotics.com/doc/automobile/driver-library
