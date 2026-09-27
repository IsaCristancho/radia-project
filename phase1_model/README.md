# Fase 1 — Modelo base: clasificación de neumonía en radiografías de tórax

## Dataset

**Chest X-Ray Images (Pneumonia)** — Kaggle
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

~5.800 imágenes, 2 clases: `NORMAL` / `PNEUMONIA`. Ya viene dividido en train/val/test.

## Cómo ejecutarlo en local (macOS con Apple Silicon)

1. Instala `uv` si no lo tienes: `brew install uv`
2. Desde `phase1_model/`: `uv venv && source .venv/bin/activate`
3. `uv pip install -r requirements.txt`
4. Consigue tu token de la API de Kaggle (kaggle.com → perfil → Settings → API →
   "Create New Token"), guárdalo en `~/.kaggle/kaggle.json` y dale permisos:
   `chmod 600 ~/.kaggle/kaggle.json`
5. `uv pip install kaggle`
6. `kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p ./data --unzip`
7. `python train.py`

`train.py` detecta automáticamente tu GPU: usa **MPS** (el backend de aceleración de
Apple Silicon — M1 a M5) si está disponible, si no CUDA, si no CPU. No necesitas tocar
nada para que use tu GPU integrada.

## Alternativa: Kaggle Notebooks / Google Colab

Si prefieres no instalar nada en local, puedes correr esto igual en un notebook de
Kaggle (GPU gratuita, dataset ya alojado ahí) o en Colab. Solo cambia `DATA_DIR` en
`train.py` por la ruta correspondiente:
- Kaggle: `/kaggle/input/chest-xray-pneumonia/chest_xray`
- Colab: la ruta donde hayas descargado/montado el dataset

## Prácticas actuales de la industria que ya estás usando aquí

- **`uv`** en vez de `pip`/`conda` puro — es el gestor de entornos/paquetes que más
  está ganando terreno en equipos de ML en 2025-2026 por velocidad y reproducibilidad.
- **MPS (Apple Silicon)** — cada vez más gente entrena/prototipa en Mac; saber que
  existe y cómo detectarlo en código es relevante, no solo CUDA.
- **`weights_only=True`** al cargar checkpoints — práctica de seguridad recomendada
  actualmente por PyTorch (evita ejecutar código arbitrario al deserializar un modelo).
- Cuando lleguemos a fases posteriores, añadiremos **tracking de experimentos** (tipo
  Weights & Biases) y **contenedores Docker** para reproducibilidad — ambos muy
  esperados en entrevistas de MLOps/healthtech.

## Qué demuestra este código (para la entrevista)

- **Transfer learning** con ResNet18 preentrenada en ImageNet, congelando capas iniciales.
- **Manejo de desbalance de clases** (hay más casos de neumonía que normales en este
  dataset) vía `class_weight` ponderado en la loss, en vez de ignorarlo.
- **Métricas clínicas correctas**: no solo accuracy, sino sensibilidad (recall en la clase
  positiva — clave porque un falso negativo en clínica es mucho más grave que un falso
  positivo), especificidad, AUC-ROC y matriz de confusión.
- **Reproducibilidad**: seeds fijadas, split train/val/test respetado tal cual viene.

## Próximo paso (Fase 2)

El modelo entrenado aquí (`model.pt`) se reutiliza en `phase2_pipeline/` dentro de un
pipeline que recibe un DICOM real en vez de un .jpeg suelto.
