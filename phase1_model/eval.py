"""
eval.py
Carga un modelo ya entrenado (model.pt) y lo evalúa en el set de test, o corre
inferencia sobre una sola imagen. Esta función `predict_single_image` es la que
reutilizaremos en la Fase 2 dentro del pipeline DICOM.
"""

import torch

from dataset import eval_transforms, get_dataloaders
from model import build_model
from train import evaluate, DEVICE, DATA_DIR

CLASSES = ["NORMAL", "PNEUMONIA"]


def load_model(weights_path="model.pt", num_classes=2):
    model = build_model(num_classes=num_classes, freeze_backbone=True)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict_single_image(model, pil_image):
    """
    pil_image: PIL.Image ya en modo L o RGB (una radiografía).
    Devuelve (etiqueta_predicha, probabilidad_pneumonia).
    Esta es la función que la Fase 2 llamará tras extraer los píxeles de un DICOM.
    """
    tensor = eval_transforms(pil_image).unsqueeze(0).to(DEVICE)
    output = model(tensor)
    prob_pneumonia = torch.softmax(output, dim=1)[0, 1].item()
    pred_label = CLASSES[int(prob_pneumonia > 0.5)]
    return pred_label, prob_pneumonia


if __name__ == "__main__":
    _, _, test_loader, _, classes = get_dataloaders(DATA_DIR)
    model = load_model()
    evaluate(model, test_loader, classes)
